"""PoSTCache (arXiv 2610.25664): revisable caching with postdictive smoothing, as a sampler.

The model runs only at anchor steps. Between anchors a and b the sampler reuses the last velocity (the
paper's default forward proposal). At b the innovation r = u_b - u_a corrects every skipped step with the
relative-progress gain w_k = (k - a) / (b - a), and the interval is replayed from the stored anchor state x_a
with the corrected velocities, without extra model calls. u_b is reused at the replayed state (as in the paper).
With the reuse proposal the corrected velocities are the linear interpolation between u_a and u_b.

Anchors: dense prefix + final step + either a scheduler-aware spread or Posterior Variance Reduction (PVR),
a greedy choice on the cross-step covariance of full trajectories recorded in "calibrate" mode.
Euler update in flow form, deterministic (the replay needs an ODE path).
"""

import logging
import os

import torch

import comfy.samplers
import folder_paths
from comfy_api.latest import io


def profile_path(name):
    base = os.path.join(folder_paths.get_user_directory(), "flowlab", "postcache")
    os.makedirs(base, exist_ok=True)
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_.") or "default"
    return os.path.join(base, safe + ".pt")


def scheduler_weights(sigmas, power=0.5, clip=(0.5, 2.0)):
    """gamma_k: larger where the schedule is dense (Eq. 75-78)."""
    delta = (sigmas[1:] - sigmas[:-1]).abs().double()
    g = (delta.median() / (delta + 1e-12)) ** power
    g = g / g.mean()
    g = g.clamp(*clip)
    return g / g.mean()


def dense_prefix(budget):
    # about half the budget, as in the paper's tables (FLUX 13 -> 6, 16 -> 7; Wan 16 -> 8, 20 -> 10); on Moss
    # (30 steps, 12 calls) a prefix of 6 gave 30.7 dB against 27.4 dB with 5
    return max(1, budget // 2)


def spread_anchors(sigmas, budget, prefix):
    """Without calibration: remaining anchors at equal steps of the cumulative scheduler weight."""
    n = len(sigmas) - 1
    base = set(range(min(prefix, n))) | {n - 1}
    rest = budget - len(base)
    if rest <= 0:
        return sorted(base)
    g = scheduler_weights(sigmas)
    free = [k for k in range(n) if k not in base]
    cum = torch.cumsum(g[free], 0)
    for q in range(1, rest + 1):
        target = cum[-1] * q / (rest + 1)
        base.add(free[int(torch.searchsorted(cum, target))])
    return sorted(base)


def pvr_anchors(kernel, gamma, budget, prefix):
    """Greedy Posterior Variance Reduction: add the step that most reduces sum_t gamma_t V_t(A) (Eq. 72-74)."""
    n = kernel.shape[0]
    K = kernel.double() + 1e-9 * torch.eye(n, dtype=torch.float64) * kernel.diagonal().mean()
    A = sorted(set(range(min(prefix, n))) | {n - 1})
    while len(A) < min(budget, n):
        KAA_inv = torch.linalg.inv(K[A][:, A])
        C = K - K[:, A] @ KAA_inv @ K[A, :]  # conditional covariance
        V = C.diagonal().clamp_min(1e-30)
        gain = (gamma[:, None] * C ** 2).sum(0) / V
        gain[A] = -1
        A = sorted(A + [int(torch.argmax(gain))])
    return A


class TrajectoryRecorder:
    """Running statistics of full trajectories for the PVR kernel: sum_n <u_i, u_j> and sum_n u_i (subsampled)."""

    def __init__(self, path, subsample=4):
        self.path = path
        self.subsample = subsample

    def add(self, us):
        u = torch.stack([v[..., ::self.subsample, ::self.subsample].flatten(1).float().cpu() for v in us])  # (T, B, D)
        state = torch.load(self.path) if os.path.exists(self.path) else None
        T = u.shape[0]
        if state is not None and state["outer"].shape[0] != T:
            logging.warning(f"FlowLab PoSTCache: profile {self.path} was recorded with {state['outer'].shape[0]} steps, starting over for {T}.")
            state = None
        if state is None:
            state = {"outer": torch.zeros(T, T, dtype=torch.float64), "sum": torch.zeros(T, u.shape[2], dtype=torch.float64), "count": 0, "dim": u.shape[2]}
        for b in range(u.shape[1]):
            ub = u[:, b].double()
            state["outer"] += ub @ ub.T
            state["sum"] += ub
            state["count"] += 1
        torch.save(state, self.path)
        logging.info(f"FlowLab PoSTCache: profile {os.path.basename(self.path)} now has {state['count']} trajectories.")

    @staticmethod
    def kernel(path):
        state = torch.load(path)
        n, d = state["count"], state["dim"]
        mean = state["sum"] / n
        return (state["outer"] / n - mean @ mean.T) / d, n


def make_sampler_function(mode, budget, prefix, profile):
    def sample_postcache(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        n = len(sigmas) - 1
        pre = dense_prefix(budget) if prefix < 0 else prefix
        sig = sigmas.float().cpu()
        if mode == "calibrate":
            anchors = list(range(n))
        elif mode == "pvr":
            path = profile_path(profile)
            if not os.path.exists(path):
                raise FileNotFoundError(f"PoSTCache: no calibration profile '{profile}'. Run some images with mode 'calibrate' first.")
            kernel, count = TrajectoryRecorder.kernel(path)
            if kernel.shape[0] != n:
                raise ValueError(f"PoSTCache: profile '{profile}' was calibrated with {kernel.shape[0]} steps, this run has {n}.")
            anchors = pvr_anchors(kernel, scheduler_weights(sig), budget, pre)
        else:
            anchors = spread_anchors(sig, budget, pre)
        is_anchor = set(anchors)
        logging.info(f"FlowLab PoSTCache: {len(anchors)}/{n} model calls ({n / len(anchors):.2f}x), anchors {anchors}")

        trajectory = []
        a, x_a, u_a = None, None, None
        for i in range(n):
            sigma = sigmas[i]
            if i in is_anchor:
                denoised = model(x, sigma * s_in, **extra_args)
                u = (x - denoised) / sigma
                if mode == "calibrate":
                    trajectory.append(u)
                if a is not None and i - a > 1:
                    # postdictive smoothing: velocities between the anchors move linearly from u_a to u, then replay
                    x = x_a
                    for k in range(a, i):
                        w = (k - a) / (i - a)
                        x = x + (sigmas[k + 1] - sigmas[k]) * (u_a + w * (u - u_a))
                a, x_a, u_a = i, x, u
            else:
                u = u_a
            x_next = x + (sigmas[i + 1] - sigma) * u
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": x - sigma * u})
            x = x_next
        if mode == "calibrate":
            TrajectoryRecorder(profile_path(profile)).add(trajectory)
        return x
    return sample_postcache


class PoSTCacheSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowLabPoSTCacheSampler",
            display_name="PoSTCache Sampler",
            category="FlowLab/sampling",
            description="Euler sampler that calls the model only at anchor steps and revises the skipped steps afterwards "
                        "from the next anchor (arXiv 2610.25664). Use with SamplerCustomAdvanced; pays off from about 20 steps.",
            inputs=[
                io.Combo.Input("mode", options=["spread", "pvr", "calibrate"], default="spread",
                               tooltip="spread: anchors by scheduler weight, no calibration. pvr: anchors from a calibration profile (paper method). calibrate: full run that adds this trajectory to the profile (the paper uses 10)."),
                io.Int.Input("budget", default=14, min=2, max=1000, tooltip="Model calls per run, including the dense prefix and the last step. Paper (50 steps): 13-16."),
                io.Int.Input("dense_prefix", default=-1, min=-1, max=1000, tooltip="Steps computed at the start; -1 = half the budget, as in the paper."),
                io.String.Input("profile", default="default", tooltip="Name of the calibration profile (stored in ComfyUI/user/flowlab/postcache). Use one profile per model and step count."),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, mode, budget, dense_prefix, profile) -> io.NodeOutput:
        return io.NodeOutput(comfy.samplers.KSAMPLER(make_sampler_function(mode, budget, dense_prefix, profile)))


NODES = [PoSTCacheSampler]
