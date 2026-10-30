"""Step caches that forecast the guided velocity at skipped sampling steps.

Both methods sit around the guided prediction (PREDICT_NOISE, after CFG), so they work for any model and
skip the conditional and unconditional passes together. On a computed step the velocity Y = (x - D) / sigma is
recorded; on a skipped step the forecast Y is turned back into a denoised estimate D = x - sigma * Y.

SpecKCast (arXiv 2610.26837): orthonormal 2D DCT per channel, 7 radial frequency bands, a Kalman filter with a
damped local-linear trend per coefficient on a band-specific warped log-SNR time, and a risk-based scheduler.
BeziCast (arXiv 2610.16448): ridge-fitted cubic Bezier curve over the last full steps with a simplex guardrail.
The forecast target here is the guided velocity, not BeziCast's representation before the final layer.

Steps are matched to the sampler's sigma schedule, so these caches need samplers with one model call per step
(euler, er_sde, dpmpp_2m, ...). Other calls are computed normally.
"""

import json
import logging
import math

import torch

import comfy.patcher_extension
from comfy_api.latest import io

KEY = "flowlab_stepcache"

# Signal power p_l of the clean latent per frequency band, measured with the "measure band powers" mode
# (x0 = x - sigma * Y over the second half of 30-step euler runs, mean of 4 prompts at 1024x1024, model space).
BAND_POWERS = {
    "Krea2": [275.5, 31.1, 5.78, 1.324, 0.2396, 0.0424, 0.0169],
    "MOSSImage": [159.5, 57.3, 13.54, 3.767, 1.297, 0.412, 0.1783],
}
BAND_EDGES = (0.0, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)


def dct_matrix(n, device):
    k = torch.arange(n, device=device, dtype=torch.float64)[:, None]
    i = torch.arange(n, device=device, dtype=torch.float64)[None, :]
    m = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    m[0] /= math.sqrt(2.0)
    return m.float()


class Spectrum:
    """Channel-wise orthonormal 2D DCT and the radial band of every coefficient."""

    def __init__(self, h, w, device):
        self.mh, self.mw = dct_matrix(h, device), dct_matrix(w, device)
        a = torch.arange(h, device=device, dtype=torch.float32)[:, None] / h
        b = torch.arange(w, device=device, dtype=torch.float32)[None, :] / w
        r = torch.sqrt(a ** 2 + b ** 2) / math.sqrt(2.0)
        self.band = torch.bucketize(r, torch.tensor(BAND_EDGES[1:-1], device=device), right=True)
        self.bands = len(BAND_EDGES) - 1
        self.counts = torch.bincount(self.band.flatten(), minlength=self.bands).float()

    def forward(self, y):
        return self.mh @ y @ self.mw.T

    def inverse(self, z):
        return self.mh.T @ z @ self.mw

    def band_mean(self, t):
        """Mean of t (B, C, H, W) over batch, channels and the coefficients of each band -> (bands,)."""
        flat = t.mean(dim=(0, 1)).flatten()
        sums = torch.zeros(self.bands, device=t.device).index_add_(0, self.band.flatten(), flat)
        return sums / self.counts

    def per_coefficient(self, values):
        """(bands,) -> (H, W) map."""
        return values.to(self.band.device)[self.band]


def step_of(sigmas, sigma):
    """Index of sigma in the schedule, or None if the call is not on a scheduled step (multi-stage solvers)."""
    j = int(torch.argmin((sigmas[:-1] - sigma).abs()))
    if abs(float(sigmas[j]) - sigma) > 1e-4 * max(1.0, abs(sigma)):
        return None
    return j


class StepCache:
    """Wrapper plumbing shared by the forecasting caches."""
    name = "StepCache"

    def __init__(self):
        self.sigmas = None

    def start(self, sigmas, y):
        self.sigmas = sigmas
        self.steps = len(sigmas) - 1
        self.computed = []
        self.offgrid = 0

    def summary(self):
        if self.sigmas is None:
            return
        n = len(self.computed)
        logging.info(f"FlowLab {self.name}: {n}/{self.steps} model calls ({self.steps / max(n, 1):.2f}x), computed steps {self.computed}"
                     + (f", {self.offgrid} off-schedule calls computed normally" if self.offgrid else ""))


def predict_noise_wrapper(executor, x, timestep, model_options={}, seed=None):
    opts = model_options.get("transformer_options", {})
    cache: StepCache = opts[KEY]
    sigmas = opts.get("sample_sigmas")
    sigma = float(timestep.flatten()[0])
    if sigmas is None or sigma <= 0:
        return executor(x, timestep, model_options, seed)
    sigmas = sigmas.float().cpu()
    if cache.sigmas is None or len(cache.sigmas) != len(sigmas) or not torch.equal(cache.sigmas, sigmas):
        cache.summary()
        cache.start(sigmas, x)
    j = step_of(sigmas, sigma)
    if j is None:
        cache.offgrid += 1
        return executor(x, timestep, model_options, seed)
    if cache.compute(j):
        denoised = executor(x, timestep, model_options, seed)
        cache.computed.append(j)
        cache.observe(j, (x - denoised).float() / sigma, x)
        return denoised
    return x - sigma * cache.forecast(j).to(x.dtype)


def outer_sample_wrapper(executor, *args, **kwargs):
    cache: StepCache = executor.class_obj.model_options["transformer_options"][KEY]
    cache.sigmas = None
    try:
        return executor(*args, **kwargs)
    finally:
        cache.summary()
        cache.release()


def install(model, cache):
    m = model.clone()
    m.model_options["transformer_options"][KEY] = cache
    m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREDICT_NOISE, KEY, predict_noise_wrapper)
    m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, KEY, outer_sample_wrapper)
    return m


class SpecKCastCache(StepCache):
    name = "SpecKCast"

    def __init__(self, mode, budget, threshold, band_powers, tau=0.1, t_w=2.0, beta=0.6, warmup=3, max_skip=10, rho_sc=0.9):
        super().__init__()
        self.mode = mode
        self.budget = budget
        self.threshold = threshold
        self.band_powers = band_powers
        self.tau, self.t_w, self.beta = tau, t_w, beta
        self.warmup, self.max_skip, self.rho_sc = warmup, max_skip, rho_sc
        self.release()

    def release(self):
        self.spec = self.m = self.nu = None
        self.measured = []

    def start(self, sigmas, x):
        super().start(sigmas, x)
        n = self.steps
        self.spec = Spectrum(x.shape[-2], x.shape[-1], x.device)
        bands = self.spec.bands
        sig = sigmas[:-1].double().clamp(1e-4, 1 - 1e-4)
        lam = 2 * torch.log((1 - sig) / sig)  # log-SNR of every step
        if self.mode == "measure band powers" or self.band_powers is None:
            p = torch.ones(bands, dtype=torch.float64)
            t = torch.sigmoid(lam / self.t_w)[None].expand(bands, n)  # calibration-free: raw log-SNR for every band
        else:
            p = torch.tensor(self.band_powers, dtype=torch.float64)
            t = torch.sigmoid((lam[None] + torch.log(p)[:, None]) / self.t_w)
        span = (t[:, -1] - t[:, 0]).clamp_min(1e-12)
        self.s = 2 * (t - t[:, :1]) / span[:, None] - 1  # (bands, steps), every band spans [-1, 1]
        self.p = p
        self.omega = torch.exp(-self.beta * torch.arange(n, dtype=torch.float64) / max(n - 1, 1))
        # shared per-band covariances: mean filter (p11, p12, p22) and the risk sidecar
        self.P = torch.zeros(3, bands, dtype=torch.float64)
        self.S = torch.zeros(3, bands, dtype=torch.float64)
        self.kappa = torch.ones(bands, dtype=torch.float64)
        self.next_compute = 0
        self.last = None  # last step the state was advanced to
        self.m = self.nu = None
        if self.mode == "fixed budget":
            self.schedule = self.fixed_schedule(n, self.budget)

    def fixed_schedule(self, n, budget):
        """Warm-up steps, then gaps that grow linearly (j_r = floor(alpha r (r + 1) / 2)), then the last step."""
        base = set(range(min(self.warmup, n))) | {n - 1}
        if budget >= n:
            return set(range(n))
        lo, hi = 1e-3, float(n * n)
        best = base
        for _ in range(60):
            alpha = math.sqrt(lo * hi)
            steps = base | {int(alpha * r * (r + 1) / 2) for r in range(1, n + 1) if int(alpha * r * (r + 1) / 2) < n - 1}
            if len(steps) > budget:
                lo = alpha
            else:
                hi = alpha
                best = steps
        return best

    def delta(self, j):
        return self.s[:, j] - self.s[:, j - 1]

    def advance(self, j):
        """Kalman prediction from j - 1 to j: m += delta * nu, nu *= rho; covariances grow (Eq. 10/11, 46)."""
        d = self.delta(j)
        rho = torch.exp(-d / self.tau)
        self.m += self.spec.per_coefficient(d.float()) * self.nu
        self.nu *= self.spec.per_coefficient(rho.float())
        p11, p12, p22 = self.P
        self.P = torch.stack([p11 + 2 * d * p12 + d * d * p22 + d ** 3 / 3,
                              rho * (p12 + d * p22) + d * d / 2,
                              rho * rho * p22 + d])
        self.S = self.sidecar_step(self.S, d)

    def sidecar_step(self, S, d):
        s11, s12, s22 = S
        return torch.stack([s11 + 2 * d * s12 + d * d * s22 + d ** 3 / 3,
                            self.rho_sc * (s12 + d * s22) + d * d / 2,
                            self.rho_sc ** 2 * s22 + d])

    def risk(self, S, j):
        # The paper weights bands by LPIPS-calibrated perceptual weights (default: coefficient count). Without that
        # calibration the count lets the many, late-active high-frequency coefficients dominate and the scheduler
        # skips far too early; weighting by count * power turns the relative errors into the absolute latent error.
        weights = self.spec.counts.double().cpu() * self.p
        return float(self.omega[j] * (self.kappa * weights * S[0]).sum() / weights.sum())

    def plan_next(self, j):
        """After a computed step: roll the sidecar forward and place the next computation at the first risk crossing."""
        n = self.steps
        if j + 1 < self.warmup:
            return j + 1
        S = self.S.clone()
        for k in range(1, self.max_skip + 1):
            t = j + k
            if t >= n - 1:
                return n - 1
            S = self.sidecar_step(S, self.delta(t))
            if self.risk(S, t) > self.threshold:
                return t
        return min(j + self.max_skip + 1, n - 1)

    def compute(self, j):
        if self.mode == "measure band powers":
            return True
        if self.mode == "fixed budget":
            return j in self.schedule
        return j >= self.next_compute or j == self.steps - 1

    def catch_up(self, j):
        if self.last is not None:
            for t in range(self.last + 1, j + 1):
                self.advance(t)
        self.last = j

    def observe(self, j, y, x):
        z = self.spec.forward(y)
        if self.mode == "measure band powers" and j >= self.steps // 2:
            x0 = x.float() - float(self.sigmas[j]) * y
            self.measured.append(self.spec.band_mean(self.spec.forward(x0) ** 2).double().cpu())
        if self.m is None:
            self.m, self.nu = z.clone(), torch.zeros_like(z)
            self.P = torch.tensor([[0.0] * self.spec.bands, [0.0] * self.spec.bands, [100.0] * self.spec.bands], dtype=torch.float64)
            self.last = j
            self.next_compute = j + 1
            return
        self.catch_up(j)
        p11, p12, p22 = self.P
        innovation = z - self.m
        # exact observation (R = 0): the value snaps to the observation, the trend moves by p12 / p11 of the innovation
        gain = torch.where(p11 > 0, p12 / p11.clamp_min(1e-30), torch.zeros_like(p11))
        self.nu += self.spec.per_coefficient(gain.float()) * innovation
        self.m = z.clone()
        self.P = torch.stack([torch.zeros_like(p11), torch.zeros_like(p11), torch.where(p11 > 0, p22 - p12 * p12 / p11.clamp_min(1e-30), p22)])
        # sidecar: compare the observed forecast error with the predicted variance, then reset (Eq. 47/48)
        s11, s12, s22 = self.S
        if j > 0:
            observed = self.spec.band_mean(innovation ** 2).double().cpu() / self.p
            ratio = torch.where(s11 > 0, observed / s11.clamp_min(1e-30), torch.ones_like(s11))
            self.kappa = (0.8 * self.kappa + 0.2 * ratio).clamp(0.1, 10.0)
        self.S = torch.stack([torch.zeros_like(s11), torch.zeros_like(s11), torch.where(s11 > 0, s22 - s12 * s12 / s11.clamp_min(1e-30), s22)])
        if self.mode == "adaptive":
            self.next_compute = self.plan_next(j)

    def forecast(self, j):
        self.catch_up(j)
        return self.spec.inverse(self.m)

    def summary(self):
        super().summary()
        if self.measured:
            p = torch.stack(self.measured).mean(0)
            logging.info("FlowLab SpecKCast band powers: " + json.dumps([float(f"{v:.6g}") for v in p.tolist()]))


def project_simplex(w):
    """Euclidean projection of a vector onto the probability simplex."""
    u, _ = torch.sort(w, descending=True)
    css = torch.cumsum(u, 0) - 1
    idx = torch.arange(1, len(w) + 1, dtype=w.dtype)
    cond = u - css / idx > 0
    r = int(torch.nonzero(cond)[-1]) + 1
    theta = css[r - 1] / r
    return (w - theta).clamp_min(0)


class BeziCastCache(StepCache):
    name = "BeziCast"

    def __init__(self, warmup, period, history, ridge, u_max, guard_max, guard_threshold, guard_width, degree=3):
        super().__init__()
        self.warmup, self.period, self.history = warmup, period, history
        self.ridge, self.u_max = ridge, u_max
        self.guard_max, self.guard_threshold, self.guard_width = guard_max, guard_threshold, guard_width
        self.degree = degree
        self.release()

    def release(self):
        self.anchors = []

    def start(self, sigmas, x):
        super().start(sigmas, x)
        self.anchors = []
        n = self.steps
        self.full = set(range(min(self.warmup, n))) | {s for s in range(self.warmup, n) if (s - self.warmup + 1) % self.period == 0} | {n - 1}

    def compute(self, j):
        return j in self.full or len(self.anchors) < self.degree + 1

    def observe(self, j, y, x):
        self.anchors.append((1.0 - float(self.sigmas[j]), y))
        del self.anchors[:-self.history]

    def bernstein(self, u):
        n = self.degree
        return torch.stack([math.comb(n, k) * u ** k * (1 - u) ** (n - k) for k in range(n + 1)], dim=-1)

    def weights(self, taus, tau):
        t0, t1 = taus[0], taus[-1]
        u = (torch.tensor(taus, dtype=torch.float64) - t0) / (t1 - t0 + 1e-8)
        us = torch.tensor(min((tau - t0) / (t1 - t0 + 1e-8), self.u_max), dtype=torch.float64)
        A = self.bernstein(u)
        w = self.bernstein(us) @ torch.linalg.solve(A.T @ A + self.ridge * torch.eye(A.shape[1], dtype=torch.float64), A.T)
        risk = abs(float(w.sum()) - 1) + (float(w.abs().sum()) - abs(float(w.sum())))
        beta = self.guard_max * min(max((risk - self.guard_threshold) / self.guard_width, 0.0), 1.0)
        if beta > 0:
            w = (1 - beta) * w + beta * project_simplex(w)
        return w

    def forecast(self, j):
        taus = [a[0] for a in self.anchors]
        w = self.weights(taus, 1.0 - float(self.sigmas[j]))
        out = torch.zeros_like(self.anchors[0][1])
        for wk, (_, y) in zip(w.tolist(), self.anchors):
            out.add_(y, alpha=wk)
        return out


class SpecKCast(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowLabSpecKCast",
            display_name="SpecKCast (Spectral Kalman Forecast Cache)",
            category="FlowLab/cache",
            description="Skips model calls and forecasts the guided velocity per DCT frequency band with Kalman filters "
                        "(arXiv 2610.26837). Model-agnostic. Pays off from about 20 steps; use a sampler with one model call per step.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("mode", options=["fixed budget", "adaptive", "measure band powers"], default="fixed budget",
                               tooltip="fixed budget: a fixed number of model calls with growing gaps (best in tests). adaptive (experimental): compute when the predicted forecast risk exceeds the threshold; the paper's calibrated risk tables are not published, so it allocates calls worse than the fixed plan. measure band powers: compute every step and log the band powers for this model."),
                io.Float.Input("threshold", default=0.01, min=0.0, max=10.0, step=0.0005, tooltip="adaptive: risk threshold (relative squared error). Higher = fewer model calls."),
                io.Int.Input("budget", default=12, min=2, max=1000, tooltip="fixed budget: model calls per run, including the warm-up steps and the last step."),
                io.Int.Input("warmup", default=6, min=2, max=100, tooltip="Steps always computed at the start. Paper: 3; on Moss (30 steps, 12 calls) 6 gave 27.9 dB instead of 23.9 dB, because the layout is decided early."),
                io.String.Input("band_powers", default="auto", tooltip="'auto' uses the built-in values for Krea 2 and Moss-Image, 'none' uses raw log-SNR for all bands, or a JSON list of 7 values from 'measure band powers'."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, mode, threshold, budget, warmup, band_powers) -> io.NodeOutput:
        text = band_powers.strip().lower()
        if text == "auto":
            powers = BAND_POWERS.get(type(model.model).__name__)
        elif text == "none":
            powers = None
        else:
            powers = json.loads(band_powers)
        return io.NodeOutput(install(model, SpecKCastCache(mode, budget, threshold, powers, warmup=warmup)))


class BeziCast(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowLabBeziCast",
            display_name="BeziCast (Bezier Forecast Cache)",
            category="FlowLab/cache",
            description="Computes the first steps and then every N-th step; forecasts the guided velocity in between with a "
                        "ridge-fitted cubic Bezier curve and a simplex guardrail (arXiv 2610.16448). Model-agnostic.",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("warmup", default=6, min=4, max=100, tooltip="Steps computed at the start (paper: 6)."),
                io.Int.Input("period", default=4, min=1, max=20, tooltip="After warm-up one computed step every N steps (paper: 4-6)."),
                io.Int.Input("history", default=6, min=4, max=20, tooltip="Computed steps the curve is fitted to (paper: 6)."),
                io.Float.Input("ridge", default=0.5, min=0.0, max=10.0, step=0.05, tooltip="Tikhonov strength lambda (paper: 0.5)."),
                io.Float.Input("u_max", default=1.5, min=1.0, max=3.0, step=0.05, tooltip="Cap on the extrapolation coordinate (not given in the paper)."),
                io.Float.Input("guard_max", default=1.0, min=0.0, max=1.0, step=0.05, tooltip="Maximum pull toward the convex combination of cached outputs (not given in the paper)."),
                io.Float.Input("guard_threshold", default=0.5, min=0.0, max=10.0, step=0.05, tooltip="Risk at which the guardrail starts (not given in the paper)."),
                io.Float.Input("guard_width", default=1.0, min=0.01, max=10.0, step=0.05, tooltip="Risk range over which the guardrail ramps up (not given in the paper)."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, warmup, period, history, ridge, u_max, guard_max, guard_threshold, guard_width) -> io.NodeOutput:
        return io.NodeOutput(install(model, BeziCastCache(warmup, period, history, ridge, u_max, guard_max, guard_threshold, guard_width)))


NODES = [SpecKCast, BeziCast]
