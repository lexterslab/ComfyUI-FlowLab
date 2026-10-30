"""CacheScope (arXiv 2610.12530): probe-normalized scheduling of exact computation for a residual block cache.

Every step runs the first transformer block as a probe (x = its input tokens, y = its output). The signal
  s_t = d(y_t, y_t-1) / (d(x_t, x_t-1) + eps),   d(u, v) = mean|u - v| / (mean|v| + eps)
is the network's response per unit of input movement. It is accumulated since the last exact step; once the
sum reaches tau the step is computed exactly, otherwise the remaining blocks are skipped and their residual is
reconstructed with the bounded first-order rule of DiCache: r = r2 + g * (r2 - r1), g = clip(d(y, y2) / d(y2, y1), 1, 1.5).
Warm-up and tail steps are always exact (the fixed base policy). Separate state per CFG branch (per batch layout).

Hooks every transformer block of Krea 2 or Moss-Image through object patches of their forward.
"""

import logging

import comfy.ldm.krea2.model
import comfy.ldm.moss_image.model
import comfy.patcher_extension
from comfy_api.latest import io

from .common import sampling_progress

CTX = "flowlab_cachescope"
EPS = 1e-6


def rel_change(u, v):
    return float((u - v).abs().mean() / (v.abs().mean() + EPS))


class Branch:
    def __init__(self):
        self.x = self.y = None  # previous probe input / output
        self.acc = 0.0
        self.residuals = []  # (probe output, residual) of the last two exact steps


class CacheScopeState:
    def __init__(self, threshold, warmup, tail, gamma_max):
        self.threshold, self.warmup, self.tail, self.gamma_max = threshold, warmup, tail, gamma_max
        self.reset()

    def reset(self):
        self.branches = {}
        self.exact = {}
        self.steps = 0

    def summary(self):
        for key, steps in self.exact.items():
            logging.info(f"FlowLab CacheScope branch {key}: {len(steps)}/{self.steps} exact ({self.steps / max(len(steps), 1):.2f}x), exact steps {steps}")


def forward_wrapper(executor, *args, **kwargs):
    opts = args[-1] if isinstance(args[-1], dict) else kwargs["transformer_options"]
    state: CacheScopeState = opts[CTX + "_state"]
    progress = sampling_progress(opts, opts["sigmas"])
    if progress is None:
        return executor(*args, **kwargs)
    j, n = progress
    state.steps = n
    key = tuple(opts.get("cond_or_uncond", [])) + tuple(args[0].shape)
    branch = state.branches.setdefault(key, Branch())
    base_exact = j < max(2, round(state.warmup * n)) or j >= n - state.tail
    ctx = {"branch": branch, "base_exact": base_exact, "exact": True, "y": None, "step": j, "key": key[:len(opts.get("cond_or_uncond", []))]}
    opts = {**opts, CTX: ctx}
    if isinstance(args[-1], dict):
        args = args[:-1] + (opts,)
    else:
        kwargs = {**kwargs, "transformer_options": opts}
    out = executor(*args, **kwargs)
    if ctx["exact"]:
        state.exact.setdefault(str(ctx["key"]), []).append(j)
    return out


def make_block_forward(state, orig, role):
    def forward(*args, **kwargs):
        opts = kwargs.get("transformer_options")
        if opts is None and len(args) > 3 and isinstance(args[3], dict):
            opts = args[3]
        ctx = (opts or {}).get(CTX)
        if ctx is None or kwargs.get("kv_out") is not None:
            return orig(*args, **kwargs)
        branch: Branch = ctx["branch"]
        if role == "first":
            x_in = args[0].clone()  # Moss updates the token tensor in place
            y = orig(*args, **kwargs)
            y_saved = y.clone()
            s = 0.0
            if branch.x is not None and branch.x.shape == x_in.shape:
                s = rel_change(y_saved, branch.y) / (rel_change(x_in, branch.x) + EPS)
            branch.x, branch.y = x_in, y_saved
            exact = ctx["base_exact"] or len(branch.residuals) == 0 or branch.acc + s >= state.threshold
            branch.acc = 0.0 if exact else branch.acc + s
            ctx["exact"], ctx["y"] = exact, y_saved
            return y
        if not ctx["exact"]:
            if role == "middle":
                return args[0]
            # last block: reconstruct the residual of blocks 2..L from the two most recent exact steps
            y = ctx["y"]
            y2, r2 = branch.residuals[-1]
            if len(branch.residuals) == 2:
                y1, r1 = branch.residuals[0]
                gamma = min(max(rel_change(y, y2) / (rel_change(y2, y1) + EPS), 1.0), state.gamma_max)
                return y + r2 + gamma * (r2 - r1)
            return y + r2
        out = orig(*args, **kwargs)
        if role == "last":
            branch.residuals.append((ctx["y"], out - ctx["y"]))
            del branch.residuals[:-2]
        return out
    return forward


def outer_sample_wrapper(executor, *args, **kwargs):
    state: CacheScopeState = executor.class_obj.model_options["transformer_options"][CTX + "_state"]
    state.reset()
    try:
        return executor(*args, **kwargs)
    finally:
        state.summary()
        state.reset()


class CacheScope(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowLabCacheScope",
            display_name="CacheScope (probe-scheduled block cache)",
            category="FlowLab/cache",
            description="Runs the first transformer block every step as a probe and skips the remaining blocks until the "
                        "accumulated response-per-movement reaches the threshold (arXiv 2610.12530). Krea 2 and Moss-Image only. "
                        "Skipped blocks still stream their weights when the model is offloaded.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("threshold", default=3.0, min=0.0, max=100.0, step=0.05, tooltip="tau: higher = fewer exact steps. Calibrate per model to the wanted speed (paper, FLUX: 4.6 for 68% skipped, 6.4 for 74%)."),
                io.Float.Input("warmup_percent", default=0.2, min=0.0, max=1.0, step=0.01, tooltip="Share of steps at the start that are always exact (DiCache prefix ratio 0.15-0.2)."),
                io.Int.Input("tail_steps", default=1, min=0, max=100, tooltip="Steps at the end that are always exact."),
                io.Float.Input("gamma_max", default=1.5, min=1.0, max=3.0, step=0.05, tooltip="Upper bound of the first-order extrapolation (paper: 1.5)."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, threshold, warmup_percent, tail_steps, gamma_max) -> io.NodeOutput:
        m = model.clone()
        dm = m.model.diffusion_model
        if isinstance(dm, comfy.ldm.krea2.model.SingleStreamDiT):
            name, blocks = "blocks", dm.blocks
        elif isinstance(dm, comfy.ldm.moss_image.model.MOSSImage):
            name, blocks = "layers", dm.layers
        else:
            raise ValueError(f"FlowLab CacheScope: {type(dm).__name__} is not supported (only Krea 2 and Moss-Image).")
        state = CacheScopeState(threshold, warmup_percent, tail_steps, gamma_max)
        m.model_options["transformer_options"][CTX + "_state"] = state
        last = len(blocks) - 1
        for i, block in enumerate(blocks):
            role = "first" if i == 0 else "last" if i == last else "middle"
            # bind the class method: block.forward may still be another patch's function while that model is loaded
            m.add_object_patch(f"diffusion_model.{name}.{i}.forward", make_block_forward(state, type(block).forward.__get__(block), role))
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, CTX, forward_wrapper)
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, CTX, outer_sample_wrapper)
        return io.NodeOutput(m)


NODES = [CacheScope]
