"""TrajCFG (arXiv 2610.14153): token-wise CFG scale from the direction changes of the guided velocity.

For each latent site the cosine q between the guided velocities of the two most recent steps sets the scale
  g(q) = s_min + (s_max - s_min) * clip((1 - q) / (1 - rho_lo), 0, 1)
so sites whose direction still changes get stronger guidance and settled sites weaker guidance. From the
fourth step on, a stride-2 comparison (angle rescaled to the adjacent interval) is mixed in with weight 0.2.
The first two steps use the plain CFG scale. No extra model calls.

The guided velocity is (x - denoised) / sigma; the cosine ignores the positive factor 1 / sigma, so the
history stores x - denoised directly.
"""

import math

import torch

import comfy.patcher_extension
from comfy_api.latest import io

from .common import in_window, patch_size

STATE = "flowlab_trajcfg"


class TrajCFGState:
    def __init__(self, s_min_delta, s_max_delta, rho_lo, long_weight, start, end, patch):
        self.s_min_delta = s_min_delta
        self.s_max_delta = s_max_delta
        self.rho_lo = rho_lo
        self.long_weight = long_weight
        self.start = start
        self.end = end
        self.patch = patch
        self.reset()

    def reset(self):
        self.history = []  # (sigma, guided x - denoised) of the most recent completed steps
        return self

    def sites(self, t):
        """(B, C, H, W) -> (B, sites, features); a site is one transformer token (patch x patch latents)."""
        p = self.patch
        b, c, h, w = t.shape
        if p > 1:
            t = torch.nn.functional.pad(t, (0, (-w) % p, 0, (-h) % p))
            t = t.unfold(2, p, p).unfold(3, p, p)
            return t.permute(0, 2, 3, 1, 4, 5).reshape(b, -1, c * p * p)
        return t.flatten(2).transpose(1, 2)

    def to_map(self, s, shape):
        """Per-site scale (B, sites) -> broadcastable (B, 1, H, W)."""
        b, _, h, w = shape
        p = self.patch
        s = s.view(b, 1, math.ceil(h / p), math.ceil(w / p))
        if p > 1:
            s = s.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)[:, :, :h, :w]
        return s

    def cosine(self, a, b):
        a, b = self.sites(a.float()), self.sites(b.float())
        floor = 1e-6 * max(float(a.norm(dim=-1).mean()), float(b.norm(dim=-1).mean()), 1e-12)
        den = a.norm(dim=-1).clamp_min(floor) * b.norm(dim=-1).clamp_min(floor)
        return ((a * b).sum(-1) / den).clamp(-1.0, 1.0)

    def scale_map(self, q, s_base):
        s_min, s_max = s_base + self.s_min_delta, s_base + self.s_max_delta
        return s_min + (s_max - s_min) * ((1.0 - q) / (1.0 - self.rho_lo)).clamp(0.0, 1.0)


def cfg_function(args):
    x = args["input"]
    cond, uncond = args["cond"], args["uncond"]  # x - denoised of each branch
    s_base = args["cond_scale"]
    sigma = args["sigma"]
    opts = args["model_options"]["transformer_options"]
    state: TrajCFGState = opts[STATE]
    if args.get("input_uncond") is None:
        return cond  # CFG 1 skips the unconditional branch, nothing to adapt
    s = float(sigma.flatten()[0])
    if state.history and s >= state.history[-1][0]:
        state.reset()  # a new sampling run (or a second pass) started

    h = state.history
    if len(h) >= 2 and in_window(opts, sigma, state.start, state.end):
        q1 = state.cosine(h[-2][1], h[-1][1])
        scale = state.scale_map(q1, s_base)
        if len(h) >= 3 and state.long_weight > 0:
            d1, d2 = abs(h[-1][0] - h[-2][0]), abs(h[-1][0] - h[-3][0])
            q2 = state.cosine(h[-3][1], h[-1][1])
            q2 = torch.cos((d1 / d2) * torch.arccos(q2)) if d2 > 0 else q2
            scale = (1 - state.long_weight) * scale + state.long_weight * state.scale_map(q2, s_base)
        scale = state.to_map(scale, x.shape).to(cond.dtype)
        guided = uncond + scale * (cond - uncond)
    else:
        guided = uncond + s_base * (cond - uncond)

    h.append((s, guided.detach()))
    del h[:-3]
    return guided


def outer_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    guider.model_options["transformer_options"][STATE].reset()
    try:
        return executor(*args, **kwargs)
    finally:
        guider.model_options["transformer_options"][STATE].reset()


class TrajectoryCFG(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowLabTrajCFG",
            display_name="Trajectory-Adaptive CFG (TrajCFG)",
            category="FlowLab/guidance",
            description="Token-wise CFG scale from how much each token's guided velocity still turns between steps "
                        "(arXiv 2610.14153). Uses the KSampler cfg as base scale; needs cfg > 1. No extra model calls.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("min_offset", default=-1.0, min=-20.0, max=0.0, step=0.1, tooltip="s_min = cfg + min_offset. Paper (T2I): -1."),
                io.Float.Input("max_offset", default=3.0, min=0.0, max=20.0, step=0.1, tooltip="s_max = cfg + max_offset. Paper (T2I): +2 (SD3) to +4 (Qwen-Image)."),
                io.Float.Input("rho_lo", default=0.95, min=-1.0, max=0.999, step=0.005, tooltip="Cosine at which the scale reaches s_max. Paper: 0.95 for images."),
                io.Float.Input("long_span_weight", default=0.2, min=0.0, max=1.0, step=0.05, tooltip="Weight of the stride-2 feedback (paper: 0.2)."),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, min_offset, max_offset, rho_lo, long_span_weight, start_percent, end_percent) -> io.NodeOutput:
        m = model.clone()
        m.model_options["transformer_options"][STATE] = TrajCFGState(min_offset, max_offset, rho_lo, long_span_weight, start_percent, end_percent,
                                                                     patch_size(m.model.diffusion_model))
        m.set_model_sampler_cfg_function(cfg_function)
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, STATE, outer_sample_wrapper)
        return io.NodeOutput(m)


NODES = [TrajectoryCFG]
