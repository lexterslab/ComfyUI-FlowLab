"""Semantic Feedback Guidance (arXiv 2610.13941): scale the cross-modal messages of the joint attention.

In a joint text-image attention the output of a query splits into a same-modality part and a cross-modal
message. The paper scales two messages after the softmax: image -> text (a_X, the "feedback" that writes
image content into the prompt tokens) and text -> image (a_S, the prompt conditioning of the image).

  SFG:  auxiliary branch with a_X = 1 + u, a_S = 1 - u, guided away from:  D + w * (D_cond - D_aux)
  iSFG: in place on the conditional branch, a_X = 1 + gamma (gamma < 0), a_S = 1 + gamma_s (gamma_s > 0)

Scaling the values of the source rows by a before the attention gives exactly A_src V_src * a for the
target queries, because the softmax weights stay untouched.
"""

import torch

import comfy.patcher_extension
import comfy.samplers
from comfy_api.latest import io

from .common import (chain_attention_override, forward_args, in_window, replace_transformer_options,
                     text_span_fn)

SCALES = "flowlab_sfg_scales"
AUX = "flowlab_sfg_aux"
ISFG = "flowlab_isfg"
WRAPPER_KEY = "flowlab_sfg"


def message_scaled_attention(func, q, k, v, heads, *args, **kwargs):
    opts = kwargs.get("transformer_options") or {}
    scales = opts.get(SCALES)
    if scales is None or "block_index" not in opts:
        return func(q, k, v, heads, *args, **kwargs)
    t0, t1, a_x, a_s = scales
    n_txt = t1 - t0
    seq = 2 if kwargs.get("skip_reshape", False) else 1
    out_seq = 2 if kwargs.get("skip_output_reshape", False) else 1
    per_batch = (-1,) + (1,) * (v.ndim - 1)

    v_txt_rows = v.narrow(seq, t0, n_txt)
    if a_s is not None:
        # image queries: text values scaled by a_S
        v_img = v.clone()
        v_img.narrow(seq, t0, n_txt).mul_(a_s.view(per_batch).to(v.dtype))
        out = func(q, k, v_img, heads, *args, **kwargs)
        del v_img
    else:
        out = func(q, k, v, heads, *args, **kwargs)

    # text queries: image (and reference) values scaled by a_X, text values unchanged
    if a_x is not None:
        v_txt = v * a_x.view(per_batch).to(v.dtype)
        v_txt.narrow(seq, t0, n_txt).copy_(v_txt_rows)
    else:
        v_txt = v
    txt_kwargs = kwargs
    mask = kwargs.get("mask")
    if mask is not None and mask.ndim >= 2 and mask.shape[-2] > 1:
        txt_kwargs = {**kwargs, "mask": mask.narrow(-2, t0, n_txt)}
    out_txt = func(q.narrow(seq, t0, n_txt).contiguous(), k, v_txt, heads, *args, **txt_kwargs)
    out.narrow(out_seq, t0, n_txt).copy_(out_txt)
    return out


def make_forward_wrapper(span_fn):
    def forward_wrapper(executor, *args, **kwargs):
        x, context, opts = forward_args(args, kwargs)
        batch = x.shape[0]
        aux = opts.get(AUX)
        isfg = opts.get(ISFG)
        if aux is not None:
            a_x = torch.full((batch,), aux[0], device=x.device)
            a_s = torch.full((batch,), aux[1], device=x.device)
        elif isfg is not None and in_window(opts, opts["sigmas"], isfg["start"], isfg["end"]):
            # only the conditional chunks are edited; cond_or_uncond lists 0 for cond, 1 for uncond
            chunks = opts.get("cond_or_uncond", [0])
            is_cond = torch.tensor([c == 0 for c in chunks], device=x.device).repeat_interleave(batch // len(chunks))
            a_x = torch.where(is_cond, 1.0 + isfg["gamma"], 1.0)
            a_s = torch.where(is_cond, 1.0 + isfg["gamma_s"], 1.0)
        else:
            return executor(*args, **kwargs)
        t0, t1 = span_fn(x, context)
        scales = (t0, t1, None if bool((a_x == 1).all()) else a_x, None if bool((a_s == 1).all()) else a_s)
        if scales[2] is None and scales[3] is None:
            return executor(*args, **kwargs)
        args, kwargs = replace_transformer_options(args, kwargs, {**opts, SCALES: scales})
        return executor(*args, **kwargs)
    return forward_wrapper


def install(model):
    """Clone the model and install the attention override and the forward wrapper once."""
    m = model.clone()
    span_fn = text_span_fn(m.model.diffusion_model)
    chain_attention_override(m.model_options, message_scaled_attention)
    if not m.get_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY):
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, make_forward_wrapper(span_fn))
    return m


class SemanticFeedbackGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowLabSFG",
            display_name="Semantic Feedback Guidance (SFG)",
            category="FlowLab/guidance",
            description="Guides away from an auxiliary branch with amplified image-to-text feedback and weakened prompt "
                        "conditioning in the joint attention (arXiv 2610.13941). Costs one extra model call per step inside the window. "
                        "Works with CFG 1. Krea 2 and Moss-Image only.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("scale", default=5.0, min=0.0, max=30.0, step=0.1, tooltip="Guidance scale omega. The paper's best values: 5 (SD3M, CFG 1), 7.5 (FLUX with CFG), about 14 (FLUX, CFG 1)."),
                io.Float.Input("strength", default=0.3, min=0.0, max=1.0, step=0.01, tooltip="|u|: the auxiliary branch multiplies image->text messages by 1+|u| and text->image messages by 1-|u|. The paper tests 0.2-0.4."),
                io.Combo.Input("components", options=["both", "feedback (SFG-X)", "conditioning (SFG-S)"], default="both", tooltip="SFG-X alone was the stronger component in the paper."),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Window in sampling steps. The paper uses the first 20% of the steps; longer windows did not help."),
                io.Float.Input("end_percent", default=0.2, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, scale, strength, components, start_percent, end_percent) -> io.NodeOutput:
        m = install(model)
        a_x = 1.0 + strength if components != "conditioning (SFG-S)" else 1.0
        a_s = 1.0 - strength if components != "feedback (SFG-X)" else 1.0

        def post_cfg_function(args):
            denoised = args["denoised"]
            model_options = args["model_options"]
            if scale == 0 or not in_window(model_options["transformer_options"], args["sigma"], start_percent, end_percent):
                return denoised
            model_options = {**model_options, "transformer_options": {**model_options["transformer_options"], AUX: (a_x, a_s)}}
            (aux,) = comfy.samplers.calc_cond_batch(args["model"], [args["cond"]], args["input"], args["sigma"], model_options)
            return denoised + (args["cond_denoised"] - aux) * scale

        m.set_model_sampler_post_cfg_function(post_cfg_function)
        return io.NodeOutput(m)


class InPlaceSemanticFeedbackGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowLabISFG",
            display_name="In-place Semantic Feedback Guidance (iSFG)",
            category="FlowLab/guidance",
            description="Suppresses image-to-text feedback and strengthens prompt conditioning directly in the conditional "
                        "forward (arXiv 2610.13941). No extra model calls. Krea 2 and Moss-Image only.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("feedback", default=-0.2, min=-1.0, max=1.0, step=0.01, tooltip="gamma: image->text messages are multiplied by 1+gamma. Negative suppresses the feedback (paper: -0.2)."),
                io.Float.Input("conditioning", default=0.2, min=-1.0, max=2.0, step=0.01, tooltip="gamma_S: text->image messages are multiplied by 1+gamma_S (paper: +0.2)."),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="The paper applies iSFG at every step."),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, feedback, conditioning, start_percent, end_percent) -> io.NodeOutput:
        m = install(model)
        m.model_options["transformer_options"][ISFG] = {"gamma": feedback, "gamma_s": conditioning, "start": start_percent, "end": end_percent}
        return io.NodeOutput(m)


NODES = [SemanticFeedbackGuidance, InPlaceSemanticFeedbackGuidance]
