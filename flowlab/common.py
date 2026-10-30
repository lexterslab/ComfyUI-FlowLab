"""Shared helpers: token layout of the supported models, sampling progress, attention-override chaining."""

import torch

import comfy.ldm.krea2.model
import comfy.ldm.moss_image.model


def text_span_fn(diffusion_model):
    """Return f(x, context) -> (start, end) of the prompt tokens in the joint sequence.

    Krea 2 puts the prompt first, Moss-Image puts the image first (patch size 1, so H*W tokens).
    """
    if isinstance(diffusion_model, comfy.ldm.krea2.model.SingleStreamDiT):
        return lambda x, context: (0, context.shape[1])
    if isinstance(diffusion_model, comfy.ldm.moss_image.model.MOSSImage):
        return lambda x, context: (x.shape[-2] * x.shape[-1], x.shape[-2] * x.shape[-1] + context.shape[1])
    raise ValueError(f"FlowLab: {type(diffusion_model).__name__} is not supported here (only Krea 2 and Moss-Image).")


def patch_size(diffusion_model):
    """Latent pixels per transformer token along each axis (Krea 2: 2, Moss-Image: 1, others: 1)."""
    if isinstance(diffusion_model, comfy.ldm.krea2.model.SingleStreamDiT):
        return diffusion_model.patch
    return 1


def forward_args(args, kwargs):
    """Split the DIFFUSION_MODEL wrapper arguments into (x, context, transformer_options)."""
    if isinstance(args[-1], dict):
        return args[0], args[2], args[-1]
    return args[0], args[2], kwargs["transformer_options"]


def replace_transformer_options(args, kwargs, opts):
    if isinstance(args[-1], dict):
        return args[:-1] + (opts,), kwargs
    return args, {**kwargs, "transformer_options": opts}


def sampling_progress(transformer_options, sigma):
    """(step index, number of steps) of the current model call within the running sampler, or None."""
    sigmas = transformer_options.get("sample_sigmas")
    if sigmas is None or len(sigmas) < 2:
        return None
    s = float(sigma.flatten()[0])
    steps = len(sigmas) - 1
    idx = int(torch.argmin((sigmas[:-1].float().cpu() - s).abs()))
    return idx, steps


def in_window(transformer_options, sigma, start, end):
    progress = sampling_progress(transformer_options, sigma)
    if progress is None:
        return False
    idx, steps = progress
    return start <= idx / steps < end


def chain_attention_override(model_options, override):
    """Install override(func, q, k, v, heads, ...) in front of an override another node may have set."""
    opts = model_options["transformer_options"]
    previous = opts.get("optimized_attention_override")
    if previous is None or getattr(previous, "flowlab", False):
        wrapped = override
    else:
        def wrapped(func, *args, **kwargs):
            return override(lambda *a, **kw: previous(func, *a, **kw), *args, **kwargs)
    wrapped.flowlab = True
    opts["optimized_attention_override"] = wrapped
