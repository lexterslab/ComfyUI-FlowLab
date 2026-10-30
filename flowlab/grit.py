"""GRIT - Guided Realism via Injected Texture: photographic texture and colour for flow models.

Measured on 60 DIV2K photos against Kromoss generations of the same scenes (Gemini captions), the "AI look"
is too much mid-frequency contrast (+0.6 to +1.4 dB at 0.007-0.12 cycles/px), too little of the finest grain
(-2 dB at 0.35 cycles/px, 30 % lower noise floor) and too much colour (+12 % saturation, +25 % colourfulness).

GRIT Realism (model patch, works with any sampler, er_sde or RES4LYF alike) corrects the guided denoised
estimate inside a sigma window, so the remaining steps integrate the change instead of a post filter:
  - spectral realism: one gain per latent channel and radial DCT band, calibrated from the paired sets
    (Moss/Flux2 latents are unpacked from 128 ch at 1/16 to 32 ch at 1/8 so the bands resolve finer structure);
  - chroma: the latent's colour deviation from grey, read through the latent format's linear latent->RGB
    preview map, is scaled and written back as the minimum-norm latent change;
  - detail lie (optional): inside its window the model is told a lower sigma (the Detail Daemon idea).

Measured (30 scenes, distance to real-photo statistics, lower is better), each with its own calibration:
er_sde/beta 1.64 -> 0.67; RES4LYF ClownSampler (eta 0.6, bongmath, beta57) res_2m 2.00 -> 1.04 (unseen scenes
2.18 -> 1.22), res_3m 1.94 -> 0.94, res_2s 2.32 -> 1.50. Settings per sampler in the README.

Extra stochastic noise renewal late in sampling (an er_sde variant) was tested and dropped: the distilled
model denoised it back to the same micro texture (no measurable change, like euler_ancestral vs er_sde).
"""

import json
import math
from pathlib import Path

import torch

import comfy.patcher_extension
from comfy_api.latest import io

from .cache import Spectrum

CALIBRATION_DIR = Path(__file__).parent / "data"
# spectral calibrations per model family and sampler: the measured gap depends on the sampler (res_2s lacks fine
# detail where er_sde has too much mid-frequency contrast), so the matching table must be used
CALIBRATIONS = {"MOSSImage": {"er_sde": "grit_moss_er_sde", "res_2m": "grit_moss_res_2m", "res_3m": "grit_moss_res_3m", "res_2s": "grit_moss_res_2s"}}
SAMPLERS = ["er_sde", "res_2m", "res_3m", "res_2s"]
LUMA = (0.2126, 0.7152, 0.0722)


def on_schedule(model_options, sigma):
    """True for model calls at a scheduled step; intermediate stages of multi-stage solvers (res_2s ...) return False,
    so the correction is applied once per step whatever the solver."""
    sigmas = model_options.get("transformer_options", {}).get("sample_sigmas")
    if sigmas is None:
        return True
    return bool(((sigmas - sigma).abs() <= 1e-4 * max(sigma, 1e-3)).any())


def log_lambda(sigma):
    sigma = min(max(sigma, 1e-4), 1 - 1e-4)
    return math.log(sigma / (1 - sigma))


def window(sigma, hi, lo, ramp=0.25):
    """1 inside [lo, hi] (sigma), cosine ramps over the outer `ramp` share of the window in log lambda, 0 outside."""
    a, b = log_lambda(lo), log_lambda(hi)
    if b <= a:
        return 0.0
    u = (log_lambda(sigma) - a) / (b - a)
    if u <= 0 or u >= 1:
        return 0.0
    edge = min(u, 1 - u) / ramp
    return 1.0 if edge >= 1 else 0.5 * (1 - math.cos(math.pi * edge))


class LatentLayout:
    """Unpacking and latent->RGB preview map of a model's latent format."""

    def __init__(self, latent_format):
        self.rgb = torch.tensor(latent_format.latent_rgb_factors, dtype=torch.float32)  # (channels, 3)
        self.packed = latent_format.latent_channels != self.rgb.shape[0]  # Flux2: 128 = 32 x 2 x 2

    def unpack(self, lat):
        if not self.packed:
            return lat
        b, c, h, w = lat.shape
        return lat.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)

    def pack(self, lat, shape):
        if not self.packed:
            return lat
        b, c, h, w = shape
        return lat.reshape(b, c // 4, h, 2, w, 2).permute(0, 1, 3, 5, 2, 4).reshape(shape)


class RealismCorrection:
    def __init__(self, layout, calibration, realism, chroma, hi, lo):
        self.layout = layout
        self.log_ratio = None if calibration is None else torch.tensor(calibration["log10_ratio"], dtype=torch.float32)
        self.realism, self.chroma, self.hi, self.lo = realism, chroma, hi, lo
        F = layout.rgb
        self.back = torch.linalg.pinv(F.T)  # rgb change -> minimum-norm latent change, (channels, 3)
        self.luma = torch.tensor(LUMA)
        self.spec = None

    def spectrum(self, lat):
        if self.spec is None or self.spec.mh.shape[0] != lat.shape[-2] or self.spec.mw.shape[0] != lat.shape[-1] or self.spec.band.device != lat.device:
            self.spec = Spectrum(lat.shape[-2], lat.shape[-1], lat.device)
        return self.spec

    def __call__(self, sigma, denoised):
        w = window(sigma, self.hi, self.lo)
        if w == 0:
            return denoised
        lat = self.layout.unpack(denoised.float())
        if self.log_ratio is not None and self.realism != 0:
            spec = self.spectrum(lat)
            gains = torch.pow(10.0, 0.5 * w * self.realism * self.log_ratio.to(lat.device))  # amplitude = sqrt(power ratio)
            lat = spec.inverse(spec.forward(lat) * gains[:, spec.band][None])
        if self.chroma != 1.0:
            F = self.layout.rgb.to(lat.device)
            rgb = torch.einsum("bchw,cr->brhw", lat, F)
            grey = torch.einsum("brhw,r->bhw", rgb, self.luma.to(lat.device))[:, None]
            delta = -(1 - self.chroma) * w * (rgb - grey)
            lat = lat + torch.einsum("brhw,cr->bchw", delta, self.back.to(lat.device))
        return self.layout.pack(lat, denoised.shape).to(denoised.dtype)


class GRITRealism(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowLabGRITRealism",
            display_name="GRIT Realism",
            category="FlowLab/sampling",
            description="Pulls the denoised estimate toward real-photo statistics inside a sigma window: frequency balance "
                        "per latent channel (calibrated on DIV2K, built in for Moss/Kromoss), colour saturation and an optional "
                        "detail lie. Works with any sampler (er_sde, RES4LYF ...). All at neutral = no change.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("calibration", options=SAMPLERS, default="er_sde", tooltip="Sampler the spectral correction was calibrated for; pick the one you sample with (res_* = RES4LYF ClownSampler, eta 0.6, bongmath, beta57). With texture LoRAs such as grainscape_moss set realism 0 (they fix the mid frequencies already). Other samplers: the closest one, e.g. deis_3m/abnorsett_3m -> res_3m, dpmpp_2m -> res_2m, res_3s -> res_2s, dpmpp_2m_sde -> er_sde."),
                io.Float.Input("realism", default=0.5, min=0.0, max=3.0, step=0.05, tooltip="Strength of the calibrated spectral correction (Moss/Kromoss only). Measured: 0.5 for er_sde, res_2m and res_3m; res_2s 0.4 with detail 0.08 (texture) or 1.0 without (balanced). The correction compounds over the window, so 1.0 over-corrects most samplers."),
                io.Float.Input("chroma", default=0.96, min=0.0, max=2.0, step=0.01, tooltip="Colour saturation of the denoised estimate inside the window (1 = unchanged). Measured: 0.96 with er_sde, 0.95 with res_2m/res_2s, about 0.92 with res_3m, 0.9 for res_2s without detail. Works for every model with a latent preview map."),
                io.Float.Input("start", default=0.6, min=0.0, max=1.0, step=0.01, tooltip="Window: upper sigma (start of the correction)."),
                io.Float.Input("end", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Window: lower sigma."),
                io.Float.Input("detail", default=0.08, min=0.0, max=0.9, step=0.01, tooltip="Detail lie: inside its window the model sees sigma * (1 - detail). Raises micro texture and noise floor toward real photos with er_sde and res_2s (0.08, costs some contrast with res_2s); set 0 with res_2m/res_3m, there it hurts."),
                io.Float.Input("detail_start", default=0.8, min=0.0, max=1.0, step=0.01),
                io.Float.Input("detail_end", default=0.2, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, calibration, realism, chroma, start, end, detail, detail_start, detail_end) -> io.NodeOutput:
        m = model.clone()
        name = CALIBRATIONS.get(type(m.model).__name__, {}).get(calibration)
        table = json.loads((CALIBRATION_DIR / f"{name}.json").read_text()) if name else None
        correction = RealismCorrection(LatentLayout(m.model.latent_format), table, realism, chroma, start, end)

        def post_cfg_function(args):
            sigma = float(args["sigma"].flatten()[0])
            if not on_schedule(args["model_options"], sigma):
                return args["denoised"]
            return correction(sigma, args["denoised"])

        if (table is not None and realism != 0) or chroma != 1.0:
            m.set_model_sampler_post_cfg_function(post_cfg_function)

        if detail > 0:
            def apply_model_wrapper(executor, x, t, *args, **kwargs):
                w = window(float(t.flatten()[0]), detail_start, detail_end)
                return executor(x, t * (1 - detail * w) if w > 0 else t, *args, **kwargs)
            m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, "flowlab_grit_detail", apply_model_wrapper)
        return io.NodeOutput(m)


NODES = [GRITRealism]
