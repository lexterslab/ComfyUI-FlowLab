"""Noise-Rejuvenated Sequential Monte Carlo (arXiv 2610.25479) for flow models, as a sampler.

The latent batch is the particle population. The flow ODE runs unchanged (Euler); at resampling steps every
particle is weighted by exp(lambda * max reward so far) of its decoded endpoint estimate x0, the population is
resampled multinomially, and each particle gets fresh source noise around its endpoint estimate
  x = sigma * eps_new + (1 - sigma) * x0      (paper: z_t = (1 - t) z0_new + t z1_hat, t = 1 - sigma)
so duplicated particles explore different paths without turning the sampler into an SDE. Look-ahead: the
reward of each rejuvenated particle is measured again after one flow step and enters its running maximum
(the paper's look-ahead potential); here that state is the regular next step, so it costs no extra model call.

The paper uses ImageReward; this package uses PickScore (a CLIP-H preference model) through transformers.
"""

import logging
import os

import torch

import comfy.model_management
import comfy.samplers
import folder_paths
from comfy_api.latest import io

FOLDER = "reward_models"
if FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths[FOLDER] = ([os.path.join(folder_paths.models_dir, FOLDER)], set())
else:
    folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER))

RewardModel = io.Custom("FLOWLAB_REWARD")
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def reward_model_dirs():
    found = []
    for base in folder_paths.get_folder_paths(FOLDER):
        if os.path.isdir(base):
            found += [d for d in sorted(os.listdir(base)) if os.path.isfile(os.path.join(base, d, "config.json"))]
    return found


def resolve_reward_dir(name):
    for base in folder_paths.get_folder_paths(FOLDER):
        path = os.path.join(base, name)
        if os.path.isfile(os.path.join(path, "config.json")):
            return path
    raise FileNotFoundError(f"FlowLab: reward model '{name}' not found in {folder_paths.get_folder_paths(FOLDER)}")


class PickScore:
    """CLIP preference score: logit_scale * cos(image, text), as in the PickScore reference code."""

    def __init__(self, path):
        from transformers import AutoTokenizer, CLIPModel  # optional dependency, only needed for NR-SMC
        self.device = comfy.model_management.get_torch_device()
        self.dtype = torch.float16 if comfy.model_management.should_use_fp16(self.device) else torch.float32
        self.model = CLIPModel.from_pretrained(path, torch_dtype=self.dtype).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.size = self.model.config.vision_config.image_size

    def preprocess(self, images):
        """ComfyUI IMAGE (B, H, W, 3) in [0, 1] -> CLIP input: shortest side to size, center crop, normalize."""
        x = images.movedim(-1, 1).float()
        h, w = x.shape[-2:]
        scale = self.size / min(h, w)
        x = torch.nn.functional.interpolate(x, size=(max(self.size, round(h * scale)), max(self.size, round(w * scale))), mode="bicubic", antialias=True)
        top, left = (x.shape[-2] - self.size) // 2, (x.shape[-1] - self.size) // 2
        x = x[..., top:top + self.size, left:left + self.size].clamp(0, 1)
        mean = torch.tensor(CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def __call__(self, images, prompt):
        self.model.to(self.device)
        try:
            pixels = self.preprocess(images.to(self.device)).to(self.dtype)
            tokens = self.tokenizer([prompt], padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(self.device)
            # transformers 5 returns model outputs whose pooler_output holds the projected embedding
            img = self.model.get_image_features(pixel_values=pixels).pooler_output
            txt = self.model.get_text_features(**tokens).pooler_output
            img = img / img.norm(dim=-1, keepdim=True)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            return (self.model.logit_scale.exp() * img @ txt.T).squeeze(-1).float()
        finally:
            self.model.to(comfy.model_management.unet_offload_device())


def make_sampler_function(reward, vae, prompt, lam, interval, start, end, look_ahead, order, decode_size):
    def score(model, x0):
        latent = model.inner_model.inner_model.process_latent_out(x0)
        pixels = max(latent.shape[-2:]) * vae.spacial_compression_decode()
        if 0 < decode_size < pixels:
            # the reward model sees 224 px; decoding a smaller latent saves VAE time
            size = [max(1, round(s * decode_size / pixels)) for s in latent.shape[-2:]]
            latent = torch.nn.functional.interpolate(latent, size=size, mode="bilinear")
        images = vae.decode(latent)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        return reward(images, prompt).to(x0.device)

    def sample_nrsmc(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        n = len(sigmas) - 1
        particles = x.shape[0]
        s_in = x.new_ones([particles])
        every = interval if interval > 0 else max(1, round(n / 10))
        first, last = max(1, round(start * n)), round(end * n)
        resample_steps = set(range(first, min(last, n - 1) + 1, every))
        best = torch.full((particles,), -float("inf"), device=x.device)
        check = set()  # steps after a rejuvenation where the look-ahead reward is taken
        generator = torch.Generator(device="cpu").manual_seed(int(extra_args.get("seed", 0)) + 7919)
        log = []
        for i in range(n):
            sigma = sigmas[i]
            denoised = model(x, sigma * s_in, **extra_args)
            if i in check:
                best = torch.maximum(best, score(model, denoised))
            if i in resample_steps and particles > 1:
                best = torch.maximum(best, score(model, denoised))
                weights = torch.softmax(lam * best, 0)
                idx = torch.multinomial(weights.cpu(), particles, replacement=True, generator=generator).to(x.device)
                log.append((i, sorted(idx.tolist())))
                best, denoised = best[idx], denoised[idx]
                # noise rejuvenation: fresh source noise around the kept endpoint estimates
                eps = torch.randn(denoised.shape, generator=generator, dtype=torch.float32).to(denoised)
                x = sigma * eps + (1 - sigma) * denoised
                denoised = model(x, sigma * s_in, **extra_args)
                if look_ahead:
                    check.add(i + 1)
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
            x = x + (sigmas[i + 1] - sigma) * (x - denoised) / sigma
        logging.info(f"FlowLab NR-SMC: {particles} particles, resampled at {sorted(resample_steps)}: {log}")
        if order == "best first" and particles > 1:
            final = score(model, x)
            rank = torch.argsort(final, descending=True)
            logging.info(f"FlowLab NR-SMC final rewards: {[round(v, 3) for v in final[rank].tolist()]}")
            x = x[rank]
        return x
    return sample_nrsmc


class RewardModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowLabRewardModelLoader",
            display_name="Reward Model Loader (PickScore)",
            category="FlowLab/sampling",
            description="Loads a CLIP preference model in Hugging Face format (e.g. yuvalkirstain/PickScore_v1) from models/reward_models.",
            inputs=[io.Combo.Input("model_name", options=reward_model_dirs())],
            outputs=[RewardModel.Output()],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        return io.NodeOutput(PickScore(resolve_reward_dir(model_name)))


class NoiseRejuvenatedSMC(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowLabNRSMCSampler",
            display_name="Noise-Rejuvenated SMC Sampler",
            category="FlowLab/sampling",
            description="Reward-guided particle sampling for flow models (arXiv 2610.25479). The latent batch size is the "
                        "number of particles (paper: 4). Use with SamplerCustomAdvanced; costs one extra model call per "
                        "particle and resampling step plus VAE decodes and reward evaluations.",
            inputs=[
                RewardModel.Input("reward_model"),
                io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, tooltip="Text the reward model compares the images with (usually the generation prompt)."),
                io.Float.Input("lambda_", display_name="lambda", default=10.0, min=0.0, max=1000.0, step=0.5, tooltip="Strength of the reward tilt (paper: 10)."),
                io.Int.Input("interval", default=0, min=0, max=100, tooltip="Resample every N steps; 0 = steps / 10 (paper: 5 of 50)."),
                io.Float.Input("start_percent", default=0.1, min=0.0, max=1.0, step=0.01, tooltip="Resampling window (paper: steps 5-35 of 50)."),
                io.Float.Input("end_percent", default=0.7, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("look_ahead", default=True, tooltip="Also score each rejuvenated particle one step later (paper's look-ahead refinement)."),
                io.Combo.Input("order", options=["best first", "keep"], default="best first", tooltip="best first: sort the final batch by reward (index 0 = best)."),
                io.Int.Input("decode_size", default=512, min=0, max=4096, step=64, tooltip="Longest side of the images the reward sees during sampling; 0 = full size."),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, reward_model, vae, prompt, lambda_, interval, start_percent, end_percent, look_ahead, order, decode_size) -> io.NodeOutput:
        fn = make_sampler_function(reward_model, vae, prompt, lambda_, interval, start_percent, end_percent, look_ahead, order, decode_size)
        return io.NodeOutput(comfy.samplers.KSAMPLER(fn))


NODES = [RewardModelLoader, NoiseRejuvenatedSMC]
