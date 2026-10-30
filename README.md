# ComfyUI-FlowLab

**English** | [Deutsch](README.de.md)

Sampling, guidance and caching methods from October 2026 papers as ComfyUI nodes. Built and tested with **Krea 2** and **Moss-Image (Kromoss)** in ComfyUI 0.42. All nodes are in the `FlowLab` category.

| Node | Paper | What | Cost | Models |
|---|---|---|---|---|
| Semantic Feedback Guidance (SFG) | 2610.13941 | Guides away from an auxiliary branch with amplified image→text feedback | +1 model call per step inside the window (about +10–15 %) | Krea 2, Moss |
| In-place SFG (iSFG) | 2610.13941 | Damps the feedback directly in the cond branch | no extra call | Krea 2, Moss |
| Trajectory-Adaptive CFG (TrajCFG) | 2610.14153 | Per-token CFG strength from the change of direction | no extra call | all, needs CFG > 1 |
| Edit CFG Early Correction (TI2I) | 2610.21550 | Early in editing: unconditional branch → source image | no extra call | edit models with CFG |
| SpecKCast | 2610.26837 | Cache: Kalman forecast per DCT frequency band | saves calls | all |
| BeziCast | 2610.16448 | Cache: Bézier forecast with simplex guard | saves calls | all |
| CacheScope | 2610.12530 | Block cache with probe scheduling (DiCache reconstruction) | saves block compute | Krea 2, Moss |
| PoSTCache Sampler | 2610.25664 | Sampler: anchors plus after-the-fact correction of the skipped steps | saves calls | all (Euler) |
| Noise-Rejuvenated SMC Sampler + Reward Model Loader | 2610.25479 | Particle sampling with a reward model | +1 call per resampling, VAE decodes, scoring | all (Euler) |
| **GRIT Realism** | own method | Pulls frequency balance, grain and colour of the denoised estimate toward real-photo statistics | practically none | all samplers; spectral part calibrated for Moss/Kromoss |
| FlowBudget | 2610.20636 | Estimates the required step count from samples of real images | 2 (Euler) or 4 (Heun) calls per sample | all |

## What helps when

**Distilled runs (DMD2/Turbo, 8–16 steps, CFG 1):**
- **iSFG** costs nothing and leaves the style unchanged.
- **SFG** in `feedback (SFG-X)` mode strengthens counting and colour binding.
- **NR-SMC** gives a better single image out of 4 particles.
- **Caches don't pay off here.**

**Base runs with many steps and CFG:**
- **PoSTCache** is the best cache.
- **TrajCFG** stabilises high CFG values.

## GRIT Realism (Guided Realism via Injected Texture)

Own method for photographic authenticity. Basis of measurement: 60 photos from DIV2K, their scenes described by Gemini, against Kromoss images of the same scenes (DMD2, 16 steps, CFG 1). The AI look measurably consists of three parts:
- **too much mid-frequency contrast:** +0.6 to +1.4 dB at 0.007–0.12 cycles/px
- **too little of the finest grain:** −2 dB at 0.35 cycles/px, noise floor −30 %
- **too much colour:** saturation +12 %, colourfulness +25 %

GRIT corrects the guided denoised estimate inside a σ window, and the remaining steps integrate the change:
1. **realism:** one gain per latent channel and DCT frequency band, calibrated from the image pairs. Moss latents are unpacked to 32 channels at 1/8 resolution for this.
2. **chroma:** the colour deviation from grey is read through the linear latent→RGB preview map, scaled, and written back as the minimum-norm latent change. Luminance stays exactly unchanged.
3. **detail:** "detail lie": inside its window the model sees σ·(1 − detail).

Wiring: model → GRIT Realism → sampler. It works with KSampler, SamplerCustomAdvanced and RES4LYF. Set `calibration` to the sampler you sample with. The gap to a real photo depends on the sampler: res_2s lacks fine detail, er_sde has too much mid-frequency contrast. That is why there is one calibration per sampler. With multi-stage samplers the correction is applied only on the scheduled steps, not on the intermediate stages.

Recommended values for Moss/Kromoss (DMD2, 16 steps; RES4LYF via ClownSampler with eta 0.6, bongmath, beta57). The distance to real-photo statistics is measured on 30 scenes; lower is better.

| Sampler | calibration | realism | chroma | detail | Distance without → with GRIT |
|---|---|---|---|---|---|
| er_sde/beta | er_sde | 0.5 | 0.96 | 0.08 (default) | 1.64 → **0.67** |
| res_2m | res_2m | 0.5 | 0.95 | 0 | 2.00 → **1.04**; unseen scenes: 2.18 → **1.22** |
| res_3m | res_3m | 0.5 | 0.95 (rather 0.92, saturation stays +14 %) | 0 | 1.94 → **0.94** |
| res_2s, texture | res_2s | 0.4 | 0.95 | 0.08 | 2.32 → **1.50**; noise floor 0.93, saturation 0.97 of the real photo, contrast 0.84 |
| res_2s, balanced | res_2s | 1.0 | 0.9 | 0 | 2.32 → **1.52** |

With er_sde/beta + GRIT, noise floor, micro texture, saturation, colourfulness and contrast are all within 8 % of the real photo. The cross-check on 30 scenes that no calibration has seen confirms the gain (res_2m −44 %).

What the samplers show on their own:
- **er_sde/beta** is closest to real-photo statistics at 1.64.
- **The RES4LYF samplers** are more saturated and have more mid-frequency contrast. res_2s is the cleanest (finest grain −3.7 dB, noise floor 64 %): it looks high-end, but more postcard than photo.

Sampler and scheduler comparison (Kromoss + DMD2, 16 steps; RES4LYF via ClownSampler, eta 0.6, bongmath; distance on 30 scenes):

| without GRIT | Distance | | with GRIT | Distance |
|---|---|---|---|---|
| er_sde/beta | 1.64 | | er_sde/beta | **0.67** |
| deis_3m/beta | 1.68 | | deis_3m/beta57 (calibration res_3m, 0.5/0.95) | **0.90** |
| deis_3m/beta57 | 1.72 | | res_2m/beta (res_2m, 0.5/0.95) | **0.92** |
| res_2m/beta | 1.84 | | res_2m/beta57 | 1.04 |
| abnorsett_3m/beta57 | 1.85 | | res_2m/bong_tangent (res_2m, 0.5/0.95) | 1.05 (best individual statistics, more definition) |
| res_3m, res_2m, dpmpp_2m, kl_optimal, power_shift | 1.94–2.05 | | deis_3m/beta | 1.52 (beta puts too few steps into the GRIT window) |
| ddim, bong_tangent (res_2m), res_2s, dpmpp_sde_2s, res_3s, bong_tangent (er_sde) | 2.22–2.44 | | | |

Findings:
- **More stages per step give smoother, less photographic images.** res_2s, dpmpp_sde_2s and res_3s are at the bottom (noise floor 60–67 %).
- **With shift 5, most schedulers put 10–14 of 16 steps above σ = 0.7.** bong_tangent ignores the shift and puts 5 steps below σ = 0.3. That gives more definition but also more oversharpening: worst on its own, competitive with GRIT.
- **beta beats beta57, kl_optimal and power_shift**, except for GRIT with deis_3m.

With a texture LoRA and a hires pass (a typical Kromoss workflow: DMD2 + grainscape_moss, SLG block 22, er_sde/beta, 2 MP, ×2 Lanczos with 4 steps at denoise 0.2; measured on 28 Oct on scenes 1–30):
- **grainscape turns the gap around:** the mid frequencies are already right, but the finest detail is missing (−3 to −5 dB from 0.2 cycles/px up). Without GRIT the distance is 2.28, worse than plain er_sde (1.64).
- **The spectral part hurts here.** It pushes the mid frequencies down by about 1 dB too much. At 1 MP:

  | Variant | Distance |
  |---|---|
  | without GRIT | 2.28 |
  | realism 0.5 | 1.88 |
  | **realism 0, chroma 0.96, detail 0.08** | **1.49** |
  | detail 0.08 only | 1.51 |
  | chroma only | 2.13 |
  | detail 0.12 | 1.55 |

  A dedicated calibration for this setup (scenes 31–60) did not help (2.01). Rule of thumb: **with texture LoRAs, realism 0.**
- **Put GRIT only before the refinement,** not into the hires pass. There it takes the colour back a second time. Measured on 10 scenes, downscaled to reference size: without GRIT 2.22; GRIT in both passes 2.09 (colourfulness 0.66); first pass only 1.10. On 30 scenes in the full workflow, realism 0 reaches 1.51, realism 0.5 1.92.

Tested and dropped:
- **extra late noise renewal** (an er_sde variant): no measurable effect, because the distilled model denoises the noise back to the same micro texture.
- **detail lie with res_2m/res_3m:** makes the result worse. With res_2s, on the other hand, it brings the texture back, but costs some contrast.
- **a single table calibrated on er_sde for all samplers:** makes res_2s worse (2.32 → 2.63), hence one calibration per sampler.

Limits:
- The spectral part is calibrated for Moss/Kromoss only (res_* with 30, er_sde with 60 image pairs). chroma and detail work with all models.
- A sampler does not change the model's visual language (perfect shapes, high gloss).
- realism 1.0 already over-corrects, because the correction acts in every step of the window.

## Measurements

RTX 5070 Ti 16 GB, 1024×1024, one seed per prompt. Few prompts, so a pointer, not a benchmark.

### Caches

30 Euler steps, PSNR against the full run, mean over 3 or 2 prompts:

| Method (≈12 model calls) | Kromoss base, CFG 4 | Krea 2, CFG 1 |
|---|---|---|
| full run | 39 s | 44 s |
| plain 12 steps | 15.3 dB | 16.7 dB |
| **PoSTCache 12** (spread) | **30.7 dB**, 16 s | **26.9 dB**, 18 s |
| BeziCast (W6, R4) | 27.8 dB | 25.5 dB |
| SpecKCast fixed 12, warmup 6 | 27.9 dB | 24.9 dB |
| SpecKCast fixed 12, warmup 3 (paper) | 23.9 dB | – |
| CacheScope τ=3 | 29.4 dB at 16 full steps, 21 s | 26.8 dB at 15, 23 s |

With PoSTCache and 16 anchors, Kromoss reaches 35.2 dB.

### SFG

With Kromoss + DMD2 and with Krea 2 (turbo style, CFG 1), the style is preserved with **SFG-X (ω = 5)** and **iSFG**.
- SFG-X was the only variant that got the candle colours right with Krea 2 (2 blue / 4 white).
- **SFG `both` pulls toward illustration and produces colour blotches and noise patterns with Krea 2.** Not recommended for distilled models.

### NR-SMC

Kromoss + DMD2, 16 steps, 4 particles: 59 s instead of 42 s for 4 normal images.
- The best particle (index 0 with `best first`) got "five ducks" and "white cow + balloon" right. Of 4 normal images, only every second one did.
- Often two nearly identical pairs remain at the end: the method delivers quality, not variety.

### TI2I

Only a functional test, Kromoss is not an edit model. Setup: reference latent, CFG 4, 20 steps, "make the raincoat bright green".
- **Without correction**, CFG repaints the source image: seagulls gone, washed out, grainy.
- **With τ = 0.1** the layout is preserved and the edit still applies.
- PSNR to the source image: 16.9 → 20.7 dB (τ = 0.1) → 24.4 dB (τ = 0.3, weaker edit).

### FlowBudget

Kromoss base, CFG 4: Euler 462, Heun 111 steps. In practice 30 are enough. With DMD2 it comes out at 959, which is meaningless.
- The paper's tolerance is numerical, not perceptual, and validated only without CFG.
- Useful only for **comparing** shift, schedule or solver variants with each other.

## Wiring

- **Guidance nodes and caches** (MODEL → MODEL): put them between the model loader (or `ModelSamplingAuraFlow`) and the sampler. They can be combined, e.g. iSFG + PoSTCache.
- **PoSTCache and NR-SMC** are SAMPLERs for `SamplerCustomAdvanced` (instead of `KSamplerSelect`).
  - For NR-SMC the batch size of the latent is the number of particles (paper: 4).
  - `prompt` is the text for the reward model.
  - Index 0 of the output is the best image; select it with `LatentFromBatch` (index 0, length 1).
- **TI2I:** `source` is the VAE-encoded source image, the same one used for the reference conditioning.
- **FlowBudget:** needs a GUIDER (`CFGGuider` with prompt and CFG), a batch of VAE-encoded real images and the scheduler's SIGMAS. The result comes out as INT and STRING, e.g. for `PreviewAny`, and in the log.

## Deviations from the papers

- **SFG:** for images the paper gives no |u|; default 0.3 (0.2–0.4 tested). Krea 2 and Moss are single-stream models (Krea: text before image, Moss: image before text). The paper tests only SD3/FLUX/HunyuanVideo. Qwen-Image 2.1 is block-causal (text never sees the image), so the feedback does not exist there.
- **SpecKCast:**
  - The paper does not give process and measurement noise. Here the observation is treated as exact (R = 0). The process-noise scale then cancels out of the prediction, and the update matches Eq. 47.
  - The band powers p_ℓ are measured for Krea 2 and Moss and built in.
  - The risk of the adaptive mode is weighted by coefficient count × band power instead of the paper's LPIPS-calibrated weights. The mode is still worse than the fixed schedule and is marked experimental.
  - Default warm-up 6 instead of 3, which measures better.
- **BeziCast:** forecasts the guided velocity instead of the representation before the last layer, which makes it model-independent. The paper gives no numbers for u_max and the guard rule (ρ, γ, β_max); the defaults are set, not tuned.
- **CacheScope:** the reconstruction is r = r₂ + γ(r₂ − r₁) with γ = clip(d(y, y₂)/d(y₂, y₁), 1, 1.5); the exact DiCache rule for γ is not published. The base policy is warm-up share + last step. When cond and uncond are computed together, they share the decision. With offloaded models (Krea 2 on 16 GB), skipped blocks still load their weights.
- **PoSTCache:** "hold last value" proposal (paper default). The dense prefix is half the budget. PVR calibration is implemented (`calibrate` mode, profiles under `ComfyUI/user/flowlab/postcache`), but brought nothing over `spread` with Kromoss.
- **NR-SMC:** reward model PickScore instead of ImageReward; the ImageReward package pins old dependencies. The look-ahead uses the regular next step and therefore costs no extra model call. There is always one call per resampling for the rejuvenated state.
- **FlowBudget:** finite differences instead of JVPs. The sample σ are drawn from the actual schedule.

## Requirements

- For NR-SMC a CLIP preference model in Hugging Face format in `models/reward_models`, e.g. `yuvalkirstain/PickScore_v1`, plus `transformers` (included with ComfyUI).
- The other nodes need nothing extra.

## License

GPL-3.0, like ComfyUI itself (see `LICENSE`).
