"""FlowBudget (arXiv 2610.20636): training-free estimate of the step budget a flow model needs.

Probes are interpolants x = (1 - sigma) x0 + sigma * eps of real latents at noise levels drawn from the
sampler's own sigma schedule. With v = (x - D) / sigma (dx/dsigma along the ODE) the leading local error of
  Euler:  C_E = E[ |a| / |v| ],  a = dv/dsigma along the path        ->  K = ceil(C_E / (2 tau))
  Heun:   C_H = E[ |2 g - c| / |v| ],  g = J_x v . a,  c = d2v along (v, 1)  ->  K = ceil(sqrt(C_H / (12 tau))), NFE = 2K
The paper uses Jacobian-vector products; here they are finite differences of the guided model output, so
Euler costs 2 and Heun 4 model calls per probe. The paper uses tau = 0.01 (ODE) and validated the estimate only
on class-conditional ImageNet/CIFAR models without CFG; tau is a numerical tolerance, not a perceptual one.
"""

import logging
import math

import torch

import comfy.samplers
import comfy.sample
import comfy.utils
from comfy_api.latest import io


def make_probe_sampler(results, solver, probes, h, seed):
    def sample_probe(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        x0 = model.latent_image  # model space
        eps = model.noise
        g = torch.Generator(device="cpu").manual_seed(seed)
        pool = sigmas[:-1].float().cpu()
        pool = pool[(pool > 0.02) & (pool < 0.98)]
        if len(pool) == 0:
            raise ValueError("FlowBudget: the sigma schedule has no usable steps between 0.02 and 0.98.")
        b = x0.shape[0]
        s_in = x0.new_ones([b])

        def v(xs, s):
            d = model(xs, s * s_in, **extra_args)
            return (xs - d) / s

        ratios = []
        pbar = comfy.utils.ProgressBar(probes)
        for k in range(probes):
            s = float(pool[torch.randint(len(pool), (1,), generator=g)])
            e = eps if k == 0 else torch.randn(eps.shape, generator=g).to(eps)
            xs = (1 - s) * x0 + s * e
            v0 = v(xs, s)
            # step toward the data so sigma stays positive
            hs = -min(h, 0.5 * s)
            v1 = v(xs + hs * v0, s + hs)
            a = (v1 - v0) / hs
            if solver == "euler":
                num = a.flatten(1).norm(dim=1)
            else:
                v2 = v(xs + 2 * hs * v0, s + 2 * hs)
                c = (v2 - 2 * v1 + v0) / (hs * hs)  # second directional derivative along (v0, 1)
                ga = (v(xs + hs * a, s) - v0) / hs  # J_x v . a
                num = (2 * ga - c).flatten(1).norm(dim=1)
            ratios.append((num / (v0.flatten(1).norm(dim=1) + 1e-8)).float().cpu())
            pbar.update(1)
        results["coefficient"] = float(torch.cat(ratios).mean())
        results["probes"] = probes * b
        return x0
    return sample_probe


class FlowBudget(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowLabFlowBudget",
            display_name="FlowBudget (step budget estimate)",
            category="FlowLab/analysis",
            description="Estimates how many steps a flow model needs for a solver from probes of real images, without "
                        "generating samples (arXiv 2610.20636). Feed VAE-encoded images (a batch); the guider carries prompt and CFG. "
                        "Not meaningful for distilled few-step models.",
            inputs=[
                io.Guider.Input("guider"),
                io.Latent.Input("images", tooltip="VAE-encoded real images (any batch size). Captions in the guider should match them roughly."),
                io.Sigmas.Input("sigmas", tooltip="The schedule you sample with (shift matters); probe noise levels are drawn from it."),
                io.Combo.Input("solver", options=["euler", "heun"], default="euler"),
                io.Float.Input("tolerance", default=0.01, min=0.0001, max=1.0, step=0.001, tooltip="Relative local error target tau (paper: 0.01 for ODE solvers)."),
                io.Int.Input("probes", default=8, min=1, max=1000, tooltip="Noise draws per image (paper: 2000 probes in total)."),
                io.Float.Input("step", default=0.01, min=0.0001, max=0.2, step=0.001, tooltip="Finite-difference step in sigma."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
            ],
            outputs=[io.Int.Output(display_name="steps"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, guider, images, sigmas, solver, tolerance, probes, step, seed) -> io.NodeOutput:
        results = {}
        latent = comfy.sample.fix_empty_latent_channels(guider.model_patcher, images["samples"], images.get("downscale_ratio_spacial", None))
        noise = comfy.sample.prepare_noise(latent, seed)
        sampler = comfy.samplers.KSAMPLER(make_probe_sampler(results, solver, probes, step, seed))
        guider.sample(noise, latent, sampler, sigmas, seed=seed)
        c = results["coefficient"]
        if solver == "euler":
            k = math.ceil(c / (2 * tolerance))
            nfe = k
        else:
            k = math.ceil(math.sqrt(c / (12 * tolerance)))
            nfe = 2 * k
        report = f"FlowBudget {solver}: coefficient {c:.4f} from {results['probes']} probes, tolerance {tolerance} -> {k} steps ({nfe} model calls)"
        logging.info(report)
        return io.NodeOutput(k, report)


NODES = [FlowBudget]
