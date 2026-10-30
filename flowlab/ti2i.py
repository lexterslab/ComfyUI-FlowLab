"""Early correction of CFG for text+image-to-image editing (arXiv 2610.21550).

With CFG the empty-text branch of an editing model drifts toward the "average edit" early in sampling and
repaints the source. For flow time t < tau (sigma > 1 - tau) the paper replaces that branch's velocity by the
straight path to the source, (c_img - x_t) / (1 - t), which in denoised space is simply the source latent.
The guidance difference keeps the original unconditional prediction (correcting it too distorts badly):

  denoised = source + scale * (D_cond - D_uncond)  =  cfg_result - D_uncond + source

Implemented as a post-CFG step so it also works on top of other CFG functions (TrajCFG). No extra model calls.
"""

import comfy.utils
from comfy_api.latest import io


class EditEarlyCorrection(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowLabTI2IEarlyCorrection",
            display_name="Edit CFG Early Correction (TI2I)",
            category="FlowLab/guidance",
            description="For instruction editing with CFG: early in sampling the unconditional branch is replaced by a "
                        "no-edit path to the source image (arXiv 2610.21550). Keeps more of the source and follows the "
                        "instruction at least as well. Needs cfg > 1 and a source of the output's size.",
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("source", tooltip="The VAE-encoded image being edited (same latent the reference conditioning uses)."),
                io.Float.Input("tau", default=0.1, min=0.0, max=1.0, step=0.01, tooltip="Correction is active while flow time t < tau, i.e. sigma > 1 - tau. Paper: 0.10 (Qwen-Image-Edit, FLUX.2), 0.05 (PixelGen), 0.01 (FLUX.2 klein at 1024)."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, source, tau) -> io.NodeOutput:
        m = model.clone()
        src = m.model.process_latent_in(source["samples"])

        def post_cfg_function(args):
            denoised = args["denoised"]
            uncond = args["uncond_denoised"]
            if tau <= 0 or args["uncond"] is None or float(args["sigma"].flatten()[0]) <= 1.0 - tau:
                return denoised
            target = src
            if target.shape[-2:] != denoised.shape[-2:]:
                target = comfy.utils.common_upscale(target, denoised.shape[-1], denoised.shape[-2], "bilinear", "center")
            target = comfy.utils.repeat_to_batch_size(target, denoised.shape[0]).to(denoised)
            return denoised - uncond + target

        m.set_model_sampler_post_cfg_function(post_cfg_function)
        return io.NodeOutput(m)


NODES = [EditEarlyCorrection]
