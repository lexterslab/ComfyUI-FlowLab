from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .flowlab import cache, cachescope, flowbudget, grit, nrsmc, postcache, sfg, ti2i, trajcfg


class FlowLabExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [*sfg.NODES, *trajcfg.NODES, *ti2i.NODES, *cache.NODES, *postcache.NODES, *nrsmc.NODES, *cachescope.NODES, *flowbudget.NODES, *grit.NODES]


async def comfy_entrypoint() -> FlowLabExtension:
    return FlowLabExtension()
