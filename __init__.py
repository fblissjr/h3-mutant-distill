"""h3-mutant-distill: one node for experimental MiniMax H3 distill recipes."""

from comfy_api.latest import ComfyExtension, io

from .exact_lora import H3ExactLoRA


class _Extension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [H3ExactLoRA]


async def comfy_entrypoint() -> ComfyExtension:
    return _Extension()
