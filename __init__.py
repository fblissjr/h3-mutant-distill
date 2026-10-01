"""h3-mutant-distill: experimental MiniMax H3 distill recipes, and a keyframe canvas for i2v."""

from comfy_api.latest import ComfyExtension, io

from .exact_lora import H3ExactLoRA
from .keyframe_canvas import H3KeyframeCanvas


class _Extension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [H3ExactLoRA, H3KeyframeCanvas]


async def comfy_entrypoint() -> ComfyExtension:
    return _Extension()
