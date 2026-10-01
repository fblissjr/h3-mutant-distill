"""H3KeyframeCanvas: the video canvas from your first frame, so nothing is stretched.

Core's `MiniMaxH3ImageToVideo` takes width and height as plain inputs and
stretches the first frame onto them, so a square or portrait image on the
default 1344x768 comes out distorted in every frame. The reference pipeline
derives the canvas from the first frame instead when no size is given. This
node does that with core's own port of the rule (`adapt_canvas`: 768 short
edge, the 768*1344 area cap, rounded to 32) and resizes the image to it, so the
i2v node's own resize has nothing left to do.

A trimmed port of ComfyUI-h3-explorations' `MiniMaxH3KeyframeCanvas`
(retired there on 2026-09-29, whose geometry lives on in its
`MiniMaxH3Conditioning`): the first frame only, the reference's default mode only.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io

log = logging.getLogger(__name__)

#: The aspect range the released checkpoint was trained over. Inherited from the
#: reference pipeline, as ComfyUI-h3-explorations' `h3_rules.py` carries it.
MIN_ASPECT, MAX_ASPECT = 1 / 4, 4.0


class H3KeyframeCanvas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3KeyframeCanvas",
            display_name="H3 Keyframe Canvas",
            category="MiniMax H3/conditioning",
            description=(
                "Picks the video canvas from the first frame the way the MiniMax H3 "
                "release does (768 short edge, area capped at 1344x768, multiples of "
                "32) and resizes the image to it. Wire width, height and first_frame "
                "into MiniMax H3 Image to Video so the image is never stretched."),
            inputs=[io.Image.Input("first_frame")],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Image.Output(display_name="first_frame"),
            ],
        )

    @classmethod
    def execute(cls, first_frame) -> io.NodeOutput:
        # Imported here, not at load: core's H3 module imports `nodes`, which a
        # harness that puts another `nodes.py` first on sys.path resolves wrongly
        # (bench/check_mutant_parity.py imports this package that way).
        from comfy_extras.nodes_minimax_h3 import _resize, adapt_canvas
        src_h, src_w = int(first_frame.shape[1]), int(first_frame.shape[2])
        if not MIN_ASPECT <= src_w / src_h <= MAX_ASPECT:
            raise ValueError(
                f"MiniMax H3 was trained on aspect ratios from 1:4 to 4:1; this image is "
                f"{src_w}x{src_h}. Crop it first.")
        width, height = adapt_canvas(src_w, src_h)
        # The aspect now matches, so this is a uniform scale, not a stretch.
        image = _resize(first_frame[:1], width, height, "disabled")
        log.info("[h3-mutant] canvas %dx%d from a %dx%d first frame", width, height, src_w, src_h)
        return io.NodeOutput(width, height, image)
