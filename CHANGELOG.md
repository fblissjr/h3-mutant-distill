# Changelog

Semantic versioning. Everything here is experimental: each recipe was judged by
eye on a few renders, and the README says how far each one was tested. The
weights live at [fbjr/h3-mutant-distill](https://huggingface.co/fbjr/h3-mutant-distill)
and are not versioned here.

## 0.3.0

### Added

- `h3_i2v_flashgen`: FlashGen alone from a first frame, 4 steps. An untested
  transfer, since FlashGen was trained for text to video only. Set the canvas to
  your image's aspect: the node stretches the image to fit.
- `h3_r2v_pdd8_flashgen_finish`: PDD8 from sigma 1.0 to 0.8, then FlashGen to 0,
  on ref2va with two reference images. The t2v finish carried over to reference
  to video; one scene and one reference held by eye.
- The README lists both, in the workflow table and under "What to try first".
- This changelog.

## 0.2.0

### Added

- `h3_i2v_pdd8`, `h3_r2v_pdd8` and `h3_r2v_flashgen`: PDD8 from a first frame,
  and PDD8 and FlashGen on ref2va with two reference images. All seven example
  workflows ran once on core ComfyUI before release.

### Changed

- The README follows the model card's sections, then install and workflows.
  Install is by `git clone` or Git URL; the pack is not on the ComfyUI Registry,
  so `pyproject.toml` was removed.

## 0.1.0

- `H3ExactLoRA`, one node that applies a MiniMax H3 distill LoRA (PDD8 or
  FlashGen) at the call on the pruned int8 checkpoints, instead of merging it
  into the int8 weights.
- Four text-to-video example workflows: `h3_t2v_pdd8_flashgen_finish`,
  `h3_t2v_pdd6`, `h3_t2v_flashgen_late_blocks` and `h3_t2v_fasth3_contract`.
