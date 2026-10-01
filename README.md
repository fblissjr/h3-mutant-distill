## h3 mutant distills

Experimental distill adapters for h3 in comfyui safetensor format. YMMV.
¯\\\_(ツ)\_/¯ on if they're any good or not.

## How to run in ComfyUI

**Do not** use the standard load LoRA node (I mean you can if you want, I
guess): sorry for adding yet more custom code, but requires using the node in
this repo because
ComfyUI's LoRA loader merges a LoRA into the int8 checkpoint by
requantizing it, which rounds away most of these adapters; the node applies
them at the call instead.

One node, `H3 Exact LoRA (FlashGen / PDD)`, loads every adapter here, PDD8
and FlashGen alike. For a PDD8 file it also installs the modulation update
and the per-step output heads, so no separate PDD node is needed. On every
adapter recipe we checked (text, image and reference to video) it renders
bit-identical final latents, video and audio, to the PDD and LoRA nodes in
[ComfyUI-h3-explorations](https://github.com/fblissjr/ComfyUI-h3-explorations)
that the recipes were judged with.

1. Install this node: in `ComfyUI/custom_nodes/`, run
   `git clone https://github.com/fblissjr/h3-mutant-distill`, then restart
   ComfyUI.
2. Get the adapters from
   [fbjr/h3-mutant-distill](https://huggingface.co/fbjr/h3-mutant-distill) on
   Hugging Face and put them in your ComfyUI `models/loras/` folder.
3. Open a workflow from `example_workflows/`. The frontend offers to
   download any model that is missing.

### Workflows

ref2va seems to work but how well I don't know. Mostly tested with t2v. YMMV (again).
One workflow each, with a note on why.

| workflow | what | model evaluations |
|---|---|---|
| `h3_t2v_pdd8_flashgen_finish` | PDD8 from sigma 1.0 to 0.8, then FlashGen to 0 | 6 + 2 |
| `h3_t2v_pdd6` | the PDD8 LoRA on a 6-step schedule. Only for close-ups and low-motion scenes, and iffy even there; included anyway | 6 |
| `h3_t2v_flashgen_late_blocks` | FlashGen applied to DiT blocks 34-49 only; a curiosity | 4 |
| `h3_i2v_pdd8` | PDD8 alone from a first frame; the i2v pick | 8 |
| `h3_r2v_pdd8` | PDD8 alone on ref2va, two reference images | 8 |
| `h3_r2v_flashgen` | FlashGen alone on ref2va; an untested transfer | 4 |
| `h3_r2v_pdd8_flashgen_finish` | PDD8 from sigma 1.0 to 0.8, then FlashGen to 0, on ref2va with two reference images; the t2v finish carried over | 6 + 2 |
| `h3_i2v_flashgen` | FlashGen alone from a first frame; an untested transfer | 4 |
| `h3_t2v_fasth3_contract` | FastVideo's FastH3 on FastVideo's own sampling settings; core nodes only | 8 |

The renders these were judged on also ran Sol-Attn, the sparse attention
from [ComfyUI-h3-explorations](https://github.com/fblissjr/ComfyUI-h3-explorations),
where the node's parity check and the records behind every verdict live.
These workflows use ComfyUI's kitchen attention instead.

## What to try first

Each was judged by eye (blinded as best as I could), by me, often on 1-2 seeds
max, on an RTX 4090 at 1344x768 and 345 frames.

| try | adapters | task | verdict |
|---|---|---|---|
| **first** | PDD8, then FlashGen for the last 2 steps | text to video | The pick for t2v: 8 steps, the same time as PDD8 alone. Never worse than PDD8 alone, and fixed its garbled sign text |
| **first** | PDD8 alone | image to video | The pick for i2v: the FlashGen finish brightened the frame at once |
| worth a try | FlashGen alone, 4 steps | text to video | Fastest. Coherent motion and detail; loses track of who does what in busy multi-person scenes |
| worth a try | PDD8 alone | reference to video | The ref2va PDD8 we run; not compared against the alternatives |
| experimental | PDD8 on a 6-step schedule | text to video | Close-ups and low motion only, and iffy even there |
| experimental | PDD8, then FlashGen for the last 2 steps | reference to video | The t2v pick carried to ref2va. One scene, one reference, and the reference held by eye; not compared against PDD8 alone on ref2va |
| maybe crap | FlashGen on DiT blocks 34-49 only | text to video | More natural on one figure; people and objects fall apart in busy scenes. A curiosity |
| maybe crap | FlashGen for ref2va | reference to video | Untested transfer: FlashGen was trained for text to video only. One render held its references |
| maybe crap | FlashGen for i2v | image to video | Untested transfer: FlashGen was trained for text to video only. One render held the first frame's subject and lighting; not compared against PDD8, the i2v pick |

## Files

Nothing here was trained: each file is someone else's distill, converted for
ComfyUI and the pruned int8 checkpoints of
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3). Load
them with the pack's `H3 Exact LoRA` node, not `LoraLoaderModelOnly`.

### PDD8 for fl2va

[`minimax_h3_fl2va_pdd_8step_comfy.safetensors`](https://huggingface.co/fbjr/h3-mutant-distill/blob/main/minimax_h3_fl2va_pdd_8step_comfy.safetensors)

- **Loads on:** `minimax_h3_fl2va_pruned_int8_convrot`.
- **Taken from PDD:** alibaba-pai's FL2VA 8-step Parallel Decoding
  Distillation LoRA, all three parts: the backbone LoRA (every block and
  the token refiner, rank 64), the modulation update, and the bank of 32
  per-interval output heads.
- **Taken from the checkpoint:** its 8-column time basis, which the
  modulation update is pre-solved into, and a few fingerprint tensors the
  node checks before patching.
- **The gist:** PDD distills the base's trajectory into 8 steps, each step
  using the heads for the slice of the trajectory it covers. By eye it is
  natural, flatter and dimmer than the other distills, with the least
  motion. The recipes use it alone (PDD6 runs the same file on 6 steps), and
  for PDD8's first 6 steps before a FlashGen finish.

### PDD8 for ref2va

[`minimax_h3_ref2va_pdd_8step_comfy.safetensors`](https://huggingface.co/fbjr/h3-mutant-distill/blob/main/minimax_h3_ref2va_pdd_8step_comfy.safetensors)

- **Loads on:** `minimax_h3_ref2va_pruned_int8_convrot`.
- **Taken from PDD:** alibaba-pai's Ref2VA 8-step LoRA, the same three
  parts.
- **Taken from the checkpoint:** ref2va's own time basis and fingerprints.
  The node refuses a PDD file on the other partition.
- **The gist:** PDD8 for reference to video.

Both PDD8 files were converted by
[`bench/convert_pdd_lora.py`](https://github.com/fblissjr/ComfyUI-h3-explorations/blob/main/bench/convert_pdd_lora.py):
diffusers names to ComfyUI's, q/k/v fused, SwiGLU halves reordered, alpha
tensors added, and the modulation update pre-solved into the pruned
checkpoint's time basis. They are byte-identical to the copies on
[fbjr/MiniMax-H3-Acc-LoRAs-sidecar](https://huggingface.co/fbjr/MiniMax-H3-Acc-LoRAs-sidecar).

### FlashGen for fl2va (text to video)

[`minimax_h3_flashgen_4step_v1.0_768p_fl2va_pruned_rank64_comfy.safetensors`](https://huggingface.co/fbjr/h3-mutant-distill/blob/main/minimax_h3_flashgen_4step_v1.0_768p_fl2va_pruned_rank64_comfy.safetensors)

- **Loads on:** `minimax_h3_fl2va_pruned_int8_convrot`.
- **Taken from FlashGen:** every module of the LoRA (attention q/k/v and
  output, MLP fc1 and fc2, each block's and the final layer's modulation,
  the token refiner) at its full rank 64. Nothing is dropped or resized.
- **Taken from the checkpoint:** its 8-column time basis (`adaln_t_table`),
  which the modulation part is refit onto.
- **The gist:** FlashGen is a 4-step distill of H3 for text to video, trained
  by distribution matching (VSD, data-free). It renders in 4 steps alone. In
  the recipes it also finishes PDD8's last two steps, where it fixed sign
  text PDD8 garbled.

### FlashGen for ref2va (reference to video)

[`minimax_h3_flashgen_4step_v1.0_768p_ref2va_pruned_rank64_comfy.safetensors`](https://huggingface.co/fbjr/h3-mutant-distill/blob/main/minimax_h3_flashgen_4step_v1.0_768p_ref2va_pruned_rank64_comfy.safetensors)

- **Loads on:** `minimax_h3_ref2va_pruned_int8_convrot`.
- **Taken from FlashGen:** the same modules, at the same full rank.
- **Taken from the checkpoint:** ref2va's own time basis. fl2va's and
  ref2va's differ, so the two files are not interchangeable.
- **The gist:** an untested transfer. FlashGen was trained only for text to
  video on fl2va. One render held both references and the likeness by eye;
  it has not been compared against PDD8.

### How both FlashGen files were converted

From [Beidouqixing/minimax-h3-4step-lora-flashgen](https://huggingface.co/Beidouqixing/minimax-h3-4step-lora-flashgen) (published under Apache-2.0):

- keys renamed from PEFT to ComfyUI's, with an alpha tensor per module so
  strength 1.0 is the publisher's merge scale;
- q/k/v `lora_B` rows reordered from per-head interleaved to ComfyUI's bands;
- the modulation `lora_A` re-expressed in the checkpoint's 8-column time
  basis, its mean moved into `diff_b`.

Each file's metadata records its source and conversion. The converter is
[`reference/convert_flashgen_lora.py`](https://huggingface.co/fbjr/h3-mutant-distill/blob/main/reference/convert_flashgen_lora.py),
with the commands to rebuild both files in
[`reference/README.md`](https://huggingface.co/fbjr/h3-mutant-distill/blob/main/reference/README.md).

## License

The node's code is MIT ([`LICENSE`](https://github.com/fblissjr/h3-mutant-distill/blob/main/LICENSE) on GitHub). The
adapters are MiniMax H3 Model Derivatives under the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
([`LICENSE`](https://huggingface.co/fbjr/h3-mutant-distill/blob/main/LICENSE)
and [`NOTICE`](https://huggingface.co/fbjr/h3-mutant-distill/blob/main/NOTICE)
on Hugging Face).
