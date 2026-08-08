# LLaVA-OneVision: Single-Image, Multi-Image, Video in One Model

> Before LLaVA-OneVision (Li et al., August 2024) the open-VLM world had separate lineages: LLaVA-1.5 for single images, multi-image models like Mantis and VILA, video models like Video-LLaVA and Video-LLaMA. Each won its benchmark and failed at the others. LLaVA-OneVision argued a single curriculum could train one model to dominate all three scenarios, and that the emergent task-transfer effects (single-image skills exported to video, multi-image reasoning exported to single-image) beat the sum of specialists. The recipe is deceptively simple: a visual-token budget that stays constant across scenarios, plus an explicit curriculum that moves from single-image to OneVision (multi-image) to video. This lesson reads the budget, the curriculum, and the emergent behaviors.

**Type:** Build
**Languages:** Python (stdlib, token budget solver + curriculum planner)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 06 (any-resolution)
**Time:** ~180 minutes

## Learning Objectives

- Design a visual-token budget that holds constant across single-image, multi-image, and video inputs.
- Order a training curriculum that transfers skills from single-image to video without catastrophic forgetting.
- Explain why a single model beats specialists at the same parameter count when curriculum is done right.
- Name the three emergent capabilities reported by LLaVA-OneVision: multi-camera reasoning, set-of-mark prompting, iPhone-screenshot agent.

## The Problem

Image, multi-image, and video each stress a model differently.

Single-image wants high-resolution tokens (AnyRes, ~2880 visual tokens) to catch OCR and fine detail. Budget per sample: one image, 2880 tokens.

Multi-image wants several images at moderate resolution (~576 tokens each) so reasoning across images fits in context. Budget per sample: 4-8 images, 576 each, 2300-4600 tokens.

Video wants many frames at low resolution (~196 tokens per frame after pooling) to capture temporal dynamics. Budget per sample: 8-32 frames, 196 each, 1600-6200 tokens.

If you train separate models, you pick one budget. If you train one model, you need the budget to scale sensibly across scenarios without blowing context.

Pre-OneVision, the default answer was "train one scenario, ignore the others." Video-LLaVA retrofitted video onto an image model with extra training stages. LLaVA-NeXT added multi-image support with tiling. None handled all three cleanly.

## The Concept

### The OneVision token budget

SigLIP SO400M at 384x384 produces a 27x27 = 729 patch grid. That 729 is the unit every count below is built from. LLaVA-OneVision deliberately keeps the *maximum* token count similar across the three scenarios — the paper's Figure 3 caption says it outright: "The maximum number of visual tokens across different scenarios is designed to be similar, ensuring balanced visual representations to accommodate cross-scenario capability transfer."

- Single image: AnyResMax-9 — a thumbnail plus up to 9 grid crops, each encoded at the full 729. Maximum: `(1 + 9) * 729 = 7290` tokens.
- Multi-image: each image at base resolution, 729 tokens, no pooling. Up to 12 images per instance: `12 * 729 = 8748` tokens.
- Video: 2x2 bilinear pooling on the patch grid, `ceil(27/2) = 14`, so `14 * 14 = 196` tokens per frame. Up to 32 frames: `32 * 196 = 6272` tokens.

7290, 8748, 6272 — different geometry, one budget band. That is the design, and it is why the LLM can transfer capability between scenarios instead of learning each one's token scale separately.

Two details are easy to get wrong. First, the single-image path does *not* pool by default: crops keep all 729 tokens, and bilinear downscaling only kicks in when the crop grid grows past the budget (the config allows grids up to 6x6). Pooling every scenario by some factor is the wrong mental model — it is the *video* path that pools, because 32 frames is where the token count would otherwise explode. Second, if you see `32 * 729` for video anywhere, that is a typo in arXiv v1 of the paper, corrected to `32 * 196` in v3; Figure 3 read 6272 in both.

### The three-stage curriculum

LLaVA-OneVision trains in three stages:

1. Single-image SFT (stage SI). All data is single-image-plus-text. Train on high-resolution AnyRes input. This teaches perception, OCR, and fine-grained understanding. Uses LLaVA-NeXT data plus OneVision-specific single-image data.
2. OneVision SFT (stage OV). Mix single-image + multi-image + video (uniformly sampled frames). Train on the unified token budget. This teaches the model to handle heterogeneous batch shapes. No weight reset — continues from stage SI.
3. Task transfer (stage TT). Continue with a target task mix, typically heavier on multi-image or video depending on product. Optional fine-tune for deployment.

Critical: the curriculum order matters. Training video-first or multi-image-first produces worse image performance than single-image-first, even with the same data. The paper ablates this explicitly.

### Why curriculum works

Single-image training builds the perceptual base. Patch tokens carry fine-grained visual features; the LLM learns to integrate them with text. Multi-image and video introduce structural challenges (which image is which, what happened first) that are hard to learn without a strong perceptual base.

If you train all scenarios from scratch together, the model underfits perception (limited single-image data per batch) and overfits structure (lots of multi-image / video data). Result: a model that follows cross-image reasoning patterns but is visually shallow.

Curriculum ordering gives you perception strength from stage SI, then compositional/temporal reasoning from stage OV, without losing either.

### Emergent cross-scenario skills

The LLaVA-OneVision paper reports three emergent capabilities:

1. Multi-camera reasoning. Trained on multi-image + video separately; at inference, asked to reason about a multi-camera driving scene. The model correctly integrates the views despite never seeing that exact format in training.
2. Set-of-mark prompting. User annotates objects in an image with numbered marks; the model reasons about "what is mark 3 doing relative to mark 7." Trained on neither marks nor annotation; learned from the combination of spatial grounding + multi-image reference.
3. iPhone-screenshot agent. User provides a screenshot of an iPhone screen and asks to plan the next click. Trained on UI screenshots, video of user workflows, and multi-image before/after pairs. Generalizes to the agent use case.

These are not trained tasks; they emerge from the curriculum's compositional structure.

### Visual-token pooling

Where the budget does force pooling, OneVision uses bilinear interpolation on the 2D patch grid. The shipped video path pools by 2: `ceil(27 / 2) = 14`, so 729 patches become `14 * 14 = 196`. Pooling happens in patch-grid space, not token space, to preserve locality — which is why the count comes from the side length. Divide the token count instead and you get `729 / 4 = 182`, a number that corresponds to no grid at all.

The pooling factor is a hyperparameter, and the trade is the obvious one: less pooling means more tokens and a richer per-frame representation, more pooling means more frames fit under the same budget. OneVision's answer is to pool video and leave single images alone, because a single image is where spatial detail earns its tokens and video is where frame count does.

### LLaVA-OneVision-1.5

The 2025 follow-up (LLaVA-OneVision-1.5, arXiv 2509.23661) is "fully open" in training data, model weights, and code. Matches the proprietary gap on some benchmarks and democratizes the recipe. Same curriculum, more data, better base LLM. No architecture change.

### Contrast with Qwen2.5-VL

Qwen2.5-VL (Lesson 12.09) makes different choices. It uses M-RoPE and dynamic FPS instead of fixed pooling. Its budget scales with input — a 1-minute video uses more tokens than a 5-second video. LLaVA-OneVision fixes the budget and scales the pooling. Both work; they trade configurability for predictability.

```figure
l5-onevision-budget
```

## Use It

`code/main.py` is a curriculum and budget planner for a OneVision-style VLM. Given a token budget per sample and a target scenario mix (say 40% single-image, 30% multi-image, 30% video), it:

- Allocates resolution, pooling factor, and frames per scenario.
- Checks that every scenario fits within the shared budget.
- Reports expected token count, LLM FLOPs, and which scenarios are under-tokenized.
- Prints a stage-by-stage training schedule.

Use it to plan a OneVision fine-tune or to sanity-check a VLM deployment's per-request cost.

## Ship It

This lesson produces `outputs/skill-onevision-budget-planner.md`. Given a target task distribution and a per-sample budget, it emits the AnyRes factor, per-frame pooling, video frame count, and curriculum stage weights. Use this whenever you train or fine-tune a unified-scenario VLM.

## Exercises

1. Your product supports 80% single-image, 10% multi-image (2-4 images), 10% video (8-16 frames). Design the token budget. Where would you put the extra budget you save from not doing heavy multi-image?

2. Read LLaVA-OneVision Section 4.3 (emergent capabilities). Propose a fourth emergent skill the curriculum would likely unlock but the paper did not report.

3. Swap the curriculum order — train multi-image first, then single-image, then video. Predict which benchmarks degrade and why.

4. The paper reports video benchmarks trained on only 8 frames per sample. Does that generalize to 30-second videos at inference? What breaks first — the token budget or the temporal reasoning?

5. Bilinear pooling of 24x24 patches to 12x12 is a 4x reduction per dim. Implement the pooling in stdlib Python and verify that the mean over each 2x2 block matches the bilinear output.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| OneVision scenario | "Single-image, multi-image, or video" | One of three input shapes the unified VLM handles; the budget stays constant across |
| Token budget | "How many tokens per sample" | Total visual tokens the LLM sees per training / inference sample, typically 3000-4000 |
| Curriculum | "Training order" | Stage ordering (single-image → multi-image → video) chosen for emergent transfer |
| Bilinear pooling | "Token shrink" | Applying bilinear interpolation to the patch grid (2D) to reduce token count while preserving locality |
| Emergent skill | "Not trained, still works" | Capability that appears at inference without matching training data, due to curriculum composition |
| AnyRes-k | "k-tile setup" | k sub-tiles of fixed resolution plus one thumbnail, typical k ∈ {4, 9} |
| Task transfer | "Cross-scenario generalization" | Skills learned on single-image that apply to video (and vice versa) via shared backbone |

## Further Reading

- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5: Fully Open Framework (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin et al. — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
