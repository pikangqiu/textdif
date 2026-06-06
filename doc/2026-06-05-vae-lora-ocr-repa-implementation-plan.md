# VAE Encoder LoRA + OCR-REPA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-stage text-SR experiment: train a CODSR-inspired RAGP + LQ-modulation teacher with VAE encoder LoRA, then distill it with guided target + OCR-REPA while reusing the frozen VAE LoRA.

**Architecture:** Follow CODSR's PEFT `LoraConfig` target selection for the VAE encoder and `quant_conv`. During teacher training, only the LQ encoding uses the trainable adapter; the HQ target is encoded with adapters disabled so the flow-matching target space stays fixed. Checkpoints save DiT and VAE LoRA separately; distillation and inference load and freeze the same adapter.

**Tech Stack:** PyTorch, Diffusers `AutoencoderKL`, PEFT, Accelerate, safetensors, pytest.

---

### Task 1: VAE Encoder LoRA Utility

**Files:**
- Create: `models/vae_encoder_lora.py`
- Create: `tests/test_vae_encoder_lora.py`
- Modify: `requirements.txt`

- [ ] Write failing tests for CODSR-compatible target collection, adapter-only trainability, state filtering, and checkpoint path resolution.
- [ ] Run `pytest tests/test_vae_encoder_lora.py -v` and confirm failure because the utility does not exist.
- [ ] Implement target normalization/collection, PEFT adapter setup, trainable parameter selection, adapter enable/disable context, and safetensors save/load helpers.
- [ ] Run `pytest tests/test_vae_encoder_lora.py -v` and confirm all tests pass.

### Task 2: Multi-Step Teacher Training

**Files:**
- Modify: `train_vosr.py`
- Create: `configs/train_yml/one_step/text_codsr/VOSR_0.5B_text_ragp_lq_mod_vae_lora.yml`

- [ ] Add tests for paired encoding: adapted LQ latent plus base-VAE HQ latent.
- [ ] Initialize rank-4 `default_encoder` adapter only when `use_vae_encoder_lora: true`.
- [ ] Add VAE LoRA parameters to the optimizer with `vae_lora_learning_rate`.
- [ ] Keep the VAE frozen and existing combined no-grad encoding unchanged when the feature is disabled.
- [ ] Save `clean_weights/vae_encoder_lora.safetensors` with each checkpoint.

### Task 3: Distillation and Inference Loading

**Files:**
- Modify: `train_vosr_distill.py`
- Modify: `inference_vosr.py`
- Modify: `inference_vosr_onestep.py`
- Create: `configs/train_yml/one_step/text_codsr_distill/VOSR_0.5B_text_ragp_lq_mod_vae_lora_guided_target_ocr_repa.yml`

- [ ] Load the teacher checkpoint's VAE LoRA before latent encoding.
- [ ] Freeze all VAE parameters during distillation.
- [ ] Reuse adapted LQ/base HQ encoding in distillation and FD queue initialization.
- [ ] Auto-resolve VAE LoRA beside a checkpoint during inference, while allowing an explicit CLI/config path.
- [ ] Keep legacy inference unchanged when no VAE LoRA is configured or present.

### Task 4: Four-GPU Commands and Documentation

**Files:**
- Modify: `scripts/train_text_codsr_ablation.sh`
- Modify: `scripts/train_text_codsr_distill.sh`
- Modify: `scripts/infer_text_codsr_ablation_multigpu.sh`
- Create: `doc/vae_lora_ocr_repa_experiment.md`

- [ ] Add `ragp_lq_mod_vae_lora` teacher mode.
- [ ] Add `vae_lora_ocr_repa` distillation mode with checkpoint validation.
- [ ] Add inference modes for both stages.
- [ ] Document exact 4x4090 commands, expected paths, and fair comparison controls.

### Task 5: Verification

- [ ] Run focused pytest tests.
- [ ] Run `python -m py_compile` on every modified Python entrypoint.
- [ ] Parse both new YAML configurations and validate all referenced paths.
- [ ] Run `bash -n` on all modified scripts.
- [ ] Verify disabled-feature configs do not contain VAE LoRA keys and retain the original code path.
