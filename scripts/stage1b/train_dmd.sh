#!/bin/bash
# Stage 1b: 3-step bidirectional rgb2thermal student by FastVideo LoRA DMD (the phase-1 recipe that matched the 30-step
# CFG teacher at 3 steps): student / real score / fake score all start from the merged stage-1 teacher; LoRA r32,
# lr 1e-4, generator updated every 5 critic steps, EMA from step 0, student steps 1000/757/522, real-score guidance 1.0
# (= CFG 2 on the condition video), no GAN. 33-frame clips (9 latents).
#   source scripts/env.sh && bash scripts/stage1b/train_dmd.sh [extra FastVideo args]
# Needs: scripts/stage1b/setup_fastvideo.sh, the merged teacher (merge_teacher.py) and the parquet data
# (latents_to_parquet.py). Resumes from the newest checkpoint in the output dir. Multi-GPU: NPROC (single node).
set -euo pipefail
: "${FASTVIDEO_DIR:?source scripts/env.sh first}"
TEACHER=${TEACHER:-$THERMAL_RUNS/stage1b_teacher/Wan2.2-TI2V-5B-Diffusers}
DATA=${DATA:-$THERMAL_DATA/fastvideo/rgb2thermal_33f/train}
OUT=${OUT:-$THERMAL_RUNS/stage1b_dmd3}
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
mkdir -p "$OUT"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA TOKENIZERS_PARALLELISM=false WANDB_MODE=${WANDB_MODE:-offline}
export TORCH_NCCL_ENABLE_MONITORING=0 PYTHONUNBUFFERED=1
cd "$FASTVIDEO_DIR"
ARGS=(
  --pipeline-class WanICLDistillationPipeline --pipeline-module fastvideo.training.wan_icl_distillation_pipeline
  --num_gpus "$NPROC" --sp_size 1 --tp_size 1 --hsdp_replicate_dim 1 --hsdp_shard_dim "$NPROC"
  --model_path "$TEACHER" --pretrained_model_name_or_path "$TEACHER" --real_score_model_path "$TEACHER" --fake_score_model_path "$TEACHER"
  --data_path "$DATA" --dataloader_num_workers 4 --output_dir "$OUT"
  --train_batch_size 1 --train_sp_batch_size 1 --gradient_accumulation_steps 1
  --num_latent_t 9 --num_height 512 --num_width 640 --num_frames 33
  --enable_gradient_checkpointing_type full
  --training_state_checkpointing_steps ${CKPT_EVERY:-250} --weight_only_checkpointing_steps ${CKPT_EVERY:-250} --checkpoints_total_limit 4
  --lora_training True --lora_rank ${LORA_RANK:-32}
  --learning_rate ${LR:-1e-4} --mixed_precision bf16 --weight_decay 0.01 --max_grad_norm 1.0
  --inference_mode False --training_cfg_rate 0.0 --dit_precision fp32
  --ema_start_step 0 --flow_shift ${FLOW_SHIFT:-5} --seed 1000
  --dmd_denoising_steps '1000,757,522' --min_timestep_ratio 0.02 --max_timestep_ratio 0.98
  --generator_update_interval 5 --real_score_guidance_scale ${GUIDANCE:-1.0}
  --max_train_steps ${MAX_STEPS:-3000}
)
# --real_score_guidance_scale is FastVideo's (cond + w*(cond-uncond)): w=1.0 == CFG 2.0 on the condition video.
RES=()
last=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | grep -v weight_only | sort -t- -k2 -n | tail -1 || true)
[ -n "$last" ] && RES=(--resume_from_checkpoint "$last") && echo "[stage1b] resuming from $last"
exec $PY_FV -m torch.distributed.run --standalone --nproc_per_node "$NPROC" fastvideo/training/runner.py "${ARGS[@]}" "${RES[@]}" "$@"
