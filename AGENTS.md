# AGENTS.md: runbook for an AI coding agent (Claude Code, Codex, …)

You are setting up and running the **thermal_v2v** training pipeline on a new GPU cluster for a human collaborator.
Read this file fully before doing anything. `README.md` explains the method; this file tells you **what to run, in
which order, how to check each step, and what to report**.

## 0. Goal and deliverables

Train six models. The model tree in `README.md` (`assets/model_tree.png`) labels every node with its step ID
S0–S8 and the scripts it runs, and every model with ①–⑥; use the same IDs when you report:

| # | Model | Produced by | Where it ends up |
|---|---|---|---|
| ① ② | bidirectional multi-step thermal→RGB and RGB→thermal (one LoRA) | stage 1 | `$THERMAL_RUNS/stage1_bidir_33f/ckpt/step_*.pt` |
| ③ | bidirectional 3-step RGB→thermal | stage 1b | `$THERMAL_RUNS/stage1b_dmd3/checkpoint-*_weight_only/ema/generator_ema.safetensors` |
| ④ | streaming autoregressive multi-step RGB→thermal | stage 2 | `$THERMAL_RUNS/stage2_ar/checkpoint_model_*/model.pt` (+ `snapshots/`) |
| ⑤ | streaming 4-step RGB→thermal (self-forcing DMD LoRA on ④) | S7 | `$THERMAL_RUNS/stage3_dmd/checkpoint_model_*/model.pt` |
| ⑥ | streaming 4-step RGB→thermal for long video (⑤ + long-rollout tuning; default S8b = + GT regression) | S8 | `$THERMAL_RUNS/stage3_long/checkpoint_model_*/model.pt` |

Dependencies:
- S4 → (S5, S6). S5 and S6 both start from a chosen S4 checkpoint and can run in parallel on different GPUs.
- S6 → S7 → S8.
- ⑤ ⑥ are LoRAs that are used on top of the ④ checkpoint they were trained on.

⑤ ⑥ follow a recipe validated only on a 1-GPU prototype (TartanRGBT, a single-direction teacher), so check their evals
closely.

## 1. Rules

- **Ask the human before:**
  - submitting multi-node jobs;
  - deleting anything you did not create;
  - using more GPUs than they allotted;
  - spending > ~1 day of GPU time on a single step.
- **Do not change** these without asking:
  - the caption format (`datasets/captions.py`);
  - the LoRA gating (`model.lora.gating: all`; stages 1b and 2 merge the LoRA, which only works for `all`);
  - the two patches (`scripts/stage1b/*.patch`, `scripts/stage2/*.patch`).

  The patches fix real bugs; see README §9. Never run LongLive or FastVideo unpatched.
- **Datasets** are third-party, mostly non-commercial licences. Download them with the scripts; never re-upload or
  redistribute them.
- **Paths** all come from `scripts/env.sh`. Do not hard-code cluster paths in code; put them in `env.sh`.
- **Shell state:** every new shell must `source scripts/env.sh` first.
- **On failure:** read the log, fix the root cause, re-run the step. All steps are resumable/idempotent. Record
  every non-trivial fix in your report.

## 2. S0 · Environment setup (once)

```bash
nvidia-smi                      # GPU model / memory / driver; CUDA 12.8-capable driver needed
df -h <data dir> <runs dir>     # need ~400 GB for data + latents, plus checkpoints (stage-2 ckpt ~50 GB each)
conda --version; git --version
```

Edit `scripts/env.sh`: set `WAN22_5B_DIR`, `THERMAL_DATA`, `THERMAL_RUNS` (and `THERMAL_EXT` if needed) to
locations on a large shared filesystem visible from the GPU nodes. Then:

```bash
source scripts/env.sh
bash scripts/setup/create_envs.sh           # conda env: stage 1 + data
bash scripts/stage1b/setup_fastvideo.sh     # FastVideo + patch + env
bash scripts/stage2/setup_longlive.sh       # LongLive + patch + env + Wan2.2 original-format VAE/tokenizer
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir "$WAN22_5B_DIR"
```

Then `source scripts/env.sh` again (or open a new shell): `env.sh` resolves `PY`, `PY_LL` and `PY_FV` from the conda envs,
which only exist after the setup scripts ran.

**Check:**
- `$PY -c "import torch;print(torch.cuda.is_available())"` prints True; same for `$PY_LL` and `$PY_FV`.
- `git -C $THERMAL_EXT/LongLive log --oneline | head -3` shows thermal-v2v commits on top of `c6a27e7`.
- `git -C $THERMAL_EXT/FastVideo log --oneline | head -3` shows thermal_v2v commits on top of `2b164405`.

If `conda run` is unavailable in batch jobs, set `PY`, `PY_LL`, `PY_FV` in `env.sh` to the absolute python paths of
the three envs.

## 3. S1 · Download data (once; mostly network-bound)

```bash
$PY scripts/data/download_tartanrgbt.py --out $THERMAL_DATA/raw/TartanRGBT \
    --sequences_file configs/data/tartanrgbt_download.txt --delete_zips
bash scripts/data/download_kaist.sh $THERMAL_DATA/raw/KAIST
bash scripts/data/download_llvip.sh $THERMAL_DATA/raw/LLVIP
```

**Check:**
- TartanRGBT: 38 sequence folders, each with `rgb_in_thermal/`, `thermal_left_rect_16/` and `thermal_left_ffc/`.
- KAIST: `$THERMAL_DATA/raw/KAIST/kaist-cvpr15/images/set00..set11`. The md5 is checked by the script.
- LLVIP: `$THERMAL_DATA/raw/LLVIP/LLVIP/{visible,infrared}/{train,test}`, with 12025 train images.

## 4. S2 · Smoke test (MANDATORY before any real run)

```bash
bash scripts/smoke_test.sh       # 1 GPU, ~30 min on an H200, outputs in $THERMAL_RUNS/smoke
```

It must end with `=== … SMOKE TEST PASSED`. On the way it prints six `ok: … in-training eval` lines, one each for
stages 1, 1b, 2, S7, S8a and S8b. Each part can be re-run alone, e.g. `SMOKE_STEPS="1b 2"`. The numbers in the smoke outputs are
meaningless (20 training steps). Only the plumbing is being tested.

**Report:**
- whether it passed;
- the per-step time and peak memory printed by stage 1 (`… s/step mem …G`);
- any fixes you had to make.

## 5. S3 · Encode latents (once; GPU, embarrassingly parallel)

```bash
bash scripts/data/encode_all.sh all          # one process per local GPU; multi-node: NNODES / NODE_RANK
```

**Check:**

| Latent dir | Clips |
|---|---|
| `latents/tartanrgbt_33f/train` | ~7.8k |
| `latents/tartanrgbt_33f/val` | ~130 |
| `latents/kaist_33f/train` | a few thousand |
| `latents/llvip_33f/train` | ~2.4k |
| `latents/tartanrgbt_81f/{train,val}` | |
| `latents/kaist_81f/train` | |

Logs are in `latents/logs/`; grep them for `Error`.

Then run the round-trip check. It decodes one clip per latent dir back to pixels and compares it with the raw frames:

```bash
$PY scripts/data/check_latents.py --out $THERMAL_RUNS/check_latents $THERMAL_DATA/latents/*/train $THERMAL_DATA/latents/*/val
```

It must print `ALL OK`. Look at `$THERMAL_RUNS/check_latents/*.png` (raw RGB | decoded RGB | raw thermal | decoded
thermal); the pairs must line up.

## 6. S4 · Stage 1: bidirectional teacher → models ① ②

```bash
bash scripts/stage1/train.sh                  # single node, all GPUs
# multi-node: NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT on each node (ask the human first)
```

- Defaults are in `configs/stage1/bidir_33f.yaml`: per-GPU batch 2, lr 1e-4, 3000 steps, val every 250, checkpoint
  every 250 (all kept).
- If the GPU count differs from 32, keep per-GPU batch 2, and scale `training.max_steps` so that
  `max_steps × global_batch ≈ 190k` samples.

**Monitor** (README §7):
- `train_log.jsonl`: loss should drop from ~0.6–0.9 to ~0.2–0.35 and must never be NaN;
- the val losses in `train_log.jsonl`;
- the hourly `eval/step_*/{rgb2thermal,thermal2rgb}/clip*.mp4` and `eval/eval_log.jsonl`;
- wandb.

A sound teacher, by the end:
- rgb2thermal `pearson` ≳ 0.7 on TartanRGBT val, with `gain_db` clearly > 0;
- videos with no flicker;
- RGB-invisible structure (sky cold/black, people/engines warm) right.

**Choose the checkpoint** with the human: lowest val loss + best-looking eval videos. Then evaluate it fully:

```bash
bash scripts/stage1/eval.sh step_XXXX.pt rgb2thermal
bash scripts/stage1/eval.sh step_XXXX.pt thermal2rgb
```

## 7. S5 · Stage 1b: 3-step student → model ③ (from the chosen stage-1 checkpoint)

```bash
$PY scripts/stage1b/merge_teacher.py --run_dir $THERMAL_RUNS/stage1_bidir_33f --ckpt step_XXXX.pt \
    --out $THERMAL_RUNS/stage1b_teacher/Wan2.2-TI2V-5B-Diffusers
$PY_FV scripts/stage1b/latents_to_parquet.py --prompt_embeds $THERMAL_RUNS/stage1_bidir_33f/prompt_embeds.pt \
    --src $THERMAL_DATA/latents/tartanrgbt_33f/train:0.6 --src $THERMAL_DATA/latents/kaist_33f/train:0.25 \
    --src $THERMAL_DATA/latents/llvip_33f/train:0.15 --out $THERMAL_DATA/fastvideo/rgb2thermal_33f/train
bash scripts/stage1b/train_dmd.sh             # all local GPUs (single node)
bash scripts/stage1b/eval.sh <step>           # 3-step student vs. the stage-1 30-step evals
```

**Check:**
- hourly `stage1b_dmd3/eval/step_*/rgb2thermal/clip*.mp4`;
- a healthy student is at least as good as the 30-step teacher on psnr_affine / ssim_affine;
- if quality collapses (noise textures), stop and report.

## 8. S6 · Stage 2: streaming AR multi-step RGB→thermal → model ④ (from the chosen stage-1 checkpoint)

```bash
$PY scripts/stage2/export_teacher.py --run_dir $THERMAL_RUNS/stage1_bidir_33f --ckpt step_XXXX.pt \
    --out $THERMAL_RUNS/stage2_teacher
INFER_CFG=configs/thermal/stage2_infer_bidir.yaml bash scripts/stage2/eval_ar.sh $THERMAL_RUNS/stage2_teacher_bidir_eval
bash scripts/stage2/train_ar.sh stage2_ar     # FSDP, all local GPUs
bash scripts/stage2/eval_ar.sh $THERMAL_RUNS/stage2_ar/eval_sNNNN \
    $THERMAL_RUNS/stage2_ar/snapshots/step_NNNNNN.pt --steps 30 --n 8
```

**Check:**
- The bidirectional reference eval (the second command) is the quality ceiling.
- Hourly `stage2_ar/eval/step_*/rgb2thermal/clip*.mp4`.
- `jump_ratio` should approach the real value (~1.02). Values ≫ 1.1 mean visible chunk seams: report them with the
  mp4.
- `pearson` should approach the bidirectional reference.
- A stage-2 checkpoint is ~50 GB; `max_checkpoints: 1` keeps one plus 3 EMA snapshots. Watch disk space.

## 8b. S7 · Self-forcing DMD → model ⑤ (from the chosen S6 checkpoint)

```bash
bash scripts/data/encode_all.sh stage3           # once: 481f long clips (S8) + the 4-min val stream (evals)
STAGE2_CKPT=$THERMAL_RUNS/stage2_ar/snapshots/step_XXXXXX.pt bash scripts/stage3/train_dmd.sh
STAGE2_CKPT=... bash scripts/stage3/eval.sh $THERMAL_RUNS/stage3_dmd/checkpoint_model_XXXXXX/model.pt $THERMAL_RUNS/stage3_dmd/eval_sXXXX
```

**Check** the hourly in-training evals in `stage3_dmd/eval/`: short val clips plus `rgb2thermal_long/stream.mp4`, the
4-min val stream.
- A good ⑤ matches ④ (30 steps) on the short clips at 4 steps. In the prototype that happened by step ~100, and the
  run plateaued after.
- Pick the step by the short-clip evals.
- The long stream may slowly wash out (contrast collapsing for tens of seconds). That is what S8 fixes.

## 8c. S8 · Long-rollout tuning → model ⑥ (from the chosen S7 checkpoint)

```bash
STAGE2_CKPT=<same as S7> STAGE3_LORA=$THERMAL_RUNS/stage3_dmd/checkpoint_model_XXXXXX/model.pt \
    bash scripts/stage3/train_long.sh
STAGE2_CKPT=... bash scripts/stage3/eval.sh $THERMAL_RUNS/stage3_long/checkpoint_model_XXXXXX/model.pt $THERMAL_RUNS/stage3_long/eval_sXXXX long
```

**Check** the 4-min stream in `stage3_long/eval/`:
- the contrast should track the real thermal for the whole stream;
- the short clips must not get worse than ⑤.

S8 has two variants:
- **S8b, recommended and the default:** `configs/thermal/stage3_long_gtreg.yaml`, which adds a ground-truth regression
  term (`gt_regression_weight: 0.25`).
- **S8a, plain DMD:** `STAGE3_CONFIG=configs/thermal/stage3_long.yaml bash scripts/stage3/train_long.sh`.

Prototype reference, step 200:
- stream pearson 0.72 (④, 30 steps) → 0.90 (⑥, 4 steps); short clips unchanged.
- S8a vs S8b, raw weights, 8 short clips:
  - pearson 0.636 → 0.689;
  - near-white pixel excess 4.4% → 2.2%;
  - std ratio 1.17 → 1.00;
  - sharpness ratio 1.13 → 0.99.
- S8a vs S8b at step 400 (raw weights):
  - short clips: pearson 0.645 → 0.685, near-white excess 1.6% → 0.7%;
  - 4-min stream: pearson 0.892 → 0.911, gain over constant +6.89 → +7.70 dB.
- Plain DMD drifts whiter and over-sharpened.

If S8b blurs, or its pearson drops below ⑤, lower `gt_regression_weight` to 0.05–0.1.

⑥ must be run with the 1-pair frame sink it was trained with (`eval.sh … long`, i.e.
`configs/thermal/stage3_infer_long.yaml`).

## 8d. Step counts on more GPUs (S6–S8)

- Stages 2 and 3 use per-GPU batch 1, so the global batch = number of GPUs. Their step counts
  (`max_iters` in `stage2_ar.yaml` 3000, `stage3_dmd.yaml` 1000, `stage3_long{,_gtreg}.yaml` 1100) are the 1-GPU
  prototype values.
- With N GPUs each step sees N clips. As a starting point, divide those counts by roughly N/2 and pick the step by the
  in-training evals, not by the counter.
- S8's `max_iters` counts on from the S7 step it starts from: S7 step 100 + 1000 S8 iterations = 1100.

## 9. Common failures

| Symptom | Likely cause / fix |
|---|---|
| `No space left on device` | Free space or move `THERMAL_RUNS`. Stage-2 checkpoints are ~50 GB, the 1b merged teacher ~20 GB. |
| hangs at start on a shared filesystem | `flock` on NFS. `HF_HOME` / caches must be on a lock-capable disk; `HDF5_USE_FILE_LOCKING=FALSE` is already set. |
| triton / inductor `.so` errors (`failed to map segment from shared object`, permission denied) | `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` must be on an exec-able disk (not a noexec `/dev/shm`). |
| CUDA "driver too old" | The driver must support CUDA 12.8 (stage 1/2) / 12.6 (stage 1b). Use matching wheels. |
| NCCL timeout during the hourly eval | Stage 1 evals on rank 0 while the others wait (timeout raised to 90 min). If still hit, lower `training.eval_clips` / `eval_steps`. |
| wandb errors | Never fatal; runs offline without `WANDB_API_KEY`. `wandb sync` later. |
| `prompt(s) not in LLV2_PROMPT_EMBEDS cache` | Stage 2 must use the stage-1 run's `prompt_embeds.pt` (copied by `export_teacher.py`). |

## 10. What to report back (after each step; always name the step ID S0–S8 and model ①–⑥)

1. The commands you ran and any changes you made (diffs).
2. GPU type/count, wall-clock time, s/step, peak memory.
3. Loss curve summary (start / end / any spikes) from `train_log.jsonl`.
4. The `eval/eval_log.jsonl` metrics over time, plus 2–3 eval mp4s at the latest step, with absolute paths.
5. The checkpoint you propose to keep, and why.
