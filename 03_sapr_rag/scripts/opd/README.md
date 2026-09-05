# SAPR-OPD

External-teacher, state-aligned on-policy distillation for the existing
SAPR-RAG multi-turn retrieval pipeline.

## Fixed layout on an 8×H20 Worker

```text
GPU0    retrieval daemon
GPU1    frozen Qwen2.5-14B teacher server
GPU2-6  Qwen2.5-7B SFT student LoRA training
GPU7    student rollout server
```

The teacher receives the same student-generated messages and retrieval
observations. No `teacher_prompt`, gold answer, gold evidence, or R3 query plan
is included in the teacher context.

## Execution order

```bash
# 1. Verify that the 14B model is stronger than the SFT student.
bash 03_sapr_rag/scripts/opd/evaluate_teacher_ceiling.sh

# 2. Run a two-step plumbing smoke.
bash 03_sapr_rag/scripts/opd/run_opd_smoke.sh

# 3. Start the gated 1,000-step run after the ceiling and smoke pass.
bash 03_sapr_rag/scripts/opd/run_opd_formal.sh
```

Run detached on a Worker:

```bash
RUN_NAME=opd_sft_14b_failed_em_s1000_$(date +%Y%m%d_%H%M%S)
LOG=03_sapr_rag/scripts/opd/logs/$RUN_NAME/launcher.log
mkdir -p "$(dirname "$LOG")"
nohup setsid env RUN_NAME="$RUN_NAME" \
  bash 03_sapr_rag/scripts/opd/run_opd_formal.sh \
  >"$LOG" 2>&1 < /dev/null &
```

## Main controls

```text
OPD_MODE=pure|hybrid
TEACHER_SEQUENCE_GATE=none|failed_em|failed_f1
TEACHER_SEQUENCE_GATE_THRESHOLD=1.0
TEACHER_KL_COEF=0.01
MAX_STEPS=1000
```

`pure` zeros the GRPO base advantage and trains only from the gated teacher
log-ratio. `hybrid` adds the same gated teacher signal to the existing GRPO
advantage.

## Required checks

- `teacher_sequence_gate_ratio` is finite and between 0 and 1.
- `teacher_kl` and `teacher_kl_scoped` are finite.
- Correct EM trajectories have no teacher contribution.
- Retrieval observations remain masked.
- Student rollout really loads SFT `checkpoint-1650`.
- Teacher and student tokenizers have identical hashes.

See `docs/opd_plan.md` for the complete method and experiment matrix.
