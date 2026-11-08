#!/usr/bin/env bash
# Inference-only control (~6 min on an L4, no training, no sealed data): how does M2 behave under the `typed` reminder it was
# trained with, versus the `typed_abstain` reminder used for the hard benchmark? Separates a reminder effect from a GRPO effect.
set -euo pipefail
export TRL_EXPERIMENTAL_SILENCE=1
S=outputs/study_hard; SFT=outputs/study_sft; CFG=configs/train_hard_grpo.yaml
for SPLIT in hard_dev twohop_dev dev; do
  [ -f $S/eval_${SPLIT}_M2_typed.json ] || python -m src.evaluate --config $CFG --checkpoint $SFT/B/adapter --split $SPLIT \
      --answer-format typed --output $S/eval_${SPLIT}_M2_typed.json | tail -1
done
echo "format control written: $S/eval_{hard_dev,twohop_dev,dev}_M2_typed.json"
