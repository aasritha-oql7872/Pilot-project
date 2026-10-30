#!/usr/bin/env bash
# Mixed one-hop + two-hop training (outputs/mixed), then evaluation on the held-out splits.
# Compares base, the one-hop model (outputs/main_v2) and the mixed model on the same tasks.
set -euo pipefail
export TRL_EXPERIMENTAL_SILENCE=1
CFG=configs/train_mixed.yaml
python -m src.train_grpo --config $CFG --output outputs/mixed
for SPLIT in test twohop; do
  python -m src.evaluate --config $CFG --checkpoint base --split $SPLIT --output outputs/mixed/eval_${SPLIT}_base.json
  python -m src.evaluate --config $CFG --checkpoint outputs/main_v2/adapter --split $SPLIT --output outputs/mixed/eval_${SPLIT}_onehop.json
  python -m src.evaluate --config $CFG --checkpoint outputs/mixed/adapter --split $SPLIT --output outputs/mixed/eval_${SPLIT}_mixed.json
  python -m src.compare --base outputs/mixed/eval_${SPLIT}_base.json \
    --trained outputs/mixed/eval_${SPLIT}_onehop.json outputs/mixed/eval_${SPLIT}_mixed.json
done
python -m src.plot_metrics --run outputs/mixed --output outputs/mixed/figures
