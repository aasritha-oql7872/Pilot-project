#!/usr/bin/env bash
# Extra seeds for the main configuration (seed 42 is outputs/main_v2). ~4 min per seed on an L4.
# Evaluates base once per split, then every seed's adapter on the same tasks.
set -euo pipefail
export TRL_EXPERIMENTAL_SILENCE=1
for SEED in 43 44; do
  RUN=outputs/seed$SEED
  python -m src.train_grpo --config configs/train.yaml --seed $SEED --output $RUN
  python -m src.evaluate --checkpoint $RUN/adapter --split test --output $RUN/eval_trained.json
  python -m src.evaluate --checkpoint $RUN/adapter --split twohop --output $RUN/eval_twohop_trained.json
done
python -m src.evaluate --checkpoint base --split twohop --output outputs/main_v2/eval_twohop_base.json
python -m src.evaluate --checkpoint outputs/main_v2/adapter --split twohop --output outputs/main_v2/eval_twohop_trained.json
python -m src.compare --base outputs/main_v2/eval_base.json \
  --trained outputs/main_v2/eval_trained.json outputs/seed43/eval_trained.json outputs/seed44/eval_trained.json
python -m src.compare --base outputs/main_v2/eval_twohop_base.json \
  --trained outputs/main_v2/eval_twohop_trained.json outputs/seed43/eval_twohop_trained.json outputs/seed44/eval_twohop_trained.json
