#!/usr/bin/env bash
# Everything from scratch on a GPU: tests, seed-42 run, seeds 43/44, mixed one-hop+two-hop run.
# ~35-40 min on an L4. Outputs: outputs/main_v2, outputs/seed43, outputs/seed44, outputs/mixed.
set -euo pipefail
export TRL_EXPERIMENTAL_SILENCE=1
python -m pytest -q
python -m src.train_grpo --config configs/train.yaml --output outputs/main_v2
python -m src.evaluate --checkpoint base --split test --output outputs/main_v2/eval_base.json
python -m src.evaluate --checkpoint outputs/main_v2/adapter --split test --output outputs/main_v2/eval_trained.json
python -m src.plot_metrics --run outputs/main_v2 --output outputs/main_v2/figures
bash run_seeds.sh
bash run_mixed.sh
