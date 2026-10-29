#!/usr/bin/env bash
# Main experiment: train, evaluate base vs trained on identical test tasks, plot. Run on a GPU box.
set -euo pipefail
RUN=${1:-outputs/main}
export TRL_EXPERIMENTAL_SILENCE=1
mkdir -p "$RUN"
pytest -q 2>&1 | tee "$RUN/pytest.txt"
python -m src.train_grpo --config configs/train.yaml --output "$RUN" 2>&1 | tee "$RUN/train.log"
python -m src.evaluate --checkpoint base --split test --output "$RUN/eval_base.json" 2>&1 | tee "$RUN/eval_base.log"
python -m src.evaluate --checkpoint "$RUN/adapter" --split test --output "$RUN/eval_trained.json" 2>&1 | tee "$RUN/eval_trained.log"
python -m src.plot_metrics --run "$RUN" --output "$RUN/figures"
