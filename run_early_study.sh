#!/usr/bin/env bash
# Priority 2: can GRPO improve a PARTIAL warm-start (policy that only sometimes makes the second search)?
#   E0 early SFT checkpoint | E1 E0 + original-reward GRPO | E2 E0 + evidence-reward GRPO  (same init/seeds/budget).
# Development-only (twohop_dev, dev). Sealed holdouts are never touched here. Never deletes/overwrites: every stage is
# skipped if its output exists. Requires sealed evaluation to be finished first (ALLOW_BEFORE_SEALED=1 to override).
set -euo pipefail
export TRL_EXPERIMENTAL_SILENCE=1
[ -f outputs/study_sft/summary_sealed.md ] || [ "${ALLOW_BEFORE_SEALED:-0}" = "1" ] || \
  { echo "Finish and save the sealed evaluation first (run_sealed.sh)."; exit 4; }
S=outputs/study_early; mkdir -p $S
CFG=configs/train_early_grpo.yaml
SEEDS=${SEEDS:-"42 43"}
STEPS=${STEPS:-"1 2 3 4 5 6 9 12 15 21 30 45 75"}   # candidate checkpoints probed on twohop_dev

# 1. SFT with intermediate checkpoints (same data and recipe as B, training splits only)
[ -f $S/sft/adapter/adapter_config.json ] || python -m src.train_sft --config configs/sft_early.yaml --output $S/sft

# 2. choose the early checkpoint on DEVELOPMENT data: sampled (T=0.7) second-search rate in [0.2, 0.7]
for N in $STEPS; do
  P=$(printf "%03d" $N)
  [ -d $S/sft/ckpt_step$P ] || [ $N = 75 ] || continue
  CK=$S/sft/ckpt_step$P; [ $N = 75 ] && CK=$S/sft/adapter
  [ -f $S/probe_step$P.json ] || python -m src.probe explore --config $CFG --checkpoint $CK --split twohop_dev \
      --samples 4 --temperature 0.7 --answer-format typed --output $S/probe_step$P.json > /dev/null
done
python -m src.explore_gate early --out $S/selection.json $S/probe_step*.json
SEL=$(python -c "import json;print(json.load(open('$S/selection.json'))['selected'] or '')")
if [ -z "$SEL" ]; then
  echo "NO EARLY CHECKPOINT IN THE TARGET RANGE (see $S/selection.json). Stopping: nothing is forced and no GRPO is run."
  exit 3
fi
INIT=$S/sft/ckpt_step$(printf "%03d" $SEL); echo "selected early checkpoint: step $SEL -> $INIT"

ev() { local label=$1 ckpt=$2 split=$3; shift 3
  [ -f $S/eval_${split}_${label}.json ] || python -m src.evaluate --config $CFG --checkpoint $ckpt --split $split \
      --output $S/eval_${split}_${label}.json "$@" | tail -1; }

# 3. E0 evaluation, then matched E1/E2 GRPO (same init, tasks, seeds, sampling, budget)
for SPLIT in dev twohop_dev; do ev E0 $INIT $SPLIT; done
[ -f $S/explore_E0.json ] || python -m src.probe explore --config $CFG --checkpoint $INIT --split twohop_dev \
    --samples 8 --temperature 0.7 --answer-format typed --output $S/explore_E0.json > /dev/null
for SEED in $SEEDS; do
  [ -f $S/E1_s$SEED/adapter/adapter_config.json ] || python -m src.train_grpo --config $CFG --seed $SEED \
      --override init_adapter=$INIT reward.mode=original --output $S/E1_s$SEED
  [ -f $S/E2_s$SEED/adapter/adapter_config.json ] || python -m src.train_grpo --config $CFG --seed $SEED \
      --override init_adapter=$INIT reward.mode=evidence --output $S/E2_s$SEED
  for C in E1 E2; do
    for SPLIT in dev twohop_dev; do ev ${C}_s$SEED $S/${C}_s$SEED/adapter $SPLIT; done
    [ -f $S/explore_${C}_s$SEED.json ] || python -m src.probe explore --config $CFG --checkpoint $S/${C}_s$SEED/adapter \
        --split twohop_dev --samples 8 --temperature 0.7 --answer-format typed --output $S/explore_${C}_s$SEED.json > /dev/null
    python -m src.train_diag $S/${C}_s$SEED --init $INIT --out $S/diag_${C}_s$SEED.json > /dev/null
  done
done

# 4. does the evidence bonus actually separate sampled trajectories? (group-level reward variance, real rollouts)
python -m src.traj_stats $S/E1_s* $S/E2_s* --out $S/traj_stats.json > /dev/null
ENTRIES=()
for SPLIT in dev twohop_dev; do
  ENTRIES+=("E0 early SFT (step $SEL)=$S/eval_${SPLIT}_E0.json")
  for SEED in $SEEDS; do ENTRIES+=("E1 +GRPO orig (seed $SEED)=$S/eval_${SPLIT}_E1_s$SEED.json" "E2 +GRPO evidence (seed $SEED)=$S/eval_${SPLIT}_E2_s$SEED.json"); done
done
COMP=(); for SEED in $SEEDS; do COMP+=("E1_s$SEED=$S/E1_s$SEED" "E2_s$SEED=$S/E2_s$SEED"); done
python -m src.summarize_study --out $S/summary_early_dev.md "${ENTRIES[@]}" --compute "${COMP[@]}"
python -m src.manifest --roots $S --out $S/experiment_manifest.json
echo "done: $S (development-only, exploratory)"
