#!/usr/bin/env bash
# Does GRPO improve an SFT agent where the SFT agent has room to improve?  (reports/controlled_experiments.md)
#   M1 original GRPO (condition A) | M2 SFT only (B, FROZEN) | M3 SFT + original GRPO | M4 SFT + evidence-aware GRPO
#   (+ M4g SFT + grounded GRPO as a separately labelled reward variant; M3/M4/M4g differ ONLY in reward.mode and seed).
# Evidence for every decision is saved as JSON. Selection uses hard_dev only. hard_final is NOT touched here (run_hard_final.sh).
# Stages are gated: headroom -> signal -> training. A failed gate exits 3 with the measured reason; nothing is forced.
# Never deletes/overwrites: finished stages are skipped, trainers refuse to overwrite adapters.
#
# PRE-REGISTERED selection rule (written before any hard_dev result existed): a GRPO variant replaces M2 as the headline
# only if its mean hard_dev accuracy over seeds exceeds M2's by >= 5 points AND dev and twohop_dev each drop by <= 3 points;
# otherwise M2 stays the headline. Compute budget (L4): ~10 min inference for stage 1, ~10 min for stage 2, ~10 min per
# GRPO run (6 runs with SEEDS="42 43") and ~2 min per evaluation (~20 evaluations): about 2.5 h in total.
set -euo pipefail
export TRL_EXPERIMENTAL_SILENCE=1
S=outputs/study_hard; mkdir -p $S
SFT=outputs/study_sft
CFG=configs/train_hard_grpo.yaml
SEEDS=${SEEDS:-"42 43"}
FMT=typed_abstain
[ -f $SFT/B/adapter/adapter_config.json ] || { echo "missing $SFT/B/adapter (the frozen SFT checkpoint)"; exit 2; }
if [ -f outputs/main_v2/adapter/adapter_config.json ]; then A=outputs/main_v2/adapter; else A=outputs/main_v2_retrained/adapter; fi
[ -f $A/adapter_config.json ] || { echo "missing condition A adapter ($A)"; exit 2; }

# 0. benchmark integrity (no model): structure, solvability, disjointness, locked hashes
python -m src.manifest --verify-hard
python -m src.manifest --verify-sealed
python -m src.validate_hard > $S/validate_hard.json
python -m pytest -q tests/test_hard.py tests/test_grpo_mechanics.py

ev() { local label=$1 ckpt=$2 split=$3; shift 3
  [ -f $S/eval_${split}_${label}.json ] || python -m src.evaluate --config $CFG --checkpoint $ckpt --split $split \
      --answer-format $FMT --output $S/eval_${split}_${label}.json "$@" | tail -1; }

# 1. HEADROOM: existing models on hard_dev (greedy, identical reminder/budget). M1 never saw this reminder (limitation noted).
ev base base hard_dev
ev M1 $A hard_dev
ev M2 $SFT/B/adapter hard_dev
ev C_old $SFT/C/adapter hard_dev
ev D_old $SFT/D/adapter hard_dev
python -m src.explore_gate headroom --out $S/decision_headroom.json $S/eval_hard_dev_M2.json
if [ "$(python -c "import json;print(json.load(open('$S/decision_headroom.json'))['proceed'])")" != "True" ]; then
  echo "GATE: the SFT model leaves no headroom on hard_dev (see $S/decision_headroom.json). Not running GRPO."; exit 3
fi

# 2. SIGNAL: would GRPO have informative groups? sampled rollouts of M2 on hard_train only (no weight update)
for T in 0.7 1.0; do
  [ -f $S/signal_T$T.json ] || python -m src.signal_probe --config $CFG --checkpoint $SFT/B/adapter --split hard_train \
      --temperature $T --samples 8 --tasks 48 --answer-format $FMT --output $S/signal_T$T.json > /dev/null
done
python -m src.explore_gate signal --out $S/decision_signal.json $S/signal_T*.json
if [ "$(python -c "import json;print(json.load(open('$S/decision_signal.json'))['proceed'])")" != "True" ]; then
  echo "GATE: fewer than 15% of GRPO groups carry a learning signal in every tested setting (see $S/decision_signal.json). Not training."; exit 3
fi
TEMP=$(python -c "import json;print(json.load(open('$S/decision_signal.json'))['selected']['temperature'])")
GSZ=$(python -c "import json;print(json.load(open('$S/decision_signal.json'))['selected']['group_size'])")
BATCH=16; echo "selected sampling: T=$TEMP group size=$GSZ (training split probe)"

# 3. matched GRPO runs from the SAME frozen checkpoint: only reward.mode and the seed differ
for SEED in $SEEDS; do
  for R in "M3 original" "M4 evidence" "M4g grounded"; do
    set -- $R; NAME=$1_s$SEED
    # a run that crashed before saving its adapter is set aside (never deleted) so the retrain starts clean
    if [ -d $S/$NAME ] && [ ! -f $S/$NAME/adapter/adapter_config.json ]; then
      mv $S/$NAME $S/${NAME}_crashed_$(date +%Y%m%d_%H%M%S); echo "set aside incomplete run -> ${NAME}_crashed_*"
    fi
    [ -f $S/$NAME/adapter/adapter_config.json ] || python -m src.train_grpo --config $CFG --seed $SEED --output $S/$NAME \
        --override reward.mode=$2 grpo.temperature=$TEMP grpo.num_generations=$GSZ grpo.per_device_train_batch_size=$BATCH
    python -m src.train_diag $S/$NAME --init $SFT/B/adapter --out $S/diag_$NAME.json > /dev/null
  done
done
python -m src.traj_stats $S/M3_s* $S/M4_s* $S/M4g_s* --out $S/traj_stats.json > /dev/null
python -m src.audit_lineage M2=$SFT/B $(for SEED in $SEEDS; do echo M3_s$SEED=$S/M3_s$SEED M4_s$SEED=$S/M4_s$SEED M4g_s$SEED=$S/M4g_s$SEED; done) --out $S/lineage.json > /dev/null

# 4. development evaluation (greedy): hard_dev + regression checks on dev (one-hop) and twohop_dev
for SEED in $SEEDS; do for M in M3 M4 M4g; do
  for SPLIT in hard_dev dev twohop_dev; do ev ${M}_s$SEED $S/${M}_s$SEED/adapter $SPLIT; done
done; done
for SPLIT in dev twohop_dev; do ev M2 $SFT/B/adapter $SPLIT; done
if [ "${SAMPLED:-0}" = "1" ]; then   # robustness: sampled decoding, 3 seeds
  for SD in 1 2 3; do
    ev M2_T0.7_r$SD $SFT/B/adapter hard_dev --temperature 0.7 --seed $SD
    for M in M3 M4 M4g; do ev ${M}_s42_T0.7_r$SD $S/${M}_s42/adapter hard_dev --temperature 0.7 --seed $SD; done
  done
fi

# 5. summary tables (per category, evidence-supported rate, abstention, paired tests vs M2)
ENT=("base=$S/eval_hard_dev_base.json" "M1 original GRPO=$S/eval_hard_dev_M1.json" "M2 SFT=$S/eval_hard_dev_M2.json"
     "C_old (SFT+GRPO, easy tasks)=$S/eval_hard_dev_C_old.json" "D_old=$S/eval_hard_dev_D_old.json")
for SEED in $SEEDS; do for M in M3 M4 M4g; do ENT+=("${M} seed $SEED=$S/eval_hard_dev_${M}_s$SEED.json"); done; done
python -m src.report_categories --out $S/summary_hard_dev.md --reference "M2 SFT=$S/eval_hard_dev_M2.json" "${ENT[@]}"
REG=("M2=$S/eval_dev_M2.json")
for SEED in $SEEDS; do for M in M3 M4 M4g; do REG+=("${M} s$SEED=$S/eval_dev_${M}_s$SEED.json"); done; done
python -m src.report_categories --out $S/summary_dev_regression.md --reference "M2=$S/eval_dev_M2.json" "${REG[@]}"
REG2=("M2=$S/eval_twohop_dev_M2.json")
for SEED in $SEEDS; do for M in M3 M4 M4g; do REG2+=("${M} s$SEED=$S/eval_twohop_dev_${M}_s$SEED.json"); done; done
python -m src.report_categories --out $S/summary_twohop_dev_regression.md --reference "M2=$S/eval_twohop_dev_M2.json" "${REG2[@]}"
python -m src.manifest --roots $S --out $S/experiment_manifest.json
echo "done: $S  (hard_dev only; apply the pre-registered selection rule, then run_hard_final.sh once)"
