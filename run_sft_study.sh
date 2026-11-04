#!/usr/bin/env bash
# Supervised warm-start study. Conditions: A existing one-hop GRPO | B SFT only | C SFT+GRPO (original reward)
# | D SFT+GRPO (evidence reward). GPU required (~45 min on an L4). Never deletes or overwrites existing results:
# every stage is skipped if its output exists, and trainers refuse to overwrite an adapter directory.
# Model selection uses ONLY dev and twohop_dev. Sealed holdouts need FINAL=1 AND a file outputs/study_sft/FROZEN
# (create it yourself after choosing the method).
set -euo pipefail
export TRL_EXPERIMENTAL_SILENCE=1
S=outputs/study_sft; mkdir -p $S
CFG=configs/train_sft_grpo.yaml     # shared eval settings: 384-token budget, greedy, typed reminder, 2-search cap

python -m pytest -q
python -m src.sft_data --output $S/sft_data.jsonl                               # inspect the trajectories
python -m src.train_sft --config configs/sft.yaml --output $S/B_dryrun --dry-run  # tokenization/mask statistics

# A: existing one-hop model (restore from Drive beforehand, or it is retrained with the original recipe)
if [ -f outputs/main_v2/adapter/adapter_config.json ]; then A=outputs/main_v2/adapter
else
  [ -f outputs/main_v2_retrained/adapter/adapter_config.json ] || \
    python -m src.train_grpo --config configs/train.yaml --output outputs/main_v2_retrained
  A=outputs/main_v2_retrained/adapter
fi

ev() { # label checkpoint split [extra evaluate args]
  local label=$1 ckpt=$2 split=$3; shift 3
  [ -f $S/eval_${split}_${label}.json ] || python -m src.evaluate --config $CFG --checkpoint $ckpt --split $split \
      --output $S/eval_${split}_${label}.json "$@"
}

# ---- B: SFT only, then the gate -------------------------------------------------------------------------
[ -f $S/B/adapter/adapter_config.json ] || python -m src.train_sft --config configs/sft.yaml --output $S/B
ev B $S/B/adapter twohop_dev
ev B $S/B/adapter dev
# A and base under both reminders, before the gate (cached; the later matched loop reuses them)
for SPLIT in twohop_dev dev; do
  ev base_default base $SPLIT --answer-format default
  ev base base $SPLIT
  ev A_default $A $SPLIT --answer-format default
  ev A $A $SPLIT
done
python -m src.gate $S/eval_twohop_dev_B.json --min-second-search 0.5 --min-accuracy 0.3   # exits 3 and stops here if B fails
python -m src.probe explore --config $CFG --checkpoint $S/B/adapter --split twohop_dev --samples 8 \
    --answer-format typed --output $S/explore_B.json

# ---- C, D: GRPO from the warm-start (3-step smoke first; the trainer raises if LoRA weights do not move) ------
[ -f $S/smoke_D/adapter/adapter_config.json ] || \
  python -m src.train_grpo --config configs/train_sft_grpo_evidence.yaml --smoke --output $S/smoke_D
[ -f $S/C/adapter/adapter_config.json ] || python -m src.train_grpo --config configs/train_sft_grpo.yaml          --output $S/C
[ -f $S/D/adapter/adapter_config.json ] || python -m src.train_grpo --config configs/train_sft_grpo_evidence.yaml --output $S/D

# ---- matched dev evaluation ------------------------------------------------------------------------------
for SPLIT in dev twohop_dev; do
  ev base_default base $SPLIT --answer-format default
  ev base base $SPLIT
  ev A_default $A $SPLIT --answer-format default     # A as it was trained
  ev A $A $SPLIT                                     # A under the typed reminder (matched with B/C/D)
  ev C $S/C/adapter $SPLIT
  ev D $S/D/adapter $SPLIT
done

# ---- diagnostics -----------------------------------------------------------------------------------------
python -m src.audit episodes $S/eval_twohop_dev_{base,A_default,A,B,C,D}.json | tee $S/audit_episodes.json
for NAME in base A B C D; do
  case $NAME in base) CK=base;; A) CK=$A;; *) CK=$S/$NAME/adapter;; esac
  [ -f $S/forced_$NAME.json ] || python -m src.probe forced --config $CFG --checkpoint $CK --split twohop_dev \
      --answer-format typed --output $S/forced_$NAME.json
done
for NAME in C D; do
  [ -f $S/explore_$NAME.json ] || python -m src.probe explore --config $CFG --checkpoint $S/$NAME/adapter \
      --split twohop_dev --samples 8 --answer-format typed --output $S/explore_$NAME.json
done
for NAME in B D; do   # ORACLE-STATE reference (uses gold evidence ids): not comparable to Qwen's information
  python -m src.mdp_baseline --episodes $S/eval_twohop_dev_$NAME.json --forced $S/forced_base.json \
      --output $S/mdp_oracle_$NAME.json > /dev/null || echo "mdp $NAME: skipped (state without observations)"
done

python -m src.summarize_study --out $S/summary_dev.md \
  "base (default reminder)=$S/eval_dev_base_default.json" "A one-hop GRPO (default reminder)=$S/eval_dev_A_default.json" \
  "A (typed reminder)=$S/eval_dev_A.json" "B SFT only=$S/eval_dev_B.json" "C SFT+GRPO orig=$S/eval_dev_C.json" \
  "D SFT+GRPO evidence=$S/eval_dev_D.json" \
  "base (default reminder)=$S/eval_twohop_dev_base_default.json" "A one-hop GRPO (default reminder)=$S/eval_twohop_dev_A_default.json" \
  "A (typed reminder)=$S/eval_twohop_dev_A.json" "B SFT only=$S/eval_twohop_dev_B.json" \
  "C SFT+GRPO orig=$S/eval_twohop_dev_C.json" "D SFT+GRPO evidence=$S/eval_twohop_dev_D.json" \
  --compute B=$S/B C=$S/C D=$S/D
for NAME in B C D; do python -m src.plot_metrics --run $S/$NAME --output $S/$NAME/figures || true; done

# ---- sealed holdouts: once, after the method is frozen -------------------------------------------------------
if [ "${FINAL:-0}" = "1" ]; then
  [ -f $S/FROZEN ] || { echo "FINAL=1 requires $S/FROZEN (write the chosen method into it first)"; exit 4; }
  for SPLIT in final twohop_final; do
    ev base base $SPLIT; ev A $A $SPLIT; ev B $S/B/adapter $SPLIT; ev C $S/C/adapter $SPLIT; ev D $S/D/adapter $SPLIT
  done
  python -m src.summarize_study --out $S/summary_sealed.md \
    "base=$S/eval_final_base.json" "A=$S/eval_final_A.json" "B=$S/eval_final_B.json" "C=$S/eval_final_C.json" "D=$S/eval_final_D.json" \
    "base=$S/eval_twohop_final_base.json" "A=$S/eval_twohop_final_A.json" "B=$S/eval_twohop_final_B.json" \
    "C=$S/eval_twohop_final_C.json" "D=$S/eval_twohop_final_D.json"
fi
echo "done: $S"
