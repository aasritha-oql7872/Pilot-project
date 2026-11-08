#!/usr/bin/env bash
# LOCKED final evaluation on hard_final. Run ONCE, only after (1) run_headroom_study.sh finished and (2) `python -m src.freeze_hard`
# wrote outputs/study_hard/FROZEN_HARD.json (immutable, read-only). Every model is evaluated with IDENTICAL settings (greedy,
# typed_abstain reminder, 2-search budget, same system prompt) so the table is complete; the headline is whatever the freeze
# manifest says, not what looks best here. Existing result files are never overwritten: a crashed run resumes by re-running this
# script (finished evaluations are skipped; partial ones have no output file and are redone).
set -euo pipefail
export TRL_EXPERIMENTAL_SILENCE=1
S=outputs/study_hard; SFT=outputs/study_sft; CFG=configs/train_hard_grpo.yaml; SEEDS=${SEEDS:-"42 43"}
[ -f $S/FROZEN_HARD.json ] || { echo "run: python -m src.freeze_hard  (writes $S/FROZEN_HARD.json) first"; exit 4; }
[ -f $S/summary_hard_dev.md ] || { echo "run_headroom_study.sh has not finished"; exit 4; }
python -m src.manifest --verify-hard
python -m src.manifest --verify-sealed
python - <<'PY'
import json
m = json.load(open("outputs/study_hard/FROZEN_HARD.json"))
print("FROZEN:", m["headline_label"], "|", m["checkpoint_path"], "| frozen", m["frozen_at_utc"])
print(m["sealed_view_statement"])
PY
# record when hard_final was first evaluated (append-only)
[ -f $S/HARD_FINAL_FIRST_EVALUATION ] || date -u +%FT%TZ > $S/HARD_FINAL_FIRST_EVALUATION
if [ -f outputs/main_v2/adapter/adapter_config.json ]; then A=outputs/main_v2/adapter; else A=outputs/main_v2_retrained/adapter; fi
ev() { local label=$1 ckpt=$2 fmt=${3:-typed_abstain}; [ -f $S/eval_hard_final_${label}.json ] || python -m src.evaluate --config $CFG --checkpoint $ckpt \
      --split hard_final --answer-format $fmt --output $S/eval_hard_final_${label}.json | tail -1; }
ev base base; ev M1 $A; ev M2 $SFT/B/adapter
ev M2_typed $SFT/B/adapter typed        # format control: M2 under the reminder it was trained with (secondary, pre-declared)
ENT=("base=$S/eval_hard_final_base.json" "M1 original GRPO=$S/eval_hard_final_M1.json" "M2 SFT=$S/eval_hard_final_M2.json"
     "M2 SFT (typed reminder, format control)=$S/eval_hard_final_M2_typed.json")
for SEED in $SEEDS; do for M in M3 M4 M4g; do
  [ -f $S/${M}_s$SEED/adapter/adapter_config.json ] || continue
  ev ${M}_s$SEED $S/${M}_s$SEED/adapter; ENT+=("${M} seed $SEED=$S/eval_hard_final_${M}_s$SEED.json")
done; done
python -m src.report_categories --out $S/summary_hard_final.md --reference "M2 SFT=$S/eval_hard_final_M2.json" "${ENT[@]}"
python -m src.manifest --roots $S --out $S/experiment_manifest_final.json
echo "final evaluation done: $S/summary_hard_final.md"
