#!/usr/bin/env bash
# LOCKED final evaluation on hard_final. Run ONCE, only after (1) run_headroom_study.sh finished and (2) you wrote the
# headline model into outputs/study_hard/FROZEN_HARD using the pre-registered rule. All models are evaluated with identical
# settings so the table is complete; the headline is whatever FROZEN_HARD says, not what looks best here.
set -euo pipefail
export TRL_EXPERIMENTAL_SILENCE=1
S=outputs/study_hard; SFT=outputs/study_sft; CFG=configs/train_hard_grpo.yaml; SEEDS=${SEEDS:-"42 43"}
[ -f $S/FROZEN_HARD ] || { echo "write the chosen headline model into $S/FROZEN_HARD first"; exit 4; }
[ -f $S/summary_hard_dev.md ] || { echo "run_headroom_study.sh has not finished"; exit 4; }
python -m src.manifest --verify-hard
cat $S/FROZEN_HARD
if [ -f outputs/main_v2/adapter/adapter_config.json ]; then A=outputs/main_v2/adapter; else A=outputs/main_v2_retrained/adapter; fi
ev() { local label=$1 ckpt=$2; [ -f $S/eval_hard_final_${label}.json ] || python -m src.evaluate --config $CFG --checkpoint $ckpt \
      --split hard_final --answer-format typed_abstain --output $S/eval_hard_final_${label}.json | tail -1; }
ev base base; ev M1 $A; ev M2 $SFT/B/adapter
ENT=("base=$S/eval_hard_final_base.json" "M1 original GRPO=$S/eval_hard_final_M1.json" "M2 SFT=$S/eval_hard_final_M2.json")
for SEED in $SEEDS; do for M in M3 M4 M4g; do
  [ -f $S/${M}_s$SEED/adapter/adapter_config.json ] || continue
  ev ${M}_s$SEED $S/${M}_s$SEED/adapter; ENT+=("${M} seed $SEED=$S/eval_hard_final_${M}_s$SEED.json")
done; done
python -m src.report_categories --out $S/summary_hard_final.md --reference "M2 SFT=$S/eval_hard_final_M2.json" "${ENT[@]}"
python -m src.manifest --roots $S --out $S/experiment_manifest_final.json
echo "final evaluation done: $S/summary_hard_final.md"
