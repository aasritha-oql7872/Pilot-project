#!/usr/bin/env bash
# CPU-only analysis + packaging. Reads saved evaluation/training files; trains and evaluates nothing.
#   MODE=pre    before the freeze: claims check, existing-artifact audit tables (safe at any time)
#   MODE=post   after run_hard_final.sh: paired RL-improvement, failure modes, figures, case studies, report, package check
# Output: submission/  (tables/, figures/, raw/, case_studies.md, FINAL_PILOT_REPORT.md, FROZEN_HARD.json)
set -euo pipefail
MODE=${MODE:-post}
S=outputs/study_hard; SFT=outputs/study_sft; SUB=submission; SEEDS=${SEEDS:-"42 43"}
mkdir -p $SUB/tables $SUB/figures
python -m src.verify_claims --study $S --out $SUB/tables/claims_check.md
if [ "$MODE" = "pre" ]; then echo "pre-freeze analysis done (claims_check.md). Next: python -m src.freeze_hard"; exit 0; fi

[ -f $S/FROZEN_HARD.json ] || { echo "no FROZEN_HARD.json: freeze first"; exit 4; }
[ -f $S/eval_hard_final_M2.json ] || { echo "hard_final not evaluated: run run_hard_final.sh first"; exit 4; }
read HV HS <<< "$(python -c "import json;m=json.load(open('$S/FROZEN_HARD.json'));print(m['headline_model'], m['training_seed'])")"
if [ "$HV" = "M2" ]; then HL=M2; else HL=${HV}_s$HS; fi
AFTER=""; for SD in $SEEDS; do for M in M3 M4 M4g; do AFTER="$AFTER ${M}_s$SD"; done; done
echo "headline label: $HL"

# paired RL improvement + behaviour decomposition (same M2 checkpoint before/after)
for SPLIT in hard_final hard_dev twohop_dev; do
  python -m src.rl_improvement --study $S --split $SPLIT --after $AFTER --out $SUB/tables/rl_$SPLIT.json --md $SUB/tables/rl_$SPLIT.md > /dev/null
done

# failure modes from traces (headline + others, M2 as reference)
for SPLIT in hard_final hard_dev; do
  ENT=("M2=$S/eval_${SPLIT}_M2.json"); [ -f $S/eval_${SPLIT}_M1.json ] && ENT+=("M1=$S/eval_${SPLIT}_M1.json")
  for L in $AFTER; do [ -f $S/eval_${SPLIT}_$L.json ] && ENT+=("$L=$S/eval_${SPLIT}_$L.json"); done
  python -m src.failure_analysis --split $SPLIT --out $SUB/tables/failures_$SPLIT.md --json $SUB/tables/failures_$SPLIT.json "${ENT[@]}" > /dev/null
done

# summary tables as produced by the evaluation scripts
cp $S/summary_hard_dev.md $SUB/tables/summary_hard_dev.md
cp $S/summary_hard_final.md $SUB/tables/summary_hard_final.md

# training diagnostics table
TS=("A original GRPO (no SFT)=outputs/main_v2")
for L in $AFTER; do [ -d $S/$L ] && TS+=("$L=$S/$L"); done
python -m src.train_summary --out $SUB/tables/train_summary.md "${TS[@]}" > /dev/null

# figures
python -m src.make_figures --out $SUB/figures --original outputs/main_v2 --study $S --splits hard_dev hard_final --seeds $SEEDS

# case studies: original-GRPO and SFT traces on twohop_dev are located by name, never guessed
shopt -s nullglob
ORIG=(outputs/main_v2/eval_twohop_dev*.json outputs/study_sft/eval_twohop_dev_A*typed*.json outputs/study_sft/eval_twohop_dev_A.json)
SFTE=(outputs/study_sft/eval_twohop_dev_B.json outputs/study_sft/B/eval_twohop_dev*.json)
EXTRA=(); [ ${#ORIG[@]} -gt 0 ] && [ ${#SFTE[@]} -gt 0 ] && EXTRA=(--orig-grpo "${ORIG[0]}" --sft "${SFTE[0]}") && echo "cases 1-2 from ${ORIG[0]} / ${SFTE[0]}"
python -m src.case_studies --split hard_final --before $S/eval_hard_final_M2.json --after $S/eval_hard_final_$HL.json "${EXTRA[@]}" --out $SUB/case_studies.md

python -m src.package_submission --study $S --sft $SFT --sub $SUB
python -m src.build_report --sub $SUB --template reports/FINAL_PILOT_REPORT.template.md
python -m src.submission_check --sub $SUB
