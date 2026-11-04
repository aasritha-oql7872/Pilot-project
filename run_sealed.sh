#!/usr/bin/env bash
# Sealed final evaluation (final, twohop_final), run ONCE after the headline method is frozen (B).
# Matched settings for every model: configs/train_sft_grpo.yaml decoding (greedy, 384-token budget, 2-search cap) and the
# typed reminder; base and A are ALSO evaluated under the default reminder to separate the format effect.
# Nothing is deleted or overwritten: existing eval files are skipped. Needs outputs/study_sft/{B,C,D}/adapter.
set -euo pipefail
export TRL_EXPERIMENTAL_SILENCE=1
S=outputs/study_sft; CFG=configs/train_sft_grpo.yaml
python -m src.manifest --verify-sealed                       # exit 1 if a sealed file was edited
[ -f $S/FROZEN ] || echo "chosen method: B (SFT only), chosen on dev only (see results/frozen_selection.json); recorded $(date -u +%FT%TZ) BEFORE any sealed evaluation" > $S/FROZEN
cat $S/FROZEN
if [ -f outputs/main_v2/adapter/adapter_config.json ]; then A=outputs/main_v2/adapter; else A=outputs/main_v2_retrained/adapter; fi
for ck in $S/B $S/C $S/D; do [ -f $ck/adapter/adapter_config.json ] || { echo "missing $ck/adapter"; exit 2; }; done
[ -f $A/adapter_config.json ] || { echo "missing condition A adapter ($A)"; exit 2; }

ev() { # label checkpoint split [extra evaluate args]
  local label=$1 ckpt=$2 split=$3; shift 3
  [ -f $S/eval_${split}_${label}.json ] || python -m src.evaluate --config $CFG --checkpoint $ckpt --split $split \
      --output $S/eval_${split}_${label}.json "$@" | tail -1
}
for SPLIT in final twohop_final; do
  ev base_default base $SPLIT --answer-format default
  ev base base $SPLIT
  ev A_default $A $SPLIT --answer-format default
  ev A $A $SPLIT
  ev B $S/B/adapter $SPLIT
  ev C $S/C/adapter $SPLIT
  ev D $S/D/adapter $SPLIT
done
ENTRIES=()
for SPLIT in final twohop_final; do
  ENTRIES+=("base (default reminder)=$S/eval_${SPLIT}_base_default.json" "base (typed reminder)=$S/eval_${SPLIT}_base.json"
            "A (default reminder)=$S/eval_${SPLIT}_A_default.json" "A (typed reminder)=$S/eval_${SPLIT}_A.json"
            "B SFT only [HEADLINE]=$S/eval_${SPLIT}_B.json" "C SFT+GRPO orig=$S/eval_${SPLIT}_C.json"
            "D SFT+GRPO evidence=$S/eval_${SPLIT}_D.json")
done
python -m src.summarize_study --out $S/summary_sealed.md "${ENTRIES[@]}"
# paired comparisons against the headline model (exact sign test on discordant tasks)
for SPLIT in final twohop_final; do
  echo "== paired vs B on $SPLIT (columns: wins/losses of the other run relative to B)"
  python -m src.compare --base $S/eval_${SPLIT}_B.json --trained $S/eval_${SPLIT}_C.json $S/eval_${SPLIT}_D.json $S/eval_${SPLIT}_A.json
done | tee $S/paired_sealed.txt
python -m src.audit episodes $S/eval_twohop_final_{base,A_default,A,B,C,D}.json > $S/audit_sealed_twohop.json
python -m src.manifest --roots $S --out $S/experiment_manifest.json
echo "sealed evaluation done: $S/summary_sealed.md"
