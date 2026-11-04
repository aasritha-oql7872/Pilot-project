#!/usr/bin/env bash
# Priority 4: can GRPO learn multi-step retrieval WITHOUT any supervised demonstrations? All runs start from the pretrained
# Qwen3-0.6B (no SFT adapter, no gold action sequences). Staged and gated; development-only (twohop_dev, dev).
#   F0 reproduce original-reward GRPO (mixed tasks)      F1 exploration diversity (temperature / group size)
#   F2 evidence-aware (and potential) reward              F3 explicit exploration incentive (gold evidence = privileged
#   F4 optional RL-only curriculum (RUN_F4=1)             reward signal, never shown to the policy)
# Stage rule: a short 30-step PILOT of each condition; the full 100-step run only if the pilot contains >=1 two-hop rollout
# with a second search (a reward cannot reinforce an action that is never sampled). Every decision is saved as JSON.
# Never deletes/overwrites; sealed holdouts untouched. Requires the sealed evaluation first (ALLOW_BEFORE_SEALED=1 overrides).
set -euo pipefail
export TRL_EXPERIMENTAL_SILENCE=1
[ -f outputs/study_sft/summary_sealed.md ] || [ "${ALLOW_BEFORE_SEALED:-0}" = "1" ] || \
  { echo "Finish and save the sealed evaluation first (run_sealed.sh)."; exit 4; }
S=outputs/study_rl_only; mkdir -p $S
CFG=configs/train_mixed.yaml          # eval/probe decoding settings (384-token budget, greedy eval)
PILOT_STEPS=${PILOT_STEPS:-30}; FULL_STEPS=${FULL_STEPS:-100}

ev() { local label=$1 ckpt=$2 split=$3; shift 3
  [ -f $S/eval_${split}_${label}.json ] || python -m src.evaluate --config $CFG --checkpoint $ckpt --split $split \
      --output $S/eval_${split}_${label}.json "$@" | tail -1; }
evaluate_run() { # name (evaluated under the training reminder (default) AND the typed reminder)
  for SPLIT in dev twohop_dev; do
    ev $1 $S/$1/adapter $SPLIT --answer-format default
    ev $1_typed $S/$1/adapter $SPLIT --answer-format typed
  done
  python -m src.train_diag $S/$1 --out $S/diag_$1.json > /dev/null
}
train() { # name config [override...]   (skips if the adapter already exists)
  local name=$1 cfg=$2; shift 2
  [ -f $S/$name/adapter/adapter_config.json ] || python -m src.train_grpo --config $cfg --output $S/$name --override "$@"
}
pilot_then_full() { # label config [override...] : pilot -> decision -> full run
  local label=$1 cfg=$2; shift 2
  train ${label}_pilot $cfg grpo.max_steps=$PILOT_STEPS "$@"
  python -m src.explore_gate pilot --out $S/decision_${label}.json $S/${label}_pilot > /dev/null
  if [ "$(python -c "import json;print(json.load(open('$S/decision_${label}.json'))['proceed'])")" = "True" ]; then
    train $label $cfg grpo.max_steps=$FULL_STEPS "$@"; evaluate_run $label
  else
    echo "$label: pilot produced NO two-hop rollout with a second search -> full run NOT started (see $S/decision_${label}.json)"
  fi
}

# ---- F0: reproduce the failure (original reward, base init, mixed tasks) with exploration statistics -------------------
train F0 $CFG grpo.max_steps=$FULL_STEPS
evaluate_run F0
python -m src.traj_stats $S/F0 --out $S/traj_stats_F0.json > /dev/null

# ---- F1: exploration diversity. Stage 1: no-training probe of the base model's second-search rate ----------------------
for T in 0.7 1.0 1.3; do
  [ -f $S/probe_base_T$T.json ] || python -m src.probe explore --config $CFG --checkpoint base --split twohop_dev \
      --samples 8 --temperature $T --answer-format default --output $S/probe_base_T$T.json > /dev/null
done
python -m src.explore_gate temps --out $S/decision_F1_temps.json $S/probe_base_T*.json
TSEL=$(python -c "import json;d=json.load(open('$S/decision_F1_temps.json'));print(d['selected_temperature'] or 0.7)")
if [ "$(python -c "import json;print(json.load(open('$S/decision_F1_temps.json'))['proceed'])")" = "True" ]; then
  # more samples per question (8 instead of 4: the group, not the batch, is where a rare 2nd search must appear)
  pilot_then_full F1 $CFG grpo.temperature=$TSEL grpo.num_generations=8
else
  echo "F1: no second search at ANY tested temperature ($S/decision_F1_temps.json) -> F1 training NOT run"
fi

# ---- F2: evidence-aware / potential reward from base ---------------------------------------------------------------------
pilot_then_full F2 configs/train_evidence.yaml grpo.temperature=$TSEL
pilot_then_full F2p configs/train_potential.yaml grpo.temperature=$TSEL

# ---- F3: explicit exploration incentive (evidence bonuses + gamma for a valid non-duplicate 2nd search from partial evidence)
pilot_then_full F3 configs/train_explore.yaml grpo.temperature=$TSEL

# ---- F4 (optional): RL-only curriculum, one-hop GRPO adapter (condition A) -> GRPO on two-hop tasks only -----------------
if [ "${RUN_F4:-0}" = "1" ] && [ -f outputs/main_v2/adapter/adapter_config.json ]; then
  pilot_then_full F4 configs/train_evidence.yaml train_splits='[twohop_train]' init_adapter=outputs/main_v2/adapter
fi

# ---- reward variance / exploration statistics over every training run, then the comparison table -----------------------
RUNS=(); for d in $S/F*; do if [ -f $d/trajectories.jsonl ]; then RUNS+=($d); fi; done
python -m src.traj_stats "${RUNS[@]}" --out $S/traj_stats_all.json > /dev/null
ENTRIES=()
for SPLIT in dev twohop_dev; do
  for d in $S/F[0-9]*; do n=$(basename $d); case $n in *_pilot) continue;; esac
    if [ -f $S/eval_${SPLIT}_${n}.json ]; then ENTRIES+=("$n (train reminder)=$S/eval_${SPLIT}_${n}.json" "$n (typed reminder)=$S/eval_${SPLIT}_${n}_typed.json"); fi
  done
done
if [ ${#ENTRIES[@]} -gt 0 ]; then python -m src.summarize_study --out $S/summary_rl_only_dev.md "${ENTRIES[@]}"; fi
python -m src.manifest --roots $S --out $S/experiment_manifest.json
echo "done: $S (development-only, exploratory)"
