# Pilot final status (2026-11-03)

Branch `experiment/evidence-aware-grpo`, nothing committed or pushed from this session. 66 tests pass on CPU.
Colab GPU outputs are on Drive (`MyDrive/SearchLite-RL/`), not in the repository.

## Done and verified
* [local] Audit: retrieval is not the failure (100% two-query chain); base never samples a second search (0/160).
* [local] Answer-extraction fix: forced-search accuracy 10/30 (default reminder) -> 26/30 (typed reminder).
* [Colab, pasted] Dev results: A 0/30 two-hop; B/C/D 30/30; one-hop 57/59/60 of 60 (A-typed/B/C=D). GRPO after SFT
  changed nothing measurable (95% zero-variance groups, adapter moved 2.6e-4, C and D identical).
* Headline model frozen as B on dev evidence only (`results/frozen_selection.json`).
* Fixed: summary-order bug in `run_sft_study.sh`; Wilson intervals replace the bootstrap interval; **explore-probe defect**
  (`--answer-format` was ignored), so the earlier sampled B/C/D numbers (73.8% / 72.1% second search; 41% / 38% correct) are
  confounded by a reminder mismatch and must be re-run (notebook cell 7).

## Implemented and unit-tested, NOT yet run (needs Colab)
| Item | Command | Gate / output |
|---|---|---|
| Verify reported numbers vs saved JSONs | `python -m src.verify_reported outputs/study_sft` | MATCH/MISMATCH per cell |
| Sealed final evaluation (once) | `bash run_sealed.sh` | `summary_sealed.md`, paired tests, audit |
| E0/E1/E2 partial SFT + GRPO | `bash run_early_study.sh` | exits 3 if no checkpoint has a 20-70% sampled second-search rate |
| F0-F3 (F4 optional) RL-only exploration | `bash run_rl_only_study.sh` | 30-step pilot per condition; full run only if a pilot has a two-hop second search |
| Reward-variance / exploration statistics | `python -m src.traj_stats RUN_DIR` | per-window second-search rate, informative groups, one-hop controls |
| Optimization diagnostics | `python -m src.train_diag RUN --init ADAPTER` | grad norms, zero-variance fraction, weight change |
| Manifest | `python -m src.manifest --roots ... --out ...` | run -> init/reward/seed/checkpoint/decoding/split |
| Demo | `python demo.py --a eval_A.json --b eval_B.json` | question -> search -> record -> search -> answer |

## Not done / blocked
* No sealed number, no E/F number exists yet. None is quoted anywhere.
* Local trainer-build dry-run for the new F3 config was NOT executed (Mac swap ~98% full); config parsing and overrides were
  checked on CPU. The Colab pilot (30 steps) is the first end-to-end test of that config.
* The old Colab outputs for v1/v2/seeds/mixed are still reported-only.
* README.md still says "no performance results" and needs a final pass after the sealed numbers arrive.

## Order of work (deadline 2026-11-09)
1. Push the branch (user), open `SearchLite_RL_Final_Study.ipynb`, run cells 1-4: verify, then sealed evaluation. Paste
   `summary_sealed.md` -> fill REPORT 5.4.
2. Cell 7 (typed explore probes) and cell 5 (E study). 3. Cell 6 (F study, pilots gate the full runs).
4. Fill REPORT 5.5/8/10, plots (`python -m src.plot_metrics --run RUN --output RUN/figures`), finish slides.

## Headroom study (branch `experiment/headroom-grpo`, 2026-11-06) — implemented, NOT yet run
* Hard benchmark (`data/make_hard.py`, 6 categories incl. decoys, a stale record that outranks the gold one, reverse two-hop,
  unanswerable) with validator and locked hashes; 94 CPU tests pass. Reports in `reports/` (phase1_audit, grpo_diagnostics,
  evaluation_design, controlled_experiments, final_rl_findings).
* Run order on Colab: `SearchLite_RL_Headroom_Study.ipynb` (cell 4 audit of the existing C/D artifacts, cell 5 gated headroom /
  signal / training / hard_dev, freeze by the pre-registered rule, cell 8 locked `hard_final` once).
* Question "does evidence-aware GRPO beat SFT where SFT has headroom?" is OPEN. No M1–M4 hard-benchmark number exists yet.
