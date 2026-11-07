# Evaluation design: a held-out benchmark with room for improvement

## 1. Why a new benchmark
`twohop_dev` (30 tasks) is saturated by SFT (30/30), and `final`/`twohop_final` use the same templates. A saturated benchmark
cannot show an RL gain. The hard benchmark keeps the same environment (BM25 over a per-split corpus, ≤ 2 searches, same tool
and verifier) and adds categories where an SFT policy trained on plain two-hop trajectories may fail.

## 2. Categories (generator: `data/make_hard.py`; validator: `src/validate_hard.py`)
| Category | What is hard | Ground truth |
|---|---|---|
| standard (control) | plain two-hop | PI's institution |
| distractor_name | decoy "Project <N> Prime" has its own PI/affiliation records that match the query | the PI of exactly "Project <N>" |
| distractor_person | a decoy person shares the PI's surname; both affiliation records match the second query | the institution of the exact full name |
| stale_affiliation | a "previous affiliation" record **outranks** the current one for "<PI> affiliation" (misleading first result) | the *current* institution |
| reverse_two_hop | "city/funder/year of the project led by <PI>": PI record → project record (combines two records in the opposite direction) | the project attribute |
| unanswerable | the PI's affiliation record does not exist | `<answer>insufficient evidence</answer>` |

Splits (disjoint entities, separate corpora): `hard_train` 120 tasks (20/category), `hard_dev` 60 (10/category), `hard_final`
120 (20/category, **locked**). The abstention option is announced in the tool-output reminder `typed_abstain` (environment v2).

## 3. Cases considered and not generated
* **Three-hop / alias chains**: need ≥ 3 searches; `MAX_SEARCHES = 2` is a constant shared by training and evaluation.
* **Failed first search needing reformulation**: the rare project-name token always makes the first BM25 query succeed; a valid
  case would need alias documents (three hops). Not generated, to avoid inventing an unrealistic situation.
* **Irrelevant but similar evidence** is covered by the decoy categories (similar names/surnames, stale affiliation).

## 4. Validation (CPU, no model) [test]
* **Solvable from observed text.** A scripted agent that uses only retrieved text (≤ 2 searches, exact-name matching) answers all
  300 tasks correctly and reaches all required evidence documents; unanswerable tasks verifiably have no affiliation record.
* **Defensible ground truth.** For answerable tasks the gold string occurs in the last required evidence document.
* **No leakage.** Generated entity names are disjoint across the three hard splits and from every other split (incl. sealed),
  no duplicate questions, no gold in the question, decoy documents are never required evidence.
* **Hash lock.** `data/manifest_hard.json` stores file hashes; `python -m src.manifest --verify-hard` and `tests/test_hard.py`
  fail if any file changes. Revision 2 was regenerated *before any model evaluation* to make the stale record outrank the gold
  record (diagnosed on hard_dev ranks: 0/10 gold-first after the change; it was 10/10 before). That is recorded in the manifest.
* **What a "trust the top record" reader gets on hard_dev:** stale_affiliation 0/10; the other categories 10/10. So only
  stale_affiliation is a ranking trap; the distractor and abstention categories test in-context discrimination or the decision to
  abstain, which is an empirical question for the language model, not something the generator guarantees.

## 5. Headroom check [pending, run on Colab]
`run_headroom_study.sh` evaluates base, M1, M2 (frozen SFT B), and the existing C/D on `hard_dev`, then
`explore_gate headroom` proceeds only if overall accuracy ≤ 0.85 or some category ≤ 0.8. If M2 is already above that, GRPO is not
run and the finding is "no headroom". Results are reported by category (`summary_hard_dev.md`). No number is quoted here.

## 6. Metrics (`src/report_categories.py`)
Accuracy (k/n, 95% Wilson interval), evidence-supported accuracy (correct and all required documents retrieved), required-evidence
recall, unsupported-answer rate, false-abstention rate, mean searches, invalid-action rate; per category and overall. Paired
comparisons against M2: exact sign test on discordant tasks and paired-bootstrap CI of the accuracy difference. Sampled-decoding
repeats (`SAMPLED=1`, 3 seeds) are reported as mean ± sd over runs. Categories have n = 10 (dev) or 20 (final): per-category
differences are only suggestive.

## 7. Known weaknesses
Synthetic fixed templates; single-seed generator; the decoys are lexical, not semantic; M1 never saw the abstention reminder;
GRPO runs additionally train on `hard_train`, whereas M2 did not (an SFT-on-hard_train control is *not* included, so any gain
cannot be attributed to RL as opposed to extra data on those categories).
