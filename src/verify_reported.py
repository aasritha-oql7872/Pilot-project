"""Check the previously reported development results against the evaluation JSONs actually saved on disk.

    python -m src.verify_reported outputs/study_sft

Prints MATCH / MISMATCH / MISSING per cell. The reported numbers below are the ones written in the project notes
(first read from Colab output); this script is what turns them into verified results.
"""

import argparse
import json
import sys
from pathlib import Path

REPORTED = {  # (file label, split) -> (correct, n)
    ("A", "twohop_dev"): (0, 30), ("A", "dev"): (57, 60),          # A under the typed reminder
    ("B", "twohop_dev"): (30, 30), ("B", "dev"): (59, 60),
    ("C", "twohop_dev"): (30, 30), ("C", "dev"): (60, 60),
    ("D", "twohop_dev"): (30, 30), ("D", "dev"): (60, 60),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    bad = 0
    for (label, split), (k, n) in REPORTED.items():
        path = Path(args.root) / f"eval_{split}_{label}.json"
        if not path.exists():
            print(f"MISSING  {path}"); bad += 1; continue
        eps = json.loads(path.read_text())["episodes"]
        got = (sum(bool(e["correct"]) for e in eps), len(eps))
        ok = got == (k, n)
        bad += not ok
        print(f"{'MATCH   ' if ok else 'MISMATCH'} {label:2} {split:11} reported {k}/{n}  saved {got[0]}/{got[1]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
