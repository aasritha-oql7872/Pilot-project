"""Go/no-go check between pipeline stages. Exits 3 (stopping the shell script) if the check fails.

    python -m src.gate outputs/study_sft/eval_twohop_dev_B.json --min-second-search 0.5 --min-accuracy 0.3
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--min-second-search", type=float, required=True)
    parser.add_argument("--min-accuracy", type=float, required=True)
    args = parser.parse_args()
    eps = [e for e in json.loads(Path(args.report).read_text())["episodes"] if e["kind"] == "twohop"]
    second = sum(e["num_searches"] >= 2 for e in eps) / len(eps)
    acc = sum(e["correct"] for e in eps) / len(eps)
    print(json.dumps({"n_twohop": len(eps), "second_search_rate": round(second, 3), "accuracy": round(acc, 3)}))
    if second < args.min_second_search or acc < args.min_accuracy:
        print("GATE FAILED: the warm-start does not produce independent second searches / correct answers. "
              "Diagnose (inspect the eval episodes, python -m src.audit episodes) before spending GPU time on GRPO.")
        sys.exit(3)
    print("gate passed")


if __name__ == "__main__":
    main()
