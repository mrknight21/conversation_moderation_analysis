#!/usr/bin/env python
"""aggregate_eval.py
====================
Utilities for corpus‑level evaluation of GPT predictions against human
annotations in the *roundtable* (or any supported) dataset.

Three sub‑commands
------------------
* **joint-eval**          – evaluate the *joint* (5‑label) model outputs stored
  within `agg/<split>.json`.
* **single-task-eval**    – evaluate *separate* single‑label GPT runs (one file
  per attribute) located in an `output/` folder.
* **comparison-excel**    – build an XLSX sheet that juxtaposes context,
  targets, and GPT vs. human labels for quick manual inspection.

All parameters (corpus, split, folder paths) are configurable via CLI flags.
The script returns JSON evaluation reports and, where relevant, an Excel file.

Example usage
~~~~~~~~~~~~~
```bash
# 1️⃣ joint evaluation on test split
python aggregate_eval.py joint-eval \
    --corpus roundtable --split test \
    --agg ./data/roundtable/agg \
    --output ./data/roundtable/output

# 2️⃣ single‑task evaluation
python aggregate_eval.py single-task-eval --corpus roundtable --split test

# 3️⃣ produce comparison sheet for dev split
python aggregate_eval.py comparison-excel --corpus insq --split dev
```
"""
from __future__ import annotations

import argparse
import json
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import krippendorff  # type: ignore
from sklearn.metrics import classification_report, f1_score

# silencing pandas FutureWarnings in report tabulation
warnings.filterwarnings("ignore")

###############################################################################
# Constants & helper mappings
###############################################################################
_SINGLE_TASKS = [
    "informational motive",
    "social motive",
    "coordinative motive",
    "dialogue act",
    "target speaker",
]

_DIALOGUE_ACTS = {
    "Probing": 0,
    "Confronting": 1,
    "Instruction": 2,
    "Interpretation": 3,
    "Supplement": 4,
    "All Utility": 5,
}

###############################################################################
# Generic helpers
###############################################################################

def _label_tag(attr: str) -> str:
    return "".join(w[0] for w in attr.split())


def _load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _safe_counter(data: Dict[str, int]) -> Counter:
    """Convert vote‑dict (str→int) into *Counter*, ignoring malformed entries."""
    try:
        return Counter(data.values())
    except Exception:
        return Counter()


def _krippendorff_alpha(a: Sequence[int], b: Sequence[int]) -> float:
    try:
        return float(krippendorff.alpha(reliability_data=[a, b], level_of_measurement="nominal"))
    except Exception:
        return float("nan")


def _random_macro_f1(human: Sequence[int], k: int = 5, runs: int = 5) -> float:
    labels = sorted(set(human))
    scores = []
    for _ in range(runs):
        rnd = random.choices(labels, k=len(human))
        scores.append(f1_score(human, rnd, labels=labels, average="macro"))
    return float(np.mean(scores))

###############################################################################
# Joint evaluation (single combined GPT output file)
###############################################################################

def joint_evaluation(args: argparse.Namespace) -> None:
    agg_path = Path(args.agg) / f"{args.split}.json"
    data = _load_json(agg_path)

    # structure: attr -> field -> list
    D: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))

    for sample in data:
        try:
            gpt = sample["answer"]["gpt"]
            human = sample["answer"]["human"]
        except KeyError:
            continue  # skip malformed rows

        # motives -------------------------------------------------------
        for m in ("informational motive", "social motive", "coordinative motive"):
            D[m]["gpt"].append(1 if m in gpt["motives"] else 0)
            hm_label = int(human["motives"][m]["label"])
            D[m]["human"].append(hm_label)
            votes = list(human["motives"][m]["vote"].values())
            D[m]["rl_gt"].extend([hm_label] * len(votes))
            D[m]["rl_vote"].extend(votes)
            D[m]["rl_llm"].extend([D[m]["gpt"][-1]] * len(votes))

        # dialogue act ---------------------------------------------------
        D["dialogue act"]["gpt"].append(gpt["dialogue act"])
        hm_da = int(human["dialogue act"]["label"])
        D["dialogue act"]["human"].append(hm_da)
        da_votes = list(human["dialogue act"]["vote"].values())
        D["dialogue act"]["rl_gt"].extend([hm_da] * len(da_votes))
        D["dialogue act"]["rl_vote"].extend(da_votes)
        D["dialogue act"]["rl_llm"].extend([gpt["dialogue act"]] * len(da_votes))

        # target speaker -------------------------------------------------
        D["target speaker"]["gpt"].append(gpt["target speaker(s)"])
        hm_ts = int(human["target speaker(s)"]["label"])
        D["target speaker"]["human"].append(hm_ts)
        ts_votes = list(human["target speaker(s)"]["vote"].values())
        D["target speaker"]["rl_gt"].extend([hm_ts] * len(ts_votes))
        D["target speaker"]["rl_vote"].extend(ts_votes)
        D["target speaker"]["rl_llm"].extend([gpt["target speaker(s)"]] * len(ts_votes))

    report = _compile_joint_report(D)
    out_path = Path(args.output) / f"{args.corpus}_{args.split}_gpt_joint_eval.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"✓ Joint evaluation saved to {out_path}")


def _compile_joint_report(D: Dict[str, Dict[str, List]]) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    for attr, v in D.items():
        gpt, human = v["gpt"], v["human"]
        result[attr] = classification_report(human, gpt, output_dict=True)
        result[attr]["krippendorff_alpha"] = _krippendorff_alpha(human, gpt)
        result[attr]["random_macro_f1"] = _random_macro_f1(human)
    return result

###############################################################################
# Excel comparison sheet
###############################################################################

def comparison_excel(args: argparse.Namespace) -> None:
    agg_path = Path(args.agg) / f"{args.split}.json"
    data = _load_json(agg_path)

    rows = []
    for s in data:
        rows.append(
            {
                "id": s["id"],
                "topic": s["topic"],
                "speakers": ",\n".join(s["speakers"]),
                "prior context": "".join(
                    f"{p[0]} ({p[1]}): {p[2]}\n" for p in s["context"]["prior_context"]
                ),
                "post context": "".join(
                    f"{p[0]} ({p[1]}): {p[2]}\n" for p in s["context"]["post_context"]
                ),
                "target": f"{s['target']['speaker']} ({s['target']['role']}): {s['target']['content']}\n",
                # copy label fields below
                **_extract_label_row(s),
            }
        )
    df = pd.DataFrame(rows)
    out_xlsx = Path(args.output) / f"{args.split}_comparison.xlsx"
    df.to_excel(out_xlsx, index=False)
    print(f"✓ Excel sheet saved to {out_xlsx}")


def _extract_label_row(s: dict) -> dict:
    gpt = s["answer"]["gpt"]
    human = s["answer"]["human"]
    row = {}
    for m in ("informational", "social", "coordinative"):
        key = f"{m} motive"
        row[key] = int(human["motives"][key]["label"])
        row[f"{key} vote"] = human["motives"][key]["vote"]
        row[f"{key} gpt"] = 1 if key in gpt["motives"] else 0
    # dialogue act & target speaker
    row["dialogue act"] = int(human["dialogue act"]["label"])
    row["dialogue act vote"] = human["dialogue act"]["vote"]
    row["dialogue act gpt"] = gpt["dialogue act"]
    row["target speaker"] = int(human["target speaker(s)"]["label"])
    row["target speaker vote"] = human["target speaker(s)"]["vote"]
    row["target speaker gpt"] = gpt["target speaker(s)"]
    row["gpt prompt"] = gpt.get("prompt", "")
    row["gpt reason"] = gpt.get("reason", "")
    return row

###############################################################################
# Single‑task evaluation
###############################################################################

def single_task_eval(args: argparse.Namespace) -> None:
    agg_path = Path(args.agg) / f"{args.split}.json"
    agg_data = _load_json(agg_path)

    # load all single‑task prediction files
    task_preds = {}
    for t in _SINGLE_TASKS:
        tag = _label_tag(t)
        pfile = Path(args.output) / f"gpt-4o_{args.split}_{tag}_output.json"
        task_preds[t] = {x["custom_id"]: x for x in _load_json(pfile)}

    # accumulate metrics --------------------------------------------------
    eval_result: Dict[str, dict] = {}
    attr_map = {
        "informational motive": (lambda p: p["answer"]["verdict"], lambda h: int(h["motives"]["informational motive"]["label"])),
        "social motive": (lambda p: p["answer"]["verdict"], lambda h: int(h["motives"]["social motive"]["label"])),
        "coordinative motive": (lambda p: p["answer"]["verdict"], lambda h: int(h["motives"]["coordinative motive"]["label"])),
        "dialogue act": (
            lambda p: _DIALOGUE_ACTS[p["answer"]["dialogue act"]],
            lambda h: int(h["dialogue act"]["label"]),
        ),
        "target speaker": (
            lambda p: int(p["answer"]["target speaker(s)"].split()[0]),
            lambda h: int(h["target speaker(s)"]["label"]),
        ),
    }

    for attr, (pred_fn, human_fn) in attr_map.items():
        preds, gts = [], []
        for samp in agg_data:
            cid = samp["id"]
            try:
                p_obj = task_preds[attr][cid]
            except KeyError:
                continue  # missing prediction
            preds.append(pred_fn(p_obj))
            gts.append(human_fn(samp["answer"]["human"]))
        rep = classification_report(gts, preds, output_dict=True)
        rep["krippendorff_alpha"] = _krippendorff_alpha(gts, preds)
        eval_result[attr] = rep

    out_path = Path(args.output) / f"gpt_{args.corpus}_{args.split}_single_task_eval.json"
    out_path.write_text(json.dumps(eval_result, indent=2, ensure_ascii=False))
    print(f"✓ Single‑task evaluation saved to {out_path}")

###############################################################################
# CLI entry‑point
###############################################################################

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate GPT evaluation utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _sub(name: str, handler):
        sp = sub.add_parser(name)
        sp.add_argument("--corpus", default="roundtable")
        sp.add_argument("--split", default="test", help="data split (dev/test/…)")
        sp.add_argument("--agg", default="./data/roundtable/agg", help="agg folder")
        sp.add_argument("--output", default="./data/roundtable/output", help="output folder")
        sp.set_defaults(func=handler)
        return sp

    _sub("joint-eval", joint_evaluation)
    _sub("comparison-excel", comparison_excel)
    _sub("single-task-eval", single_task_eval)
    return p


def main():
    random.seed(2025)
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
