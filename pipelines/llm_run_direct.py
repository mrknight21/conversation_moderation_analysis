#!/usr/bin/env python
"""direct_llm_run.py
========================
It runs an LLM over moderator utterances, compares the model output with human
annotations, and produces per‑episode and corpus‑level evaluation reports.


Example
~~~~~~~
```bash
python direct_llm_run.py \
    --input ./data/slmod \
    --output ./data/slmod \
    --eval ./data/evaluation_scores \
    --mode dev \
    --model gpt-4o \
    --prior 5 --post 2 --max-trials 3 --repair
```
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import classification_report
from tqdm import tqdm

# external project helpers -----------------------------------------------------
from utils.dataloaders import get_episode_meta  # noqa: E402
from static.attributes_prompts import construct_prompt_unit  # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
# Constants & default mappings
# ---------------------------------------------------------------------------
_DA_ENCODER = {
    "probing": 0,
    "confronting": 1,
    "instruction": 2,
    "interpretation": 3,
    "supplement": 4,
    "all utility": 5,
}
_ALLOWED_DA = {k.title() for k in _DA_ENCODER}
_ALLOWED_MOTIVES = {
    "informational motive",
    "social motive",
    "coordinative motive",
}


###############################################################################
# OpenAI helpers
###############################################################################

def openai_client() -> OpenAI:
    """Return an authenticated OpenAI client based on env vars."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    org = os.getenv("OPENAI_ORG_ID") or None
    return OpenAI(api_key=key, organization=org)


###############################################################################
# Validation helpers (re‑used across batches)
###############################################################################
_JSON_RE = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}")


def _check_single_answer(raw: str) -> bool:
    """Validate that *raw* is a JSON string with required keys & values."""
    try:
        obj = json.loads(raw)
    except Exception:
        return False

    if not {"motives", "dialogue act", "target speaker(s)"}.issubset(obj):
        return False

    if any(m not in _ALLOWED_MOTIVES for m in obj.get("motives", [])):
        return False

    if obj.get("dialogue act") not in _ALLOWED_DA:
        return False
    # target speaker(s) is free‑form but must start with a digit
    try:
        int(str(obj["target speaker(s)"]).split()[0])
    except Exception:
        return False
    return True


###############################################################################
# GPT wrapper utilities
###############################################################################

def _call_chat(client: OpenAI, prompt: str, model: str) -> str:
    """Single chat completion call with minimal retry/back‑off logic."""
    delay = 1.0
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content
        except Exception as exc:
            logging.warning("%s (attempt %d) – sleeping %.1fs", exc, attempt + 1, delay)
            time.sleep(delay)
            delay *= 2  # exponential back‑off
    raise RuntimeError("OpenAI API failed after several retries.")


def gpt_batch(
    instances: Sequence[Dict[str, Any]],
    *,
    model: str,
    max_trials: int,
    client: OpenAI,
) -> List[Dict[str, Any]]:
    """Run instances through GPT with validation + retry."""
    outputs: List[Dict[str, Any]] = []
    for inst in tqdm(instances, desc="↻ GPT batch"):
        for trial in range(max_trials):
            raw = _call_chat(client, inst["prompt"], model)
            matches = _JSON_RE.findall(raw)
            if matches and _check_single_answer(matches[0]):
                inst["output"] = json.loads(matches[0])
                break
        else:  # exhausted trials
            logging.warning("Validation failed for id=%s", inst["id"])
            inst["output"] = None
        outputs.append(inst)
    return outputs


###############################################################################
# Episode processing & evaluation
###############################################################################

def _build_context_masks(
    df: pd.DataFrame,
    utt: int,
    sent: int,
    prior: int,
    post: int,
) -> Tuple[pd.Series, pd.Series]:
    """Return boolean masks for prior and post context selection."""
    def _in_prior(_id: str) -> bool:
        u, s = map(int, _id.split("_"))
        return (utt - prior <= u < utt) or (u == utt and s < sent)

    def _in_post(_id: str) -> bool:
        u, s = map(int, _id.split("_"))
        return (utt < u <= utt + post) or (u == utt and s > sent)

    return df.id.apply(_in_prior), df.id.apply(_in_post)


def process_episode(
    path: Path,
    *,
    client: OpenAI,
    model: str,
    prior: int,
    post: int,
    max_trials: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate prompts, query GPT, and compute evaluation for one episode."""
    meta = get_episode_meta(str(path))
    df = pd.read_excel(path, index_col=0)
    df["id"] = df["id"].astype(str)

    prompts: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        if row.role != "mod":
            continue
        utt, sent = map(int, row.id.split("_"))
        prior_mask, post_mask = _build_context_masks(df, utt, sent, prior, post)
        context = {
            "prior_context": [
                (r.speaker, r.role, r.text) for _, r in df[prior_mask].iterrows()
            ],
            "post_context": [
                (r.speaker, r.role, r.text) for _, r in df[post_mask].iterrows()
            ],
        }
        instance = {
            "id": row.id,
            "meta": meta,
            "context": context,
            "target": (row.speaker, row.role, row.text),
        }
        instance["prompt"] = construct_prompt_unit(instance)
        prompts.append(instance)

    # ------------------------------------------------------------------ GPT
    results = gpt_batch(
        prompts, model=model, max_trials=max_trials, client=client
    )

    # ---------------------------------------------------------------- evaluation
    eval_data = defaultdict(lambda: {"human": [], "llm": [], "votes": []})  # type: ignore

    # build fast look‑ups
    idx_map = {row.id: i for i, row in df.iterrows()}
    motive_cols = {
        "informational motive": "informational motive",
        "social motive": "social motive",
        "coordinative motive": "coordinative motive",
    }

    for res in results:
        if res["output"] is None:
            continue  # skip invalid
        out = res["output"]
        i = idx_map[res["id"]]

        # dialogue act ---------------------------------------------------
        human_da = df.loc[df.index[i], "dialogue act"]
        if not pd.isna(human_da):
            eval_data["dialogue act"]["human"].append(human_da)
            eval_data["dialogue act"]["llm"].append(_DA_ENCODER[out["dialogue act"].lower()])
            eval_data["dialogue act"]["votes"].append(
                _get_votes(df.loc[df.index[i], "dialogue act vote"])
            )

        # target speaker -----------------------------------------------
        human_sp = df.loc[df.index[i], "target speaker"]
        if not pd.isna(human_sp):
            eval_data["target speaker"]["human"].append(human_sp)
            eval_data["target speaker"]["llm"].append(
                meta["speakers"].index(out["target speaker(s)"])
            )
            eval_data["target speaker"]["votes"].append(
                _get_votes(df.loc[df.index[i], "target speaker vote"])
            )

        # motives -------------------------------------------------------
        motives = set(out.get("motives", []))
        for name, col in motive_cols.items():
            human_val = df.loc[df.index[i], col]
            if pd.isna(human_val):
                continue
            eval_data[name]["human"].append(human_val)
            eval_data[name]["llm"].append(int(name in motives))
            eval_data[name]["votes"].append(
                _get_votes(df.loc[df.index[i], f"{col} vote"])
            )

        # write predictions back for inspection ------------------------
        df.at[df.index[i], "dialogue act llm"] = out["dialogue act"]
        df.at[df.index[i], "target speaker llm"] = out["target speaker(s)"]
        df.at[df.index[i], "llm reason"] = out.get("reason", "")
        for m, col in motive_cols.items():
            df.at[df.index[i], f"{col} llm"] = int(m in motives)

    # classification reports
    eval_report = {"topic": meta["topic"]}
    for lbl, data in eval_data.items():
        eval_report[lbl] = _evaluate(data)

    # save per‑episode annotated sheet
    return df, eval_data, eval_report


###############################################################################
# Evaluation helpers
###############################################################################

def _get_votes(vote_str: str) -> Counter:
    vote_str = vote_str.replace("'", '"')
    return Counter(json.loads(vote_str).values())


def _weighted_accuracy(votes: Sequence[Counter], preds: Sequence[Any]) -> float:
    scores = []
    for vote, pred in zip(votes, preds):
        max_v = max(vote.values()) if vote else 1
        scores.append(vote.get(pred, 0) / max_v)
    return float(np.mean(scores)) if scores else np.nan


def _evaluate(data: Dict[str, List[Any]]) -> Dict[str, Any]:
    report = classification_report(data["human"], data["llm"], output_dict=True)
    if data["votes"]:
        report["weighted_accuracy"] = _weighted_accuracy(data["votes"], data["llm"])
    return report


###############################################################################
# CLI & main
###############################################################################

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SL‑Mod GPT evaluation pipeline")
    p.add_argument("--input", default="./data/slmod", help="Folder with episode .xlsx")
    p.add_argument("--output", default="./data/slmod", help="Folder to write updated .xlsx")
    p.add_argument("--eval", default="./data/evaluation_scores", help="Folder for JSON reports")
    p.add_argument("--mode", default="dev", help="Data split tag (dev/test/etc.)")
    p.add_argument("--model", default="gpt-4o", help="OpenAI model name")
    p.add_argument("--prior", type=int, default=5, help="Prior context window size")
    p.add_argument("--post", type=int, default=2, help="Post context window size")
    p.add_argument("--max-trials", type=int, default=3, help="Max validation retries per utterance")
    p.add_argument("--seed", type=int, default=2023, help="Random seed")
    p.add_argument("--log-level", default="INFO", help="logging level")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=args.log_level, format="[%(levelname)s] %(message)s")

    random.seed(args.seed)
    np.random.seed(args.seed)

    in_dir, out_dir, eval_dir = map(Path, (args.input, args.output, args.eval))
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    client = openai_client()

    corpus_eval: Dict[str, Dict[str, List[Any]]] = defaultdict(
        lambda: {"human": [], "llm": [], "votes": []}
    )
    full_reports: Dict[str, Any] = {}

    for ep_path in sorted(in_dir.glob("*.xlsx")):
        if ep_path.name.startswith("~$"):
            continue  # skip Excel temp files
        logging.info("Processing %s", ep_path.name)
        df, ep_data, ep_report = process_episode(
            ep_path,
            client=client,
            model=args.model,
            prior=args.prior,
            post=args.post,
            max_trials=args.max_trials,
        )
        # save annotated sheet
        df.to_excel(out_dir / f"{ep_path.stem}_{args.mode}_llm.xlsx")

        # accumulate corpus‑level data
        for lbl in ep_data:
            corpus_eval.setdefault(lbl, {"human": [], "llm": [], "votes": []})
            for k in ("human", "llm", "votes"):
                corpus_eval[lbl][k].extend(ep_data[lbl][k])

        full_reports[ep_report["topic"]] = ep_report

    # overall evaluation
    overall = {lbl: _evaluate(data) for lbl, data in corpus_eval.items()}
    full_reports["overall"] = overall

    (eval_dir / f"{args.mode}_llm_eva_scores.json").write_text(
        json.dumps(full_reports, ensure_ascii=False, indent=2)
    )
    logging.info("✓ Finished. Reports written to %s", eval_dir)


if __name__ == "__main__":
    main()
