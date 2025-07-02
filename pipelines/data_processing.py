#!/usr/bin/env python
"""data_processing.py
=======================
A unified, command‑line interface that bundles the miscellaneous data‑munging
utilities previously scattered across *roundtable* preprocessing scripts. Each
original top‑level helper (train‑set build, aggregation, anonymisation, etc.) is
now exposed as a sub‑command under a single executable.

Sub‑commands
------------
* **build-train-json**      – combine GPT outputs & episode metadata into `train.json`.
* **aggregate**             – merge GPT joint outputs with human labels (creates `<split>.json`).
* **convert-gpt-human**     – build unified JSON from XLSX sheets already containing LLM columns.
* **tie-break**             – update dev/test aggs using additional *valid* annotations.
* **anonymise**             – strip voter identities & save cleaned XLSX per split.
* **gen-meta**              – create `_meta.json` for each episode lacking one.

Run `python data_processing_cli.py <sub-command> -h` for detailed options.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from llm_run_direct import (
    get_episode_meta,
    check_single_answer,
    gpt_single,
    gpt4,
)
from pipelines.prompts.attributes_prompts import construct_prompt_unit, annotators, dialogue_acts
from utils.dataloaders import load_json_data

###############################################################################
# Constants (tunable via CLI)
###############################################################################

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Roundtable data‑processing CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # global opts ---------------------------------------------------------
    p.add_argument("--corpus", default="roundtable", help="corpus name tag")
    p.add_argument("--mode", default="train", help="data split tag (train/dev/test)")
    p.add_argument("--input", default="./data", help="root data folder")
    p.add_argument("--output", default="./data", help="root output folder")
    p.add_argument("--model", default="gpt-4o", help="model tag in file names")
    p.add_argument("--prior", type=int, default=5, help="prior context window size")
    p.add_argument("--post", type=int, default=2, help="post context window size")
    p.add_argument("--log-level", default="INFO", help="logging level")

    # build‑train‑json ----------------------------------------------------
    sp_train = sub.add_parser("build-train-json", help="create train.json from GPT outputs + xlsx")
    sp_train.add_argument("--gpt-files", nargs="*", required=True, help="json files with GPT outputs")
    sp_train.set_defaults(func=cmd_build_train)

    # aggregate -----------------------------------------------------------
    sp_agg = sub.add_parser("aggregate", help="merge GPT outputs with human labels")
    sp_agg.add_argument("--gpt-output", required=True, help="joint gpt output json file")
    sp_agg.set_defaults(func=cmd_aggregate)

    # convert‑gpt‑human ---------------------------------------------------
    sp_conv = sub.add_parser("convert-gpt-human", help="produce agg json from llm‑annotated xlsx sheets")
    sp_conv.add_argument("--annotated", required=True, help="folder containing *_llm.xlsx files")
    sp_conv.set_defaults(func=cmd_convert)

    # tie‑break -----------------------------------------------------------
    sp_tb = sub.add_parser("tie-break", help="update dev/test aggs using *_valid.xlsx sheets")
    sp_tb.add_argument("--splits", nargs="*", default=["dev", "test"], help="splits to update")
    sp_tb.set_defaults(func=cmd_tie_break)

    # anonymise -----------------------------------------------------------
    sp_anon = sub.add_parser("anonymise", help="create cleaned XLSX (remove identities)")
    sp_anon.add_argument("--splits", nargs="*", default=["train", "dev", "test"], help="splits to export")
    sp_anon.set_defaults(func=cmd_anonymise)

    # gen‑meta ------------------------------------------------------------
    sp_meta = sub.add_parser("gen-meta", help="generate <episode>_meta.json if missing")
    sp_meta.set_defaults(func=cmd_gen_meta)

    return p

###############################################################################
# Shared helpers
###############################################################################

def _episode_paths(root: Path, corpus: str, mode: str) -> List[Path]:
    return sorted((root / corpus / f"{mode}_data").glob("*.xlsx"))


def _hide_votes(votes: str | dict) -> dict:
    if not isinstance(votes, dict):
        votes = json.loads(str(votes).replace("'", '"'))
    return {f"annotator_{annotators[k]}": v for k, v in votes.items()}

###############################################################################
# Sub‑command implementations
###############################################################################

def cmd_build_train(args):
    root = Path(args.input)
    out_dir = Path(args.output) / args.corpus / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # load GPT jsons ------------------------------------------------------
    gpt_data = {}
    for gfile in args.gpt_files:
        for entry in load_json_data(gfile):
            gpt_data[entry["custom_id"]] = entry
    logging.info("Loaded %d GPT entries", len(gpt_data))

    # build train set -----------------------------------------------------
    episodes = _episode_paths(root, args.corpus, args.mode)
    train_set = []
    for ep in tqdm(episodes, desc="building train set"):
        for inst in _process_episode(
            ep,
            prior=args.prior,
            post=args.post,
            corpus=args.corpus,
        ):
            cid = f"{ep.stem}_{inst['id']}"
            if cid not in gpt_data:
                logging.warning("missing GPT answer for %s", cid)
                continue
            rec = gpt_data[cid]
            # ensure valid JSON answer --------------------------------
            ans_json = json.dumps(rec["answer"])
            if not check_single_answer(ans_json, inst):
                ans_json = gpt_single(inst["prompt"], gpt4)
            rec.update(
                {
                    "topic_id": "_".join(cid.split("_")[:-2]),
                    "instance_id": inst["id"],
                    "topic": inst["meta"]["topic"],
                    "speakers": inst["meta"]["speakers"],
                    "context": inst["context"],
                    "target": {
                        "speaker": inst["target"][0],
                        "role": inst["target"][1],
                        "content": inst["target"][2],
                    },
                    "prompt": inst["prompt"],
                }
            )
            train_set.append(rec)

    out_path = out_dir / "train.json"
    out_path.write_text(json.dumps(train_set, ensure_ascii=False))
    print(f"✓ train.json written to {out_path}")


# ---------------------------------------------------------------------------

def cmd_aggregate(args):
    root = Path(args.input)
    out_dir = Path(args.output) / args.corpus / "agg"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_map: Dict[str, dict] = {}
    for entry in load_json_data(args.gpt_output):
        data_map[entry["custom_id"]] = entry

    episodes = _episode_paths(root, args.corpus, args.mode)
    for ep in tqdm(episodes, desc="aggregating"):
        ep_name = ep.stem
        meta = get_episode_meta(str(ep))
        df = pd.read_excel(ep, index_col=0)
        for _, row in df.iterrows():
            if row.role != "mod":
                continue
            cid = f"{ep_name}_{row.id}"
            if cid not in data_map:
                continue
            rec = data_map[cid]
            # build human answer ---------------------------------------
            human = {
                "motives": {
                    m: {
                        "label": row[m],
                        "vote": _hide_votes(row[f"{m} vote"]),
                    }
                    for m in [
                        "informational motive",
                        "social motive",
                        "coordinative motive",
                    ]
                },
                "dialogue act": {
                    "label": int(row["dialogue act"]),
                    "vote": _hide_votes(row["dialogue act vote"]),
                },
                "target speaker(s)": {
                    "label": int(row["target speaker"]),
                    "vote": _hide_votes(row["target speaker vote"]),
                },
            }
            rec["answer"] = {"gpt": rec.pop("answer"), "human": human}
            # normalise gpt values -------------------------------------
            rec["answer"]["gpt"]["dialogue act"] = dialogue_acts[rec["answer"]["gpt"]["dialogue act"]]
            speaker_pred = rec["answer"]["gpt"]["target speaker(s)"]
            rec["answer"]["gpt"]["target speaker(s)"] = _speaker_to_code(
                speaker_pred, meta["speakers"]
            )

    out_path = out_dir / f"{args.mode}.json"
    out_path.write_text(json.dumps(list(data_map.values()), ensure_ascii=False))
    print(f"✓ aggregated json saved to {out_path}")


# ---------------------------------------------------------------------------

def cmd_convert(args):
    root = Path(args.input)
    out_dir = Path(args.output) / args.corpus / "agg"
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated_folder = Path(args.annotated)
    data: Dict[str, dict] = {}

    for sheet in annotated_folder.glob("*_llm.xlsx"):
        ep_name = sheet.stem.replace("_llm", "")
        meta = get_episode_meta(str(root / args.corpus / f"{args.mode}_data" / f"{ep_name}.xlsx"))
        df = pd.read_excel(sheet, index_col=0)
        for _, r in df.iterrows():
            if r.role != "mod":
                continue
            cid = f"{ep_name}_{r.id}"
            inst = {
                "id": cid,
                "topic_id": "_".join(cid.split("_")[:-2]),
                "instance_id": r.id,
                "topic": meta["topic"],
                "speakers": meta["speakers"],
                "context": {},  # omitted for brevity
                "target": {
                    "speaker": r.speaker,
                    "role": r.role,
                    "content": r.text,
                },
            }
            human_ans = {
                # same structure as earlier
                "motives": {
                    m: {
                        "label": r[m],
                        "vote": _hide_votes(r[f"{m} vote"]),
                    }
                    for m in [
                        "informational motive",
                        "social motive",
                        "coordinative motive",
                    ]
                },
                "dialogue act": {
                    "label": r["dialogue act"],
                    "vote": _hide_votes(r["dialogue act vote"]),
                },
                "target speaker(s)": {
                    "label": r["target speaker"],
                    "vote": _hide_votes(r["target speaker vote"]),
                },
            }
            gpt_ans = {
                "motives": [
                    m
                    for m in [
                        "informational motive",
                        "social motive",
                        "coordinative motive",
                    ]
                    if r[f"{m} llm"] == 1
                ],
                "dialogue act": r["dialogue act llm"],
                "target speaker(s)": r["target speaker llm"],
                "reason": r["llm reason"],
            }
            inst["answer"] = {"gpt": gpt_ans, "human": human_ans}
            data[cid] = inst

    out_path = out_dir / f"{args.mode}.json"
    out_path.write_text(json.dumps(list(data.values()), ensure_ascii=False))
    print(f"✓ converted json saved to {out_path}")


# ---------------------------------------------------------------------------

def cmd_tie_break(args):
    root = Path(args.input)
    for split in args.splits:
        agg_path = root / args.corpus / "agg" / f"{split}.json"
        if not agg_path.exists():
            logging.warning("missing %s", agg_path)
            continue
        data = load_json_data(agg_path)
        valid_files = list((root / args.corpus / "valid").glob("*.xlsx"))
        valid_map = {
            re.sub(r"_valid.*", "", v.stem): pd.read_excel(v, index_col=0) for v in valid_files
        }
        for samp in data:
            topic_id = samp["topic_id"]
            if topic_id not in valid_map:
                continue
            row = valid_map[topic_id][valid_map[topic_id].id == samp["instance_id"]].iloc[0]
            _apply_tie_break(row, samp["answer"]["human"])
        agg_path.with_name(f"{split}_valid.json").write_text(json.dumps(data, ensure_ascii=False))
        print(f"✓ tie‑broken {split} saved")

# helpers ------------------------------------------------------------------

def _apply_tie_break(valid_row: pd.Series, human_ans: dict):
    for attr in _even_ties(human_ans):
        if "motive" in attr:
            human_ans["motives"][attr]["label"] = valid_row[attr]
            human_ans["motives"][attr]["vote"] = _hide_votes(valid_row[f"{attr} vote"])
        elif attr == "target speaker(s)":
            human_ans[attr]["label"] = valid_row["target speaker"]
            human_ans[attr]["vote"] = _hide_votes(valid_row["target speaker vote"])
        else:
            human_ans[attr]["label"] = valid_row[attr]
            human_ans[attr]["vote"] = _hide_votes(valid_row[f"{attr} vote"])


def _even_ties(human_ans) -> List[str]:
    def _is_even(votes):
        c = Counter(votes.values())
        if c.total() <= 1:
            return False
        most = c.most_common()
        return len(most) > 1 and most[0][1] == most[1][1]

    ties = []
    for k, v in human_ans.items():
        if k == "motives":
            for m, d in v.items():
                if _is_even(d["vote"]):
                    ties.append(m)
        else:
            if _is_even(v["vote"]):
                ties.append(k)
    return ties


# ---------------------------------------------------------------------------

def cmd_anonymise(args):
    root = Path(args.input)
    clean_root = root / args.corpus / "cleaned"

    for split in args.splits:
        in_dir = root / args.corpus / f"{split}_data"
        if not in_dir.exists():
            continue
        out_dir = clean_root / f"{split}_data"
        out_dir.mkdir(parents=True, exist_ok=True)

        agg_files = {
            split: load_json_data(root / args.corpus / "agg" / f"{split}.json")
            for split in ("train", "dev", "test")
            if (root / args.corpus / "agg" / f"{split}.json").exists()
        }

        for xlsx in in_dir.glob("*.xlsx"):
            if xlsx.name.startswith("~$"):
                continue
            df = pd.read_excel(xlsx, index_col=0)
            rows = []
            for _, r in df.iterrows():
                row = r.to_dict()
                if r.role == "mod":
                    cid = f"{xlsx.stem}_{r.id}"
                    label_src = agg_files[split if split in agg_files else "train"]
                    label_row = next((x for x in label_src if x["id"] == cid), None)
                    if label_row:
                        _apply_clean_labels(row, label_row, is_train=(split == "train"))
                else:
                    _set_default_labels(row, is_train=(split == "train"))
                rows.append(row)
            pd.DataFrame(rows).to_excel(out_dir / xlsx.name)
        print(f"✓ anonymised {split} split written to {out_dir}")


def _apply_clean_labels(row: dict, label_row: dict, *, is_train: bool):
    if is_train:
        ans = label_row["answer"]
        row["dialogue act"] = dialogue_acts[ans["dialogue act"]]
        row["target speaker"] = int(str(ans["target speaker(s)"]).split()[0])
        for m in ("informational", "coordinative", "social"):
            row[f"{m} motive"] = int(f"{m} motive" in ans["motives"])
    else:
        gpt = label_row["answer"]["gpt"]
        row["dialogue act(gpt)"] = int(gpt["dialogue act"])
        row["target speaker(gpt)"] = int(gpt["target speaker(s)"])
        for m in ("informational", "coordinative", "social"):
            row[f"{m} motive(gpt)"] = int(f"{m} motive" in gpt["motives"])
    # anonymise votes ---------------------------------------------------
    for c in list(row.keys()):
        if "vote" in c:
            row[c] = _hide_votes(row[c])
        if any(x in c for x in ("consensus", "completion")):
            row.pop(c, None)


def _set_default_labels(row: dict, *, is_train: bool):
    default = -1
    keys = [
        "dialogue act",
        "target speaker",
        "informational motive",
        "coordinative motive",
        "social motive",
    ]
    if not is_train:
        keys = [k + suffix for k in keys for suffix in ("", "(gpt)")]
    for k in keys:
        row[k] = default


# ---------------------------------------------------------------------------

def cmd_gen_meta(args):
    root = Path(args.input)
    eps = _episode_paths(root, args.corpus, args.mode)
    tags = ["(Support team)", "(Against team)", "(All speakers)"]
    for ep in eps:
        meta_file = ep.with_suffix("_meta.json")
        if meta_file.exists():
            continue
        labels = pd.read_excel(ep, sheet_name="labels")
        speakers = labels.speakers.tolist()
        idx = len(speakers)
        for t in tags:
            if not any(t in s for s in speakers):
                speakers.append(f"{idx} {t}")
                idx += 1
        meta = {"default": {"topic": ep.stem, "speakers": speakers}}
        meta_file.write_text(json.dumps(meta, ensure_ascii=False))
        print(f"+ wrote {meta_file.name}")

###############################################################################
# Utility functions
###############################################################################

def _process_episode(path: Path, *, prior: int, post: int, corpus: str):
    meta = get_episode_meta(str(path))
    df = pd.read_excel(path, index_col=0)
    df["id"] = df["id"].astype(str)
    for _, r in df.iterrows():
        if r.role != "mod":
            continue
        utt, sent = map(int, r.id.split("_"))
        prior_mask = df.id.apply(lambda x: _ctx_pred(x, utt, sent, prior, before=True))
        post_mask = df.id.apply(lambda x: _ctx_pred(x, utt, sent, post, before=False))
        context = {
            "prior_context": [(v.speaker, v.role, v.text) for _, v in df[prior_mask].iterrows()],
            "post_context": [(v.speaker, v.role, v.text) for _, v in df[post_mask].iterrows()],
        }
        inst = {
            "id": r.id,
            "meta": meta,
            "context": context,
            "target": (r.speaker, r.role, r.text),
        }
        inst["prompt"] = construct_prompt_unit(inst, corpus)
        yield inst


def _ctx_pred(idx: str, utt: int, sent: int, win: int, *, before: bool) -> bool:
    u, s = map(int, idx.split("_"))
    if before:
        return (utt - win <= u < utt) or (u == utt and s < sent)
    return (utt < u <= utt + win) or (u == utt and s > sent)


def _speaker_to_code(pred: str, speakers: Sequence[str]) -> int:
    s_lower = pred.lower()
    matches = [i for i, sp in enumerate(speakers) if s_lower in sp.lower()]
    return int(matches[0] if matches else str(pred).split()[0])

###############################################################################
# Main entry‑point
###############################################################################

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="[%(levelname)s] %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
