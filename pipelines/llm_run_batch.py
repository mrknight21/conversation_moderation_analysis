#!/usr/bin/env python
"""
batch_processing.py
-------------------
A refactored, command‑line–driven utility for generating, submitting, and
retrieving OpenAI batch jobs used in the ISQ corpus annotation workflow.


Example
~~~~~~~
```bash
# 1️⃣ build & upload batches for dev set
python batch_processing.py create-batches \
    --corpus insq --mode dev --model gpt-4o \
    --input ./data/insq/dev_data --output ./data/insq/output

# 2️⃣ monitor progress later on
python batch_processing.py check-status --record ./log/batch_upload_record.json

# 3️⃣ download once completed (will auto‑repair invalid json)
python batch_processing.py download-output --record ./log/batch_upload_record.json
```
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from openai import OpenAI

# External project‑specific helpers (remain unchanged)
from llm_run_direct import get_episode_meta, gpt_batch
from pipelines.prompts.attributes_prompts import construct_prompt_unit

###############################################################################
# Configuration helpers
###############################################################################

def default_labels() -> List[str]:
    """Return the full 5‑label set expected by the original pipeline."""
    return [
        "informational motive",
        "social motive",
        "coordinative motive",
        "dialogue act",
        "target speaker",
    ]


def get_openai_client() -> OpenAI:
    """Initialise an OpenAI client from env‑vars and return it."""
    key = os.getenv("OPENAI_API_KEY", "")
    org = os.getenv("OPENAI_ORG_ID", "")
    if not key:
        sys.exit("[ERR] OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=key, organization=org or None)


###############################################################################
# Prompt generation utilities
###############################################################################

def process_episode(
    episode_path: Path,
    model_name: str,
    labels: List[str],
    prior_ctx: int,
    post_ctx: int,
    corpus: str,
) -> List[Dict[str, Any]]:
    """Generate a list of batch‑submission dictionaries for one episode file."""

    meta = get_episode_meta(str(episode_path))
    debate = pd.read_excel(episode_path, index_col=0)
    debate["id"] = debate["id"].astype(str)

    prompts: List[Dict[str, Any]] = []

    for _, row in debate.iterrows():
        if row.role != "mod":
            continue  # we only label moderator utterances

        utt, sent = map(int, row.id.split("_"))

        prior_mask = debate.id.apply(
            lambda x: not (
                not (utt - prior_ctx <= int(x.split("_")[0]) < utt)
                and not (int(x.split("_")[0]) == utt and int(x.split("_")[1]) < sent)
            )
        )
        post_mask = debate.id.apply(
            lambda x: not (
                not (utt + post_ctx >= int(x.split("_")[0]) > utt)
                and not (int(x.split("_")[0]) == utt and int(x.split("_")[1]) > sent)
            )
        )

        context = {
            "prior_context": [
                (v.speaker, v.role, v.text) for _, v in debate[prior_mask].iterrows()
            ],
            "post_context": [
                (v.speaker, v.role, v.text) for _, v in debate[post_mask].iterrows()
            ],
        }

        instance = {
            "id": row.id,
            "meta": meta,
            "context": context,
            "target": (row.speaker, row.role, row.text),
        }
        prompt = construct_prompt_unit(instance, corpus, labels)

        prompts.append(
            {
                "custom_id": f"{episode_path.stem}_{row.id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 1,
                    "response_format": {"type": "json_object"},
                    "max_tokens": 300,
                },
            }
        )
    return prompts


###############################################################################
# IO helpers
###############################################################################

def write_jsonl(data: Iterable[Dict[str, Any]], path: Path) -> None:
    """Write an iterable of dictionaries to *path* in JSON‑Lines format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf‑8") as fh:
        for item in data:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


#####################################
# Validation helpers (unchanged logic)
#####################################
_JSON_RE = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}")


def _is_valid_answer(raw: str, labels: List[str]) -> bool:
    """Replicates original *check_single_answer* logic (simplified)."""
    try:
        answer = json.loads(raw)
    except Exception:
        return False

    # 5‑label variant (status‑quo)
    if labels == default_labels():
        required = {"motives", "dialogue act", "target speaker(s)"}
        if not required.issubset(answer):
            return False
        if not all(
            m in ["informational motive", "social motive", "coordinative motive"]
            for m in answer.get("motives", [])
        ):
            return False
        if answer.get("dialogue act") not in [
            "Probing",
            "Confronting",
            "Supplement",
            "Interpretation",
            "Instruction",
            "All Utility",
        ]:
            return False
        return True  # predicates satisfied

    # Reduced‑label variants (1‑label tasks)
    key = labels[0]
    if key == "dialogue act":
        return answer.get("dialogue act") in [
            "Probing",
            "Confronting",
            "Supplement",
            "Interpretation",
            "Instruction",
            "All Utility",
        ]
    if "motive" in key:
        return str(answer.get("verdict")) in {"0", "1"}
    return "target speaker(s)" in answer  # final fallback


###############################################################################
# Batch‑level OpenAI helpers
###############################################################################

def openai_upload_batch(
    client: OpenAI,
    data_path: Path,
    description: str,
    labels: List[str],
) -> Tuple[str, str]:
    """Upload the *data_path* jsonl and create a 24 h chat batch."""
    upload = client.files.create(file=data_path.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description, "labels": json.dumps(labels)},
    )
    return upload.id, batch.id


###############################################################################
# Sub‑command implementations
###############################################################################

def cmd_create_batches(args: argparse.Namespace) -> None:
    client = get_openai_client()
    episodes = sorted(Path(args.input).glob("*.xlsx"))
    if not episodes:
        sys.exit(f"[ERR] No .xlsx files found under {args.input}.")

    batch_index, buffer = 0, []
    record: Dict[int, Dict[str, str]] = {}

    for i, episode in enumerate(episodes, 1):
        if episode.name.startswith("~$"):
            continue  # skip Excel temp files
        buffer.extend(
            process_episode(
                episode, args.model, args.labels, args.prior, args.post, args.corpus
            )
        )
        if i % args.chunk == 0:
            batch_path = _emit_and_upload(
                client, buffer, batch_index, args.output, args.mode, args.corpus, args.labels
            )
            record[batch_index] = batch_path
            batch_index += 1
            buffer = []

    # remainder
    if buffer:
        batch_path = _emit_and_upload(
            client, buffer, batch_index, args.output, args.mode, args.corpus, args.labels
        )
        record[batch_index] = batch_path

    # write log
    log_file = Path(args.log) / "batch_upload_record.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(json.dumps(record, indent=2))
    print(f"✓ Upload record written to {log_file}")


def _emit_and_upload(
    client: OpenAI,
    data: List[Dict[str, Any]],
    index: int,
    out_dir: str | Path,
    mode: str,
    corpus: str,
    labels: List[str],
) -> Dict[str, str]:
    label_tag = "" if len(labels) == 5 else "_" + "".join(w[0] for w in labels[0].split())
    jsonl_path = (
        Path(out_dir)
        / f"{corpus}_batch_{mode}{label_tag}_part{index}.jsonl"
    )
    write_jsonl(data, jsonl_path)

    print(f"↥ Uploading batch part {index} ({len(data)} requests)…")
    file_id, batch_id = openai_upload_batch(
        client,
        jsonl_path,
        description=f"{mode} part {index}",
        labels=labels,
    )
    print(f"  • file_id = {file_id}\n  • batch_id = {batch_id}\n")
    return {"file_id": file_id, "batch_id": batch_id}


# ---------------------------------------------------------------------------

def cmd_check_status(args: argparse.Namespace) -> None:
    client = get_openai_client()
    record = _load_record(args.record)
    for part, ids in record.items():
        status = client.batches.retrieve(ids["batch_id"]).status
        print(f"part {part}: {status}")


# ---------------------------------------------------------------------------

def cmd_download_output(args: argparse.Namespace) -> None:
    client = get_openai_client()
    record = _load_record(args.record)
    all_valid, all_invalid = [], []

    for part, ids in record.items():
        valid, invalid = _download_single_batch(
            client,
            ids["batch_id"],
            part_index=int(part),
            labels=args.labels,
            model=args.model,
            mode=args.mode,
            out_dir=Path(args.output),
            repair=args.repair,
        )
        all_valid.extend(valid)
        all_invalid.extend(invalid)

    # aggregate‑level dump
    if all_valid:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        Path(args.output, f"{args.model}_{args.mode}_full_output.json").write_text(
            json.dumps(all_valid, ensure_ascii=False)
        )
    if all_invalid:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        Path(args.output, f"{args.model}_{args.mode}_full_invalid_output.json").write_text(
            json.dumps(all_invalid, ensure_ascii=False)
        )


# helpers --------------------------------------------------------------------

def _download_single_batch(
    client: OpenAI,
    batch_id: str,
    *,
    part_index: int,
    labels: List[str],
    model: str,
    mode: str,
    out_dir: Path,
    repair: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    info = client.batches.retrieve(batch_id)
    print(f"batch {batch_id} (part {part_index}) → {info.status}")
    if info.status != "completed":
        return [], []

    content = client.files.content(info.output_file_id)
    valid, invalid = [], []
    for line in content.iter_lines():
        rec = json.loads(line)
        cid, body = rec["custom_id"], rec["response"]["body"]
        if rec["response"]["status_code"] != 200:
            invalid.append(rec)
            continue
        answer_raw = body["choices"][0]["message"]["content"]
        matches = _JSON_RE.findall(answer_raw)
        if matches and _is_valid_answer(matches[0], labels):
            valid.append(
                {
                    "custom_id": cid,
                    "model_used": body["model"],
                    "answer": json.loads(matches[0]),
                    "usage": body["usage"],
                }
            )
        else:
            invalid.append(rec)

    # optional repair pass --------------------------------------------------
    if invalid and repair:
        print(f"↻ Attempting repair for {len(invalid)} invalid outputs…")
        tasks = [
            {
                "id": rec["custom_id"],
                "prompt": rec["response"]["body"]["choices"][0]["message"]["content"],
                "meta": {},
            }
            for rec in invalid
        ]
        repairs = gpt_batch(tasks, model_name="gpt4", labels=labels)
        for rep in repairs:
            matched = next((r for r in invalid if r["custom_id"] == rep["id"]), None)
            if matched and _is_valid_answer(rep["output"], labels):
                body = matched["response"]["body"]
                body["choices"][0]["message"]["content"] = rep["output"]
                body["usage"]["repaired"] = True
                valid.append(
                    {
                        "custom_id": matched["custom_id"],
                        "model_used": body["model"],
                        "answer": json.loads(rep["output"]),
                        "usage": body["usage"],
                    }
                )
                invalid.remove(matched)

    # dump part‑level jsonl
    out_dir.mkdir(parents=True, exist_ok=True)
    label_tag = "" if len(labels) == 5 else "_" + "".join(w[0] for w in labels[0].split())
    part_file = out_dir / f"batch_{model}_{mode}{label_tag}_output_part{part_index}.jsonl"
    write_jsonl(valid + invalid, part_file)
    print(f"  • Saved to {part_file} (valid={len(valid)}, invalid={len(invalid)})")

    return valid, invalid


# ---------------------------------------------------------------------------

def cmd_process_invalid(args: argparse.Namespace) -> None:
    client = get_openai_client()
    invalid_cases = json.loads(Path(args.invalid).read_text())
    invalid_ids = {c["custom_id"] for c in invalid_cases}
    print(f"Invalid cases: {len(invalid_ids)}")

    episodes = sorted(Path(args.input).glob("*.xlsx"))
    prompts = []
    for ep in episodes:
        prompts.extend(
            process_episode(
                ep, args.model, args.labels, args.prior, args.post, args.corpus
            )
        )
    retry_prompts = [p for p in prompts if p["custom_id"] in invalid_ids]

    retry_file = Path(args.output) / f"{args.corpus}_batch_{args.mode}_invalid_cases.jsonl"
    write_jsonl(retry_prompts, retry_file)

    file_id, batch_id = openai_upload_batch(
        client,
        retry_file,
        description="invalid case repair",
        labels=args.labels,
    )
    print(f"Retry batch submitted (file={file_id}, batch={batch_id})")


###############################################################################
# Argument parsing machinery
###############################################################################

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ISQ batch‑processing utility")
    sub = p.add_subparsers(dest="cmd", required=True)

    # shared opts
    def _add_common(sp):
        sp.add_argument("--corpus", default="insq", help="Corpus name tag")
        sp.add_argument("--mode", default="test", help="Data split (train/dev/test)")
        sp.add_argument("--model", default="gpt-4o", help="OpenAI model name")
        sp.add_argument("--labels", nargs="*", default=default_labels(), help="Label list")
        sp.add_argument("--prior", type=int, default=5, help="Prior context window size")
        sp.add_argument("--post", type=int, default=2, help="Post context window size")

    # create‑batches -------------------------------------------------------
    sp_create = sub.add_parser("create-batches", help="Generate + upload batch jobs")
    _add_common(sp_create)
    sp_create.add_argument("--input", default="./data/insq/test_data", help="Episode XLSX folder")
    sp_create.add_argument("--output", default="./data/insq/output", help="Output folder")
    sp_create.add_argument("--log", default="./log", help="Log folder")
    sp_create.add_argument("--chunk", type=int, default=70, help="Requests per batch part")
    sp_create.set_defaults(func=cmd_create_batches)

    # check‑status ---------------------------------------------------------
    sp_status = sub.add_parser("check-status", help="Check batch status via log file")
    sp_status.add_argument("--record", required=True, help="Path to batch_upload_record.json")
    sp_status.set_defaults(func=cmd_check_status)

    # download‑output ------------------------------------------------------
    sp_dl = sub.add_parser("download-output", help="Download & validate batch outputs")
    _add_common(sp_dl)
    sp_dl.add_argument("--record", required=True, help="Path to batch_upload_record.json")
    sp_dl.add_argument("--output", default="./data/insq/output", help="Where to write jsonl/json")
    sp_dl.add_argument("--repair", action="store_true", help="Attempt repair via gpt4")
    sp_dl.set_defaults(func=cmd_download_output)

    # process‑invalid ------------------------------------------------------
    sp_inv = sub.add_parser("process-invalid", help="Resubmit prompts for invalid cases")
    _add_common(sp_inv)
    sp_inv.add_argument("--invalid", required=True, help="Invalid output json file")
    sp_inv.add_argument("--input", default="./data/insq/test_data", help="Episode XLSX folder")
    sp_inv.add_argument("--output", default="./data/insq/output", help="Output folder")
    sp_inv.set_defaults(func=cmd_process_invalid)

    return p


###############################################################################
# Entry point
###############################################################################

def _load_record(path: str | Path) -> Dict[str, Dict[str, str]]:
    return json.loads(Path(path).read_text())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)  # dispatch


if __name__ == "__main__":
    main()
