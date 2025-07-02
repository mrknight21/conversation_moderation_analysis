#!/usr/bin/env python
"""make_meta.py
================
Generate `<episode>_meta.json` files containing a canonical speaker list for the
`insq` and `roundtable` corpora.

Key rules
---------
* **Universal base tags**: `(Unknown)`, `(Self)`, `(Everyone)`, `(Audience)`.
* **insq**:
  • Drop rows where `role` is *mod* / *moderator*.
  • Speakers with role *for* must precede those with role *against*.
  • Append `(Support team)`, `(Against team)`, `(All speakers)`.
* **roundtable**:
  • Insert `(_NO_SPEAKER)` placeholder after base tags.
  • Always append `(All speakers)`.
* The final list is **unique & order‑preserving**. Each tag is prefixed with a
  numeric code (`0 … n`).

Example (*insq*)
~~~~~~~~~~~~~~~~
```json
{
  "default": {
    "topic": "34_8444_Dont_Eat_Anything_With_A_Face",
    "speakers": [
      "0 (Unknown)",
      "1 (Self)",
      "2 (Everyone)",
      "3 (Audience)",
      "4 (John R. Lott - for)",
      "5 (Stephen Halbrook - for)",
      "6 (Gary Kleck - for)",
      "7 (R. Gil Kerlikowske - against)",
      "8 (John J. Donohue - against)",
      "9 (Paul Helmke - against)",
      "10 (Support team)",
      "11 (Against team)",
      "12 (All speakers)"
    ]
  }
}
```

Usage
~~~~~
```bash
python make_meta.py \
  --corpus insq --mode train \
  --input ./insq/train_data --output ./insq/train_data
```
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List
import glob

import pandas as pd

###############################################################################
# Tag configuration
###############################################################################
BASE_TAGS = ["Unknown", "Self", "Everyone", "Audience"]

TAIL_TAGS = {
    "insq": ["Support team", "Against team", "All speakers"],
    "roundtable": ["All speakers"],
}

###############################################################################
# Helper functions
###############################################################################

def _unique_preserve(seq: List[str]) -> List[str]:
    """Return list with order preserved and duplicates removed."""
    seen = set()
    out = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _read_speakers_insq(xlsx: Path) -> List[str]:
    df = pd.read_excel(xlsx, sheet_name="labels") if "labels" in pd.ExcelFile(xlsx).sheet_names else pd.read_excel(xlsx)
    if "role" not in df.columns:
        return df.speaker.astype(str).tolist()
    # filter out moderators, split by stance
    for_mask = (df.role.str.lower() == "for")
    against_mask = (df.role.str.lower() == "against")
    fors = [f"{s} - for" for s in df.loc[for_mask, "speaker"].astype(str)]
    against = [f"{s} - against" for s in df.loc[against_mask, "speaker"].astype(str)]
    return fors + against


def _read_speakers_roundtable(xlsx: Path) -> List[str]:
    try:
        df = pd.read_excel(xlsx, sheet_name="labels")
    except ValueError:
        df = pd.read_excel(xlsx)
    return df.speaker.astype(str).tolist()


def _get_speaker_labels(xlsx: Path, corpus: str) -> List[str]:
    if corpus == "insq":
        return _read_speakers_insq(xlsx)
    return _read_speakers_roundtable(xlsx)

###############################################################################
# Main routine
###############################################################################

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate <episode>_meta.json files")
    p.add_argument("--corpus", choices=["insq", "roundtable"], default="roundtable")
    p.add_argument("--mode", default="test", help="data split folder")
    return p


def generate_meta() -> None:
    args = build_parser().parse_args()
    in_dir, out_dir = f"./{args.corpus}/{args.mode}_data/", f"./{args.corpus}/{args.mode}_data/"

    tail_tags = TAIL_TAGS[args.corpus]

    for xlsx in sorted(glob.glob(in_dir + "*.xlsx")):
        labels = _get_speaker_labels(xlsx, args.corpus)

        # assemble full list -------------------------------------------------
        full = BASE_TAGS.copy()
        if args.corpus == "roundtable":
            full.append("(_NO_SPEAKER)")
        full.extend(labels)
        for tag in tail_tags:
            if tag not in full:
                full.append(tag)
        full = _unique_preserve(full)

        # enumerate ---------------------------------------------------------
        speakers = [f"{idx} ({tag})" for idx, tag in enumerate(full)]
        topic = xlsx.split("/")[-1].replace(".xlsx", "")
        meta = {"default": {"topic": topic, "speakers": speakers}}

        out_file = out_dir + f"{topic}_meta.json"
        with open(out_file, "w") as f:
            json.dumps(meta, f, ensure_ascii=False, indent=2)
        print(f"✓ {out_file}")

def clean_meta() -> None:
    args = build_parser().parse_args()
    meta_dir = f"./{args.corpus}/{args.mode}_data/"

    for meta_file in sorted(glob.glob(meta_dir + "*_meta.json")):
        with open(meta_file, "r") as f:
            meta = json.load(f)

        annotator = list(meta.keys())[0]
        topic = meta[annotator]["topic"]
        speakers = meta[annotator]["speakers"]

        cleaned_meta = {"default": {"topic": topic, "speakers": speakers}}

        with open(meta_file, "w") as f:
            json.dump(cleaned_meta, f, ensure_ascii=False, indent=2)

        if "agg" in meta_file:
            new_name = topic + "_meta.json"
            old_path = Path(meta_file).expanduser().resolve()
            new_path = old_path.with_name(new_name)
            old_path.rename(new_path)


if __name__ == "__main__":
    clean_meta()
