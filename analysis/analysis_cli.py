#!/usr/bin/env python
"""analysis_cli.py
=================
End‑to‑end analysis toolkit for *INSQ* and *Roundtable* corpora. Functions that
were previously scattered across `analysis.py` are now organised into a
command‑line utility with the following sub‑commands:

* **token-stats**          – per‑episode token/utterance statistics dump (CSV).
* **cooc-matrix**          – compute motive‑×‑dialogue‑act co‑occurrence matrices
  (mean & std across episodes) and optional Welch t‑test GPT vs human.
* **transition-plot**      – plot state‑transition heat‑map (PDF/PNG).

Every command shares consistent flags for corpus, splits, and whether to use
GPT or human annotations.

Example
~~~~~~~
```bash
# 1️⃣ token statistics for roundtable dev split
python analysis_cli.py token-stats --corpus roundtable --splits dev

# 2️⃣ motive×DA matrix & t‑test
python analysis_cli.py cooc-matrix --corpus insq --mode gpt --splits dev test \
    --compare human --out ./results

# 3️⃣ transition heat‑map for moderator dialogue‑act flow
python analysis_cli.py transition-plot --corpus roundtable --mode human \
    --filter mod --output fig.pdf
```
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from attributes_prompts import dialogue_acts, dialogue_acts_decode, motive_decode

###############################################################################
# Helper / loader utilities
###############################################################################

def _load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _agg_dict(corpus_root: Path) -> Dict[str, Dict[str, dict]]:
    """Return mapping *split* → {id → sample}."""
    adict = {}
    for js in (corpus_root / "agg").glob("*.json"):
        adict[js.stem] = {x["id"]: x for x in _load_json(js)}
    return adict


def _episode_paths(root: Path, corpus: str, split: str) -> List[Path]:
    return sorted((root / corpus / f"{split}_data").glob("*.xlsx"))


def _get_meta(xlsx: Path) -> dict:
    meta_file = xlsx.with_suffix("_meta.json")
    return list(_load_json(meta_file).values())[0]

###############################################################################
# 1) Token & utterance statistics
###############################################################################

def token_stats(args: argparse.Namespace) -> None:
    root = Path(args.input)
    rows = []
    for split in args.splits:
        for ep in _episode_paths(root, args.corpus, split):
            df = pd.read_excel(ep, index_col=0)
            stats = defaultdict(int)
            lengths = defaultdict(list)
            for _, r in df.iterrows():
                if pd.isna(r.text):
                    continue
                role = r.role if r.role in ("mod", "for", "against", "speakers") else "others"
                n_tok = len(str(r.text).split())
                stats[role] += n_tok
                stats["total"] += n_tok
                lengths[role].append(n_tok)
                lengths["total"].append(n_tok)
            rows.append(
                {
                    "split": split,
                    "episode": ep.stem,
                    **{f"tok_{k}": v for k, v in stats.items()},
                    **{f"avg_len_{k}": (np.mean(v) if v else 0) for k, v in lengths.items()},
                }
            )
    out = pd.DataFrame(rows)
    out_path = Path(args.out) / f"{args.corpus}_token_stats.csv"
    out.to_csv(out_path, index=False)
    print(f"✓ Token stats saved to {out_path}")

###############################################################################
# 2) Motive × Dialogue Act co‑occurrence matrix + Welch t‑test
###############################################################################

def _collect_sequences(
    root: Path,
    corpus: str,
    agg: Dict[str, Dict[str, dict]],
    split: str,
    mode: str,
) -> List[List[dict]]:
    seqs = []
    for ep in _episode_paths(root, corpus, split):
        df = pd.read_excel(ep, index_col=0)
        meta = _get_meta(ep)
        seq = []
        for _, r in df.iterrows():
            if pd.isna(r.text) or r.role != "mod":
                continue
            cid = f"{ep.stem}_{r.id}"
            if split == "train" and mode != "human":
                ans = agg[split][cid]["answer"]
                da = str(dialogue_acts[ans["dialogue act"]])
                ts = ans["target speaker(s)"]
                mot = "".join(m[0] for m in ans["motives"])
            else:
                ans = agg[split][cid]["answer"][mode]
                da = str(ans["dialogue act"])
                ts = ans["target speaker(s)"]
                mot = "".join(m[0] for m in ans["motives"]) if mode == "gpt" else "".join(
                    m[0] for m, v in ans["motives"].items() if v["label"] == 1
                )
            seq.append({"labels": {"da": da, "m": list(mot)}, "meta": meta})
        seqs.append(seq)
    return seqs


def _cooc_matrix(sequences: List[List[dict]]) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    items, counts = [], {"da": [0] * 6, "m": [0] * 3, "total": 0}
    for sents in sequences:
        for s in sents:
            counts["total"] += 1
            d, motives = s["labels"]["da"], s["labels"]["m"]
            counts["da"][int(d)] += 1
            for m in motives:
                idx = "ics".index(m)
                counts["m"][idx] += 1
                items.append((dialogue_acts_decode[d], motive_decode[m]))
    idx = sorted({i for i, _ in items})
    col = [motive_decode[m] for m in "ics"]
    df = pd.DataFrame(0, index=idx, columns=col)
    for a, b in items:
        df.at[a, b] += 1
    df = df.T.div(counts["m"], axis=0).round(4)
    df["total"] = [c / counts["total"] for c in counts["m"]]
    df.loc["total"] = [* (np.array(counts["da"]) / counts["total"]), counts["total"]]
    df.columns = [re.sub(r"[^A-Za-z]", "", c)[:4] if c != "total" else "total" for c in df.columns]
    return df, counts


def cooc_matrix(args: argparse.Namespace) -> None:
    root = Path(args.input)
    agg = _agg_dict(root / args.corpus)
    target_mode = args.mode
    cmp_mode = args.compare

    # gather per‑episode matrices ---------------------------------------
    mats_target, mats_cmp = [], []
    for split in args.splits:
        seq_target = _collect_sequences(root, args.corpus, agg, split, target_mode)
        seq_cmp = _collect_sequences(root, args.corpus, agg, split, cmp_mode) if cmp_mode else None
        for seqs in seq_target:
            mats_target.append(_cooc_matrix([seqs])[0])
        if seq_cmp:
            for seqs in seq_cmp:
                mats_cmp.append(_cooc_matrix([seqs])[0])

    mean_t = pd.concat(mats_target).groupby(level=0).mean()
    std_t = pd.concat(mats_target).groupby(level=0).std()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_t.to_csv(out_dir / f"{args.corpus}_{target_mode}_cooc_mean.csv")
    std_t.to_csv(out_dir / f"{args.corpus}_{target_mode}_cooc_std.csv")
    print("✓ Saved mean/std csv files")

    if cmp_mode:
        mean_c = pd.concat(mats_cmp).groupby(level=0).mean()
        std_c = pd.concat(mats_cmp).groupby(level=0).std()
        ttest = _welch_t(mean_t, std_t, mean_c, std_c)
        ttest.to_csv(out_dir / f"{args.corpus}_{target_mode}_vs_{cmp_mode}_ttest.csv")
        print("✓ Welch t‑test csv saved")


def _welch_t(a_mean, a_std, b_mean, b_std):
    res = pd.DataFrame(False, index=a_mean.index, columns=a_mean.columns)
    for i in res.index:
        for c in res.columns:
            if c == "total":
                continue
            n1 = int(a_mean.at[i, "total"] * a_mean.at["total", "total"])
            n2 = int(b_mean.at[i, "total"] * b_mean.at["total", "total"])
            t, p = stats.ttest_ind_from_stats(
                mean1=a_mean.at[i, c], std1=a_std.at[i, c], nobs1=n1,
                mean2=b_mean.at[i, c], std2=b_std.at[i, c], nobs2=n2, equal_var=False
            )
            res.at[i, c] = p < 0.05
    return res

###############################################################################
# 3) Transition matrix plotting
###############################################################################

def transition_plot(args: argparse.Namespace) -> None:
    root = Path(args.input)
    agg = _agg_dict(root / args.corpus)
    sequences = []
    for split in args.splits:
        sequences.extend(_collect_sequences(root, args.corpus, agg, split, args.mode))
    # Build state list ----------------------------------------------------
    transitions, states = defaultdict(lambda: defaultdict(int)), set()
    for seq in sequences:
        seq = [s for s in seq if (not args.filter or s["role"] == args.filter)]
        for s1, s2 in zip(seq[:-1], seq[1:]):
            a = dialogue_acts_decode[s1["labels"]["da"]]
            b = dialogue_acts_decode[s2["labels"]["da"]]
            transitions[a][b] += 1
            states.update([a, b])
    states = sorted(states)
    mat = pd.DataFrame(0, index=states, columns=states)
    for a, d in transitions.items():
        for b, v in d.items():
            mat.at[a, b] = v
    mat = mat.div(mat.sum(axis=1), axis=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, annot=True, cmap="Blues", fmt=".2f")
    plt.xlabel("To state")
    plt.ylabel("From state")
    out = Path(args.output)
    plt.savefig(out, bbox_inches="tight", dpi=300)
    print(f"✓ Transition matrix plot saved to {out}")

###############################################################################
# CLI factory
###############################################################################

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM dialogue analysis CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _global(sp):
        sp.add_argument("--corpus", default="roundtable")
        sp.add_argument("--splits", nargs="*", default=["dev", "test"], help="data splits")
        sp.add_argument("--mode", choices=["gpt", "human"], default="gpt")
        sp.add_argument("--input", default="./data")
        sp.add_argument("--out", default="./results")
        sp.add_argument("--log-level", default="INFO")

    # token-stats --------------------------------------------------------
    sp_tok = sub.add_parser("token-stats", help="episode token/utterance stats")
    _global(sp_tok)
    sp_tok.set_defaults(func=token_stats)

    # cooc-matrix --------------------------------------------------------
    sp_cooc = sub.add_parser("cooc-matrix", help="motive × dialogue‑act matrix")
    _global(sp_cooc)
    sp_cooc.add_argument("--compare", choices=["gpt", "human", None], default=None,
                         help="optional second mode for t‑test comparison")
    sp_cooc.set_defaults(func=cooc_matrix)

    # transition-plot ----------------------------------------------------
    sp_tr = sub.add_parser("transition-plot", help="state transition heat‑map")
    _global(sp_tr)
    sp_tr.add_argument("--filter", default=None, help="role filter (e.g. mod)")
    sp_tr.add_argument("--output", default="transition.pdf")
    sp_tr.set_defaults(func=transition_plot)

    return p

###############################################################################
# Entry
###############################################################################

def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=args.log_level, format="[%(levelname)s] %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
