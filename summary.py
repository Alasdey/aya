#!/usr/bin/env python3
"""
Summarize MECI run_*.json logs into an Excel sheet.

Extracted columns:
- timestamp_utc
- max_examples
- split
- model
- langsmith_project
- tools_enabled
- micro_precision, micro_recall, micro_f1
- per-label metrics (e.g., CauseEffect_precision/recall/f1/support, etc.)
- skipped_docs

Usage:
  python summarize_runs.py --logs ./logs --out runs_summary.xlsx
"""

import argparse
import glob
import json
import os
from typing import Dict, Any, List

import pandas as pd


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def flatten_per_label(results: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    per_label = results.get("per_label", {}) or {}
    for label, metrics in per_label.items():
        if not isinstance(metrics, dict):
            continue
        out[f"{label}_precision"] = metrics.get("precision")
        out[f"{label}_recall"] = metrics.get("recall")
        out[f"{label}_f1"] = metrics.get("f1")
        out[f"{label}_support"] = metrics.get("support")
    return out


# NEW: Flatten per-language metrics (micro only, as requested)
def flatten_per_lang(results: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    per_lang_metrics = results.get("per_lang_metrics", {}) or {}
    known_langs = ["causal-en", "causal-da", "causal-es", "causal-tr", "causal-ur"]  # Based on dataset
    for lang in known_langs:
        mc = safe_get(per_lang_metrics, [lang, "multiclass"], {})
        out[f"{lang}_micro_precision"] = mc.get("micro_precision")
        out[f"{lang}_micro_recall"] = mc.get("micro_recall")
        out[f"{lang}_micro_f1"] = mc.get("micro_f1")
        # Optional: Add more (e.g., macro_f1, total_pairs) if needed
        # out[f"{lang}_macro_f1"] = mc.get("macro_f1")
        # out[f"{lang}_total_pairs"] = per_lang_metrics.get(lang, {}).get("total_pairs")
    return out


def extract_row(data: Dict[str, Any], fname: str) -> Dict[str, Any]:
    row = {
        # helpful identifiers
        "file": os.path.basename(fname),
        "run_id": data.get("run_id"),
        "status": data.get("status"),

        # requested top-level fields
        "timestamp_utc": data.get("timestamp_utc"),

        # cli_args
        "max_examples": safe_get(data, ["cli_args", "max_examples"]),
        "split": safe_get(data, ["cli_args", "split"]),

        # config
        "model": safe_get(data, ["config", "model"]),
        "langsmith_project": safe_get(data, ["config", "langsmith_project"]),
        "tools_enabled": ", ".join(safe_get(data, ["config", "tools", "enabled"], []) or []),

        # micro metrics
        "micro_precision": safe_get(data, ["results", "micro_precision"]),
        "micro_recall": safe_get(data, ["results", "micro_recall"]),
        "micro_f1": safe_get(data, ["results", "micro_f1"]),

        # skipped docs (handle variant keys defensively)
        "skipped_docs": safe_get(data, ["results", "skipped_docs"]) or safe_get(data, ["results", "skipped"]) or 0,
    }

    # per-label metric columns
    per_label_cols = flatten_per_label(safe_get(data, ["results"]) or {})
    row.update(per_label_cols)

    # NEW: per-language metric columns
    per_lang_cols = flatten_per_lang(safe_get(data, ["results"]) or {})
    row.update(per_lang_cols)

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs", help="Folder containing run_*.json files")
    ap.add_argument("--out", default="runs_summary.xlsx", help="Path to output Excel file")
    args = ap.parse_args()

    pattern = os.path.join(args.logs, "run_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files matched {pattern}")

    rows: List[Dict[str, Any]] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows.append(extract_row(data, fp))
        except Exception as e:
            # Keep going; record error for visibility
            rows.append({
                "file": os.path.basename(fp),
                "parse_error": str(e),
            })

    df = pd.DataFrame(rows)

    # Nice-ish column ordering: put the requested keys up front if present
    preferred = [
        "file", "run_id", "status",
        "timestamp_utc", "max_examples", "split", "model", "langsmith_project", "tools_enabled",
        "micro_precision", "micro_recall", "micro_f1",
        "skipped_docs",
    ]
    # NEW: Add per-language columns to preferred
    for lang in ["causal-en", "causal-da", "causal-es", "causal-tr", "causal-ur"]:
        preferred.extend([f"{lang}_micro_precision", f"{lang}_micro_recall", f"{lang}_micro_f1"])

    # Move preferred to front, keep the rest (including per-label columns) afterward
    cols_front = [c for c in preferred if c in df.columns]
    cols_rest = [c for c in df.columns if c not in cols_front]
    df = df[cols_front + cols_rest]

    # Write to Excel
    # (openpyxl is the default writer; ensure it's installed for .xlsx)
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="runs")

    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
