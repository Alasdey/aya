#!/usr/bin/env python3
"""
Reconstitute per-language micro precision/recall/F1 from existing MECI run_*.json logs.

Usage:
  python reconstitute_per_lang_from_logs.py --logs ./logs --repo Nofing/MECI-v0.1-public-span --split test --out per_lang_summary.xlsx
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Any

import pandas as pd
from datasets import load_dataset

def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def reconstruct_counts(precision: float, recall: float, support: int) -> Dict[str, int]:
    """
    Reverse-engineer TP, FP, FN from precision, recall, support.
    Uses rounding to handle float precision issues.
    """
    if support == 0:
        return {"tp": 0, "fp": 0, "fn": 0}
    
    tp = round(recall * support)
    if precision > 0:
        fp = round(tp / precision - tp)
    else:
        fp = 0  # If precision=0, likely tp=0
    fn = support - tp
    
    # Sanity check: ensure non-negative
    tp = max(0, tp)
    fp = max(0, fp)
    fn = max(0, fn)
    
    return {"tp": tp, "fp": fp, "fn": fn}

def compute_micro_from_counts(total_tp: int, total_fp: int, total_fn: int) -> Dict[str, float]:
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }

def reconstitute_per_lang(per_doc_metrics: List[Dict[str, Any]], ds: Any) -> Dict[str, Dict[str, float]]:
    """
    Reconstitute per-language micro metrics by aggregating TP/FP/FN from per-doc per-label metrics.
    """
    per_lang_counts = defaultdict(lambda: {"total_tp": 0, "total_fp": 0, "total_fn": 0})
    positive_labels = ["CauseEffect", "EffectCause"]  # Exclude NoRel, as in your micro metrics
    
    for doc in per_doc_metrics:
        doc_idx = doc.get("doc_idx")
        if doc_idx is None or doc_idx >= len(ds):
            continue  # Skip invalid doc_idx
        
        # Get lang from dataset
        row = ds[doc_idx]
        lang = row.get("lang", "unknown")
        
        # Get per_label from log
        per_label = doc.get("per_label", {})
        
        # Accumulate counts for positive labels
        for label in positive_labels:
            metrics = per_label.get(label, {})
            prec = metrics.get("precision", 0.0)
            rec = metrics.get("recall", 0.0)
            supp = metrics.get("support", 0)
            
            counts = reconstruct_counts(prec, rec, supp)
            per_lang_counts[lang]["total_tp"] += counts["tp"]
            per_lang_counts[lang]["total_fp"] += counts["fp"]
            per_lang_counts[lang]["total_fn"] += counts["fn"]
    
    # Compute micro metrics per lang
    per_lang_metrics = {}
    for lang, counts in per_lang_counts.items():
        per_lang_metrics[lang] = compute_micro_from_counts(
            counts["total_tp"], counts["total_fp"], counts["total_fn"]
        )
    
    return per_lang_metrics

def flatten_per_lang(per_lang_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    out = {}
    known_langs = ["causal-en", "causal-da", "causal-es", "causal-tr", "causal-ur"]  # Based on your dataset
    for lang in known_langs:
        mc = per_lang_metrics.get(lang, {})
        out[f"{lang}_micro_precision"] = mc.get("micro_precision")
        out[f"{lang}_micro_recall"] = mc.get("micro_recall")
        out[f"{lang}_micro_f1"] = mc.get("micro_f1")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs", help="Folder containing run_*.json files")
    ap.add_argument("--repo", required=True, help="Hugging Face dataset repo (e.g., Nofing/MECI-v0.1-public-span)")
    ap.add_argument("--split", default="test", help="Dataset split")
    ap.add_argument("--out", default="per_lang_summary.xlsx", help="Output Excel file")
    args = ap.parse_args()

    ds = load_dataset(args.repo, split=args.split)

    pattern = os.path.join(args.logs, "run_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files matched {pattern}")

    rows: List[Dict[str, Any]] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract key info from log
            row = {
                "file": os.path.basename(fp),
                "run_id": data.get("run_id"),
                "timestamp_utc": data.get("timestamp_utc"),
                "max_examples": safe_get(data, ["cli_args", "max_examples"]),
                "split": safe_get(data, ["cli_args", "split"]),
                "model": safe_get(data, ["config", "model"]),
                "langsmith_project": safe_get(data, ["config", "langsmith_project"]),
                "tools_enabled": ", ".join(safe_get(data, ["config", "tools", "enabled"], []) or []),
                "micro_precision": safe_get(data, ["results", "micro_precision"]),
                "micro_recall": safe_get(data, ["results", "micro_recall"]),
                "micro_f1": safe_get(data, ["results", "micro_f1"]),
                "skipped_docs": safe_get(data, ["results", "skipped_docs"], 0),
            }

            # Reconstitute per-lang metrics
            per_doc_metrics = safe_get(data, ["results", "per_doc_metrics"], [])
            if per_doc_metrics:
                per_lang_metrics = reconstitute_per_lang(per_doc_metrics, ds)
                per_lang_cols = flatten_per_lang(per_lang_metrics)
                row.update(per_lang_cols)
            else:
                row["error"] = "No per_doc_metrics in log"

            rows.append(row)
        except Exception as e:
            rows.append({
                "file": os.path.basename(fp),
                "error": str(e),
            })

    df = pd.DataFrame(rows)

    # Column ordering
    preferred = [
        "file", "run_id", "timestamp_utc", "max_examples", "split", "model", "langsmith_project", "tools_enabled",
        "micro_precision", "micro_recall", "micro_f1", "skipped_docs",
    ]
    for lang in ["causal-en", "causal-da", "causal-es", "causal-tr", "causal-ur"]:
        preferred.extend([f"{lang}_micro_precision", f"{lang}_micro_recall", f"{lang}_micro_f1"])
    cols_front = [c for c in preferred if c in df.columns]
    cols_rest = [c for c in df.columns if c not in cols_front]
    df = df[cols_front + cols_rest]

    # Write to Excel
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="per_lang_runs")

    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()