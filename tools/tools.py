from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import List, Dict, Any, Tuple, Optional

from langchain_core.tools import tool
from langsmith.run_helpers import traceable  # <- LangSmith tracing


DATASET_DEFAULT = Path(__file__).resolve().parents[1] / "data" / "sample.jsonl"


def _load_rows(dataset_path: str | Path = DATASET_DEFAULT) -> List[Dict[str, Any]]:
    p = Path(dataset_path)
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@tool
def mean_price(dataset_path: str = str(DATASET_DEFAULT)) -> str:
    """Compute the mean price over all items in the JSONL dataset."""
    rows = _load_rows(dataset_path)
    prices = [float(r["price"]) for r in rows if "price" in r]
    return f"{mean(prices):.2f}"


@tool
def find_by_keyword(keyword: str, dataset_path: str = str(DATASET_DEFAULT)) -> str:
    """Return items whose title or tags contain the keyword (case-insensitive)."""
    rows = _load_rows(dataset_path)
    k = keyword.lower().strip()
    hits = [r for r in rows if k in r.get("title", "").lower() or any(k in t.lower() for t in r.get("tags", []))]
    return "[]" if not hits else json.dumps(hits, ensure_ascii=False)


CAUSAL = {"CauseEffect", "EffectCause"}

def _inverse_label(lab: str) -> str:
    return {"CauseEffect": "EffectCause", "EffectCause": "CauseEffect"}.get(lab, lab)

@tool
def coherence_check(
    *,
    pairs: List[Dict[str, str]],
    conversation: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    LangSmith-traced function tool.

    Args:
        pairs: [{"pair":"Ti,Tj","label":"CauseEffect|EffectCause|NoRel"}, ...]
        conversation: (optional) the running chat history; included only for trace context.

    Returns:
        {
          "considered": int,
          "symmetric_ok": List[List[str]],
          "conflicts": List[{a,b,lab_ab,lab_ba,suggest_ba}],
          "missing_reverse": List[{a,b,lab_ab,suggest_ba}],
          "coherence_rate": float
        }
    """
    dir_map: Dict[Tuple[str, str], str] = {}
    for obj in pairs or []:
        p = (obj.get("pair") or "").strip()
        lab = (obj.get("label") or "").strip()
        if p and lab and "," in p:
            a, b = [t.strip() for t in p.split(",", 1)]
            dir_map[(a, b)] = lab

    unordered = set()
    for (a, b) in dir_map:
        if a != b:
            unordered.add(tuple(sorted((a, b))))

    considered = 0
    symmetric_ok: List[List[str]] = []
    conflicts: List[Dict[str, str]] = []
    missing_reverse: List[Dict[str, str]] = []

    for (x, y) in unordered:
        lab_xy = dir_map.get((x, y))
        lab_yx = dir_map.get((y, x))
        cx = lab_xy in CAUSAL if lab_xy else False
        cy = lab_yx in CAUSAL if lab_yx else False
        if not (cx or cy):
            continue

        considered += 1
        if lab_xy and lab_yx:
            if (lab_xy, lab_yx) in (("CauseEffect", "EffectCause"), ("EffectCause", "CauseEffect")):
                symmetric_ok.append([x, y])
            else:
                conflicts.append({
                    "a": x, "b": y,
                    "lab_ab": lab_xy, "lab_ba": lab_yx,
                    "suggest_ba": _inverse_label(lab_xy) if lab_xy in CAUSAL else lab_yx
                })
        else:
            if lab_xy in CAUSAL and not lab_yx:
                missing_reverse.append({"a": x, "b": y, "lab_ab": lab_xy, "suggest_ba": _inverse_label(lab_xy)})
            if lab_yx in CAUSAL and not lab_xy:
                missing_reverse.append({"a": y, "b": x, "lab_ab": lab_yx, "suggest_ba": _inverse_label(lab_yx)})

    rate = (len(symmetric_ok) / considered) if considered else 0.0
    return {
        "considered": considered,
        "symmetric_ok": symmetric_ok,
        "conflicts": conflicts,
        "missing_reverse": missing_reverse,
        "coherence_rate": rate
    }


# This is really great
TOOLS = [coherence_check]