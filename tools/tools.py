# from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import List, Dict, Any

from langchain_core.tools import tool
from langsmith.run_helpers import traceable  # <- LangSmith tracing

DATASET_DEFAULT = Path(__file__).resolve().parents[1] / "data" / "sample.jsonl"


def _load_rows(dataset_path: str | Path = DATASET_DEFAULT) -> List[Dict[str, Any]]:
    p = Path(dataset_path)
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@traceable(name="tool:mean_price", run_type="tool")
def _mean_price_impl(dataset_path: str = str(DATASET_DEFAULT)) -> str:
    """Compute the mean price over all items in the JSONL dataset."""
    rows = _load_rows(dataset_path)
    prices = [float(r["price"]) for r in rows if "price" in r]
    return f"{mean(prices):.2f}"


@traceable(name="tool:find_by_keyword", run_type="tool")
def _find_by_keyword_impl(keyword: str, dataset_path: str = str(DATASET_DEFAULT)) -> str:
    """Return items whose title or tags contain the keyword (case-insensitive)."""
    rows = _load_rows(dataset_path)
    k = keyword.lower().strip()
    hits = [r for r in rows if k in r.get("title", "").lower() or any(k in t.lower() for t in r.get("tags", []))]
    return "[]" if not hits else json.dumps(hits, ensure_ascii=False)


# Wrap the traced functions as LangChain tools
mean_price = tool(_mean_price_impl)
find_by_keyword = tool(_find_by_keyword_impl)


# This is really great
TOOLS = [mean_price, find_by_keyword]