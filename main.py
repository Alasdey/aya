from __future__ import annotations

import os
import argparse
import json
import sys
import re
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional  # <- add these for type hints

from datasets import load_dataset
import yaml
from langsmith.run_helpers import tracing_context  # <- LangSmith top-level run context

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langchain_openai import ChatOpenAI

import asyncio
import uuid
from collections import defaultdict  # NEW: For per_lang defaultdict

# Tools
from tools.tools import TOOLS
from prompts.meci import (
    _meci_system_prompt,
    _meci_user_template_prefix,
    _pair_line,
)
from utils.metrics import compute_multiclass_metrics, compute_binary_metrics
from utils.logger import capture_git_state, log_run

print("Imports done")

ROOT = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

# ---- LangSmith v2 tracing (env) ----
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_PROJECT"] = CONFIG["langsmith_project"]

print("Config loaded")

if not os.environ.get("LANGSMITH_API_KEY"):
    raise RuntimeError("LANGSMITH_API_KEY not set.")

if not os.environ.get("OPENROUTER_API_KEY"):
    raise RuntimeError("OPENROUTER_API_KEY not set. See https://openrouter.ai/docs/quickstart")

# ---- MECI parsing helpers (inline span format) ----
PAIR_LABELS = {"CauseEffect", "EffectCause", "NoRel"}
TAG_RE = re.compile(r"<T(\d+)\s+([^>]+)>")

ANN_PAIR_RE = re.compile(
    r"<T(\d+)\s+[^>]+>\s+(CauseEffect|EffectCause|NoRel)\s+<T(\d+)\s+[^>]+>",
    re.IGNORECASE
)

def extract_event_spans(doc_text: str) -> Dict[str, str]:
    spans = {}
    for m in TAG_RE.finditer(doc_text or ""):
        idx = m.group(1)
        token_text = m.group(2).strip()
        spans[f"T{idx}"] = token_text
    return spans

def parse_annotations(ann_text: str) -> List[tuple[str, str, str]]:
    triples: List[tuple[str, str, str]] = []
    for m in ANN_PAIR_RE.finditer(ann_text or ""):
        i, label, j = m.groups()
        norm = {"causeeffect": "CauseEffect", "effectcause": "EffectCause", "norel": "NoRel"}.get(label.lower(), label)
        if norm in PAIR_LABELS:
            triples.append((f"T{i}", norm, f"T{j}"))
    return triples

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _extract_last_json_array(messages: List[AnyMessage]) -> List[Dict[str, Any]]:
    arr_text = None
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            raw = (m.content or "").strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = next((p for p in parts if "[" in p and "]" in p), raw)
            l = raw.find("[")
            r = raw.rfind("]")
            if l != -1 and r != -1 and r > l:
                arr_text = raw[l:r+1]
                break
    if not arr_text:
        return []
    try:
        return json.loads(arr_text)
    except Exception:
        return []

def _pred_map_from_json(arr: List[Dict[str, Any]], requested_pairs: List[tuple[str,str,str]]) -> Dict[tuple[str,str], str]:
    pred_map: Dict[tuple[str,str], str] = {}
    for obj in arr or []:
        p = (obj.get("pair") or "").strip()
        lab = (obj.get("label") or "").strip()
        lab = {"causeeffect":"CauseEffect","effectcause":"EffectCause","norel":"NoRel"}.get(lab.lower(), lab)
        if lab in PAIR_LABELS and "," in p:
            a,b = [t.strip() for t in p.split(",",1)]
            pred_map[(a,b)] = lab
    for (Ti, _gold, Tj) in requested_pairs:
        pred_map.setdefault((Ti,Tj), "NoRel")
    return pred_map


# ---- Retry utility functions ----
def _should_retry_exception(e: Exception) -> bool:
    """Determine if an exception is worth retrying."""
    # Retry on network errors, rate limits, timeouts, and general API errors
    error_msg = str(e).lower()
    retryable_patterns = [
        "timeout", "connection", "rate", "limit", "service unavailable",
        "internal server error", "bad gateway", "gateway timeout",
        "temporary", "transient", "overloaded", "throttled", 
        "JSONDecodeError",
    ]
    # return any(pattern in error_msg for pattern in retryable_patterns)
    return True

async def _async_retry_with_backoff(
    coro_func, 
    max_retries: int = 3, 
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    *args, 
    **kwargs
):
    """
    Retry an async function with exponential backoff.
    
    Args:
        coro_func: Async function to retry
        max_retries: Maximum number of retry attempts (default: 3)  
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        backoff_factor: Exponential backoff multiplier (default: 2.0)
        jitter: Whether to add random jitter to delays (default: True)
        *args, **kwargs: Arguments to pass to coro_func
    
    Returns:
        Result of coro_func if successful
        
    Raises:
        Last exception encountered if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):  # +1 to include initial attempt
        try:
            return await coro_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if attempt == max_retries:
                # Final attempt failed, re-raise
                print(f"[ERROR] Final retry attempt {attempt + 1}/{max_retries + 1} failed: {e}", file=sys.stderr)
                raise e
            
            # Check if this exception is worth retrying
            if not _should_retry_exception(e):
                print(f"[WARN] Non-retryable error on attempt {attempt + 1}: {e}", file=sys.stderr)
                raise e
                
            # Calculate delay with exponential backoff
            delay = min(base_delay * (backoff_factor ** attempt), max_delay)
            
            # Add jitter to prevent thundering herd
            if jitter:
                delay = delay * (0.5 + random.random() * 0.5)  # 50%-100% of delay
                
            print(f"[RETRY] Attempt {attempt + 1}/{max_retries + 1} failed: {e}. Retrying in {delay:.2f}s", file=sys.stderr)
            await asyncio.sleep(delay)
    
    # Should never reach here, but just in case
    raise last_exception

# ---- Model via OpenRouter (OpenAI-compatible) ----
llm = ChatOpenAI(
    model=CONFIG["model"],
    temperature=CONFIG.get("temperature", 0),
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url=CONFIG.get("openrouter_base_url", "https://openrouter.ai/api/v1"),
)

llm_with_tools = llm.bind_tools(TOOLS).with_config({"run_name": "chat_model+tools"})

def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


# --- NEW: cache-friendly, independent bootstrap (no tools) with retry ---
async def cached_pairwise_bootstrap(state: MessagesState, *, config=None):
    """
    Run pairwise predictions by directly calling the base LLM with a stable prefix
    (rules + doc text + 'Pairs to classify:' header). Each request only appends one
    trailing pair line, maximizing prefix/prompt cache reuse across the batch.
    Now includes retry logic for individual LLM calls.
    """
    cfg = (config or {}).get("configurable", {}) or {}
    doc_text: str = cfg.get("doc_text", "") or ""
    spans: Dict[str, str] = cfg.get("spans", {}) or {}
    triples: List[tuple[str, str, str]] = cfg.get("pairs", []) or []
    max_workers: int = int(cfg.get("bootstrap_concurrency", 16))

    # Stable, long prefix for prompt caching
    sys_seed = SystemMessage(_meci_system_prompt())
    cached_prefix = _meci_user_template_prefix(doc_text)

    sem = asyncio.Semaphore(max_workers)

    async def _classify_one_with_retry(Ti: str, Tj: str):
        async def _classify_one_attempt():
            pair_line = _pair_line(Ti, Tj, spans)
            user_msg = HumanMessage(cached_prefix + pair_line)
            async with sem:
                ai = await llm.ainvoke([sys_seed, user_msg], config=config)
                raw = (ai.content or "").strip()
                label = "NoRel"
                pair_key = f"{Ti},{Tj}"
                try:
                    if raw.startswith("```"):
                        parts = raw.split("```")
                        raw = next((p for p in parts if "[" in p and "]" in p), raw)
                    if "[" in raw and "]" in raw:
                        arr = json.loads(raw[raw.find("["): raw.rfind("]")+1])
                        if isinstance(arr, list) and arr:
                            obj = arr[0]
                            if isinstance(obj, dict) and (obj.get("pair") or "").strip() == pair_key:
                                label = (obj.get("label") or "").strip()
                    else:
                        obj = json.loads(raw)
                        if isinstance(obj, dict) and (obj.get("pair") or "").strip() == pair_key:
                            label = (obj.get("label") or "").strip()
                except Exception:
                    import re as _re
                    m = _re.search(r'"label"\s*:\s*"([^"]+)"', raw)
                    label = (m.group(1) if m else "NoRel").strip()

                labmap = {
                    "causeeffect": "CauseEffect",
                    "effectcause": "EffectCause", 
                    "norel": "NoRel",
                }
                norm = labmap.get(label.lower(), label)
                if norm not in {"CauseEffect", "EffectCause", "NoRel"}:
                    norm = "NoRel"
                return {"pair": pair_key, "label": norm}
        
        try:
            return await _async_retry_with_backoff(
                _classify_one_attempt,
                max_retries=3,
                base_delay=1.0,
                max_delay=30.0
            )
        except Exception as e:
            print(f"[ERROR] All retries failed for pair {Ti},{Tj}: {e}", file=sys.stderr)
            return {"pair": f"{Ti},{Tj}", "label": "NoRel", "error": str(e)}

    tasks = [_classify_one_with_retry(Ti, Tj) for (Ti, _gold, Tj) in triples]
    preds = await asyncio.gather(*tasks)

    seed_json = json.dumps(preds, ensure_ascii=False, indent=2)
    seed_msg = SystemMessage(
        "Initial pairwise predictions (cache-friendly seed; treat as fallible hints):\n"
        f"```json\n{seed_json}\n```"
    )
    return {"messages": [seed_msg]}


def call_model(state: MessagesState, *, config=None):
    messages: List[AnyMessage] = state["messages"]
    ai = llm_with_tools.invoke(messages, config=config)
    return {"messages": [ai]}

print("Graph building")

builder = StateGraph(MessagesState)
builder.add_node("cached_pairwise_bootstrap", cached_pairwise_bootstrap)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(TOOLS).with_config({"run_name": "tool_node"}))
if CONFIG["inference"]["initial_prediction_mode"] == "pairwise":
    print("Pairwise initial prediction")
    builder.add_edge(START, "cached_pairwise_bootstrap")
else:
    builder.add_edge(START, "call_model")
builder.add_edge("cached_pairwise_bootstrap", "call_model")
builder.add_conditional_edges("call_model", should_continue, ["tools", END])
builder.add_edge("tools", "call_model")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

print("Graph built")

async def predict_meci_hf_async(
    repo_id: str,
    split: str = "test",
    text_field: str = "text",
    ann_field: str = "annotations",
    max_examples: int = 0,
    pair_batch_size: int = 100,
    streaming: bool = False,
    ls_project: Optional[str] = None,
    max_concurrency: int = 8,
):
    """
    Async version: schedules **each batch** for a document as a separate task and runs them concurrently.
    Uses graph.ainvoke under the hood. Now includes retry logic for batch processing.
    """
    ds = load_dataset(repo_id, split=split, streaming=streaming)
    it = ds if streaming else [ds[i] for i in range(len(ds))]

    labels_all_true: List[str] = []
    labels_all_pred: List[str] = []
    skipped = 0

    sem = asyncio.Semaphore(max_concurrency)
    tasks: List[asyncio.Task] = []
    per_doc: Dict[int, Dict[str, List[str]]] = {}  # {doc_idx: {"y_true": [...], "y_pred": [...]}}
    per_lang = defaultdict(lambda: {"y_true": [], "y_pred": []})  # NEW: Collect per language

    async def run_batch_with_retry(doc_idx: int, batch_idx: int, doc_text: str,
                                  spans: Dict[str, str], batch: List[tuple[str, str, str]], lang: str):  # NEW: Added lang param
        async def run_batch_attempt():
            async with sem:
                sys_msg = SystemMessage(_meci_system_prompt())

                # Cache-friendly user text: stable prefix + trailing pair lines
                prefix = _meci_user_template_prefix(doc_text)
                pair_lines = "\n".join(_pair_line(Ti, Tj, spans) for (Ti, _gold, Tj) in batch)
                user_msg = HumanMessage(prefix + pair_lines)

                thread_id = f"meci::predict::{split}::{doc_idx}::b{batch_idx}::{uuid.uuid4().hex[:8]}"
                cfg = {
                    "configurable": {
                        "thread_id": thread_id,
                        "doc_text": doc_text,
                        "spans": spans,
                        "pairs": batch,
                        "bootstrap_concurrency": max_concurrency,
                    },
                    "recursion_limit": 100,
                    "tags": ["meci", "predict", split]
                }

                with tracing_context(
                    name="doc_predict_async",
                    metadata={"doc_idx": doc_idx, "batch_idx": batch_idx, "num_pairs": len(batch)},
                    tags=["meci", "doc", "predict", "async"],
                ):
                    final = await graph.ainvoke(
                        {"messages": [sys_msg, user_msg]}, 
                        config=cfg,
                    )
                    arr = _extract_last_json_array(final["messages"])

                pred_map = _pred_map_from_json(arr, batch)
                y_true = [gold for (_Ti, gold, _Tj) in batch]
                y_pred = [pred_map.get((Ti, Tj), "NoRel") for (Ti, _gold, Tj) in batch]
                return doc_idx, lang, y_true, y_pred  # NEW: Return lang too
        
        try:
            return await _async_retry_with_backoff(
                run_batch_attempt,
                max_retries=3,
                base_delay=2.0,
                max_delay=60.0
            )
        except Exception as e:
            print(f"[ERROR] All retries failed for doc {doc_idx} batch {batch_idx}: {e}", file=sys.stderr)
            # Return fallback predictions (all NoRel) to keep the evaluation running
            y_true = [gold for (_Ti, gold, _Tj) in batch]
            y_pred = ["NoRel" for _ in batch]
            return doc_idx, lang, y_true, y_pred  # NEW: Return lang too

    with tracing_context(name="meci_predict_hf_async", metadata={"repo_id": repo_id, "split": split}, tags=["eval","meci","predict","async"]):
        for idx, row in enumerate(it):
            if max_examples and idx >= max_examples:
                break
            doc_text = row.get(text_field, "")
            ann_text = row.get(ann_field, "")
            gold_triples = parse_annotations(ann_text)
            if not gold_triples:
                skipped += 1
                continue
            spans = extract_event_spans(doc_text)
            lang = row.get("lang", "unknown")  # NEW: Extract lang from row

            print(f"Processing document {idx} ({lang}) with {len(gold_triples)} pairs", end="\r")
            for b_idx, batch in enumerate(chunked(gold_triples, pair_batch_size)):
                tasks.append(asyncio.create_task(run_batch_with_retry(idx, b_idx, doc_text, spans, batch, lang)))  # NEW: Pass lang

        print("Gathering async results...")
        # Gather results as they complete
        completed_batches = 0
        total_batches = len(tasks)
        
        for coro in asyncio.as_completed(tasks):
            doc_idx, lang, y_true, y_pred = await coro  # NEW: Unpack lang
            completed_batches += 1
            print(f"Completed batch {completed_batches}/{total_batches}", end="\r")
            
            labels_all_true.extend(y_true)
            labels_all_pred.extend(y_pred)
            d = per_doc.setdefault(doc_idx, {"y_true": [], "y_pred": []})
            d["y_true"].extend(y_true)
            d["y_pred"].extend(y_pred)

            # NEW: Append to per_lang
            l = per_lang[lang]
            l["y_true"].extend(y_true)
            l["y_pred"].extend(y_pred)

    print("Computing metrics...")
    labels = ["CauseEffect", "EffectCause", "NoRel"]
    mc = compute_multiclass_metrics(labels_all_true, labels_all_pred, labels)
    binm = compute_binary_metrics(labels_all_true, labels_all_pred)

    print("Computing per-document metrics...")
    # ---- Per-document metrics (compute + log into trace metadata) ----
    per_doc_metrics: List[Dict[str, Any]] = []
    for doc_idx, d in sorted(per_doc.items(), key=lambda x: x[0]):
        yt, yp = d["y_true"], d["y_pred"]
        mc_d = compute_multiclass_metrics(yt, yp, labels)
        binm_d = compute_binary_metrics(yt, yp)
        doc_report = {
            "doc_idx": doc_idx,
            "pairs": len(yt),
            "macro_f1": mc_d["macro_f1"],
            "micro_precision": mc_d["micro_precision"],
            "micro_recall": mc_d["micro_recall"],
            "micro_f1": mc_d["micro_f1"],
            "per_label": mc_d["per_label"],
            "binary": binm_d,
        }
        per_doc_metrics.append(doc_report)
        # Emit a child trace with metrics in metadata so it's easily inspectable in LangSmith UI
        with tracing_context(
            name="doc_metrics",
            metadata=doc_report,
            tags=["meci", "doc", "metrics", split],
        ):
            # no-op body; metadata is what we want on the trace
            pass

    # NEW: Compute per-language metrics
    print("Computing per-language metrics...")
    per_lang_metrics: Dict[str, Dict[str, Any]] = {}
    for lg, ld in per_lang.items():
        if not ld["y_true"]:
            continue
        mc_l = compute_multiclass_metrics(ld["y_true"], ld["y_pred"], labels)
        binm_l = compute_binary_metrics(ld["y_true"], ld["y_pred"])
        per_lang_metrics[lg] = {
            "multiclass": mc_l,
            "binary": binm_l,
            "total_pairs": len(ld["y_true"]),
        }

    print("Generating report...")
    report = {
        "per_label": mc["per_label"],
        "macro_f1": mc["macro_f1"],
        "micro_precision": mc["micro_precision"],
        "micro_recall": mc["micro_recall"],
        "micro_f1": mc["micro_f1"],
        "total_pairs": mc["total"],
        "binary": binm,
        "skipped_docs": skipped,
        "per_doc_metrics": per_doc_metrics,
        "per_lang_metrics": per_lang_metrics,  # NEW: Add per-language metrics to report
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True, help="Hugging Face dataset name or path (e.g., your_org/meci_dataset)") 
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--ann_field", type=str, default="annots")
    parser.add_argument("--pair_batch_size", type=int, default=100)
    parser.add_argument("--max_examples", type=int, default=0, help="0 = all")
    parser.add_argument("--streaming", type=bool, default=False, help="Streaming")
    parser.add_argument("--ls_project", type=str, required=False, help="Langsmith project name")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent batch calls") 
    parser.add_argument("--logdir", type=str, default="logs", help="Directory to write execution logs")

    args = parser.parse_args()

    git = capture_git_state(ROOT)

    print("Launching async runs with retry logic...")

    res = asyncio.run(
        predict_meci_hf_async(
            repo_id=args.repo,
            split=args.split,
            text_field=args.text_field,
            ann_field=args.ann_field,
            max_examples=args.max_examples,
            pair_batch_size=args.pair_batch_size,
            streaming=args.streaming,
            ls_project=args.ls_project,
            max_concurrency=args.concurrency,
        )
    )

    # Logging
    artifacts = log_run(
        logdir=Path(args.logdir),
        config=CONFIG,
        args=vars(args),
        results=res,
        git_state=git,
    )

    res.pop("per_doc_metrics")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    sys.exit(0)
