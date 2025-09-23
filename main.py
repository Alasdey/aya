from __future__ import annotations

import os
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional  # <- add these for type hints

import yaml
from langsmith.run_helpers import tracing_context  # <- LangSmith top-level run context

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langchain_openai import ChatOpenAI

# Tools
from tools.tools import TOOLS
from prompts.meci import _meci_system_prompt, _meci_user_prompt
from utils.metrics import compute_multiclass_metrics, compute_binary_metrics

ROOT = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

# ---- LangSmith v2 tracing (env) ----
os.environ.setdefault("LANGSMITH_TRACING_V2", "true")
os.environ["LANGSMITH_PROJECT"] = CONFIG["langsmith_project"]

# NEW imports
import re
from collections import defaultdict

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

def predict_meci_hf(
    repo_id: str,
    split: str = "test",
    text_field: str = "text",
    ann_field: str = "annotations",
    max_examples: int = 0,
    pair_batch_size: int = 100,
    streaming: bool = False,
    ls_project: Optional[str] = None,
):
    from datasets import load_dataset
    ds = load_dataset(repo_id, split=split, streaming=streaming)
    it = ds if streaming else [ds[i] for i in range(len(ds))]

    labels_all_true: List[str] = []
    labels_all_pred: List[str] = []
    skipped = 0

    with tracing_context(name="meci_predict_hf", metadata={"repo_id": repo_id, "split": split}, tags=["eval","meci","predict"]):
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

            for batch in chunked(gold_triples, pair_batch_size):
                sys_msg = SystemMessage(_meci_system_prompt())
                user_msg = HumanMessage(_meci_user_prompt(doc_text, batch, spans))

                thread_id = f"meci::predict::{split}::{idx}"
                cfg = {"configurable": {"thread_id": thread_id}, "tags": ["meci","predict", split]}

                with tracing_context(name="doc_predict", metadata={"doc_idx": idx, "num_pairs": len(batch)}, tags=["meci","doc","predict"]):
                    final = graph.invoke({"messages": [sys_msg, user_msg]}, config=cfg)

                arr = _extract_last_json_array(final["messages"])
                pred_map = _pred_map_from_json(arr, batch)

                for (Ti, gold_label, Tj) in batch:
                    labels_all_true.append(gold_label)
                    labels_all_pred.append(pred_map.get((Ti,Tj), "NoRel"))

    labels = ["CauseEffect","EffectCause","NoRel"]
    mc = compute_multiclass_metrics(labels_all_true, labels_all_pred, labels)
    binm = compute_binary_metrics(labels_all_true, labels_all_pred)

    report = {
        "per_label": mc["per_label"],
        "macro_f1": mc["macro_f1"],
        "micro_precision": mc["micro_precision"],
        "micro_recall": mc["micro_recall"],
        "micro_f1": mc["micro_f1"],
        "total_pairs": mc["total"],
        "binary": binm,
        "skipped_docs": skipped,
    }
    return report

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

def call_model(state: MessagesState, *, config=None):
    messages: List[AnyMessage] = state["messages"]
    ai = llm_with_tools.invoke(messages, config=config)
    return {"messages": [ai]}

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(TOOLS).with_config({"run_name": "tool_node"}))
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", should_continue, ["tools", END])
builder.add_edge("tools", "call_model")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# ---- Demo helper (now returns config-based system prompt) ----
def _system_prompt() -> str:
    return CONFIG["prompts"]["system"]["base"]

def _sample_prompt() -> str:
    return "Use the coherence check tool with a made up sample to test it"

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

    args = parser.parse_args()

    res = predict_meci_hf(
        repo_id=args.repo,
        split=args.split,
        text_field=args.text_field,
        ann_field=args.ann_field,
        max_examples=args.max_examples,
        pair_batch_size=args.pair_batch_size,
        streaming=args.streaming,
        ls_project=args.ls_project,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    sys.exit(0)
