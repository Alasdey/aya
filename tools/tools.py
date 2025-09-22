from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import List, Dict, Any, Tuple, Optional

from langchain_core.tools import tool
from langsmith.run_helpers import traceable  # <- LangSmith tracing


# DATASET_DEFAULT = Path(__file__).resolve().parents[1] / "data" / "sample.jsonl"


# def _load_rows(dataset_path: str | Path = DATASET_DEFAULT) -> List[Dict[str, Any]]:
#     p = Path(dataset_path)
#     with p.open("r", encoding="utf-8") as f:
#         return [json.loads(line) for line in f]


# @tool
# def mean_price(dataset_path: str = str(DATASET_DEFAULT)) -> str:
#     """Compute the mean price over all items in the JSONL dataset."""
#     rows = _load_rows(dataset_path)
#     prices = [float(r["price"]) for r in rows if "price" in r]
#     return f"{mean(prices):.2f}"


# @tool
# def find_by_keyword(keyword: str, dataset_path: str = str(DATASET_DEFAULT)) -> str:
#     """Return items whose title or tags contain the keyword (case-insensitive)."""
#     rows = _load_rows(dataset_path)
#     k = keyword.lower().strip()
#     hits = [r for r in rows if k in r.get("title", "").lower() or any(k in t.lower() for t in r.get("tags", []))]
#     return "[]" if not hits else json.dumps(hits, ensure_ascii=False)


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
    Check directional coherence among the *predicted* pairs only.

    - Works with partial predictions (you can pass just a subset of all possible pairs).
    - Missing directions are interpreted as NoRel (i.e., they are NOT flagged unless the
      present direction is causal).
    - Does not require knowledge of the original universe of pairs.

    Args:
        pairs: [{"pair":"Ti,Tj","label":"CauseEffect|EffectCause|NoRel"}, ...]
        conversation: optional, ignored except for tracing context.

    Returns:
        {
          "considered": int,                      # number of unordered pairs with at least one causal direction
          "symmetric_ok": List[List[str]],        # [ [Ti, Tj], ... ] where causal directions are consistent inverses
          "conflicts": List[{...}],               # both directions present but inconsistent
          "missing_reverse": List[{...}],         # one causal direction present, reverse missing
          "coherence_rate": float,                # |symmetric_ok| / considered  (only over causal pairs)
          # New, backwards-compatible helpers:
          "suggested_additions": List[{...}],     # pairs to add for symmetry (incl. NoRel when appropriate)
          "suggested_updates": List[{...}],       # pairs to change to fix conflicts
          "assumed_norel": List[List[str]]        # reverse pairs assumed NoRel due to missing prediction
        }
    """
    CAUSAL = {"CauseEffect", "EffectCause"}

    def _inverse_label(lab: str) -> str:
        return {"CauseEffect": "EffectCause", "EffectCause": "CauseEffect"}.get(lab, lab)

    def _norm_label(lab: str) -> str:
        if not lab:
            return ""
        m = lab.strip().lower()
        if m in ("causeeffect", "cause-effect", "cause->effect", "causes"):
            return "CauseEffect"
        if m in ("effectcause", "effect-cause", "effect->cause", "causedby", "caused_by"):
            return "EffectCause"
        if m in ("norel", "no-rel", "no_relation", "no-relation", "none"):
            return "NoRel"
        # Fall back to original; if it's not one of the three, we drop it later.
        return lab.strip()

    # Build a directed map from the provided (possibly partial) predictions.
    dir_map: Dict[Tuple[str, str], str] = {}
    for obj in pairs or []:
        p = (obj.get("pair") or "").strip()
        lab = _norm_label(obj.get("label", ""))
        if not p or "," not in p or lab not in ("CauseEffect", "EffectCause", "NoRel"):
            continue
        a, b = [t.strip() for t in p.split(",", 1)]
        if not a or not b:
            continue
        # last-one-wins if duplicates; that’s fine for a tool check
        dir_map[(a, b)] = lab

    # Unordered set of entity pairs inferred *only* from what we were given
    unordered = set()
    for (a, b) in dir_map:
        if a != b:
            unordered.add(tuple(sorted((a, b))))

    considered = 0
    symmetric_ok: List[List[str]] = []
    conflicts: List[Dict[str, str]] = []
    missing_reverse: List[Dict[str, str]] = []

    # New helper outputs to make auto-fixing easy
    suggested_additions: List[Dict[str, str]] = []  # {"pair":"Tj,Ti","label":"..."}
    suggested_updates: List[Dict[str, str]] = []    # {"pair":"Tj,Ti","from":"...","to":"..."}
    assumed_norel: List[List[str]] = []             # [ [Ti, Tj], ... ]  (means "Tj,Ti" assumed NoRel)

    for (x, y) in unordered:
        lab_xy = dir_map.get((x, y))
        lab_yx = dir_map.get((y, x))

        # If neither direction is causal, we do not include it in "considered"
        cx = lab_xy in CAUSAL if lab_xy else False
        cy = lab_yx in CAUSAL if lab_yx else False

        if cx or cy:
            considered += 1

            if lab_xy and lab_yx:
                # Both directions present
                if (lab_xy, lab_yx) in (("CauseEffect", "EffectCause"), ("EffectCause", "CauseEffect")):
                    symmetric_ok.append([x, y])
                else:
                    # Incoherent: suggest making (y,x) the inverse of (x,y) when (x,y) is causal,
                    # otherwise invert the other direction.
                    if lab_xy in CAUSAL:
                        suggest_to = _inverse_label(lab_xy)
                        conflicts.append({
                            "a": x, "b": y, "lab_ab": lab_xy, "lab_ba": lab_yx, "suggest_ba": suggest_to
                        })
                        suggested_updates.append({
                            "pair": f"{y},{x}", "from": lab_yx, "to": suggest_to
                        })
                    else:
                        suggest_to = _inverse_label(lab_yx)
                        conflicts.append({
                            "a": y, "b": x, "lab_ab": lab_yx, "lab_ba": lab_xy, "suggest_ba": suggest_to
                        })
                        suggested_updates.append({
                            "pair": f"{x},{y}", "from": lab_xy, "to": suggest_to
                        })
            else:
                # Exactly one causal direction present -> missing reverse
                if lab_xy in CAUSAL and lab_yx is None:
                    inv = _inverse_label(lab_xy)
                    missing_reverse.append({"a": x, "b": y, "lab_ab": lab_xy, "suggest_ba": inv})
                    suggested_additions.append({"pair": f"{y},{x}", "label": inv})
                if lab_yx in CAUSAL and lab_xy is None:
                    inv = _inverse_label(lab_yx)
                    missing_reverse.append({"a": y, "b": x, "lab_ab": lab_yx, "suggest_ba": inv})
                    suggested_additions.append({"pair": f"{x},{y}", "label": inv})
        else:
            # No causal labels in either direction.
            # If exactly one side is explicitly NoRel and the reverse is missing,
            # treat the missing reverse as NoRel (do not flag it).
            if lab_xy == "NoRel" and lab_yx is None:
                assumed_norel.append([y, x])  # meaning "(y,x) assumed NoRel"
                suggested_additions.append({"pair": f"{y},{x}", "label": "NoRel"})
            if lab_yx == "NoRel" and lab_xy is None:
                assumed_norel.append([x, y])  # meaning "(x,y) assumed NoRel"
                suggested_additions.append({"pair": f"{x},{y}", "label": "NoRel"})

    rate = (len(symmetric_ok) / considered) if considered else 0.0

    return {
        "considered": considered,
        "symmetric_ok": symmetric_ok,
        "conflicts": conflicts,
        "missing_reverse": missing_reverse,
        "coherence_rate": rate,
        # helpful extras (won't break existing callers)
        "suggested_additions": suggested_additions,
        "suggested_updates": suggested_updates,
        "assumed_norel": assumed_norel,
    }

# Multi agents tools

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

def _make_llm(model: Optional[str] = None, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or CONFIG.get("model", "openai/gpt-5-mini"),
        temperature=temperature,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url=CONFIG.get("openrouter_base_url", "https://openrouter.ai/api/v1"),
    )

def _llm_content(llm: ChatOpenAI, system: str, prompt: str) -> str:
    resp = llm.invoke([SystemMessage(system), HumanMessage(prompt)])
    return (resp.content or "").strip()

def _format_artifacts(prior_artifacts: Optional[List[Dict[str, Any]]]) -> str:
    """
    Compact, role-tagged context to pass between tools.
    Expect items like {"stage": "thinker|critic|summarizer|worker", "content": "..."}.
    """
    if not prior_artifacts:
        return "[none]"
    lines = []
    for i, art in enumerate(prior_artifacts):
        role = art.get("stage", f"stage{i}")
        content = (art.get("content") or "").strip()
        if len(content) > 2000:
            content = content[:2000] + "... [truncated]"
        lines.append(f"[{i}:{role}]\n{content}")
    return "\n\n".join(lines)

def _last_of(prior_artifacts: Optional[List[Dict[str, Any]]], stage: str) -> Optional[str]:
    if not prior_artifacts:
        return None
    for art in reversed(prior_artifacts):
        if art.get("stage") == stage and art.get("content"):
            return str(art["content"])
    return None

# ---- System prompts for each role ----

_THINKER_SYS = (
    "You are Thinker. Do deep, structured reasoning.\n"
    "Produce a concise plan with numbered steps, explicit assumptions, edge cases, and data to verify.\n"
    "Do NOT write the final deliverable."
)

_CRITIC_SYS = (
    "You are Critic. Be constructive and precise.\n"
    "List concrete issues, missing information, incoherences, and constraints.\n"
    "Provide actionable fixes as bullet points; do NOT rewrite the whole prediction."
)

_SUMMARIZER_SYS = (
    "You are Summarizer. Write a crisp summary for a busy reader. Keep the <> tags\n"
    "Keep it faithful and concrete, outlining the causal relations between mentions"
)

_WORKER_SYS = (
    "You are Worker. Produce the final deliverable for the user task.\n"
    "Follow any plans and honor constraints from prior artifacts.\n"
    "Return a JSON array like:\n[\n  {\"pair\":\"T0,T1\",\"label\":\"CauseEffect\"}\n]\n"
)

# ---- Tools ----

@tool
def thinker(
    *,
    task: str,
    input_text: Optional[str] = None,
    prior_artifacts: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    THINKER tool — drafts a structured plan.

    Args:
        task: What needs to be achieved.
        input_text: Optional source/context to consider.
        prior_artifacts: Optional list of prior role outputs to reference.
        model: Optional model override.

    Returns: {"stage":"thinker","content":str,"meta":{"model":...}}
    """
    llm = _make_llm(model=model, temperature=0.0)
    prompt = (
        f"Task:\n{task}\n\n"
        f"Context text (optional):\n{input_text or '[none]'}\n\n"
        f"Prior artifacts (optional):\n{_format_artifacts(prior_artifacts)}\n\n"
        "Output: Numbered plan, assumptions, edge cases, and data to verify."
    )
    out = _llm_content(llm, _THINKER_SYS, prompt)
    return {"stage": "thinker", "content": out, "meta": {"model": model or CONFIG.get("model")}}

@tool
def critic(
    *,
    task: str,
    plan: Optional[str] = None,
    input_text: Optional[str] = None,
    prior_artifacts: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    CRITIC tool — critiques a plan and surfaces constraints/risks.

    Args:
        task: The user goal.
        plan: Optional explicit plan to critique; if omitted, will try the last THINKER artifact.
        input_text: Optional context.
        prior_artifacts: Optional list of prior role outputs.
        model: Optional model override.

    Returns: {"stage":"critic","content":str,"meta":{...}}
    """
    llm = _make_llm(model=model, temperature=0.0)
    plan_to_critique = plan or _last_of(prior_artifacts, "thinker") or "[no plan provided]"
    prompt = (
        f"Task:\n{task}\n\n"
        f"Context text (optional):\n{input_text or '[none]'}\n\n"
        f"Plan to critique:\n{plan_to_critique}\n\n"
        f"Prior artifacts (optional):\n{_format_artifacts(prior_artifacts)}\n\n"
        "Output: Bullet list of issues/risks/missing info, plus actionable fixes and constraints."
    )
    out = _llm_content(llm, _CRITIC_SYS, prompt)
    return {"stage": "critic", "content": out, "meta": {"model": model or CONFIG.get("model")}}

@tool
def summarizer(
    *,
    task: str,
    input_text: Optional[str] = None,
    focus: Optional[str] = None,
    prior_artifacts: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    SUMMARIZER tool — summarizes either input_text (preferred) or the most relevant prior artifact.

    Args:
        task: The user goal (used for context).
        input_text: If provided, summarize this text.
        focus: Optional explicit text to summarize when input_text is absent.
        prior_artifacts: Optional list of prior role outputs to pick from.
        model: Optional model override.

    Returns: {"stage":"summarizer","content":str,"meta":{...}}
    """
    llm = _make_llm(model=model, temperature=0.0)
    source = input_text or focus or _last_of(prior_artifacts, "thinker") or _last_of(prior_artifacts, "worker") \
             or _last_of(prior_artifacts, "critic") or "[nothing to summarize]"
    prompt = (
        f"Task context:\n{task}\n\n"
        f"Source to summarize:\n{source}\n\n"
        "Output: One-line TL;DR + 3–7 bullets."
    )
    out = _llm_content(llm, _SUMMARIZER_SYS, prompt)
    return {"stage": "summarizer", "content": out, "meta": {"model": model or CONFIG.get("model")}}

@tool
def worker(
    *,
    task: str,
    input_text: Optional[str] = None,
    plan: Optional[str] = None,
    constraints: Optional[str] = None,
    prior_artifacts: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    WORKER tool — produces the final deliverable.

    Args:
        task: The user goal (what to produce).
        input_text: Optional source text to use.
        plan: Optional plan to follow; if omitted, will try the last THINKER artifact.
        constraints: Optional constraints to honor; if omitted, will try the last CRITIC artifact.
        prior_artifacts: Optional list of prior role outputs to reference.
        model: Optional model override.

    Returns: {"stage":"worker","content":str,"meta":{...}}
    """
    llm = _make_llm(model=model, temperature=0.0)
    plan_final = plan or _last_of(prior_artifacts, "thinker") or "[no plan provided]"
    constraints_final = constraints or _last_of(prior_artifacts, "critic") or "[no explicit constraints]"
    prompt = (
        f"Task:\n{task}\n\n"
        f"Plan:\n{plan_final}\n\n"
        f"Constraints:\n{constraints_final}\n\n"
        f"Context text :\n{input_text or '[none]'}\n\n"
        f"Prior artifacts (optional):\n{_format_artifacts(prior_artifacts)}\n\n"
        "Output: The final deliverable."
    )
    out = _llm_content(llm, _WORKER_SYS, prompt)
    return {"stage": "worker", "content": out, "meta": {"model": model or CONFIG.get("model")}}

# Export all tools
TOOLS = [coherence_check, thinker, critic, summarizer, worker]