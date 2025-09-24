from __future__ import annotations

import os
import json
import yaml
import re
from pathlib import Path
from statistics import mean
from typing import List, Dict, Any, Tuple, Optional

from langchain_core.tools import tool
from langsmith.run_helpers import traceable  # <- LangSmith tracing

# New imports needed for role tools
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

CAUSAL = {"CauseEffect", "EffectCause"}

def _inverse_label(lab: str) -> str:
    return {"CauseEffect": "EffectCause", "EffectCause": "CauseEffect"}.get(lab, lab)

@tool
def coherence_check(
    *,
    pairs: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Check directional coherence among the *predicted* pairs only.

    - Works with partial predictions (you can pass just a subset of all possible pairs).
    - Missing directions are interpreted as NoRel (i.e., they are NOT flagged unless the
      present direction is causal).
    - Does not require knowledge of the original universe of pairs.

    Args:
        pairs: [{"pair":"Ti,Tj","label":"CauseEffect|EffectCause|NoRel"}, ...]

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
        return lab.strip()

    dir_map: Dict[Tuple[str, str], str] = {}
    for obj in pairs or []:
        p = (obj.get("pair") or "").strip()
        lab = _norm_label(obj.get("label", ""))
        if not p or "," not in p or lab not in ("CauseEffect", "EffectCause", "NoRel"):
            continue
        a, b = [t.strip() for t in p.split(",", 1)]
        if not a or not b:
            continue
        dir_map[(a, b)] = lab

    unordered = set()
    for (a, b) in dir_map:
        if a != b:
            unordered.add(tuple(sorted((a, b))))

    considered = 0
    symmetric_ok: List[List[str]] = []
    conflicts: List[Dict[str, str]] = []
    missing_reverse: List[Dict[str, str]] = []

    suggested_additions: List[Dict[str, str]] = []
    suggested_updates: List[Dict[str, str]] = []
    assumed_norel: List[List[str]] = []

    for (x, y) in unordered:
        lab_xy = dir_map.get((x, y))
        lab_yx = dir_map.get((y, x))

        cx = lab_xy in CAUSAL if lab_xy else False
        cy = lab_yx in CAUSAL if lab_yx else False

        if cx or cy:
            considered += 1

            if lab_xy and lab_yx:
                if (lab_xy, lab_yx) in (("CauseEffect", "EffectCause"), ("EffectCause", "CauseEffect")):
                    symmetric_ok.append([x, y])
                else:
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
                if lab_xy in CAUSAL and lab_yx is None:
                    inv = _inverse_label(lab_xy)
                    missing_reverse.append({"a": x, "b": y, "lab_ab": lab_xy, "suggest_ba": inv})
                    suggested_additions.append({"pair": f"{y},{x}", "label": inv})
                if lab_yx in CAUSAL and lab_xy is None:
                    inv = _inverse_label(lab_yx)
                    missing_reverse.append({"a": y, "b": x, "lab_ab": lab_yx, "suggest_ba": inv})
                    suggested_additions.append({"pair": f"{x},{y}", "label": inv})
        else:
            if lab_xy == "NoRel" and lab_yx is None:
                assumed_norel.append([y, x])
                suggested_additions.append({"pair": f"{y},{x}", "label": "NoRel"})
            if lab_yx == "NoRel" and lab_xy is None:
                assumed_norel.append([x, y])
                suggested_additions.append({"pair": f"{x},{y}", "label": "NoRel"})

    rate = (len(symmetric_ok) / considered) if considered else 0.0

    return {
        "considered": considered,
        "symmetric_ok": symmetric_ok,
        "conflicts": conflicts,
        "missing_reverse": missing_reverse,
        "coherence_rate": rate,
        "suggested_additions": suggested_additions,
        "suggested_updates": suggested_updates,
        "assumed_norel": assumed_norel,
    }

def _norm_pair_str(s: str) -> tuple[str, str]:
    if not s or "," not in s:
        raise ValueError("pair must be like 'Ti,Tj'")
    a, b = [p.strip() for p in s.split(",", 1)]
    return a, b

def _norm_label(lab: str) -> str:
    if not lab:
        return "NoRel"
    m = lab.strip().lower()
    if m in ("causeeffect", "cause-effect", "causes", "cause->effect"):
        return "CauseEffect"
    if m in ("effectcause", "effect-cause", "causedby", "caused_by", "effect->cause"):
        return "EffectCause"
    if m in ("norel", "no-rel", "no_relation", "no-relation", "none"):
        return "NoRel"
    return lab.strip()

@tool
def counterfactual_pairs(
    *,
    pair: str,
    # Pass the full document text that was shown to the model (contains <T*> tags).
    # The tool will instruct the model to read only; no rewriting needed.
    context_text: str,
    # Optional: spans for convenience/debug; not required.
    span_i: Optional[str] = None,
    span_j: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Decide the MECI label for ONE ordered pair "Ti,Tj" by running two counterfactual tests.

    - Uses a but-for analysis in BOTH directions:
        • Could Tj still happen if Ti did NOT happen?
        • Could Ti still happen if Tj did NOT happen?
      Treat enablement and prevention as causal.

    Args:
      pair: "Ti,Tj" (ordered)
      context_text: the exact document text (read-only; do NOT rewrite)
      span_i, span_j: optional trigger strings for Ti/Tj (debugging only)
      model: optional model override

    Returns:
      {
        "pair": "Ti,Tj",
        "label": "CauseEffect|EffectCause|NoRel",
        "confidence": float in [0,1],
        "tests": {
          "i_without_j": "yes|no|unclear",
          "j_without_i": "yes|no|unclear"
        }
      }
    """
    Ti, Tj = _norm_pair_str(pair)
    llm = _make_llm(model=model, temperature=0.0)

    system = (
        "You are a careful MECI counterfactual annotator. Use ONLY the label set:\n"
        '- "CauseEffect" (Ti → Tj)\n'
        '- "EffectCause" (Tj → Ti)\n'
        '- "NoRel"\n\n'
        "Causality includes enablement and prevention. Reject mere temporal order or correlation. "
        "Do not paraphrase or rewrite the context; read it as-is and decide."
    )

    human = f"""
Context (read-only; do NOT rewrite):
{context_text}

Pair: {Ti},{Tj}
Event {Ti} span: {span_i or "[from context]"}
Event {Tj} span: {span_j or "[from context]"}

Task:
1) Run the two but-for tests using ONLY the context:
   - Test A (j_without_i): Could {Tj} still happen if {Ti} did NOT happen? Answer yes/no/unclear.
   - Test B (i_without_j): Could {Ti} still happen if {Tj} did NOT happen? Answer yes/no/unclear.
   Treat enablement and prevention as causal.

2) Choose exactly ONE label from {{CauseEffect, EffectCause, NoRel}}:
   - If {Ti} is necessary for {Tj} (A = no) and {Tj} is NOT necessary for {Ti} (B = yes): label = CauseEffect.
   - If {Tj} is necessary for {Ti} (B = no) and {Ti} is NOT necessary for {Tj} (A = yes): label = EffectCause.
   - If both could still happen (A = yes and B = yes): label = NoRel.
   - If neither could still happen (A = no and B = no): pick the better-supported DIRECTION from explicit cues
     (connectives like “because/so/therefore”, enable/prevent, polarity). If still tied, prefer the direction that
     is most explicitly stated; as a last resort, default to CauseEffect.

3) Output ONLY valid JSON (no prose), with this exact shape:
{{
  "pair": "{Ti},{Tj}",
  "label": "CauseEffect|EffectCause|NoRel",
  "confidence": 0.0,
  "tests": {{
    "i_without_j": "yes|no|unclear",
    "j_without_i": "yes|no|unclear"
  }}
}}
"""

    raw = _llm_content(llm, system, human)

    # Try to parse; be robust to minor formatting issues.
    data: Dict[str, Any]
    try:
        data = json.loads(raw)
    except Exception:
        # Fallback: extract a label token if JSON parsing fails
        m = re.search(r'"label"\s*:\s*"([^"]+)"', raw)
        lab = _norm_label(m.group(1) if m else "")
        data = {
            "pair": f"{Ti},{Tj}",
            "label": lab if lab in ("CauseEffect", "EffectCause", "NoRel") else "NoRel",
            "confidence": 0.0,
            "tests": {"i_without_j": "unclear", "j_without_i": "unclear"},
        }

    # Normalize label and pair, ensure required keys
    data["pair"] = f"{Ti},{Tj}"
    data["label"] = _norm_label(str(data.get("label", "")))
    if data["label"] not in ("CauseEffect", "EffectCause", "NoRel"):
        data["label"] = "NoRel"
    if "confidence" not in data or not isinstance(data["confidence"], (int, float)):
        data["confidence"] = 0.0
    if "tests" not in data or not isinstance(data["tests"], dict):
        data["tests"] = {"i_without_j": "unclear", "j_without_i": "unclear"}

    return data



# -------- Role tools now pull prompts from config --------

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

# System prompts pulled from config
_THINKER_SYS = CONFIG["prompts"]["roles"]["thinker"].strip()
_CRITIC_SYS = CONFIG["prompts"]["roles"]["critic"].strip()
_SUMMARIZER_SYS = CONFIG["prompts"]["roles"]["summarizer"].strip()
_WORKER_SYS = CONFIG["prompts"]["roles"]["worker"].strip()

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


# -------- Tool selection from config (NEW) --------

# Registry of available tool objects by name
TOOL_REGISTRY: Dict[str, Any] = {
    "coherence_check": coherence_check,
    "thinker": thinker,
    "critic": critic,
    "summarizer": summarizer,
    "worker": worker,
    "counterfactual_pairs": counterfactual_pairs,  # NEW
}

# Read enabled tool names from config (default to just coherence_check)
ENABLED_TOOL_NAMES: List[str] = CONFIG.get("tools", {}).get("enabled", ["coherence_check"])

# Validate configuration early
_unknown = [name for name in ENABLED_TOOL_NAMES if name not in TOOL_REGISTRY]
if _unknown:
    raise ValueError(f"Unknown tools in config.tools.enabled: {', '.join(_unknown)}")

# Export the selected tools in the order specified by config
TOOLS: List[Any] = [TOOL_REGISTRY[name] for name in ENABLED_TOOL_NAMES]