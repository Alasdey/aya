from __future__ import annotations
from typing import List, Dict, Any

# ---- MECI prompts ----
def _meci_system_prompt() -> str:
    base = _system_prompt().strip()
    addon = (
        "\n\nYou are an expert MECI annotator. "
        "Draft labels for each requested pair, then you MUST call the `coherence_check` tool with those draft labels. "
        "Use the tool's output to fix conflicts and ensure symmetric reverses for causal pairs. "
    )
    return base + addon


def _meci_system_prompt() -> str:
    # Base + strict tool use
    base = _system_prompt().strip()
    addon = (
        "\n\nYou are an expert MECI annotator. "
        "Draft labels for each requested pair, then you should use call the tools"
        "Use the tool's output to fix conflicts and ensure symmetric reverses for causal pairs. "
        # "Return ONLY the final JSON array."
    )
    return base + addon

def _meci_user_prompt(doc_text: str, batch_pairs: List[tuple[str,str,str]], spans: Dict[str,str]) -> str:
    pair_lines = []
    for (Ti, _gold, Tj) in batch_pairs:
        si = spans.get(Ti, "")
        sj = spans.get(Tj, "")
        pair_lines.append(f'- "{Ti},{Tj}" ( {Ti}="{si}" , {Tj}="{sj}" )')
    return (
        "Text:\n" + doc_text + "\n\n"
        "Pairs to classify (use EXACT pair ids and order):\n" + "\n".join(pair_lines) + "\n\n"
        "Return ONLY a JSON array like:\n[\n  {\"pair\":\"T0,T1\",\"label\":\"CauseEffect\"}\n]\n"
        "Before answering, you should call the `coherence_check` tool with your draft labels; "
        "then output the corrected final array only."
    )
