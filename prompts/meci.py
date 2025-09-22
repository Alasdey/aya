
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml

# Repo root + config (to find prompts/system.txt)
ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

# ---- Base/system prompt loader ----
def _system_prompt() -> str:
    p = ROOT / CONFIG["paths"]["system_prompt"]
    return p.read_text(encoding="utf-8")


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
