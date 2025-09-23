from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import yaml

# Repo root + config
ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

# ---- Base/system prompt loader (from config) ----
def _system_prompt() -> str:
    return CONFIG["prompts"]["system"]["base"].strip()

def _meci_system_prompt() -> str:
    base = _system_prompt()
    addon = CONFIG["prompts"]["meci"]["addon"].strip()
    return f"{base}\n\n{addon}"

def _meci_user_prompt(doc_text: str, batch_pairs: List[tuple[str,str,str]], spans: Dict[str,str]) -> str:
    pair_lines_list = []
    for (Ti, _gold, Tj) in batch_pairs:
        si = spans.get(Ti, "")
        sj = spans.get(Tj, "")
        pair_lines_list.append(f'- "{Ti},{Tj}" ( {Ti}="{si}" , {Tj}="{sj}" )')
    pair_lines = "\n".join(pair_lines_list)

    tmpl = CONFIG["prompts"]["meci"]["user_template"]
    example_json = CONFIG["prompts"]["meci"]["example_json"].strip()

    return tmpl.format(
        doc_text=doc_text,
        pair_lines=pair_lines,
        example_json=example_json
    )
