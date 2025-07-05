import logging
import json
from typing import Dict
from pathlib import Path
from .prompts import (
    MITIGATE_BOTH_PROMPT_TEMPLATE,
    MITIGATE_FAITH_ONLY_PROMPT_TEMPLATE,
    MITIGATE_COMP_ONLY_PROMPT_TEMPLATE,
)

def build_mitigation_prompt(example: Dict) -> str:
    """
    Given an example from load_examples(), return the appropriate mitigation prompt string.
    """
    f = example["faithfulness_score"]
    c = example["completeness_score"]
    if f < 5 and c < 5:
        template = MITIGATE_BOTH_PROMPT_TEMPLATE
        return template.format(
            table=example["table"],
            question=example["question"],
            model_answer=example["full_answer"],
            faith_score=f,
            comp_score=c
        )
    elif f < 5:
        template = MITIGATE_FAITH_ONLY_PROMPT_TEMPLATE
        return template.format(
            table=example["table"],
            question=example["question"],
            model_answer=example["full_answer"],
            faith_score=f
        )
    elif c < 5:
        template = MITIGATE_COMP_ONLY_PROMPT_TEMPLATE
        return template.format(
            table=example["table"],
            question=example["question"],
            model_answer=example["full_answer"],
            comp_score=c
        )
    else:
        raise ValueError("This example does not need mitigation.")

def processed_ids(out_dir: Path, dataset: str, model: str) -> set:
    """
    Reads {model}_{dataset}.jsonl from the output folder and returns
    the set of original_idx values that have already been mitigated.
    """
    out_path = out_dir / f"{model}_{dataset}.jsonl"
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add(obj["original_idx"])
            except Exception:
                continue
    logging.info(f"[{dataset}] Found {len(done)} examples already mitigated.")
    return done 