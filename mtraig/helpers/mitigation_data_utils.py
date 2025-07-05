import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Set
from .prompts import MTRAIG_MITIGATION_PROMPT_TEMPLATE
def load_examples(dataset: str, model: str) -> List[Dict]:
    """
    Loads examples needing mitigation from faithfulness scores and model outputs.
    Returns a list of dicts for every example that has false claims, including serialized table info.
    """
    DATA_DIR = Path("data/outputs")
    CKPT_DIR = Path("mtraig/faithfulness_scores")
    data_file = DATA_DIR / f"model_outputs_with_scores_{dataset}.json"
    ckpt_file = CKPT_DIR / f"{model}_{dataset}.json"

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    with ckpt_file.open() as f:
        ckpt_obj = json.load(f)
    detailed_results = ckpt_obj["detailed_results"]

    with data_file.open() as f:
        raw = json.load(f)

    if len(detailed_results) != len(raw):
        raise ValueError(
            f"Length mismatch: {len(detailed_results)} results vs {len(raw)} examples"
        )

    keep: List[Dict] = []
    for idx, (ex, result) in enumerate(zip(raw, detailed_results)):
        false_claims = [
            claim for claim, is_true in zip(result["claims"], result["claim_verifications"])
            if not is_true
        ]
        if not false_claims:
            continue
        # Use serialized_table directly if present
        serialized_table = ex.get("serialized_table")
        if not serialized_table:
            metadata = ex.get("metadata", {})
            if dataset == "fetaqa":
                serialized_table = {
                    'title': f"{metadata.get('table_page_title', '')} - {metadata.get('table_section_title', '')}",
                    'header': metadata.get('table_array', [[]])[0],
                    'rows': metadata.get('table_array', [[]])[1:]
                }
            else:  # qtsumm
                serialized_table = {
                    'title': metadata.get('table', {}).get('title', []),
                    'header': metadata.get('table', {}).get('header', []),
                    'rows': metadata.get('table', {}).get('rows', [])
                }
        keep.append({
            "idx": idx,
            "question": ex["question"],
            "table": serialized_table,
            "full_answer": ex["model_output"],
            "false_claims": false_claims
        })
    logging.info(f"{dataset.upper()}: {len(keep)} / {len(raw)} examples need mitigation.")
    return keep

def processed_ids(dataset: str, model: str) -> Set[int]:
    """
    Reads {model}_{dataset}.jsonl (if it exists) in mitigation_outputs and returns the set of original_idx values already mitigated.
    """
    OUT_DIR = Path("mtraig/mitigation_outputs")
    done = set()
    out_path = OUT_DIR / f"{model}_{dataset}.jsonl"
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add(obj["original_idx"])
            except Exception:
                continue  # ignore malformed lines
    logging.info(f"[{dataset}] Found {len(done)} examples already mitigated.")
    return done 

def build_mitigation_prompt(example):
    """
    Takes one example dict from load_examples() → formatted coarse-level prompt.
    """
    prompt = MTRAIG_MITIGATION_PROMPT_TEMPLATE.format(
        false_claims=example["false_claims"],
        table=example["table"],
        question=example["question"],
        model_answer=example["full_answer"]
    )
    return prompt 