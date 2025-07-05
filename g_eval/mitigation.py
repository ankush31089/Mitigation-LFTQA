"""
G-Eval mitigation logic: modular, using helpers for prompts, schemas, OpenAI call, and mitigation utilities.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

from g_eval.helpers.mitigation_utils import build_mitigation_prompt, processed_ids
from g_eval.helpers.schemas import AnswerRewrite
from g_eval.helpers.openai_utils import call_openai_mitigation

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Load environment variables
load_dotenv()

# Output directories
MITIGATION_BASE_DIR = Path(__file__).parent / "mitigation_outputs"
NORMAL_OUT_DIR = MITIGATION_BASE_DIR / "normal"
ORACLE_OUT_DIR = MITIGATION_BASE_DIR / "oracle"
NORMAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
ORACLE_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Data and checkpoint directories
DATA_DIR = Path(__file__).parent.parent / "data" / "outputs"
FAITH_CKPT_DIR = Path(__file__).parent / "faithfulness_scores"
COMP_CKPT_DIR = Path(__file__).parent / "completeness_scores"


def load_examples(dataset: str, model: str, kind: str = "normal") -> List[Dict]:
    """
    Load examples needing mitigation (at least one score < 5.0).
    """
    assert kind in {"normal", "oracle"}, "kind must be 'normal' or 'oracle'"
    data_file = DATA_DIR / f"model_outputs_with_scores_{dataset}.json"
    with data_file.open() as f:
        raw = json.load(f)
    if kind == "normal":
        ckpt_file_faith = FAITH_CKPT_DIR / f"{model}_{dataset}.json"
        ckpt_file_comp  = COMP_CKPT_DIR  / f"{model}_{dataset}.json"
        if not ckpt_file_faith.exists() or not ckpt_file_comp.exists():
            raise FileNotFoundError("Checkpoint files for faithfulness or completeness not found.")
        with ckpt_file_faith.open() as f1, ckpt_file_comp.open() as f2:
            faith_ckpt = json.load(f1)
            comp_ckpt  = json.load(f2)
        faith_scores = faith_ckpt.get("faithfulness_scores") or faith_ckpt.get("faith_scores")
        comp_scores  = comp_ckpt.get("completeness_scores") or comp_ckpt.get("completeness_scores")
        if faith_scores is None or comp_scores is None:
            raise KeyError("One of the score keys is missing in checkpoint files.")
        if len(faith_scores) != len(comp_scores) or len(faith_scores) != len(raw):
            raise ValueError("Length mismatch among scores or with data entries.")
    else:
        faith_scores = [ex["faithfulness_score"] for ex in raw]
        comp_scores  = [ex["completeness_score"] for ex in raw]
    examples: List[Dict] = []
    for idx, (ex, fscore, cscore) in enumerate(zip(raw, faith_scores, comp_scores)):
        if fscore >= 5.0 and cscore >= 5.0:
            continue
        examples.append({
            "idx"                : idx,
            "question"           : ex["question"],
            "table"              : ex["serialized_table"],
            "full_answer"        : ex["model_output"],
            "faithfulness_score" : fscore,
            "completeness_score" : cscore
        })
    logging.info(f"{dataset.upper()} [{kind}] → {len(examples)} examples need mitigation.")
    return examples


def run_mitigation(dataset: str, kind: str, model: str = "gpt-4o-mini", max_api_retries: int = 20):
    """
    Runs coarse-level mitigation for all examples in a dataset+model+kind combo
    where either faithfulness or completeness score is < 5.
    """
    assert kind in {"normal", "oracle"}, "kind must be 'normal' or 'oracle'"
    out_dir = NORMAL_OUT_DIR if kind == "normal" else ORACLE_OUT_DIR
    out_path = out_dir / f"{model}_{dataset}.jsonl"
    examples = load_examples(dataset, model, kind=kind)
    done_ids = processed_ids(out_dir, dataset, model)
    with out_path.open("a", encoding="utf-8") as outf:
        for ex in examples:
            if ex["idx"] in done_ids:
                continue
            print(f"[{dataset}] mitigating idx {ex['idx']}  "
                  f"({len(done_ids)+1}/{len(examples)})")
            prompt = build_mitigation_prompt(ex)
            revised_answer = call_openai_mitigation(
                prompt,
                model=model,
                temperature=0.0,
                max_retries=max_api_retries
            )
            if revised_answer is None:
                logging.error(f"[{dataset}] mitigation failed – keeping original.")
                revised_answer = ex["full_answer"]
            json.dump(
                {
                    "original_idx": ex["idx"],
                    "revised_answer": revised_answer
                },
                outf,
                ensure_ascii=False
            )
            outf.write("\n")
            outf.flush()
            done_ids.add(ex["idx"])
    print(f"\nMitigation finished – total processed: {len(done_ids)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run G-Eval mitigation pipeline.")
    parser.add_argument('--dataset', type=str, default='fetaqa', help="Dataset name (fetaqa or qtsumm)")
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help="Model name (e.g., gpt-4o-mini)")
    parser.add_argument('--kind', type=str, default='normal', choices=['normal', 'oracle'], help="Mitigation kind")
    args = parser.parse_args()
    run_mitigation(args.dataset, args.kind, model=args.model) 