"""
Mitigation logic for MTRAIG approach.
"""

import os
import json
import logging
from pathlib import Path
from mtraig.helpers.mitigation_data_utils import build_mitigation_prompt, load_examples, processed_ids
from mtraig.helpers.openai_utils import get_mitigated_output

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

OUT_DIR = Path("mtraig/mitigation_outputs")
OUT_DIR.mkdir(exist_ok=True)

def run_mitigation(dataset: str, model: str = "gpt-4o-mini", max_api_retries: int = 20):
    examples  = load_examples(dataset, model)
    out_path  = OUT_DIR / f"{model}_{dataset}.jsonl"
    done_ids  = processed_ids(dataset, model)

    with out_path.open("a", encoding="utf-8") as outf:
        for ex in examples:
            if ex["idx"] in done_ids:
                continue
            logging.info(f"[{dataset}] mitigating idx {ex['idx']}  ({len(done_ids)+1}/{len(examples)})")
            prompt = build_mitigation_prompt(ex)
            revised_answer = get_mitigated_output(prompt, model=model, temperature=0.0, max_api_retries=max_api_retries)
            if revised_answer is None:
                logging.error(f"[{dataset}] mitigation failed – keeping original.")
                revised_answer = ex["full_answer"]
            json.dump({
                "original_idx": ex["idx"],
                "revised_answer": revised_answer
            }, outf, ensure_ascii=False)
            outf.write("\n")
            outf.flush()
            done_ids.add(ex["idx"])
    logging.info(f"Mitigation finished – total processed: {len(done_ids)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MT-RAIG mitigation pipeline.")
    parser.add_argument('--dataset', type=str, default='fetaqa', help="Dataset name (fetaqa or qtsumm)")
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help="Model name (e.g., gpt-4o-mini)")
    args = parser.parse_args()
    run_mitigation(args.dataset, args.model) 