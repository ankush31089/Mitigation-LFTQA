"""
G-Eval detection logic: faithfulness and completeness evaluation using OpenAI structured output.
Refactored to use modular helpers.
"""

import os
import json
import logging
import pandas as pd
from typing import Optional
import argparse

from g_eval.helpers.prompts import FAITH_PROMPT_TEMPLATE, COMP_PROMPT_TEMPLATE
from g_eval.helpers.schemas import FaithfulnessScore, CompletenessScore
from g_eval.helpers.openai_utils import call_openai_structured
from g_eval.helpers.correlation import calculate_correlation

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def evaluate(
    dataset: str,
    model_name: str = "gpt-4o-mini",
    mode: str = "faithfulness",
    data_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    results_dir: Optional[str] = None
) -> float:
    """
    Evaluate either faithfulness or completeness scores using OpenAI structured output.
    Saves checkpoints and results in the specified directories.
    """
    assert mode in {"faithfulness", "completeness"}, "Mode must be 'faithfulness' or 'completeness'"
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '../data/outputs')
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.path.dirname(__file__), f'../g_eval/{mode}_scores')
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), f'../results/g_eval_{mode}_correlation')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    data_filename    = f"model_outputs_with_scores_{dataset}.json"
    tag              = f"{model_name}_{dataset}"
    checkpoint_fname = f"{tag}.json"
    results_fname    = f"{tag}.txt"

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_fname)
    results_path    = os.path.join(results_dir, results_fname)

    # --- load data ---
    data_path = os.path.join(data_dir, data_filename)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if 'faithfulness_score' not in df.columns or 'completeness_score' not in df.columns:
        raise KeyError("Missing required columns: 'faithfulness_score' or 'completeness_score'.")
    human_scores = df['faithfulness_score'].tolist() if mode == "faithfulness" else df['completeness_score'].tolist()

    # --- model config for prompt/schema/field ---
    prompt_template = FAITH_PROMPT_TEMPLATE if mode == "faithfulness" else COMP_PROMPT_TEMPLATE
    schema_class    = FaithfulnessScore if mode == "faithfulness" else CompletenessScore
    field_name      = "faithfulness" if mode == "faithfulness" else "completeness"

    # --- resume from checkpoint if exists ---
    start_idx = 0
    model_scores = []
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, "r") as f:
            ck = json.load(f)
        start_idx     = ck.get("last_idx", -1) + 1
        model_scores  = ck.get(f"{mode}_scores", [])
        logging.info(f"Resuming from index {start_idx}")
    else:
        logging.info("No checkpoint found, starting fresh.")

    # --- evaluation loop ---
    total = len(df)
    for idx in range(start_idx, total):
        row = df.iloc[idx]
        prompt = prompt_template.format(
            table=row["serialized_table"],
            question=row["question"],
            gen_answer=row.get("model_output")
        )
        logging.info(f"idx={idx} ({idx+1}/{total}) example_id={row.get('example_id')}, model={model_name}")
        try:
            score = call_openai_structured(prompt, schema_class, field_name, model=model_name)
        except Exception:
            logging.warning(f"  → call failed at idx={idx}, defaulting to 1.0")
            score = 1.0
        model_scores.append(score)
        # checkpoint every 10 examples
        if idx % 10 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump({"last_idx": idx, f"{mode}_scores": model_scores}, f)
            logging.info(f"Checkpoint saved at idx={idx}")
    # --- final checkpoint ---
    with open(checkpoint_path, "w") as f:
        json.dump({"last_idx": total - 1, f"{mode}_scores": model_scores}, f)
    logging.info("Final checkpoint written")
    # --- correlation calculation ---
    df = df.copy()
    df["score_metric"] = model_scores
    df["score_human"]  = human_scores
    instance_r = calculate_correlation(df)
    logging.info(f"Instance-level Pearson r for {mode}: {instance_r:.4f}")
    with open(results_path, "w") as f:
        f.write(f"Instance-level Pearson r: {instance_r:.4f}\n")
    return instance_r

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run G-Eval detection pipeline.")
    parser.add_argument('--dataset', type=str, default='fetaqa', help="Dataset name (fetaqa or qtsumm)")
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help="Model name (e.g., gpt-4o-mini)")
    parser.add_argument('--mode', type=str, default='faithfulness', choices=['faithfulness', 'completeness'], help="Evaluation mode")
    args = parser.parse_args()

    print(f"Running detection for dataset={args.dataset}, model={args.model}, mode={args.mode}")
    evaluate(args.dataset, model_name=args.model, mode=args.mode) 