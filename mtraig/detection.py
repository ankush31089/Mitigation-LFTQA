"""
Detection logic for MTRAIG approach.
"""

import os
import json
import logging
from mtraig.helpers.data_utils import load_human_faith_scores
from mtraig.helpers.openai_utils import decompose_claims, verify_claims
from mtraig.helpers.score_utils import calculate_faithfulness_score, calculate_correlation

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def evaluate(dataset: str, model_name: str = "gpt-4o-mini") -> float:
    data_filename   = f"model_outputs_with_scores_{dataset}.json"
    tag             = f"{model_name}_{dataset}"
    checkpoint_fname= f"{tag}.json"
    results_fname   = f"{tag}.txt"
    temperature     = 0.0

    CHECKPOINT_DIR = "mtraig/faithfulness_scores"
    RESULTS_DIR    = "results/mtraig_correlation"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_fname)
    results_path    = os.path.join(RESULTS_DIR, results_fname)

    df, human_faith = load_human_faith_scores(data_filename)

    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, "r") as ckf:
            ck = json.load(ckf)
        detailed_results = ck.get("detailed_results", [])
        logging.info(f"Loaded {len(detailed_results)} entries from checkpoint")
    else:
        detailed_results = [{} for _ in range(len(df))]

    for idx, row in df.iterrows():
        example_id = row.get("example_id", "N/A")
        existing = detailed_results[idx] if idx < len(detailed_results) else {}
        needs_redo = (
            not existing or
            (existing.get("claims") == [] and existing.get("claim_verifications") == [])
        )
        if not needs_redo:
            continue
        logging.info(f"Re-evaluating idx={idx}, example_id={example_id}")
        try:
            claims = decompose_claims(
                schema=row["schema"],
                insight=row.get("model_output"),
                temperature=temperature,
                model=model_name
            )
            verifications = verify_claims(row["serialized_table"], claims, temperature=temperature, model=model_name)
            pred_f = calculate_faithfulness_score(verifications)
            datapoint_result = {
                "example_id": example_id,
                "claims": claims,
                "claim_verifications": verifications,
                "faithfulness_score": pred_f,
                "human_score": human_faith[idx]
            }
        except Exception as e:
            logging.warning(f" → call failed at idx={idx}: {str(e)}")
            datapoint_result = {
                "example_id": example_id,
                "claims": [],
                "claim_verifications": [],
                "faithfulness_score": 1.0,
                "human_score": human_faith[idx],
                "error": str(e)
            }
        detailed_results[idx] = datapoint_result
        with open(checkpoint_path, "w") as ckf:
            json.dump({
                "last_idx": len(detailed_results) - 1,
                "detailed_results": detailed_results
            }, ckf, indent=2)
        logging.info(f"Checkpoint saved at idx {idx}")
    df["score_metric"] = [r.get("faithfulness_score", 1.0) for r in detailed_results]
    df["score_human"] = human_faith
    instance_r = calculate_correlation(df)
    with open(results_path, "w") as rf:
        rf.write(f"Instance-level Pearson r: {instance_r:.4f}\n")
    logging.info(f"Final results written to {results_path}")
    return instance_r

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MT-RAIG detection pipeline.")
    parser.add_argument('--dataset', type=str, default='fetaqa', help="Dataset name (fetaqa or qtsumm)")
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help="Model name (e.g., gpt-4o-mini)")
    args = parser.parse_args()
    evaluate(args.dataset, args.model) 