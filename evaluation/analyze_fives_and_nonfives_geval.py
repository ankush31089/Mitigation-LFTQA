import json
import os
import argparse
from pathlib import Path

def analyze_fives_and_nonfives(human_scores, model_scores, label, score_type="Faithfulness"):
    assert len(human_scores) == len(model_scores), "Mismatch in data length"
    total = len(human_scores)
    human_5_indices = {i for i, s in enumerate(human_scores) if s == 5}
    model_5_indices = {i for i, s in enumerate(model_scores) if s == 5}
    human_non5_indices = {i for i in range(total)} - human_5_indices
    model_non5_indices = {i for i in range(total)} - model_5_indices
    matched_non5 = human_non5_indices & model_non5_indices
    missed_non5 = human_non5_indices - model_non5_indices
    wrong_non5 = model_non5_indices - human_non5_indices
    print(f"\n--- {label} ({score_type}) ---")
    print(f"Total datapoints: {total}")
    print(f"\nNon-5-score analysis:")
    print(f"  Human non-5s: {len(human_non5_indices)}")
    print(f"  Model non-5s: {len(model_non5_indices)}")
    print(f"    Matched non-5s: {len(matched_non5)}")
    print(f"    Missed (Human non-5s predicted as 5): {len(missed_non5)}")
    print(f"    Wrongly predicted as non-5: {len(wrong_non5)}")

def run_analysis_for_model(model_name: str):
    repo_root = Path(__file__).parent.parent
    lftqa_faith_dir = repo_root / "g_eval" / "faithfulness_scores"
    lftqa_comp_dir = repo_root / "g_eval" / "completeness_scores"
    data_outputs_dir = repo_root / "data" / "outputs"
    datasets = ["qtsumm", "fetaqa"]
    for dataset in datasets:
        print(f"\n{dataset.upper()} Dataset")
        with open(data_outputs_dir / f"model_outputs_with_scores_{dataset}.json", 'r') as f:
            human_entries = json.load(f)
        human_faith_scores = [entry['faithfulness_score'] for entry in human_entries]
        human_comp_scores = [entry['completeness_score'] for entry in human_entries]
        lftqa_faith_path = lftqa_faith_dir / f"{model_name}_{dataset}.json"
        lftqa_comp_path = lftqa_comp_dir / f"{model_name}_{dataset}.json"
        with open(lftqa_faith_path, 'r') as f:
            lftqa_faith_data = json.load(f)
        with open(lftqa_comp_path, 'r') as f:
            lftqa_comp_data = json.load(f)
        analyze_fives_and_nonfives(
            human_faith_scores, 
            lftqa_faith_data['faithfulness_scores'], 
            f"{dataset.upper()} LFTQA Approach",
            "Faithfulness"
        )
        analyze_fives_and_nonfives(
            human_comp_scores,
            lftqa_comp_data['completeness_scores'],
            f"{dataset.upper()} LFTQA Approach",
            "Completeness"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEVAL (LFTQA) analyze fives and nonfives for a model.")
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini", help='Model name (e.g., gpt-4o-mini)')
    args = parser.parse_args()
    run_analysis_for_model(args.model_name) 