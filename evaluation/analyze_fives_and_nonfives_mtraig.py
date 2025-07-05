import json
import os
import argparse
from pathlib import Path

def analyze_fives_and_nonfives(human_scores, model_scores, label):
    assert len(human_scores) == len(model_scores), "Mismatch in data length"
    total = len(human_scores)
    human_5_indices = {i for i, s in enumerate(human_scores) if s == 5}
    model_5_indices = {i for i, s in enumerate(model_scores) if s == 5}
    human_non5_indices = {i for i in range(total)} - human_5_indices
    model_non5_indices = {i for i in range(total)} - model_5_indices
    matched_non5 = human_non5_indices & model_non5_indices
    missed_non5 = human_non5_indices - model_non5_indices
    wrong_non5 = model_non5_indices - human_non5_indices
    print(f"\n--- {label} ---")
    print(f"Total datapoints: {total}")
    print(f"\nNon-5-score analysis:")
    print(f"  Human non-5s: {len(human_non5_indices)}")
    print(f"  Model non-5s: {len(model_non5_indices)}")
    print(f"    Matched non-5s: {len(matched_non5)}")
    print(f"    Missed (Human non-5s predicted as 5): {len(missed_non5)}")
    print(f"    Wrongly predicted as non-5: {len(wrong_non5)}")

def run_analysis_for_model(model_name: str):
    repo_root = Path(__file__).parent.parent
    mtraig_dir = repo_root / "mtraig" / "faithfulness_scores"
    data_outputs_dir = repo_root / "data" / "outputs"
    datasets = ["qtsumm", "fetaqa"]
    for dataset in datasets:
        print(f"\n{dataset.upper()} Dataset")
        with open(data_outputs_dir / f"model_outputs_with_scores_{dataset}.json", 'r') as f:
            human_entries = json.load(f)
        human_scores = [entry['faithfulness_score'] for entry in human_entries]
        mtraig_path = mtraig_dir / f"{model_name}_{dataset}.json"
        with open(mtraig_path, 'r') as f:
            mtraig_data = json.load(f)
            if "detailed_results" not in mtraig_data:
                raise ValueError("Missing 'detailed_results' key in checkpoint file.")
            model_scores = []
            for entry in mtraig_data["detailed_results"]:
                if "faithfulness_score" not in entry:
                    raise ValueError("Missing 'faithfulness_score' in an entry.")
                model_scores.append(entry["faithfulness_score"])
        analyze_fives_and_nonfives(human_scores, model_scores, f"{dataset.upper()} MTRAIG Approach")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTRAIG analyze fives and nonfives for a model.")
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini", help='Model name (e.g., gpt-4o-mini)')
    args = parser.parse_args()
    run_analysis_for_model(args.model_name) 