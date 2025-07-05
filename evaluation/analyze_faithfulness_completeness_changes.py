import json
from pathlib import Path
import argparse

def analyze_faithfulness_completeness_changes(model: str, dataset: str):
    repo_root = Path(__file__).parent.parent
    FAITH_ORIG_DIR = repo_root / "g_eval" / "faithfulness_scores"
    COMP_ORIG_DIR = repo_root / "g_eval" / "completeness_scores"
    FAITH_NEW_DIR = repo_root / "g_eval" / "automated_eval_checkpoints" / "normal" / "faithfulness"
    COMP_NEW_DIR = repo_root / "g_eval" / "automated_eval_checkpoints" / "normal" / "completeness"
    # Load original scores
    with open(FAITH_ORIG_DIR / f"{model}_{dataset}.json") as f:
        original_faith = json.load(f)["faithfulness_scores"]
    with open(COMP_ORIG_DIR / f"{model}_{dataset}.json") as f:
        original_comp = json.load(f)["completeness_scores"]
    # Load revised scores
    with open(FAITH_NEW_DIR / f"{model}_{dataset}.json") as f:
        revised_faith = {int(k): v for k, v in json.load(f)["all_new_scores"].items()}
    with open(COMP_NEW_DIR / f"{model}_{dataset}.json") as f:
        revised_comp = {int(k): v for k, v in json.load(f)["all_new_scores"].items()}
    n = len(original_faith)
    # Category 1: Original faith < 5 AND comp == 5
    cat1_indices = [i for i in revised_faith if original_faith[i] < 5 and original_comp[i] == 5]
    faith_deltas_cat1 = [revised_faith[i] - original_faith[i] for i in cat1_indices]
    # Category 2: Original comp < 5 AND faith == 5
    cat2_indices = [i for i in revised_comp if original_comp[i] < 5 and original_faith[i] == 5]
    comp_deltas_cat2 = [revised_comp[i] - original_comp[i] for i in cat2_indices]
    # Category 3: Both faith < 5 AND comp < 5
    cat3_indices = [i for i in set(revised_faith) & set(revised_comp) if original_faith[i] < 5 and original_comp[i] < 5]
    faith_deltas_cat3 = [revised_faith[i] - original_faith[i] for i in cat3_indices]
    comp_deltas_cat3 = [revised_comp[i] - original_comp[i] for i in cat3_indices]
    print(f"\nCategory 1: Faith<5 and Comp=5")
    print(f"Count: {len(cat1_indices)} ({len(cat1_indices)/n:.2%})")
    print(f"Avg Δ Faithfulness: {sum(faith_deltas_cat1)/len(faith_deltas_cat1):.4f}" if cat1_indices else "No examples")
    print(f"\nCategory 2: Comp<5 and Faith=5")
    print(f"Count: {len(cat2_indices)} ({len(cat2_indices)/n:.2%})")
    print(f"Avg Δ Completeness: {sum(comp_deltas_cat2)/len(comp_deltas_cat2):.4f}" if cat2_indices else "No examples")
    print(f"\nCategory 3: Faith<5 and Comp<5")
    print(f"Count: {len(cat3_indices)} ({len(cat3_indices)/n:.2%})")
    if cat3_indices:
        print(f"Avg Δ Faithfulness: {sum(faith_deltas_cat3)/len(faith_deltas_cat3):.4f}")
        print(f"Avg Δ Completeness: {sum(comp_deltas_cat3)/len(comp_deltas_cat3):.4f}")
    else:
        print("No examples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze faithfulness and completeness changes.")
    parser.add_argument('--model', type=str, default="gpt-4o-mini", help='Model name (e.g., gpt-4o)')
    parser.add_argument('--dataset', type=str, default="fetaqa", help='Dataset name (e.g., fetaqa or qtsumm)')
    args = parser.parse_args()
    analyze_faithfulness_completeness_changes(args.model, args.dataset) 