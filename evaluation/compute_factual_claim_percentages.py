import json
from pathlib import Path
import argparse

def compute_factual_claim_percentages(model: str, dataset: str):
    """
    Compute % factual claims in original and revised MTRAIG outputs.
    Returns a tuple: (original_factual_percentage, revised_factual_percentage)
    """
    # Paths
    repo_root = Path(__file__).parent.parent
    original_path = repo_root / "mtraig" / "faithfulness_scores" / f"{model}_{dataset}.json"
    revised_path  = repo_root / "mtraig" / "automated_eval_checkpoints" / f"{model}_{dataset}.json"
    # Load original claim verifications
    with original_path.open() as f:
        original_data = json.load(f)["detailed_results"]
    # Load revised verifications
    with revised_path.open() as f:
        revised_data = json.load(f)
    original_total = 0
    original_true = 0
    revised_total = 0
    revised_true = 0
    # Map from index to revised claim count
    revised_claims_by_idx = {entry["original_idx"]: entry["verifications"] for entry in revised_data}
    for idx, entry in enumerate(original_data):
        orig_claims = entry.get("claim_verifications", [])
        revised_claims = revised_claims_by_idx.get(idx, orig_claims)
        original_total += len(orig_claims)
        original_true += sum(orig_claims)
        revised_total += len(revised_claims)
        revised_true += sum(revised_claims)
    # Calculate percentages
    original_pct = 100 * original_true / original_total if original_total else 0
    revised_pct  = 100 * revised_true / revised_total if revised_total else 0
    print(f"Original Output: {original_true}/{original_total} factual claims ({original_pct:.2f}%)")
    print(f"Revised Output:  {revised_true}/{revised_total} factual claims ({revised_pct:.2f}%)")
    return original_pct, revised_pct

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute factual claim percentages for MTRAIG outputs.")
    parser.add_argument('--model', type=str, default="gpt-4o-mini", help='Model name (e.g., gpt-4o-mini)')
    parser.add_argument('--dataset', type=str, default="fetaqa", help='Dataset name (e.g., fetaqa or qtsumm)')
    args = parser.parse_args()
    compute_factual_claim_percentages(args.model, args.dataset) 