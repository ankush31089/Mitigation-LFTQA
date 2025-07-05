
import json
import csv
from pathlib import Path
import argparse

def create_mitigation_eval_file(model_name: str, dataset: str, num_points: int = 50):
    """
    Combines LFTQA and MTRAIG mitigation outputs and writes both JSON and CSV.
    """
    repo_root = Path(__file__).parent.parent
    data_dir  = repo_root / "data" / "outputs"
    # input files
    data_file   = data_dir / f"model_outputs_with_scores_{dataset}.json"
    lftqa_file  = repo_root / "g_eval"  / "mitigation_outputs" / "normal" / f"{model_name}_{dataset}.jsonl"
    mtraig_file = repo_root / "mtraig" / "mitigation_outputs" / f"{model_name}_{dataset}.jsonl"
    # output dir & paths
    out_dir     = repo_root / "human_mitigation_eval/raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path   = out_dir / f"{model_name}_{dataset}.json"
    csv_path    = out_dir / f"{model_name}_{dataset}.csv"
    # load base examples
    with data_file.open() as f:
        full_data = json.load(f)
    # load mitigation entries
    lftqa_entries  = [json.loads(line) for line in lftqa_file.open()]
    mtraig_entries = [json.loads(line) for line in mtraig_file.open()]
    # build maps: idx → revised_answer
    lftqa_map  = {e["original_idx"]: e["revised_answer"] for e in lftqa_entries}
    mtraig_map = {e["original_idx"]: e["revised_answer"] for e in mtraig_entries}
    # common indices up to num_points
    common_idxs = sorted(set(lftqa_map) & set(mtraig_map))[:num_points]
    # assemble rows
    rows = []
    for idx in common_idxs:
        ex = full_data[idx]
        rows.append({
            "example_id":                ex.get("example_id", ""),
            "original_model":           ex.get("model", ""),
            "original_idx":             idx,
            "question":                 ex.get("question", ""),
            "table":                    ex.get("table", ""),
            "ideal_answer":             ex.get("answer", ""),
            "original_answer":          ex.get("model_output", ""),
            "lftqa_mitigated_output":   lftqa_map[idx],
            "mtraig_mitigated_output":  mtraig_map[idx]
        })
    # write JSON
    with json_path.open("w") as f_json:
        json.dump(rows, f_json, indent=2)
    # write CSV
    fieldnames = [
        "example_id",
        "original_model",
        "original_idx",
        "question",
        "table",
        "ideal_answer",
        "original_answer",
        "lftqa_mitigated_output",
        "mtraig_mitigated_output"
    ]
    with csv_path.open("w", newline='', encoding='utf-8') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[✓] Wrote {len(rows)} examples to:")
    print(f"    JSON → {json_path.resolve()}")
    print(f"    CSV  → {csv_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mitigation eval file (JSON and CSV) for human eval.")
    parser.add_argument('--model_name', type=str, default="gpt-4o", help='Model name (e.g., gpt-4o-mini)')
    parser.add_argument('--dataset', type=str, default="qtsumm", help='Dataset name (e.g., fetaqa or qtsumm)')
    parser.add_argument('--num_points', type=int, default=50, help='Number of examples to include (default: 50)')
    args = parser.parse_args()
    create_mitigation_eval_file(args.model_name, args.dataset, args.num_points)
