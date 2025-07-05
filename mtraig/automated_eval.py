import os
import json
import logging
from pathlib import Path
from mtraig.helpers.data_utils import load_human_faith_scores
from mtraig.helpers.automated_eval_data_utils import load_faithfulness_scores_from_ckpt
from mtraig.helpers.openai_utils import decompose_claims, verify_claims
from mtraig.helpers.score_utils import calculate_faithfulness_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

MITIG_DIR   = Path("mtraig/mitigation_outputs")
AE_CKPT_DIR = Path("mtraig/automated_eval_checkpoints")
RESULTS_DIR = Path("results/mtraig_automated_eval")
for p in (AE_CKPT_DIR, RESULTS_DIR):
    p.mkdir(exist_ok=True)

CKPT_DIR = Path("mtraig/faithfulness_scores")


def evaluate_mitigation(dataset: str, model: str):
    mit_file     = MITIG_DIR   / f"{model}_{dataset}.jsonl"
    ae_ck_file   = AE_CKPT_DIR / f"{model}_{dataset}.json"
    summary_file = RESULTS_DIR / f"{model}_{dataset}.txt"
    temperature = 0.0

    if not mit_file.exists():
        raise FileNotFoundError(mit_file)

    # Load original scores and dataset
    df, _ = load_human_faith_scores(f"model_outputs_with_scores_{dataset}.json")
    old_scores = load_faithfulness_scores_from_ckpt(str(CKPT_DIR / f"{model}_{dataset}.json"))

    # Load or initialize checkpoint
    revised_entries = []
    seen_indices = set()
    if ae_ck_file.exists():
        with ae_ck_file.open() as f:
            revised_entries = json.load(f)
            seen_indices = {entry["original_idx"] for entry in revised_entries}
        logging.info(f"[{dataset}] Resuming from checkpoint: {len(revised_entries)} entries loaded")

    all_old_scores = []
    all_new_scores = []

    # Recompute only for missing entries
    with mit_file.open() as f:
        for raw in f:
            e = json.loads(raw)
            idx = e["original_idx"]
            if idx in seen_indices:
                continue
            revised_answer = " ".join(e["revised_answer"]).strip() if isinstance(e["revised_answer"], list) else str(e["revised_answer"]).strip()
            old = old_scores[idx]
            if old >= 5:
                continue
            r = df.iloc[idx]
            try:
                claims = decompose_claims(schema=r["schema"], insight=revised_answer, temperature=temperature, model=model)
                verifications = verify_claims(r["serialized_table"], claims, temperature=temperature, model=model)
                new_score = calculate_faithfulness_score(verifications)
            except Exception as err:
                logging.warning(f"{idx}: {err}; keep old score")
                new_score = old
                claims = []
                verifications = []
            entry = {
                "original_idx": idx,
                "old_score": old,
                "new_score": new_score,
                "claims": claims,
                "verifications": verifications
            }
            revised_entries.append(entry)
            # Save updated checkpoint
            json.dump(revised_entries, ae_ck_file.open("w"), indent=2)
            all_old_scores.append(old)
            all_new_scores.append(new_score)

    # Recalculate summary regardless of whether new entries were processed
    if not revised_entries:
        logging.warning("No new examples processed. Using existing checkpoint for summary.")
    if not all_old_scores:
        all_old_scores = [e["old_score"] for e in revised_entries]
        all_new_scores = [e["new_score"] for e in revised_entries]
    avg_old_updated = sum(all_old_scores) / len(all_old_scores)
    avg_new_updated = sum(all_new_scores) / len(all_new_scores)
    delta_updated = (avg_new_updated - avg_old_updated) / 5 * 100
    full_new_scores = old_scores.copy()
    for entry in revised_entries:
        full_new_scores[entry["original_idx"]] = entry["new_score"]
    avg_old_all = sum(old_scores) / len(old_scores)
    avg_new_all = sum(full_new_scores) / len(full_new_scores)
    delta_all = (avg_new_all - avg_old_all) / 5 * 100
    with summary_file.open("w") as sf:
        sf.write(f"{dataset.upper()} – MT-RAIG Mitigation Summary\n")
        sf.write(f"examples revised     : {len(all_old_scores)}\n")
        sf.write(f"\n--- On Revised Only ---\n")
        sf.write(f"before               : {avg_old_updated:.3f}\n")
        sf.write(f"after                : {avg_new_updated:.3f}\n")
        sf.write(f"improvement          : {delta_updated:+.2f}%\n")
        sf.write(f"\n--- On Full Dataset ---\n")
        sf.write(f"overall before       : {avg_old_all:.3f}\n")
        sf.write(f"overall after        : {avg_new_all:.3f}\n")
        sf.write(f"overall improvement  : {delta_all:+.2f}%\n")
    logging.info(f"[{dataset}] summary -> {summary_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MT-RAIG mitigation automated evaluation.")
    parser.add_argument('--dataset', type=str, default='fetaqa', help="Dataset name (fetaqa or qtsumm)")
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help="Model name (e.g., gpt-4o-mini)")
    args = parser.parse_args()
    evaluate_mitigation(args.dataset, args.model)
