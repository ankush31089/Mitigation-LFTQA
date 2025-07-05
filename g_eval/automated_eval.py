import logging
import json
from pathlib import Path
from g_eval.helpers.automated_eval_utils import (
    load_coarse_scores, load_dataset_rows, load_oracle_coarse_scores,
    AE_CKPT_DIR_NORMAL_FAITH, AE_CKPT_DIR_NORMAL_COMP,
    AE_CKPT_DIR_ORACLE_FAITH, AE_CKPT_DIR_ORACLE_COMP,
    MITIG_DIR, ORACLE_MIT_DIR,
    RESULTS_DIR_NORMAL_FAITH, RESULTS_DIR_NORMAL_COMP,
    RESULTS_DIR_ORACLE_FAITH, RESULTS_DIR_ORACLE_COMP
)
from g_eval.helpers.prompts import FAITH_PROMPT_TEMPLATE, COMP_PROMPT_TEMPLATE
from g_eval.helpers.schemas import FaithfulnessScore, CompletenessScore
from g_eval.helpers.openai_utils import call_openai_structured

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

MAX_API_RETRY = 20

def evaluate_mitigation(dataset: str, model: str, type: str, mode: str):
    assert mode in {"faithfulness", "completeness"}, "Invalid mode"
    assert type in {"normal", "oracle"}, "Invalid type"
    mit_file = (MITIG_DIR if type == "normal" else ORACLE_MIT_DIR) / f"{model}_{dataset}.jsonl"
    ae_ck_dir = {
        ("normal", "faithfulness"): AE_CKPT_DIR_NORMAL_FAITH,
        ("normal", "completeness"): AE_CKPT_DIR_NORMAL_COMP,
        ("oracle", "faithfulness"): AE_CKPT_DIR_ORACLE_FAITH,
        ("oracle", "completeness"): AE_CKPT_DIR_ORACLE_COMP
    }[(type, mode)]
    results_dir = {
        ("normal", "faithfulness"): RESULTS_DIR_NORMAL_FAITH,
        ("normal", "completeness"): RESULTS_DIR_NORMAL_COMP,
        ("oracle", "faithfulness"): RESULTS_DIR_ORACLE_FAITH,
        ("oracle", "completeness"): RESULTS_DIR_ORACLE_COMP
    }[(type, mode)]
    ae_ck_file = ae_ck_dir / f"{model}_{dataset}.json"
    summary_file = results_dir / f"{model}_{dataset}.txt"
    if not mit_file.exists():
        raise FileNotFoundError(mit_file)
    if type == "normal":
        old_scores = load_coarse_scores(dataset, model, mode)
    else:
        old_scores = load_oracle_coarse_scores(dataset, mode)
    rows = load_dataset_rows(dataset)
    all_new_scores = {}
    last_line = -1
    if ae_ck_file.exists():
        ck = json.load(ae_ck_file.open())
        all_new_scores = ck.get("all_new_scores", {})
        last_line = ck.get("last_line", -1)
        logging.info(f"[{dataset}] resume {mode} eval at line {last_line + 1}")
    prompt_template = FAITH_PROMPT_TEMPLATE if mode == "faithfulness" else COMP_PROMPT_TEMPLATE
    schema = FaithfulnessScore if mode == "faithfulness" else CompletenessScore
    field = "faithfulness" if mode == "faithfulness" else "completeness"
    with mit_file.open() as f:
        for ln, raw in enumerate(f):
            if ln <= last_line:
                continue
            e = json.loads(raw)
            idx = e["original_idx"]
            revised_answer = e["revised_answer"].strip()
            old = old_scores[idx]
            if old >= 5:
                last_line = ln
                continue
            r = rows[idx]
            serialized_table = r["serialized_table"]
            prompt = prompt_template.format(
                table=serialized_table,
                question=r["question"],
                gen_answer=revised_answer
            )
            try:
                new_score = call_openai_structured(
                    prompt,
                    schema=schema,
                    field=field,
                    model=model,
                    temperature=0.0,
                    max_retries=MAX_API_RETRY
                )
            except Exception as err:
                logging.warning(f"{idx}: {err}; keeping old score")
                new_score = old
            all_new_scores[str(idx)] = new_score
            last_line = ln
            json.dump({
                "last_line": last_line,
                "all_new_scores": all_new_scores
            }, ae_ck_file.open("w"), indent=2)
    if not all_new_scores:
        logging.warning("Nothing processed")
        return
    new_scores_full = old_scores.copy()
    for idx_str, new_score in all_new_scores.items():
        new_scores_full[int(idx_str)] = new_score
    avg_old_total = sum(old_scores) / len(old_scores)
    avg_new_total = sum(new_scores_full) / len(new_scores_full)
    pct_impr_total = (avg_new_total - avg_old_total) / 5 * 100
    affected_old = [old_scores[int(i)] for i in all_new_scores]
    affected_new = [all_new_scores[i] for i in all_new_scores]
    avg_old_affected = sum(affected_old) / len(affected_old)
    avg_new_affected = sum(affected_new) / len(affected_new)
    pct_impr_affected = (avg_new_affected - avg_old_affected) / 5 * 100
    with summary_file.open("w") as sf:
        sf.write(f"{dataset.upper()} – coarse {mode}\n")
        sf.write(f"examples total        : {len(old_scores)}\n")
        sf.write(f"mitigated datapoints  : {len(all_new_scores)}\n")
        sf.write(f"average before (total): {avg_old_total:.3f}\n")
        sf.write(f"average after  (total): {avg_new_total:.3f}\n")
        sf.write(f"change total          : {pct_impr_total:+.1f}%\n\n")
        sf.write(f"average before (affected): {avg_old_affected:.3f}\n")
        sf.write(f"average after  (affected): {avg_new_affected:.3f}\n")
        sf.write(f"change affected          : {pct_impr_affected:+.1f}%\n")
    logging.info(f"[{dataset}] summary written to {summary_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automated evaluation of G-Eval mitigation.")
    parser.add_argument('--dataset', type=str, default='fetaqa', help="Dataset name (fetaqa or qtsumm)")
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help="Model name (e.g., gpt-4o-mini)")
    parser.add_argument('--type', type=str, default='normal', choices=['normal', 'oracle'], help="Mitigation type")
    parser.add_argument('--mode', type=str, default='faithfulness', choices=['faithfulness', 'completeness'], help="Evaluation mode")
    args = parser.parse_args()
    evaluate_mitigation(args.dataset, args.model, args.type, args.mode) 