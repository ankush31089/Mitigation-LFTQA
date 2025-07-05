import pathlib
import json
from typing import List, Dict

DATA_DIR = pathlib.Path("data/outputs")
CKPT_DIR_FAITH = pathlib.Path("g_eval/faithfulness_scores")
CKPT_DIR_COMP = pathlib.Path("g_eval/completeness_scores")
MITIG_DIR = pathlib.Path("g_eval/mitigation_outputs/normal")
ORACLE_MIT_DIR = pathlib.Path("g_eval/mitigation_outputs/oracle")
# Automated eval checkpoints structure
AE_CKPT_DIR_NORMAL_FAITH = pathlib.Path("g_eval/automated_eval_checkpoints/normal/faithfulness")
AE_CKPT_DIR_NORMAL_COMP = pathlib.Path("g_eval/automated_eval_checkpoints/normal/completeness")
AE_CKPT_DIR_ORACLE_FAITH = pathlib.Path("g_eval/automated_eval_checkpoints/oracle/faithfulness")
AE_CKPT_DIR_ORACLE_COMP = pathlib.Path("g_eval/automated_eval_checkpoints/oracle/completeness")
# Results should be in /results/geval_automated_eval/...
RESULTS_DIR_NORMAL_FAITH = pathlib.Path("results/geval_automated_eval/normal/faithfulness")
RESULTS_DIR_NORMAL_COMP = pathlib.Path("results/geval_automated_eval/normal/completeness")
RESULTS_DIR_ORACLE_FAITH = pathlib.Path("results/geval_automated_eval/oracle/faithfulness")
RESULTS_DIR_ORACLE_COMP = pathlib.Path("results/geval_automated_eval/oracle/completeness")

for p in (
    AE_CKPT_DIR_NORMAL_FAITH, AE_CKPT_DIR_NORMAL_COMP,
    AE_CKPT_DIR_ORACLE_FAITH, AE_CKPT_DIR_ORACLE_COMP,
    RESULTS_DIR_NORMAL_FAITH, RESULTS_DIR_NORMAL_COMP,
    RESULTS_DIR_ORACLE_FAITH, RESULTS_DIR_ORACLE_COMP
):
    p.mkdir(parents=True, exist_ok=True)

def load_coarse_scores(dataset: str, model: str, mode: str = "faithfulness") -> list:
    assert mode in {"faithfulness", "completeness"}, "Invalid mode"
    ckpt_dir = CKPT_DIR_FAITH if mode == "faithfulness" else CKPT_DIR_COMP
    ckpt_file = ckpt_dir / f"{model}_{dataset}.json"
    with ckpt_file.open() as f:
        ck = json.load(f)
    key = f"{mode}_scores"
    if key not in ck:
        raise KeyError(f"'{key}' not found in {ckpt_file}")
    return ck[key]

def load_dataset_rows(dataset: str) -> list:
    file_path = DATA_DIR / f"model_outputs_with_scores_{dataset}.json"
    return json.load(file_path.open())

def load_oracle_coarse_scores(dataset: str, mode: str = "faithfulness") -> list:
    assert mode in {"faithfulness", "completeness"}, "Invalid mode"
    file_path = DATA_DIR / f"model_outputs_with_scores_{dataset}.json"
    with file_path.open() as f:
        entries = json.load(f)
    score_key = f"{mode}_score"
    return [entry[score_key] for entry in entries if score_key in entry] 