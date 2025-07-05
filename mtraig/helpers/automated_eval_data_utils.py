import json

def load_faithfulness_scores_from_ckpt(filepath: str) -> list[float]:
    """
    Given a JSON file with structure:
    {
      "last_idx": ...,
      "detailed_results": [
        { "faithfulness_score": float, ... },
        ...
      ]
    }
    Returns a list of faithfulness scores aligned with dataset indices.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if "detailed_results" not in data:
        raise ValueError("Missing 'detailed_results' key in checkpoint file.")
    scores = []
    for entry in data["detailed_results"]:
        if "faithfulness_score" not in entry:
            raise ValueError("Missing 'faithfulness_score' in an entry.")
        scores.append(entry["faithfulness_score"])
    return scores 