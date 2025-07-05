# import pandas as pd
# from collections import Counter, defaultdict
# import argparse
# from pathlib import Path

# def count_label_frequencies(csv_file_path):
#     df = pd.read_csv(csv_file_path)
#     target_columns = ['lftqa_flabel', 'lftqa_clabel', 'mtraig_flabel']
#     valid_labels = {'C', 'I', 'S', 'D'}
#     label_counts = defaultdict(lambda: {label: 0 for label in valid_labels})
#     for column in target_columns:
#         if column in df.columns:
#             column_counts = Counter(df[column])
#             for label in valid_labels:
#                 label_counts[column][label] = column_counts.get(label, 0)
#         else:
#             print(f"Warning: Column {column} not found in CSV.")
#     return dict(label_counts)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Count label frequencies in a mitigation eval CSV file.")
#     parser.add_argument('--csv_file', type=str, default="human_mitigation_eval/consolidated_annotations/gpt-4o_fetaqa.csv", help='Path to the CSV file (relative to official_repo)')
#     args = parser.parse_args()
#     repo_root = Path(__file__).parent.parent
#     csv_path = repo_root / args.csv_file
#     result = count_label_frequencies(csv_path)
#     for col, counts in result.items():
#         print(f"\nLabel counts for {col}:")
#         for label, count in counts.items():
#             print(f"  {label}: {count}") 

import pandas as pd
from collections import Counter, defaultdict
import argparse
from pathlib import Path

def count_label_frequencies(csv_file_path):
    df = pd.read_csv(csv_file_path)
    target_columns = [
        'geval_faithfulness_label',
        'geval_completeness_label',
        'mtraig_eval_faithfulness_label'
    ]
    valid_labels = {
        'Fully Factual',
        'Fully Complete',
        'Improved',
        'Unchanged',
        'Deteriorated'
    }
    label_counts = defaultdict(lambda: {label: 0 for label in valid_labels})
    for column in target_columns:
        if column in df.columns:
            column_counts = Counter(df[column])
            for label in valid_labels:
                label_counts[column][label] = column_counts.get(label, 0)
        else:
            print(f"Warning: Column {column} not found in CSV.")
    return dict(label_counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count label frequencies in a mitigation eval CSV file.")
    parser.add_argument('--csv_file', type=str, default="human_mitigation_eval/consolidated_annotations/gpt-4o_fetaqa.csv", help='Path to the CSV file (relative to official_repo)')
    args = parser.parse_args()
    repo_root = Path(__file__).parent.parent
    csv_path = repo_root / args.csv_file
    result = count_label_frequencies(csv_path)
    for col, counts in result.items():
        print(f"\nLabel counts for {col}:")
        for label, count in counts.items():
            print(f"  {label}: {count}")
