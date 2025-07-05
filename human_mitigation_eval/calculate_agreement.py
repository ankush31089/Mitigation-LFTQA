import pandas as pd
import krippendorff
import os

def calculate_aggregated_alpha():
    """
    Calculates aggregated Krippendorff's alpha for 'flabel' and 'clabel'
    across multiple datasets from two annotators.

    This script must be run from the 'human_mitigation_eval/' directory.
    """
    # --- Configuration ---
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct paths relative to the script's location
    annotator1_dir = os.path.join(script_dir, 'annotated', 'annotator1')
    annotator2_dir = os.path.join(script_dir, 'annotated', 'annotator2')
    
    # List of CSV files to process in the annotator directories
    dataset_files = ['gpt-4o_qtsumm.csv', 'gpt-4o_fetaqa.csv']

    # --- Data Accumulators ---
    # Lists to hold all labels for annotator 1 and annotator 2
    flabels_rater1, flabels_rater2 = [], []
    clabels_rater1, clabels_rater2 = [], []

    print("Starting agreement calculation...")

    # --- Loop through each dataset to collect annotations ---
    for filename in dataset_files:
        path1 = os.path.join(annotator1_dir, filename)
        path2 = os.path.join(annotator2_dir, filename)

        try:
            print(f"  - Processing '{filename}'...")
            df1 = pd.read_csv(path1)
            df2 = pd.read_csv(path2)
        except FileNotFoundError as e:
            print(f"Error: Could not find a file. {e}")
            print("Please ensure the script is run from the 'human_mitigation_eval/' directory.")
            return

        # Merge dataframes on 'original_idx' to ensure perfect row alignment
        merged_df = pd.merge(
            df1,
            df2,
            on='original_idx',
            suffixes=('_1', '_2'),
            how='inner' # Only compare rows present in both files
        )

        if merged_df.empty:
            print(f"Warning: No matching 'original_idx' found in '{filename}'. Skipping.")
            continue
            
        # 1. Aggregate 'flabel' data (lftqa_flabel + mtraig_flabel)
        flabels_rater1.extend(merged_df['lftqa_flabel_1'].tolist())
        flabels_rater1.extend(merged_df['mtraig_flabel_1'].tolist())
        
        flabels_rater2.extend(merged_df['lftqa_flabel_2'].tolist())
        flabels_rater2.extend(merged_df['mtraig_flabel_2'].tolist())

        # 2. Aggregate 'clabel' data (lftqa_clabel only)
        # Note: Your request mentioned 'cftqa_flabel' for fetaqa, which seems like a typo.
        # This script assumes you meant 'lftqa_clabel' for both datasets.
        clabels_rater1.extend(merged_df['lftqa_clabel_1'].tolist())
        clabels_rater2.extend(merged_df['lftqa_clabel_2'].tolist())

    # --- Sanity Checks and Final Calculations ---
    if not flabels_rater1 or not clabels_rater1:
        print("\nError: No data was collected. Cannot calculate alpha.")
        return

    # Prepare final data structures for the krippendorff library
    # Format: [[rater1_labels], [rater2_labels]]
    final_flabel_data = [flabels_rater1, flabels_rater2]
    final_clabel_data = [clabels_rater1, clabels_rater2]
    
    print("\n--- Aggregated Data Summary ---")
    print(f"Total 'flabel' comparisons: {len(flabels_rater1)}")
    print(f"Total 'clabel' comparisons: {len(clabels_rater1)}")

    # Calculate Krippendorff's alpha for each aggregated set
    alpha_flabel = krippendorff.alpha(
        reliability_data=final_flabel_data,
        level_of_measurement='nominal'
    )
    alpha_clabel = krippendorff.alpha(
        reliability_data=final_clabel_data,
        level_of_measurement='nominal'
    )

    # --- Print Final Results ---
    print("\n--- Final Krippendorff's Alpha Scores ---")
    print(f"Overall 'flabel' Agreement (across all datasets): {alpha_flabel:.4f}")
    print(f"Overall 'clabel' Agreement (across all datasets): {alpha_clabel:.4f}")


if __name__ == "__main__":
    calculate_aggregated_alpha()