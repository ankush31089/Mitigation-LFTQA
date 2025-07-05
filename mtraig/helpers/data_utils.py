import os
import json
import logging
import pandas as pd

def load_human_faith_scores(filename: str):
    DATA_DIR = "data/outputs"
    DATA_PATH = os.path.join(DATA_DIR, filename)
    if not os.path.exists(DATA_PATH):
        logging.error(f"Data file not found at {DATA_PATH}")
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    logging.info(f"Loading data from {DATA_PATH}")
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    logging.info(f"Dataset contains {len(df)} entries")
    if 'faithfulness_score' not in df.columns:
        raise KeyError("'faithfulness_score' column not found in the dataset.")
    if 'fetaqa' in filename:
        df['schema'] = df['metadata'].apply(lambda x: x['table_array'][0])
        df['serialized_table'] = df['metadata'].apply(lambda x: {
            'title': f"{x['table_page_title']} - {x['table_section_title']}",
            'header': x['table_array'][0],
            'rows': x['table_array'][1:]
        })
    else:
        df['schema'] = df['metadata'].apply(lambda x: x['table']['header'])
        df['serialized_table'] = df['metadata'].apply(lambda x: {
            'title': x['table']['title'],
            'header': x['table']['header'],
            'rows': x['table']['rows']
        })
    human_faith = df['faithfulness_score'].tolist()
    return df, human_faith 