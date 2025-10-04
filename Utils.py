# utils.py
import pandas as pd

def load_csv(path):
    return pd.read_csv(path)

def preview_prompts(path, n=10, prompt_col="prompt", text_col="text"):
    df = load_csv(path)
    if prompt_col in df.columns:
        return df[[prompt_col, text_col]].head(n)
    else:
        # if only a text column exists, show first n rows
        return df.head(n)
      
