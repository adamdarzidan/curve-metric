import pandas as pd
from components.metric import Metric
import json
import sys
import numpy as np
import textstat
from .config import DOCS_PATH, MODEL_PATH, DATAFRAME_PATH


from .scraper import process_hospital_documents


DOCS_INDICES = ["title", "url", "score", "features", "shap"]
SCORE_INDICES = ["custom", "flesch-kincaid-ease", "flesch-kincaid-level", "smog", "coleman-liau", "spache", "linsear-write", "gunning-fog", "dale-chall"]
COLUMNS = ["name", "directory", "state", "type", "rank", "title", "url"] + SCORE_INDICES + ["features", "shap"]


def get_scores_helper(m: Metric, text: str) -> tuple[np.ndarray, dict, np.ndarray]:
    score, features, shap = m.score(text)
    scores = [
        score,                        # custom
        textstat.flesch_reading_ease(text),    # flesch-kincaid-ease
        textstat.flesch_kincaid_grade(text),   # flesch-kincaid-level
        textstat.smog_index(text),             # smog
        textstat.coleman_liau_index(text),     # coleman-liau
        textstat.spache_readability(text),     # spache
        textstat.linsear_write_formula(text),  # linsear-write
        textstat.gunning_fog(text),            # gunning-fog
        textstat.dale_chall_readability_score(text)  # dale-chall
    ]
    return np.array(scores, dtype=float), features, shap

# Assumes text has been preprocessed 
def retrieve_scores(m: Metric, text):
    score, features, shap = get_scores_helper(m, text)
    scores = pd.Series(score, index=SCORE_INDICES)
    return scores, features, shap


# def rescore(df: pd.DataFrame):
#     for h in data.get("hospitals", []):
#             # docs per hospital with series of (title, url, text, formatted, score)
#             process_hospital_documents(h, DOCS_PATH)
#             # For each document in the hospital dictionary
#             for d in h.get("documents", []):
#                 # Get text from the formatted document and send to get scores
#                 try:
#                     # Transform title to file name: remove hospital suffix, replace dashes with underscores, add .txt
#                     hospital_suffix = f"-{h['name'].lower().replace(' ', '-')}"
#                     file_title = d["title"].replace(hospital_suffix, "").replace("-", "_") + ".txt"
#                     file_path = f"{DOCS_PATH}{h['directory']}{file_title}"
#                     print(file_path)
#                     with open(file_path) as txtf:
#                         text = txtf.read()
#                         scores, features, shap = retrieve_scores(m, text)
#                         print(f"{scores}\n{features}\n{shap}")
#                 except FileNotFoundError:
#                     print("File not found... skipping file")
#                     continue

def process_text(path: str, save=True) -> pd.DataFrame:
    # Read the data from the file
    m = Metric()
    m.load_model(MODEL_PATH)
    df = pd.DataFrame(columns=COLUMNS)
    try:
        with open(path) as f:
            data = json.load(f)
        for h in data.get("hospitals", []):
            # docs per hospital with series of (title, url, text, formatted, score)
            process_hospital_documents(h, DOCS_PATH)
            # For each document in the hospital dictionary
            for d in h.get("documents", []):
                # Get text from the formatted document and send to get scores
                try:
                    # Transform title to file name: remove hospital suffix, replace dashes with underscores, add .txt
                    hospital_suffix = f"-{h['name'].lower().replace(' ', '-')}"
                    file_title = d["title"].replace(hospital_suffix, "").replace("-", "_") + ".txt"
                    file_path = f"{DOCS_PATH}{h['directory']}{file_title}"
                    print(file_path)
                    with open(file_path) as txtf:
                        text = txtf.read()
                        scores, features, shap = retrieve_scores(m, text)
                        print(f"{scores}\n{features}\n{shap}")
                except FileNotFoundError:
                    print("File not found... skipping file")
                    continue
                
                # Append the row for this document
                row = [h["name"], h["directory"], h["state"], h["type"], h["rank"], d["title"], d["url"]] + list(scores.values) + [features, shap]
                df.loc[len(df)] = row
                
            
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
            
    except FileNotFoundError:
        print("Hospital data file does not exist")
        sys.exit()
        
    # Create metric object to record scores
    if save:
        filename = input("Please enter filename to save data:\n")
        df.to_csv(f"{DATAFRAME_PATH}{filename}.csv")
        
    return df
    
    
    
   
    
    