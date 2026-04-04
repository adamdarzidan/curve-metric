import os
import sys

from .linguistic_processor import LinguisticProcessor
from .feature_profiler import FeatureProfiler
from .data_module import SentenceFeatures, DocumentProfile
from dataclasses import fields
from typing import List, Dict, Optional
import numpy as np
import xgboost
import shap

import json
import csv
import dataclasses


DocumentFeatures = list[float]
CSVData = List[Dict[str, Optional[float]]]
PreLinearRegData = list[list, float]

OUTPUT_DIR_WEIGHT = "weights/"
os.makedirs(OUTPUT_DIR_WEIGHT, exist_ok=True)

OUTPUT_FILENAMES = [f.lower() for f in os.listdir(OUTPUT_DIR_WEIGHT)]

class Metric:
    def __init__(self, profile: list[int]):
        self.profile = profile
        self.model = None
        lp = LinguisticProcessor()
        self.fp = FeatureProfiler(lp)
        self.loaded: bool = False
        self.model_filepath = ""
        
    def __extract_doc_features(self, text, return_type=None) -> DocumentFeatures:

        # Try extracting features; ensure we always return a list
        df: DocumentProfile = self.fp.extract(text)
        if df is None:
            df = [] 

        document_features = []
        
        for feature in fields(df):
            stats = getattr(df, feature.name)
            if stats is None:
                continue
            document_features.append(stats.avg)
            

        return document_features
    
    def __prepare_train_csv(self, path, max_rows: int = None) -> CSVData:
        """
        Reads a CSV file and returns a list of dictionaries with essential readability data.
        Handles multiline excerpts, quoted fields, cleans header names, and shows live progress.
        """

        data: CSVData = []
        try:
            with open(path, encoding="utf-8", newline='') as file:
                reader = csv.DictReader(file, quotechar='"')
                
                # Clean header names
                reader.fieldnames = [name.strip().replace('\n','').replace('"','') for name in reader.fieldnames]

                # Estimate total rows if possible
                all_rows = list(reader)
                total_rows = len(all_rows) if max_rows is None else min(len(all_rows), max_rows)

                count = 0
                for row_idx, row in enumerate(all_rows, start=1):
                    # Stop if max_rows reached
                    if max_rows is not None and count >= max_rows:
                        break

                    # Clean row keys
                    row = {k.strip().replace('\n','').replace('"',''): v for k,v in row.items()}

                    # Essential fields
                    id_ = row.get("ID")
                    excerpt = row.get("Excerpt")
                    bt_easiness = row.get("BT Easiness")

                    if not id_ or not excerpt or not bt_easiness:
                        # print(f"Skipping row {row_idx}: missing essential fields")
                        continue
                    try:
                        bt_val = float(bt_easiness)
                    except ValueError:
                        # print(f"Skipping row {row_idx}: invalid BT Easiness '{bt_easiness}'")
                        continue

                    entry = {
                        "id": id_,
                        "excerpt": excerpt,
                        "bt_easiness": bt_val,
                        "flesch_reading_ease": float(row.get("Flesch-Reading-Ease", 0)) if row.get("Flesch-Reading-Ease") else None,
                        "flesch_kincaid_grade": float(row.get("Flesch-Kincaid-Grade-Level", 0)) if row.get("Flesch-Kincaid-Grade-Level") else None,
                        "ari": float(row.get("Automated Readability Index", 0)) if row.get("Automated Readability Index") else None,
                        "smog": float(row.get("SMOG Readability", 0)) if row.get("SMOG Readability") else None,
                        "new_dale_chall": float(row.get("New Dale-Chall Readability Formula", 0)) if row.get("New Dale-Chall Readability Formula") else None,
                        "kaggle_predictions": {
                            "first": float(row.get("firstPlace_pred", 0)) if row.get("firstPlace_pred") else None,
                            "second": float(row.get("secondPlace_pred", 0)) if row.get("secondPlace_pred") else None,
                            "third": float(row.get("thirdPlace_pred", 0)) if row.get("thirdPlace_pred") else None,
                            "fourth": float(row.get("fourthPlace_pred", 0)) if row.get("fourthPlace_pred") else None,
                            "fifth": float(row.get("fifthPlace_pred", 0)) if row.get("fifthPlace_pred") else None,
                            "sixth": float(row.get("sixthPlace_pred", 0)) if row.get("sixthPlace_pred") else None,
                        },
                        "kaggle_split": row.get("Kaggle split")
                    }

                    data.append(entry)
                    count += 1

                    # Print live progress
                    progress = (count / total_rows) * 100
                    sys.stdout.write(f"\rReading CSV: {count}/{total_rows} rows ({progress:.1f}%)")
                    sys.stdout.flush()

        except FileNotFoundError:
            print(f"File not found: {path}")
            return []

        print(f"\nTotal valid rows read: {len(data)}")
        return data
    

    def __formated_data_pre_lin_reg(self, csv_data: CSVData) -> PreLinearRegData:
        """
        Converts CSVData into (X_vector, Y_value) tuples for linear regression.
        Flattens sentence-level features into a single vector per text.
        Skips rows with empty feature extraction.
        """
        X = []
        Y = []
        total_rows = len(csv_data)

        for idx, line in enumerate(csv_data, start=1):
            text = line["excerpt"]
            annotated_readability = line["bt_easiness"]
            df: DocumentFeatures = self.__extract_doc_features(text, return_type=list)  # list of floats
            
            if not df:
                print(f"Skipping row {idx}: no features extracted")
                continue

            X.append(df)
            Y.append(annotated_readability)
            progress_percent = (idx / total_rows) * 100
            sys.stdout.write(f"\rExtracting features: {idx}/{total_rows} rows ({progress_percent:.1f}%)")
            sys.stdout.flush()

        print("\nFeature extraction complete.")

        if not X:
            print("Warning: No valid data available for linear regression!")

        return list(zip(X, Y))
                
        
    def load_model(self, weight_path):
        try:
            with open(weight_path, "r", encoding="utf-8") as f:
                weights_dict = json.load(f)
        except FileNotFoundError:
           print(f"weight_path entered does not exist: {weight_path}")

        model = xgboost.XGBRegressor()
        model.load_model(weight_path)
        
        self.model = model
        self.loaded = True
        self.model_filepath = weight_path
        
    def train(self, csv_path, samples=500):
        print("\nTRAINING PROCESS STARTING: ")
        csv_data: CSVData = self.__prepare_train_csv(csv_path, samples)
        pre_lin_data: PreLinearRegData = self.__formated_data_pre_lin_reg(csv_data)

        # Unzip pre_lin_data into X and y
        X, y = zip(*pre_lin_data)  # X = list of feature vectors, y = list of annotated scores

        X = list(X)
        y = list(y)

        model = xgboost.XGBRegressor().fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        
        
        file_name = input("\n\nEnter file name for weights: ")
        while(file_name == "" or file_name in OUTPUT_FILENAMES):
            if file_name in OUTPUT_FILENAMES:
                file_name = input(f"ERROR: Filename {file_name} already exists. Enter different name")
            else:
                file_name = input("Enter a valid filename")
                        
        self.model_filepath = file_name
        
        file_name = os.path.join(OUTPUT_DIR_WEIGHT, file_name) + ".json"
        
        model.save_model(file_name)

        # Saves model
        self.model = model
        self.loaded = True

        print(f"\n\nTRAINING COMPLETE! File created as {file_name}.json\n")
        
    
    def score(self, text: str, debug = False) -> float:
        text_features = self.__extract_doc_features(text)
        predicted = self.model.predict([text_features])  # sklearn expects 2D array
        return predicted[0]
    
        
    def get_sentence_scores(self, document: str ):
        sentences = document.split(".")
        results = []
        
        for span in sentences:
            score = self.score(span)
            results.append({"sentence": span, "score": score})
            
        return results
    
    
