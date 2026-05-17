import os
import sys

# Processing handling imports
from .linguistic_processor import LinguisticProcessor
from .feature_profiler import FeatureProfiler
from .data_module import SentenceFeatures, DocumentProfile, FeatureStats
from dataclasses import fields

# Model imports
import textstat
import xgboost
import shap

# Sklean imports
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Multiprocessing imports
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

# Miscelaneous imports
import csv
import numpy as np

from config import PathConfig


from typing import List, Dict, Optional

# DocumentFeatures: 
DocumentFeatures = list[float]
CSVData = List[Dict[str, Optional[float]]]
PreLinearRegData = list[list, float]
ValidationMetrics = Dict[str, float]


OUTPUT_DIR_WEIGHT = PathConfig.LOAD_PATH
os.makedirs(OUTPUT_DIR_WEIGHT, exist_ok=True)

OUTPUT_FILENAMES = [f.lower() for f in os.listdir(OUTPUT_DIR_WEIGHT)]

# ------------------------------ WORKER FUNCTIONS ------------------------------
#--------------------------------------------------------------------------------

def init_worker():
    global _GLOBAL_FP
    lp = LinguisticProcessor()
    _GLOBAL_FP = FeatureProfiler(lp)
    
def extract_tabular_features_worker(text: str, row: Dict[str, Optional[float]] | None = None) -> list[float]:
        # Extract a safe float value: returns 0 if not found or if invalid type
        def safe_float(value, default=0.0) -> float:
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
        
        # Backup just incase the row does not have any scores
        if row is None:
            features = [
                safe_float(textstat.flesch_reading_ease(text)),
                safe_float(textstat.flesch_kincaid_grade(text)),
                safe_float(textstat.automated_readability_index(text)),
                safe_float(textstat.smog_index(text)),
                safe_float(textstat.dale_chall_readability_score(text)),
            ]
            return features
        
        # Kaggle predicted scores extracted
        features = [
            safe_float(row.get("flesch_reading_ease")),
            safe_float(row.get("flesch_kincaid_grade")),
            safe_float(row.get("ari")),
            safe_float(row.get("smog")),
            safe_float(row.get("new_dale_chall"))
        ]
        return features

# Worker function for multiprocessing
def extract_row_features_worker(row):
    global _GLOBAL_FP

    text = row["excerpt"]
    label = row["bt_easiness"]

    df = _GLOBAL_FP.extract(text)

    features = []

    for feature in fields(df):
        stats = getattr(df, feature.name)
        if stats is None:
            features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([stats.avg, stats.sd, stats.min, stats.max])
            

    return features, label




# ------------------------------ METRIC CLASS ------------------------------
#--------------------------------------------------------------------------------

class Metric:
    def __init__(self):
        self.model = None
        lp = LinguisticProcessor()
        self.fp = FeatureProfiler(lp)
        self.loaded: bool = False
        self.model_filepath = ""
        self._feature_cache = {}
        
    # Returns a list of all the values of the formulaic scores
    def __extract_tabular_features(self, text: str, row: Dict[str, Optional[float]] | None = None) -> list[float]:
        # Extract a safe float value: returns 0 if not found or if invalid type
        def safe_float(value, default=0.0) -> float:
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
        
        # Backup just incase the row does not have any scores
        if row is None:
            features = [
                safe_float(textstat.flesch_reading_ease(text)),
                safe_float(textstat.flesch_kincaid_grade(text)),
                safe_float(textstat.automated_readability_index(text)),
                safe_float(textstat.smog_index(text)),
                safe_float(textstat.dale_chall_readability_score(text)),
            ]
            return features

        # Kaggle predicted scores extracted
        features = [
            safe_float(row.get("flesch_reading_ease")),
            safe_float(row.get("flesch_kincaid_grade")),
            safe_float(row.get("ari")),
            safe_float(row.get("smog")),
            safe_float(row.get("new_dale_chall"))
        ]
        return features
    
    
    def __normalize_csv_row(self, row: Dict[str, str]) -> Dict[str, Optional[float]]:
        cleaned_row = {k.strip().replace('\n', '').replace('"', ''): v for k, v in row.items()}
        # Returns a clean row of values
        return {
            "id": cleaned_row.get("ID"),
            "excerpt": cleaned_row.get("Excerpt"),
            "bt_easiness": float(cleaned_row.get("BT Easiness")) if cleaned_row.get("BT Easiness") else None,
            "flesch_reading_ease": float(cleaned_row.get("Flesch-Reading-Ease", 0)) if cleaned_row.get("Flesch-Reading-Ease") else None,
            "flesch_kincaid_grade": float(cleaned_row.get("Flesch-Kincaid-Grade-Level", 0)) if cleaned_row.get("Flesch-Kincaid-Grade-Level") else None,
            "ari": float(cleaned_row.get("Automated Readability Index", 0)) if cleaned_row.get("Automated Readability Index") else None,
            "smog": float(cleaned_row.get("SMOG Readability", 0)) if cleaned_row.get("SMOG Readability") else None,
            "new_dale_chall": float(cleaned_row.get("New Dale-Chall Readability Formula", 0)) if cleaned_row.get("New Dale-Chall Readability Formula") else None,
            "kaggle_predictions": {
                "first": float(cleaned_row.get("firstPlace_pred", 0)) if cleaned_row.get("firstPlace_pred") else None,
                "second": float(cleaned_row.get("secondPlace_pred", 0)) if cleaned_row.get("secondPlace_pred") else None,
                "third": float(cleaned_row.get("thirdPlace_pred", 0)) if cleaned_row.get("thirdPlace_pred") else None,
                "fourth": float(cleaned_row.get("fourthPlace_pred", 0)) if cleaned_row.get("fourthPlace_pred") else None,
                "fifth": float(cleaned_row.get("fifthPlace_pred", 0)) if cleaned_row.get("fifthPlace_pred") else None,
                "sixth": float(cleaned_row.get("sixthPlace_pred", 0)) if cleaned_row.get("sixthPlace_pred") else None,
            },
            "kaggle_split": cleaned_row.get("Kaggle split"),
        }

    # Extracts the document features (non worker)
    def __extract_doc_features(self, text) -> DocumentFeatures:

        # Try extracting features; ensure we always return a list
        df: DocumentProfile = self.fp.extract(text)

        features = []
        
        for feature in fields(df):
            stats = getattr(df, feature.name)
            if stats is None:
                features.extend([0.0, 0.0, 0.0, 0.0])
            else:
                features.extend([stats.avg, stats.sd, stats.min, stats.max])

        return features
    
    def __format_csv_data(self, path, train: bool ,max_rows: int = None) -> CSVData:
        """
        Reads a CSV file and returns a list of dictionaries with essential readability data.
        Handles multiline excerpts, quoted fields, cleans header names, and shows live progress.
        """

        excerpt_type = "train" if train else "test"

        data: CSVData = []
        try:
            with open(path, encoding="utf-8", newline='') as file:
                reader = csv.DictReader(file, quotechar='"')
                
                # Clean header names
                reader.fieldnames = [name.strip().replace('\n','').replace('"','') for name in reader.fieldnames]
                
                # Extract training files or testing files based off whats prompted
                filtered_rows = []
                for row in reader:
                    normalized_row = self.__normalize_csv_row(row)
                    if normalized_row.get("kaggle_split") is not None and normalized_row["kaggle_split"].lower() == excerpt_type:
                        filtered_rows.append(normalized_row)
                if max_rows is not None and max_rows > len(filtered_rows):
                    max_rows = len(filtered_rows)
                    
                total_rows = len(filtered_rows)
                if max_rows is not None and total_rows > max_rows:
                    filtered_rows = filtered_rows[:max_rows]

                # Estimate total rows if possible
                count = 0
                for row in filtered_rows:
                    # Essential fields
                    id_ = row.get("id")
                    excerpt = row.get("excerpt")
                    bt_easiness = row.get("bt_easiness")

                    if id_ is None or  excerpt is None or bt_easiness is None:
                        # print(f"Skipping row {row_idx}: missing essential fields")
                        continue

                    data.append(row)
                    count += 1

                    # Print live progress
                    progress = (count / total_rows) * 100
                    sys.stdout.write(f"\rReading CSV: {count}/{total_rows} rows ({progress:.1f}%)")
                    sys.stdout.flush()

        except FileNotFoundError:
            print(f"File not found: {path}")
            return []

        print(f"\nCSV Data Processed! Total valid rows read: {len(data)}")
        return data
    

    def __formated_data_pre_lin_reg(self, csv_data: CSVData) -> PreLinearRegData:
        X = []
        Y = []
        total_rows = len(csv_data)


        print("Starting parallel feature extraction...")

        start_time = time.time()

        results = []
        total = len(csv_data)

        with ProcessPoolExecutor(
            max_workers=os.cpu_count(),
            initializer=init_worker
        ) as executor:
            futures = [executor.submit(extract_row_features_worker, row) for row in csv_data]

            for i, future in enumerate(as_completed(futures), start=1):
                features, label = future.result()
                results.append((features, label))

                progress = (i / total) * 100
                sys.stdout.write(f"\rExtracting features: {i}/{total} rows ({progress:.1f}%)")
                sys.stdout.flush()

        end_time = time.time()

        print(f"\nFeature extraction complete in {end_time - start_time:.2f}s")

        for idx, (features, label) in enumerate(results, start=1):
            if not features:
                print(f"Skipping row {idx}: no features extracted")
                continue

            X.append(features)
            Y.append(label)

            progress_percent = (idx / total_rows) * 100
            sys.stdout.write(f"\rExtracting features: {idx}/{total_rows} rows ({progress_percent:.1f}%)")
            sys.stdout.flush()

        print("\nFeature extraction complete.")

        if not X:
            print("Warning: No valid data available for linear regression!")

        return list(zip(X, Y))
                
        
    def load_model(self, weight_path):
        
        model = xgboost.XGBRegressor()
        model.load_model(weight_path)
        
        self.model = model
        self.loaded = True
        self.model_filepath = weight_path
        
    def train(self, csv_path, samples: int | None = None):
        print("\nTRAINING PROCESS STARTING: ")
        csv_data: CSVData = self.__format_csv_data(csv_path, True ,samples)
        pre_lin_data: PreLinearRegData = self.__formated_data_pre_lin_reg(csv_data)

        # Unzip pre_lin_data into X and y
        X, y = zip(*pre_lin_data)  # X = list of feature vectors, y = list of annotated scores

        X = list(X)
        y = list(y)

        # model_config = dict(
        #     objective="reg:squarederror",
        #     n_estimators=4000,
        #     learning_rate=0.015,
        #     max_depth=4,
        #     min_child_weight=3,
        #     subsample=0.9,
        #     colsample_bytree=0.8,
        #     gamma=0.1,
        #     reg_alpha=0.15,
        #     reg_lambda=2.0,
        #     random_state=42,
        #     eval_metric="mae",
        # )
        
        model_config = dict(
            objective="reg:squarederror",
            n_estimators=4000,
            learning_rate=0.015,
            max_depth=3,           # keep shallow — 215 features makes deep trees very noisy
            min_child_weight=5,    # up from 3
            subsample=0.8,
            colsample_bytree=0.3,  # down hard from 0.8 — critical with 215 features
                                # each tree sees ~65 features, forcing real diversity
            colsample_bylevel=0.7, # add — extra randomization per level
            colsample_bynode=0.7,  # add — per-split sampling on top
            gamma=0.3,             # up from 0.1
            reg_alpha=0.5,         # up significantly — L1 will aggressively prune dead features
            reg_lambda=2.0,        # keep
            random_state=42,
            eval_metric="mae",
        )

        if len(X) >= 20:
            model = xgboost.XGBRegressor(
                **model_config,
                early_stopping_rounds=50,
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model = xgboost.XGBRegressor(**model_config)
            model.fit(X, y, verbose=False)

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
        
    
    def score(self, text: str, row: Dict[str, Optional[float]] | None = None, debug = False) -> float:
        text_features = self.__extract_doc_features(text, row)
        predicted = self.model.predict([text_features])  # sklearn expects 2D array
        return predicted[0]
    
    def __get_feature_names(self) -> list[str]:
        """
        Builds feature names in the exact order features are appended
        in __extract_doc_features / extract_row_features_worker.
        """
        # Part 1: DocumentProfile fields (in dataclass field order)
        doc_feature_names = [f.name for f in fields(DocumentProfile)]

        # Part 2: tabular features (same order as __extract_tabular_features with a row)
        tabular_names = [
            "flesch_reading_ease",
            "flesch_kincaid_grade",
            "ari",
            "smog",
            "new_dale_chall"
        ]

        return doc_feature_names + tabular_names


    def __print_shap_summary(self, shap_values: np.ndarray, X: np.ndarray, top_n: int = 20):
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        ranked = np.argsort(mean_abs_shap)[::-1]
        feature_names = self.__get_feature_names()

        print(f"\n{'='*65}")
        print(f"  SHAP Feature Importance (top {top_n} of {len(mean_abs_shap)})")
        print(f"{'='*65}")
        print(f"  {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':<14} {'Mean Value'}")
        print(f"  {'-'*60}")

        for rank, idx in enumerate(ranked[:top_n], start=1):
            name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
            mean_val = X[:, idx].mean()
            print(f"  {rank:<6} {name:<35} {mean_abs_shap[idx]:<14.4f} {mean_val:.4f}")

        print(f"{'='*65}\n")

        dead = np.sum(mean_abs_shap < 0.001)
        print(f"  Features with mean |SHAP| < 0.001: {dead}/{len(mean_abs_shap)}")

        # Print dead feature names explicitly so you know what to cut
        if dead > 0:
            print("  Dead features:")
            for idx in np.where(mean_abs_shap < 0.001)[0]:
                name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
                print(f"    - {name}")
        print()
    
    def validate(self, path: str, max_validate: int | None = None) -> ValidationMetrics:
        csv_data: CSVData = self.__format_csv_data(path, False, max_validate)

        predictions = []
        expected_values = []
        all_features = []

        for row in csv_data:
            excerpt = row.get("excerpt")
            expected_score = float(row.get("bt_easiness"))

            text_features = self.__extract_doc_features(excerpt)
            if text_features is None:
                continue
            predicted_score = self.model.predict([text_features])[0]

            print("=====================\n\n")
            print(excerpt + "\n\n\n")
            print(f"---EXPECTED: {expected_score} ----PREDICTED: {predicted_score}")
            print("=====================\n\n")

            predictions.append(float(predicted_score))
            expected_values.append(expected_score)
            all_features.append(text_features)

            total_rows = max_validate if max_validate is not None else len(csv_data)
            current_mae = mean_absolute_error(expected_values, predictions)
            progress = (len(predictions) / total_rows) * 100 if total_rows else 0.0
            msg = f"\rTesting features: {len(predictions)}/{total_rows} rows ({progress:.1f}%) Current MAE: {current_mae:.4f}"
            sys.stdout.write(msg.ljust(80))
            sys.stdout.flush()

        if not predictions:
            print("\nNo validation data available. Check your dataset filtering.")
            return {"count": 0.0, "mae": 0.0, "rmse": 0.0, "r2": 0.0}

        mae = mean_absolute_error(expected_values, predictions)
        rmse = mean_squared_error(expected_values, predictions) ** 0.5
        r2 = r2_score(expected_values, predictions)

        # --- SHAP ---
        print("\n\nComputing SHAP values...")
        X_array = np.array(all_features)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_array)  # shape: (n_samples, n_features)

        self.__print_shap_summary(shap_values, X_array)

        return {
            "count": float(len(predictions)),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "shap_values": shap_values,    
            "shap_base_value": float(explainer.expected_value),
        }


