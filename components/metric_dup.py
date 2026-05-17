import os
import sys

# Processing handling imports
from .linguistic_processor import LinguisticProcessor
from .feature_profiler import FeatureProfiler
from .data_module import  DocumentProfile, FeatureStats
from dataclasses import fields

# Model imports
import textstat
import xgboost
import shap
import numpy as np

# Sklearn imports
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# Deep learning imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Multiprocessing imports
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Miscellaneous imports
import csv

from config import PathConfig
from typing import List, Dict, Optional, Literal

# ----------------------------  TYPE ALIASES  ----------------------------

DocumentFeatures  = list[float]
CSVData           = List[Dict[str, Optional[float]]]
PreLinearRegData  = list[list, float]
ValidationMetrics = Dict[str, float]

ModelType = Literal["linreg", "xgboost", "deeplearning"]

# ----------------------------  OUTPUT DIRS  ----------------------------

_BASE_DIR  = PathConfig.LOAD_PATH
_MODEL_DIRS: Dict[ModelType, str] = {
    "linreg":      os.path.join(_BASE_DIR, "linreg"),
    "xgboost":     os.path.join(_BASE_DIR, "xgboost"),
    "deeplearning": os.path.join(_BASE_DIR, "deeplearning"),
}
for _d in _MODEL_DIRS.values():
    os.makedirs(_d, exist_ok=True)


# ----------------------------  DEEP LEARNING  ----------------------------

class _ReadabilityMLP(nn.Module):
    """Feed-forward MLP for readability regression."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ----------------------------  WORKER FUNCTIONS  ----------------------------

def init_worker():
    global _GLOBAL_FP
    lp = LinguisticProcessor()
    _GLOBAL_FP = FeatureProfiler(lp)


def extract_tabular_features_worker(
    text: str, row: Dict[str, Optional[float]] | None = None
) -> list[float]:
    def safe_float(value, default=0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    if row is None:
        return [
            safe_float(textstat.flesch_reading_ease(text)),
            safe_float(textstat.flesch_kincaid_grade(text)),
            safe_float(textstat.automated_readability_index(text)),
            safe_float(textstat.smog_index(text)),
            safe_float(textstat.dale_chall_readability_score(text)),
        ]

    return [
        safe_float(row.get("flesch_reading_ease")),
        safe_float(row.get("flesch_kincaid_grade")),
        safe_float(row.get("ari")),
        safe_float(row.get("smog")),
        safe_float(row.get("new_dale_chall")),
    ]


def extract_row_features_worker(row):
    global _GLOBAL_FP
    text  = row["excerpt"]
    label = row["bt_easiness"]
    df    = _GLOBAL_FP.extract(text)

    features = []
    for feature in fields(df):
        stats = getattr(df, feature.name)
        features.append(0.0 if stats is None else float(stats.avg))

    features.extend(extract_tabular_features_worker(text, row))
    return features, label


# ----------------------------  METRIC CLASS  ----------------------------

class Metric:
    """
    Unified readability-scoring model wrapper.

    Parameters
    ----------
    model_type : "linreg" | "xgboost" | "deeplearning"
        Which model architecture to use.  Determines the save/load directory
        and the underlying model object.
    """

    def __init__(self, model_type: ModelType = "xgboost"):
        if model_type not in ("linreg", "xgboost", "deeplearning"):
            raise ValueError(f"Unknown model_type '{model_type}'. "
                             "Choose from: linreg, xgboost, deeplearning")

        self.model_type: ModelType = model_type
        self.model      = None
        self._scaler    = None          # used by deeplearning
        self.loaded: bool = False
        self.model_filepath: str = ""

        lp       = LinguisticProcessor()
        self.fp  = FeatureProfiler(lp)
        self._feature_cache: dict = {}

        self._output_dir = _MODEL_DIRS[model_type]

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def __extract_tabular_features(
        self, text: str, row: Dict[str, Optional[float]] | None = None
    ) -> list[float]:
        def safe_float(value, default=0.0) -> float:
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        if row is None:
            return [
                safe_float(textstat.flesch_reading_ease(text)),
                safe_float(textstat.flesch_kincaid_grade(text)),
                safe_float(textstat.automated_readability_index(text)),
                safe_float(textstat.smog_index(text)),
                safe_float(textstat.dale_chall_readability_score(text)),
            ]

        return [
            safe_float(row.get("flesch_reading_ease")),
            safe_float(row.get("flesch_kincaid_grade")),
            safe_float(row.get("ari")),
            safe_float(row.get("smog")),
            safe_float(row.get("new_dale_chall")),
        ]

    def __normalize_csv_row(self, row: Dict[str, str]) -> Dict[str, Optional[float]]:
        cleaned_row = {
            k.strip().replace('\n', '').replace('"', ''): v for k, v in row.items()
        }
        return {
            "id":      cleaned_row.get("ID"),
            "excerpt": cleaned_row.get("Excerpt"),
            "bt_easiness": (
                float(cleaned_row["BT Easiness"])
                if cleaned_row.get("BT Easiness") else None
            ),
            "flesch_reading_ease": (
                float(cleaned_row["Flesch-Reading-Ease"])
                if cleaned_row.get("Flesch-Reading-Ease") else None
            ),
            "flesch_kincaid_grade": (
                float(cleaned_row["Flesch-Kincaid-Grade-Level"])
                if cleaned_row.get("Flesch-Kincaid-Grade-Level") else None
            ),
            "ari": (
                float(cleaned_row["Automated Readability Index"])
                if cleaned_row.get("Automated Readability Index") else None
            ),
            "smog": (
                float(cleaned_row["SMOG Readability"])
                if cleaned_row.get("SMOG Readability") else None
            ),
            "new_dale_chall": (
                float(cleaned_row["New Dale-Chall Readability Formula"])
                if cleaned_row.get("New Dale-Chall Readability Formula") else None
            ),
            "kaggle_predictions": {
                "first":  float(cleaned_row["firstPlace_pred"])  if cleaned_row.get("firstPlace_pred")  else None,
                "second": float(cleaned_row["secondPlace_pred"]) if cleaned_row.get("secondPlace_pred") else None,
                "third":  float(cleaned_row["thirdPlace_pred"])  if cleaned_row.get("thirdPlace_pred")  else None,
                "fourth": float(cleaned_row["fourthPlace_pred"]) if cleaned_row.get("fourthPlace_pred") else None,
                "fifth":  float(cleaned_row["fifthPlace_pred"])  if cleaned_row.get("fifthPlace_pred")  else None,
                "sixth":  float(cleaned_row["sixthPlace_pred"])  if cleaned_row.get("sixthPlace_pred")  else None,
            },
            "kaggle_split": cleaned_row.get("Kaggle split"),
        }

    def __extract_doc_features(
        self, text: str, row: Dict[str, Optional[float]] | None = None
    ) -> DocumentFeatures:
        df: DocumentProfile = self.fp.extract(text)
        document_features = []
        for feature in fields(df):
            stats = getattr(df, feature.name)
            document_features.append(0.0 if stats is None else float(stats.avg))
        document_features.extend(self.__extract_tabular_features(text, row))
        return document_features

    def __format_csv_data(
        self, path: str, train: bool, max_rows: int | None = None
    ) -> CSVData:
        excerpt_type = "train" if train else "test"
        data: CSVData = []

        try:
            with open(path, encoding="utf-8", newline='') as file:
                reader = csv.DictReader(file, quotechar='"')
                reader.fieldnames = [
                    name.strip().replace('\n', '').replace('"', '')
                    for name in reader.fieldnames
                ]

                filtered_rows = []
                for row in reader:
                    normalized = self.__normalize_csv_row(row)
                    split = normalized.get("kaggle_split")
                    if split is not None and split.lower() == excerpt_type:
                        filtered_rows.append(normalized)

                if max_rows is not None:
                    max_rows = min(max_rows, len(filtered_rows))
                    filtered_rows = filtered_rows[:max_rows]

                total_rows = len(filtered_rows)
                count = 0
                for row in filtered_rows:
                    if any(row.get(k) is None for k in ("id", "excerpt", "bt_easiness")):
                        continue
                    data.append(row)
                    count += 1
                    sys.stdout.write(
                        f"\rReading CSV: {count}/{total_rows} rows "
                        f"({count / total_rows * 100:.1f}%)"
                    )
                    sys.stdout.flush()

        except FileNotFoundError:
            print(f"File not found: {path}")
            return []

        print(f"\nCSV Data Processed! Total valid rows read: {len(data)}")
        return data

    def __formated_data_pre_lin_reg(self, csv_data: CSVData) -> PreLinearRegData:
        print("Starting parallel feature extraction...")
        start_time = time.time()
        results = []
        total   = len(csv_data)

        with ProcessPoolExecutor(
            max_workers=os.cpu_count(), initializer=init_worker
        ) as executor:
            futures = [
                executor.submit(extract_row_features_worker, row) for row in csv_data
            ]
            for i, future in enumerate(as_completed(futures), start=1):
                features, label = future.result()
                results.append((features, label))
                sys.stdout.write(
                    f"\rExtracting features: {i}/{total} rows ({i / total * 100:.1f}%)"
                )
                sys.stdout.flush()

        print(f"\nFeature extraction complete in {time.time() - start_time:.2f}s")

        X, Y = [], []
        for idx, (features, label) in enumerate(results, start=1):
            if not features:
                print(f"Skipping row {idx}: no features extracted")
                continue
            X.append(features)
            Y.append(label)

        if not X:
            print("Warning: No valid data available!")
        return list(zip(X, Y))

    def __get_feature_names(self) -> list[str]:
        doc_feature_names = [f.name for f in fields(DocumentProfile)]
        tabular_names = [
            "flesch_reading_ease",
            "flesch_kincaid_grade",
            "ari",
            "smog",
            "new_dale_chall",
        ]
        return doc_feature_names + tabular_names

    def __print_shap_summary(
        self, shap_values: np.ndarray, X: np.ndarray, top_n: int = 20
    ):
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        ranked        = np.argsort(mean_abs_shap)[::-1]
        feature_names = self.__get_feature_names()

        print(f"\n{'='*65}")
        print(f"  SHAP Feature Importance (top {top_n} of {len(mean_abs_shap)})")
        print(f"{'='*65}")
        print(f"  {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':<14} {'Mean Value'}")
        print(f"  {'-'*60}")

        for rank, idx in enumerate(ranked[:top_n], start=1):
            name     = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
            mean_val = X[:, idx].mean()
            print(f"  {rank:<6} {name:<35} {mean_abs_shap[idx]:<14.4f} {mean_val:.4f}")

        print(f"{'='*65}\n")
        dead = np.sum(mean_abs_shap < 0.001)
        print(f"  Features with mean |SHAP| < 0.001: {dead}/{len(mean_abs_shap)}")
        if dead > 0:
            print("  Dead features:")
            for idx in np.where(mean_abs_shap < 0.001)[0]:
                name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
                print(f"    - {name}")
        print()

    # ------------------------------------------------------------------ #
    #  Internal: per-model train / predict / save / load                  #
    # ------------------------------------------------------------------ #

    def _prompt_filename(self) -> str:
        """Prompt the user for a unique filename (no extension)."""
        existing = {f.lower() for f in os.listdir(self._output_dir)}
        file_name = input("\nEnter file name for weights: ").strip()
        while True:
            if file_name == "":
                file_name = input("Enter a valid (non-empty) filename: ").strip()
            elif any(file_name.lower() == e.split('.')[0] for e in existing):
                file_name = input(
                    f"ERROR: '{file_name}' already exists. Enter a different name: "
                ).strip()
            else:
                break
        return file_name

    # ---- Linear Regression ----

    def _train_linreg(self, X: list, y: list):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=1.0)),
        ])
        model.fit(X, y)
        self.model = model

        file_name = self._prompt_filename()
        path = os.path.join(self._output_dir, file_name + ".joblib")
        joblib.dump(model, path)
        self.model_filepath = path
        print(f"\nLinReg model saved → {path}")

    def _load_linreg(self, weight_path: str):
        self.model = joblib.load(weight_path)

    def _predict_linreg(self, features: list[float]) -> float:
        return float(self.model.predict([features])[0])

    # ---- XGBoost ----

    def _train_xgboost(self, X: list, y: list):
        model_config = dict(
            objective="reg:squarederror",
            n_estimators=4000,
            learning_rate=0.015,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.3,
            colsample_bylevel=0.7,
            colsample_bynode=0.7,
            gamma=0.3,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            eval_metric="mae",
        )

        if len(X) >= 20:
            model = xgboost.XGBRegressor(**model_config, early_stopping_rounds=50)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model = xgboost.XGBRegressor(**model_config)
            model.fit(X, y, verbose=False)

        self.model = model

        file_name = self._prompt_filename()
        path = os.path.join(self._output_dir, file_name + ".json")
        model.save_model(path)
        self.model_filepath = path
        print(f"\nXGBoost model saved → {path}")

    def _load_xgboost(self, weight_path: str):
        model = xgboost.XGBRegressor()
        model.load_model(weight_path)
        self.model = model

    def _predict_xgboost(self, features: list[float]) -> float:
        return float(self.model.predict([features])[0])

    # ---- Deep Learning ----

    def _train_deeplearning(self, X: list, y: list):
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_np      = np.array(X, dtype=np.float32)
        y_np      = np.array(y, dtype=np.float32)

        # Normalise inputs
        scaler = StandardScaler()
        X_np   = scaler.fit_transform(X_np)
        self._scaler = scaler

        X_train, X_val, y_train, y_val = train_test_split(
            X_np, y_np, test_size=0.2, random_state=42
        )

        train_ds = TensorDataset(
            torch.tensor(X_train), torch.tensor(y_train)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val), torch.tensor(y_val)
        )
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=256)

        input_dim = X_np.shape[1]
        model     = _ReadabilityMLP(input_dim).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state    = None
        patience      = 30
        no_improve    = 0
        epochs        = 300

        for epoch in range(1, epochs + 1):
            # --- train ---
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            # --- validate ---
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    val_losses.append(criterion(model(xb), yb).item() * len(xb))
            val_loss = sum(val_losses) / len(val_ds)
            scheduler.step(val_loss)

            sys.stdout.write(
                f"\r[DL] Epoch {epoch}/{epochs}  val_loss={val_loss:.4f}"
            )
            sys.stdout.flush()

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        model.eval()
        self.model = model.to("cpu")

        file_name = self._prompt_filename()
        model_path  = os.path.join(self._output_dir, file_name + ".pt")
        scaler_path = os.path.join(self._output_dir, file_name + "_scaler.joblib")

        torch.save({"model_state": best_state, "input_dim": input_dim}, model_path)
        joblib.dump(scaler, scaler_path)

        self.model_filepath = model_path
        print(f"\nDeep learning model saved → {model_path}")
        print(f"Scaler saved              → {scaler_path}")

    def _load_deeplearning(self, weight_path: str):
        checkpoint = torch.load(weight_path, map_location="cpu")
        model       = _ReadabilityMLP(checkpoint["input_dim"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        self.model  = model

        # Expect scaler saved alongside: <stem>_scaler.joblib
        scaler_path = weight_path.replace(".pt", "_scaler.joblib")
        if os.path.exists(scaler_path):
            self._scaler = joblib.load(scaler_path)
        else:
            print(f"Warning: scaler not found at {scaler_path}. Predictions may be off.")
            self._scaler = None

    def _predict_deeplearning(self, features: list[float]) -> float:
        x = np.array([features], dtype=np.float32)
        if self._scaler is not None:
            x = self._scaler.transform(x)
        with torch.no_grad():
            out = self.model(torch.tensor(x))
        return float(out.item())

    # ------------------------------------------------------------------ #
    #  Public API                                                          
    # ------------------------------------------------------------------ #

    def load_model(self, weight_path: str):
        """Load a previously saved model from *weight_path*."""
        loaders = {
            "linreg":      self._load_linreg,
            "xgboost":     self._load_xgboost,
            "deeplearning": self._load_deeplearning,
        }
        loaders[self.model_type](weight_path)
        self.loaded         = True
        self.model_filepath = weight_path
        print(f"[{self.model_type}] model loaded from {weight_path}")

    def train(self, csv_path: str, samples: int | None = None):
        print(f"\nTRAINING PROCESS STARTING ({self.model_type.upper()}): ")
        csv_data: CSVData       = self.__format_csv_data(csv_path, True, samples)
        pre_lin_data            = self.__formated_data_pre_lin_reg(csv_data)

        X, y = zip(*pre_lin_data)
        X, y = list(X), list(y)

        trainers = {
            "linreg":      self._train_linreg,
            "xgboost":     self._train_xgboost,
            "deeplearning": self._train_deeplearning,
        }
        trainers[self.model_type](X, y)
        self.loaded = True
        print(f"\nTRAINING COMPLETE! ({self.model_type.upper()})\n")

    def score(
        self,
        text: str,
        row: Dict[str, Optional[float]] | None = None,
        debug: bool = False,
    ) -> float:
        text_features = self.__extract_doc_features(text, row)
        predictors = {
            "linreg":      self._predict_linreg,
            "xgboost":     self._predict_xgboost,
            "deeplearning": self._predict_deeplearning,
        }
        return predictors[self.model_type](text_features)

    def validate(
        self, path: str, max_validate: int | None = None
    ) -> ValidationMetrics:
        csv_data: CSVData = self.__format_csv_data(path, False, max_validate)

        predictions, expected_values, all_features = [], [], []

        for row in csv_data:
            excerpt        = row.get("excerpt")
            expected_score = float(row.get("bt_easiness"))
            text_features  = self.__extract_doc_features(excerpt, row)
            if text_features is None:
                continue

            predicted_score = self.score(excerpt, row)

            print("=====================\n\n")
            print(excerpt + "\n\n\n")
            print(f"---EXPECTED: {expected_score} ----PREDICTED: {predicted_score}")
            print("=====================\n\n")

            predictions.append(float(predicted_score))
            expected_values.append(expected_score)
            all_features.append(text_features)

            total_rows  = max_validate if max_validate is not None else len(csv_data)
            current_mae = mean_absolute_error(expected_values, predictions)
            progress    = (len(predictions) / total_rows) * 100 if total_rows else 0.0
            sys.stdout.write(
                f"\rTesting: {len(predictions)}/{total_rows} rows "
                f"({progress:.1f}%) MAE={current_mae:.4f}".ljust(80)
            )
            sys.stdout.flush()

        if not predictions:
            print("\nNo validation data. Check dataset filtering.")
            return {"count": 0.0, "mae": 0.0, "rmse": 0.0, "r2": 0.0}

        mae  = mean_absolute_error(expected_values, predictions)
        rmse = mean_squared_error(expected_values, predictions) ** 0.5
        r2   = r2_score(expected_values, predictions)

        # SHAP is only natively supported for tree models
        shap_values, shap_base = None, None
        if self.model_type in ("xgboost", "linreg"):
            print("\n\nComputing SHAP values...")
            X_array   = np.array(all_features)
            explainer = shap.TreeExplainer(self.model) if self.model_type == "xgboost" \
                        else shap.LinearExplainer(self.model, X_array)
            shap_values = explainer.shap_values(X_array)
            shap_base   = float(explainer.expected_value
                                if not hasattr(explainer.expected_value, '__len__')
                                else explainer.expected_value[0])
            self.__print_shap_summary(shap_values, X_array)
        else:
            print("\n\nSHAP skipped for deep learning (use captum or shap.DeepExplainer separately).")

        result: ValidationMetrics = {
            "count": float(len(predictions)),
            "mae":   float(mae),
            "rmse":  float(rmse),
            "r2":    float(r2),
        }
        if shap_values is not None:
            result["shap_values"]     = shap_values
            result["shap_base_value"] = shap_base

        return result