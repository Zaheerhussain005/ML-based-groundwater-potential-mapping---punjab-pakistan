"""
ML training pipeline: Random Forest, XGBoost, LightGBM.
Outputs: trained models, accuracy metrics, feature importance.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("outputs/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["Elevation", "Slope", "Aspect", "TWI", "NDVI", "NDWI", "Rainfall", "Soil", "LST"]
TARGET = "GW_Potential"  # 0=Low, 1=Medium, 2=High


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    path = PROCESSED_DIR / "training_samples.csv"
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}\nRun label_samples.py first.")
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df[TARGET]
    logger.info(f"Training data: {X.shape} | Classes: {y.value_counts().to_dict()}")
    return X, y


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=63,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, name: str):
    y_pred = model.predict(X_test)
    logger.info(f"\n── {name} ──")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Low','Medium','High'])}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_test, y_test, cv=cv, scoring="f1_macro")
    logger.info(f"Cross-val F1 (5-fold): {scores.mean():.3f} ± {scores.std():.3f}")
    return y_pred


def save_model(model, name: str):
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    logger.info(f"Saved: {path}")


def run_training():
    X, y = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    models = {
        "random_forest": train_random_forest(X_train, y_train),
        "xgboost": train_xgboost(X_train, y_train),
        "lightgbm": train_lightgbm(X_train, y_train),
    }

    for name, model in models.items():
        evaluate(model, X_test, y_test, name)
        save_model(model, name)

    logger.info("All models trained and saved.")
    return models


if __name__ == "__main__":
    run_training()
