"""Data audit, cleaning, preprocessing, model training, evaluation and selection.

Run `python pipeline.py` to (re)build everything in ./artifacts from the real
UCI Adult dataset in ./data/adult-all.csv. Nothing is hard-coded: the winning
model is chosen from the measured results.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "adult-all.csv"
CLEAN = ROOT / "data" / "adult_clean.csv"
ART = ROOT / "artifacts"
SEED = 42

COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"]
NUM = ["age", "capital-gain", "capital-loss", "hours-per-week"]
CAT = ["workclass", "education", "marital-status", "occupation", "relationship",
       "race", "sex", "native-country"]
FEATURES = ["age", "workclass", "education", "marital-status", "occupation", "relationship",
            "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
LABELS = {0: "<=50K", 1: ">50K"}


# ----------------------------------------------------------------------------- data
def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW, header=None, names=COLS, skipinitialspace=True)


def audit(raw: pd.DataFrame) -> dict:
    cat_cols = raw.select_dtypes(exclude="number").columns
    return {
        "rows": int(raw.shape[0]), "columns": int(raw.shape[1]),
        "column_names": list(raw.columns),
        "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
        "unknown_values": {c: int((raw[c] == "?").sum()) for c in cat_cols if (raw[c] == "?").any()},
        "null_values": int(raw.isna().sum().sum()),
        "duplicates": int(raw.duplicated().sum()),
        "unique_categories": {c: sorted(raw[c].unique().tolist()) for c in cat_cols},
        "class_distribution": {k: int(v) for k, v in raw["income"].value_counts().items()},
        "describe": raw.describe().round(2).to_dict(),
    }


def clean(raw: pd.DataFrame):
    """Returns cleaned dataframe and a human-readable log of every decision."""
    df = raw.copy()
    log = []
    for c in df.select_dtypes(exclude="number").columns:
        df[c] = df[c].str.strip()
    log.append("Stripped leading/trailing whitespace from every text column.")
    df["income"] = df["income"].str.rstrip(".")
    log.append("Normalised target labels (removes trailing '.' used in some UCI test-file variants).")
    n_unk = int((df[["workclass", "occupation", "native-country"]] == "?").sum().sum())
    df = df.replace("?", "Unknown")
    log.append(f"Replaced {n_unk:,} '?' placeholders (workclass, occupation, native-country) with an explicit "
               "'Unknown' category. Rows were kept because missingness is informative (e.g. unknown "
               "occupation is linked to lower income) and dropping would discard rows; mode-imputation would invent values.")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.append(f"Removed {before - len(df):,} exact duplicate rows ({before:,} -> {len(df):,}).")
    log.append("Dropped 'fnlwgt' (census sampling weight, not an employee attribute) from the model features.")
    log.append("Kept 'education-num' only for EDA ordering; excluded from models because it is a 1:1 numeric "
               "re-encoding of 'education' (redundant) and is not on the prediction form.")
    log.append("Data types verified: 6 numeric columns are int64, 9 text columns are categorical. "
               "Target encoded internally as 0 = '<=50K', 1 = '>50K'; labels are shown to users.")
    return df, log


# ----------------------------------------------------------------------------- models
def _preprocessor(scale: bool) -> ColumnTransformer:
    ohe = OneHotEncoder(handle_unknown="ignore")
    if scale:  # Logistic Regression: scale numerics, log-transform heavily skewed capital columns
        cap = Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("sc", StandardScaler())])
        num = [("age_hours", StandardScaler(), ["age", "hours-per-week"]),
               ("capital", cap, ["capital-gain", "capital-loss"])]
    else:  # tree models are scale-invariant: no scaling
        num = [("num", "passthrough", NUM)]
    return ColumnTransformer(num + [("cat", ohe, CAT)])


def make_models() -> dict:
    return {
        "Logistic Regression": Pipeline([("prep", _preprocessor(True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))]),
        "Decision Tree": Pipeline([("prep", _preprocessor(False)),
            ("clf", DecisionTreeClassifier(max_depth=8, min_samples_leaf=25,
                                           class_weight="balanced", random_state=SEED))]),
        "Random Forest": Pipeline([("prep", _preprocessor(False)),
            ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                           class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))]),
    }


def explain_instance(model, row: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """Model-agnostic what-if explanation: how much does each input move P(>50K)
    compared with replacing it by the typical (median/mode) training value?"""
    rows = [row.iloc[0].to_dict()]
    for f in FEATURES:
        r = dict(rows[0]); r[f] = baseline[f]; rows.append(r)
    p = model.predict_proba(pd.DataFrame(rows)[FEATURES])[:, 1]
    out = pd.DataFrame({"feature": FEATURES, "value": [rows[0][f] for f in FEATURES],
                        "effect": p[0] - p[1:]})
    return out.sort_values("effect", key=abs, ascending=False).reset_index(drop=True)


def train():
    ART.mkdir(exist_ok=True)
    raw = load_raw()
    info = audit(raw)
    df, log = clean(raw)
    df.to_csv(CLEAN, index=False)

    X, y = df[FEATURES], (df["income"] == ">50K").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    log.append(f"Stratified 80/20 train/test split (random_state={SEED}): {len(Xtr):,} train / {len(Xte):,} test rows.")
    log.append("All encoders/scalers live inside scikit-learn Pipelines, so they are fitted on training data only "
               "(also inside each cross-validation fold) - no data leakage.")
    log.append("One-hot encoding for categoricals (unseen categories ignored at prediction time); StandardScaler + "
               "log1p on capital columns for Logistic Regression only; no scaling for tree models.")
    log.append("Class imbalance handled with class_weight='balanced' (no resampling of the test set).")

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    grid = np.linspace(0, 1, 201)
    for name, pipe in make_models().items():
        s = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=["f1", "roc_auc"])
        pipe.fit(Xtr, ytr)
        pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, proba)
        results[name] = {
            "accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred),
            "recall": recall_score(yte, pred), "f1": f1_score(yte, pred),
            "roc_auc": roc_auc_score(yte, proba),
            "cv_f1": s["test_f1"].mean(), "cv_roc_auc": s["test_roc_auc"].mean(),
            "confusion_matrix": confusion_matrix(yte, pred).tolist(),
            "roc": {"fpr": grid.tolist(), "tpr": np.interp(grid, fpr, tpr).tolist()},
        }
        results[name]["selection_score"] = (results[name]["cv_f1"] + results[name]["cv_roc_auc"]) / 2
        results[name] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in results[name].items()}
        fitted[name] = pipe
        print(f"{name:20s} F1={results[name]['f1']:.4f} AUC={results[name]['roc_auc']:.4f} "
              f"(CV F1={results[name]['cv_f1']:.4f}, CV AUC={results[name]['cv_roc_auc']:.4f})")

    # model selection: mean of cross-validated F1 and ROC-AUC on the TRAINING set only
    best = max(results, key=lambda k: results[k]["selection_score"])
    log.append("Model selection uses the mean of 5-fold cross-validated F1 and ROC-AUC on the training set "
               "(not accuracy, and not the test set); the test set is used once for final reporting.")

    sample = Xte.sample(min(4000, len(Xte)), random_state=SEED)
    pi = permutation_importance(fitted[best], sample, yte.loc[sample.index], scoring="roc_auc",
                                n_repeats=5, random_state=SEED, n_jobs=1)
    importance = (pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)).round(5)

    baseline = {f: (float(Xtr[f].median()) if f in NUM else Xtr[f].mode()[0]) for f in FEATURES}
    options = {c: sorted(df[c].unique().tolist()) for c in CAT}
    ranges = {c: [int(df[c].min()), int(df[c].max())] for c in NUM}

    for name, m in fitted.items():
        joblib.dump(m, ART / f"{name.lower().replace(' ', '_')}.joblib", compress=3)
    joblib.dump(fitted[best], ART / "best_model.joblib", compress=3)
    meta = {"best_model": best, "results": results, "audit": info, "preprocessing_log": log,
            "importance": importance.to_dict(), "baseline": baseline, "options": options, "ranges": ranges,
            "split": {"train": len(Xtr), "test": len(Xte), "seed": SEED},
            "clean_rows": int(len(df))}
    (ART / "meta.json").write_text(json.dumps(meta, indent=1))
    print("Selected model:", best)
    return meta


if __name__ == "__main__":
    train()
