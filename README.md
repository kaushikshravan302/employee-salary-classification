# Employee Salary Intelligence & Classification System

Classifies employees as `<=50K` or `>50K` using the real UCI Adult Census Income dataset
(48,842 records, 15 columns). Compares Logistic Regression, Decision Tree and Random Forest;
the best model is selected automatically from cross-validated F1 + ROC-AUC.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py          # trains automatically on first run if artifacts/ is missing
    python pipeline.py            # (optional) retrain and regenerate artifacts/

## Deploy (free, live URL)
1. Push this folder to a GitHub repo.
2. Go to share.streamlit.io -> New app -> choose the repo, main file `app.py` -> Deploy.

## Files
- `pipeline.py` data audit, cleaning, preprocessing, training, evaluation, selection
- `app.py` Streamlit dashboard (Overview, EDA, Model Comparison, Live Prediction, Insights)
- `data/adult-all.csv` original dataset (UCI Adult, full 48,842 rows), `data/adult_clean.csv` cleaned copy
- `artifacts/` trained pipelines (joblib) and `meta.json` (metrics, audit, importances)
