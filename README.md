
# Loan Approval Capstone — Jupyter + FastAPI

This repository contains a complete, end-to-end **Loan Approval** project built in **Jupyter** and deployed via **FastAPI**.

- Notebook: `Loan_Approval_Project.ipynb` (15 markdown cells, 23 code cells)
- API: `main.py` — **Loan Approval API** v1.0
- Models saved under `models/` (Random Forest and optionally XGBoost)
- Business rule: **EMI at interest rate r must be ≤ 50% of monthly income** (tenure in **years** → converted to **months** for EMI)

---

## 1) Problem & Goal

**Task:** Predict whether a loan application is **Approved (1)** or **Rejected (0)** using applicant and loan features.  
**Objective:** Train a robust classifier (Random Forest baseline; optionally XGBoost) and **enforce an affordability rule** using EMI.

---

## 2) Dataset

Place your dataset at:
```
data/raw/loan_approval_dataset.csv
```
Typical columns: `loan_id, no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term (YEARS), cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value, loan_status`.

- **Column cleanup:** names normalized to lower_snake_case
- **Target mapping:** `loan_status` → {0=Reject, 1=Approve}

---

## 3) Project Structure
```
.
├─ data/
│  └─ raw/loan_approval_dataset.csv
├─ models/
│  ├─ loan_approval_rf_pipeline.joblib        # Random Forest pipeline
│  └─ feature_columns.json                    # exact training feature list
├─ reports/                                   # (optional) metrics, plots
├─ notebooks/
│  └─ Loan_Approval_Project.ipynb
└─ fastapi_app/                               # if you later move the API here
   └─ main.py
```

---

## 4) Notebook Workflow (short guide)

**Preview of sections (from your notebook headings):**
- ## Imports for Predicting Approval or rejection
- ## Load data & quick peek
- #### confirm rows/columns and spot obvious issues early.
- ## Clean column names, choose target, map labels ->{0,1}
- ### Binary target for classification
- ## Data Audit and finding null values
- ## Split Train/Test
- ## Preprocessing + Random Forest pipeline
- ## Train the model
- ## XGBoost training
- ## Evalute (Accuracy)
- ### Precision, recall, F1, Auc to get clear picture

**Core steps inside the notebook:**
1. **Load & clean** data; map target to 0/1.
2. **Train/Test split** with `stratify=y`.
3. **Preprocess** via `ColumnTransformer`: median impute numeric; most_frequent + one-hot for categoricals.
4. **Train models**: Random Forest (and optionally XGBoost with same `preprocess`).
5. **Evaluate**: Accuracy, Precision, Recall, F1, ROC AUC; confusion matrices; ROC curves.
6. **Explainability**: feature importance (colored bars); optional permutation importance/SHAP.
7. **Save pipelines** to `models/` and export `feature_columns.json` used by the API.

**Saving models (example):**
```python
from joblib import dump
from pathlib import Path
import json

# Random Forest
dump(rf_pipe, Path('models/loan_approval_rf_pipeline.joblib'))

# XGBoost (if trained)
# dump(xgb_pipe, Path('models/loan_approval_xgb_pipeline.joblib'))

# Exact training feature names
feature_columns = X.columns.tolist()
Path('models/feature_columns.json').write_text(json.dumps({"features": feature_columns}, indent=2))
```

---

## 5) Business Rule — EMI Affordability

Policy: at annual rate **r**, compute **EMI** with tenure in **months** (convert **years × 12**).  
Reject if `EMI > 0.5 * (income_annum / 12)`.

- The **notebook** exposes helper functions.
- The **API** replicates the rule; it uses **tenure in YEARS** for the model and converts to **months** for EMI.
- Interest rate defaults to 8.2% (0.082) in the API.

---

## 6) FastAPI Service

Your API file: `main.py`

**Detected endpoints:**
- `GET /ping`
- `POST /predict_batch`

**Key behaviors:**
- Loads model pipeline from `models/loan_approval_rf_pipeline.joblib`
- Aligns request JSON to the exact training feature set (`models/feature_columns.json`)
- Computes ML probability, **then applies EMI policy** before returning `final_pred`
- Tenure is expected in **years** for the model; EMI uses **months** internally.

**Run locally:**
```bash
# From the directory holding main.py:
uvicorn main:APP --reload --host 0.0.0.0 --port 8000

# If you move it under fastapi_app/, from project root:
# uvicorn fastapi_app.main:APP --reload --host 0.0.0.0 --port 8000
```

**Docs:** http://127.0.0.1:8000/docs

**Example single prediction:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "income_annum": 840000,
    "loan_amount": 600000,
    "loan_term": 3,          // YEARS
    "cibil_score": 720,
    "education": "Graduate",
    "self_employed": "No"
  }'
```

**Response (example):**
```json
{
  "model_pred": 1,
  "final_pred": 1,
  "prob_approve": 0.83,
  "emi": 19230.14,
  "monthly_income": 70000.0,
  "threshold": 35000.0,
  "rule_applied": true,
  "rule_reject": false,
  "reason": "EMI within safe limit"
}
```

---

## 7) Picking a Best Model

Compare Random Forest vs XGBoost on the holdout set; optionally save a `best_model.joblib` for deployment.

```python
# dump(best_pipe, Path('models/best_model.joblib'))
```

---

## 8) Troubleshooting

- **ASGI import error**: run uvicorn from the correct directory (see above).
- **Missing columns**: ensure `models/feature_columns.json` matches training features.
- **Interest rate**: pass in body as `"annual_rate": 9.5` or `0.095` (if supported).
- **Tenure**: dataset uses **years**; EMI converts to **months** internally.
- **XGBoost/Sklearn mismatch**: `pip install -U "scikit-learn>=1.4.2" "xgboost>=2.0.3"` then restart the kernel.

---

## 9) Next Steps

- Cross-validation & hyperparameter tuning
- Export plots to `reports/plots/`
- Add CORS middleware
- Dockerize for deployment

---

© Priyank Rupera — Educational use.
