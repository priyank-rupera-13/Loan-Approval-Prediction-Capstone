
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from joblib import load
import pandas as pd
import numpy as np
from pathlib import Path
import json

APP = FastAPI(title="Loan Approval API", version="1.3")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "loan_approval_rf_pipeline.joblib"
FEAT_PATH  = BASE_DIR / "models" / "feature_columns.json"

PIPELINE = None
FEATURES: List[str] = []

# ---------------- EMI helpers (tenure in YEARS) ----------------
def compute_emi(principal: float, annual_rate: float, months: int) -> Optional[float]:
    if principal is None or months is None or months <= 0:
        return None
    r = (annual_rate or 0.0) / 12.0
    if r <= 0:
        return principal / months
    pow_ = (1 + r) ** months
    return float(principal) * r * pow_ / (pow_ - 1)

def apply_emi_policy_years(income_annum: Optional[float],
                           loan_amount: Optional[float],
                           loan_term_years: Optional[float],
                           annual_rate: float = 0.082) -> Dict[str, Any]:
    if income_annum is None or loan_amount is None or loan_term_years is None or loan_term_years <= 0:
        return {
            "emi": None, "monthly_income": None, "threshold": None,
            "rule_applied": False, "rule_reject": False,
            "reason": "Policy skipped (missing income_annum/loan_amount/loan_term_years)."
        }
    months = int(round(float(loan_term_years) * 12))
    monthly_income = float(income_annum) / 12.0
    emi = compute_emi(float(loan_amount), annual_rate, months)
    if emi is None:
        return {
            "emi": None, "monthly_income": monthly_income, "threshold": None,
            "rule_applied": False, "rule_reject": False,
            "reason": "Policy skipped (EMI not computable)."
        }
    threshold = 0.5 * monthly_income
    rule_reject = emi > threshold
    return {
        "emi": float(emi),
        "monthly_income": float(monthly_income),
        "threshold": float(threshold),
        "rule_applied": True,
        "rule_reject": bool(rule_reject),
        "reason": "EMI exceeds 50% of monthly income" if rule_reject else "EMI within safe limit"
    }

# ---------------- Feature alignment helpers ----------------
def canonicalize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        kk = (
            str(k).strip().lower()
            .replace("/", "_").replace("-", "_").replace(" ", "_")
        )
        out[kk] = v
    return out

def align_dataframe(payloads: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    1) Canonicalize keys to match training cleaning.
    2) Ensure model feature 'loan_term' is in YEARS regardless of what client sends
       (loan_term_years preferred; fallback to loan_term [treated as years];
        or derive years from loan_term_months).
    3) Add any missing training FEATURES as NaN and drop extras.
    """
    canon = [canonicalize_keys(p) for p in payloads]
    rows = []
    for row in canon:
        if "loan_term" in FEATURES:
            years = None
            if "loan_term_years" in row and row["loan_term_years"] is not None:
                years = float(row["loan_term_years"])
            elif "loan_term" in row and row["loan_term"] is not None:
                years = float(row["loan_term"])
            elif "loan_term_months" in row and row["loan_term_months"] is not None:
                years = float(row["loan_term_months"]) / 12.0
            row["loan_term"] = years
        rows.append(row)

    X = pd.DataFrame(rows)
    missing = [c for c in FEATURES if c not in X.columns]
    for c in missing:
        X[c] = np.nan
    X = X[FEATURES]
    return X

# ---------------- Pydantic Schemas ----------------
class LoanApplication(BaseModel):
    # Training features
    loan_id: Optional[float] = None
    no_of_dependents: Optional[float] = None
    education: Optional[str] = None
    self_employed: Optional[str] = None
    income_annum: Optional[float] = None
    loan_amount: Optional[float] = None
    loan_term: Optional[float] = None              # YEARS (dataset)
    cibil_score: Optional[float] = None
    residential_assets_value: Optional[float] = None
    commercial_assets_value: Optional[float] = None
    luxury_assets_value: Optional[float] = None
    bank_asset_value: Optional[float] = None

    # Convenience
    loan_term_years: Optional[float] = Field(None, description="Preferred: tenure in YEARS")
    loan_term_months: Optional[float] = Field(None, description="Alternative: tenure in months; will be converted to YEARS")

class PredictionResponse(BaseModel):
    model_pred: int
    final_pred: int
    prob_approve: float
    emi: Optional[float] = None
    monthly_income: Optional[float] = None
    threshold: Optional[float] = None
    rule_applied: bool
    rule_reject: bool
    reason: str

# ---------------- App lifecycle ----------------
APP = APP  # alias to keep name stable for uvicorn
@APP.on_event("startup")
def load_artifacts():
    global PIPELINE, FEATURES
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train & save it first.")
    PIPELINE = load(MODEL_PATH)
    if not FEAT_PATH.exists():
        raise FileNotFoundError(
            f"Feature list not found at {FEAT_PATH}. "
            "In your notebook, save models/feature_columns.json with the training feature names."
        )
    data = json.loads(FEAT_PATH.read_text())
    FEATURES = data.get("features", [])
    if not isinstance(FEATURES, list) or not FEATURES:
        raise RuntimeError("Invalid feature_columns.json content. Expected key 'features' with a non-empty list.")

@APP.get("/ping")
def ping():
    return {"status": "ok", "num_features": len(FEATURES)}

@APP.get("/expected_features")
def expected_features():
    return {"features": FEATURES}

# ---------------- Endpoints ----------------
@APP.post("/predict", response_model=PredictionResponse)
def predict(app: LoanApplication, threshold: float = 0.5, annual_rate: float = 0.082):
    if PIPELINE is None:
        raise HTTPException(500, "Model not loaded")
    X = align_dataframe([app.dict(exclude_none=True)])
    try:
        prob = float(PIPELINE.predict_proba(X)[:, 1][0])
    except Exception as e:
        raise HTTPException(400, f"Inference failed: {e}")
    model_pred = int(prob >= threshold)

    term_years = (
        app.loan_term_years
        if app.loan_term_years is not None
        else (app.loan_term if app.loan_term is not None else (app.loan_term_months / 12.0 if app.loan_term_months is not None else None))
    )

    policy = apply_emi_policy_years(
        income_annum=app.income_annum,
        loan_amount=app.loan_amount,
        loan_term_years=term_years,
        annual_rate=annual_rate
    )

    final_pred = 0 if (policy["rule_applied"] and policy["rule_reject"]) else model_pred

    return {
        "model_pred": model_pred,
        "final_pred": final_pred,
        "prob_approve": prob,
        "emi": policy["emi"],
        "monthly_income": policy["monthly_income"],
        "threshold": policy["threshold"],
        "rule_applied": policy["rule_applied"],
        "rule_reject": policy["rule_reject"],
        "reason": policy["reason"]
    }

@APP.post("/predict_batch")
def predict_batch(apps: List[LoanApplication], threshold: float = 0.5, annual_rate: float = 0.082):
    if PIPELINE is None:
        raise HTTPException(500, "Model not loaded")
    payloads = [a.dict(exclude_none=True) for a in apps]
    X = align_dataframe(payloads)
    try:
        probs = PIPELINE.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(400, f"Inference failed: {e}")
    model_preds = (probs >= threshold).astype(int)

    results = []
    for app, prob, mpred in zip(apps, probs, model_preds):
        term_years = (
            app.loan_term_years
            if app.loan_term_years is not None
            else (app.loan_term if app.loan_term is not None else (app.loan_term_months / 12.0 if app.loan_term_months is not None else None))
        )
        policy = apply_emi_policy_years(
            income_annum=app.income_annum,
            loan_amount=app.loan_amount,
            loan_term_years=term_years,
            annual_rate=annual_rate
        )
        final_pred = 0 if (policy["rule_applied"] and policy["rule_reject"]) else int(mpred)
        results.append({
            "model_pred": int(mpred),
            "final_pred": int(final_pred),
            "prob_approve": float(prob),
            "emi": policy["emi"],
            "monthly_income": policy["monthly_income"],
            "threshold": policy["threshold"],
            "rule_applied": policy["rule_applied"],
            "rule_reject": policy["rule_reject"],
            "reason": policy["reason"]
        })
    return {"items": results}
