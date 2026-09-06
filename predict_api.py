"""
Cold Chain Predictive Intelligence — REST API
=============================================
Wraps the trained ensemble model (RF + XGB + LightGBM soft voting).

Usage:
    pip install fastapi uvicorn joblib scikit-learn xgboost lightgbm
    uvicorn predict_api:app --host 0.0.0.0 --port 8000

Then POST to  http://localhost:8000/predict
with a JSON body of sensor feature values.

This is a research prototype. Validate against real SUS sensor data
before using in any production cold chain monitoring context.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, json, numpy as np
from pathlib import Path
from typing import Optional

# ── Load models at startup ────────────────────────────────────────
MODEL_DIR = Path("production_models")

try:
    rf_model   = joblib.load(MODEL_DIR / "rf_model.pkl")
    xgb_model  = joblib.load(MODEL_DIR / "xgb_model.pkl")
    lgbm_model = joblib.load(MODEL_DIR / "lgbm_model.pkl")

    with open(MODEL_DIR / "feature_cols.json") as f:
        FEATURE_COLS = json.load(f)

    with open(MODEL_DIR / "ensemble_config.json") as f:
        config = json.load(f)
    THRESHOLD = config["threshold"]

    print(f"✓ Models loaded — {len(FEATURE_COLS)} features — threshold {THRESHOLD}")

except FileNotFoundError as e:
    raise RuntimeError(
        f"Model files not found: {e}\n"
        "Run SAVE_MODELS_FOR_PRODUCTION.py in your notebook first, "
        "then copy the production_models/ folder here."
    )

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Cold Chain Predictive Intelligence API",
    description=(
        "Research prototype — predicts temperature excursion risk "
        "hours before failure using an ensemble of RF, XGBoost, and LightGBM. "
        "Not validated on SUS production hardware."
    ),
    version="0.1.0-research",
)


# ── Input schema ─────────────────────────────────────────────────
class SensorReading(BaseModel):
    """
    One row of engineered features from the cold chain sensor.
    Keys must exactly match the feature_cols.json saved with the model.
    Send the same features your notebook creates from raw temperature readings.
    """
    features: dict  # {feature_name: float_value}
    sensor_id: Optional[str] = None   # optional — for logging
    timestamp: Optional[str] = None   # optional — for logging


# ── Prediction endpoint ───────────────────────────────────────────
@app.post("/predict")
def predict(reading: SensorReading):
    """
    Accept one sensor reading (as engineered features) and return a risk score.

    Response:
        risk_score    – probability of excursion (0.0 – 1.0)
        alert         – True if risk_score >= threshold
        threshold     – the decision boundary used
        hours_warning – estimated lead time if alert is True
        model         – which ensemble produced this
        note          – research disclaimer
    """
    # Build feature vector in correct order
    missing = [f for f in FEATURE_COLS if f not in reading.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {missing}. "
                   f"Required: {FEATURE_COLS}"
        )

    X = np.array([[reading.features[f] for f in FEATURE_COLS]])

    # Soft-voting ensemble
    p_rf   = float(rf_model.predict_proba(X)[0, 1])
    p_xgb  = float(xgb_model.predict_proba(X)[0, 1])
    p_lgbm = float(lgbm_model.predict_proba(X)[0, 1])
    risk   = round((p_rf + p_xgb + p_lgbm) / 3, 4)

    alert = risk >= THRESHOLD

    return {
        "risk_score":    risk,
        "alert":         alert,
        "threshold":     THRESHOLD,
        "hours_warning": "~8 hrs (research estimate)" if alert else "none",
        "breakdown": {
            "random_forest": round(p_rf,   4),
            "xgboost":       round(p_xgb,  4),
            "lightgbm":      round(p_lgbm, 4),
        },
        "model":  "ensemble_rf_xgb_lgbm_soft_voting",
        "sensor": reading.sensor_id,
        "ts":     reading.timestamp,
        "note":   (
            "Research prototype trained on NAB public dataset. "
            "Not validated on SUS production hardware. "
            "8-hour estimate is from research data only."
        ),
    }


# ── Batch prediction ─────────────────────────────────────────────
@app.post("/predict/batch")
def predict_batch(readings: list[SensorReading]):
    """Predict for multiple readings in one call (max 1000)."""
    if len(readings) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 readings per batch.")
    return [predict(r) for r in readings]


# ── Health check ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":    "ok",
        "features":  len(FEATURE_COLS),
        "threshold": THRESHOLD,
        "models":    ["random_forest", "xgboost", "lightgbm"],
        "note":      "Research prototype — not production validated",
    }


# ── Feature list ─────────────────────────────────────────────────
@app.get("/features")
def features():
    """Returns the exact feature names and order the model expects."""
    return {"feature_count": len(FEATURE_COLS), "features": FEATURE_COLS}


# ── Run directly ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
