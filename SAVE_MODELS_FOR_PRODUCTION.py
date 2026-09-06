"""
Run this script inside your notebook (add as a new cell at the end of Section 11)
after the ensemble threshold has been tuned and all three models are trained.

Variables expected to already exist in the notebook kernel:
    rf_nosmote  – trained Random Forest
    xgb         – trained XGBoost
    lgbm        – trained LightGBM
    X_test_c    – test feature DataFrame (gives us the feature column list)
    ens_threshold – the tuned threshold (default 0.5 if you haven't tuned it)
"""

import joblib, json, os

# ── 1. Save the three base models ────────────────────────────────
SAVE_DIR = "production_models"
os.makedirs(SAVE_DIR, exist_ok=True)

joblib.dump(rf_nosmote, f"{SAVE_DIR}/rf_model.pkl")
joblib.dump(xgb,        f"{SAVE_DIR}/xgb_model.pkl")
joblib.dump(lgbm,       f"{SAVE_DIR}/lgbm_model.pkl")

print("✓ Models saved")

# ── 2. Save the feature list (critical — order must be exact) ────
feature_cols = list(X_test_c.columns)
with open(f"{SAVE_DIR}/feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print(f"✓ {len(feature_cols)} features saved: {feature_cols[:5]} ...")

# ── 3. Save the ensemble threshold ───────────────────────────────
# Use the tuned value if it exists, otherwise default 0.5
try:
    threshold = float(ens_threshold)
except NameError:
    threshold = 0.5
    print("  (ens_threshold not found — using default 0.5)")

with open(f"{SAVE_DIR}/ensemble_config.json", "w") as f:
    json.dump({"threshold": threshold, "models": ["rf", "xgb", "lgbm"],
               "strategy": "soft_voting_mean"}, f, indent=2)

print(f"✓ Threshold saved: {threshold}")

# ── 4. Quick sanity check ────────────────────────────────────────
rf2   = joblib.load(f"{SAVE_DIR}/rf_model.pkl")
xgb2  = joblib.load(f"{SAVE_DIR}/xgb_model.pkl")
lgbm2 = joblib.load(f"{SAVE_DIR}/lgbm_model.pkl")

sample = X_test_c.iloc[:5]
p_rf   = rf2.predict_proba(sample)[:, 1]
p_xgb  = xgb2.predict_proba(sample)[:, 1]
p_lgbm = lgbm2.predict_proba(sample)[:, 1]
p_ens  = (p_rf + p_xgb + p_lgbm) / 3

print("\n✓ Sanity check — ensemble probs on 5 test samples:")
for i, p in enumerate(p_ens):
    alert = "🚨 ALERT" if p >= threshold else "  ok"
    print(f"   Sample {i}: {p:.3f}  {alert}")

print(f"\nAll files saved to: {SAVE_DIR}/")
print("  rf_model.pkl")
print("  xgb_model.pkl")
print("  lgbm_model.pkl")
print("  feature_cols.json")
print("  ensemble_config.json")
print("\nCopy the production_models/ folder next to predict_api.py to deploy.")
