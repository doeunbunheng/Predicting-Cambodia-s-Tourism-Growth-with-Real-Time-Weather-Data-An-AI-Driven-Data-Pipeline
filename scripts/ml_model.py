## scripts/ml_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pickle
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 55)
print(" ML MODEL — Predict Tourist Arrivals")
print("=" * 55)

# ── LOAD GOLD TABLE ───────────────────────────────────
print("\n Loading Gold table...")
gold = pd.read_csv("data/gold/fact_tourism_monthly.csv")
gold["date"] = pd.to_datetime(gold["date"])
print(f"    Gold rows : {len(gold)}")
print(f"    Columns   : {list(gold.columns)}")

# ── PREPARE FEATURES ──────────────────────────────────
print("\n Preparing features...")

# Select weather feature columns
weather_cols = [c for c in gold.columns
                if any(x in c for x in
                       ["avg_temp","total_rain","avg_humidity"])]
print(f"   Weather features: {weather_cols}")

# Feature set
features = ["month", "is_covid", "prev_month_arrivals"] + weather_cols
target   = "arrivals"

# Drop rows with nulls in features
gold_clean = gold[features + [target, "year", "date"]].dropna()
print(f"    Clean rows : {len(gold_clean)}")

X = gold_clean[features]
y = gold_clean[target]

# ── TRAIN / TEST SPLIT ────────────────────────────────
print("\n Splitting data...")

# Train: 2012-2022, Test: 2023-2025
train = gold_clean[gold_clean["year"] <= 2022]
test  = gold_clean[gold_clean["year"] >= 2023]

X_train = train[features]
y_train = train[target]
X_test  = test[features]
y_test  = test[target]

print(f"   Train rows : {len(X_train)} (2012-2022)")
print(f"   Test rows  : {len(X_test)}  (2023-2025)")

# ── TRAIN MODELS ──────────────────────────────────────
print("\n Training RandomForest...")

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred  = rf.predict(X_test)
rf_mae   = mean_absolute_error(y_test, rf_pred)
rf_r2    = r2_score(y_test, rf_pred)

print(f"    RandomForest MAE : {rf_mae:>10,.0f} arrivals")
print(f"    RandomForest R²  : {rf_r2:.4f}")

print("\n Training Linear Regression...")

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_mae  = mean_absolute_error(y_test, lr_pred)
lr_r2   = r2_score(y_test, lr_pred)

print(f"    Linear Reg MAE   : {lr_mae:>10,.0f} arrivals")
print(f"    Linear Reg R²    : {lr_r2:.4f}")

# ── BEST MODEL ────────────────────────────────────────
best_model = rf if rf_mae < lr_mae else lr
best_name  = "RandomForest" if rf_mae < lr_mae else "LinearRegression"
print(f"\n Best model: {best_name} (lower MAE)")

# ── FEATURE IMPORTANCE ────────────────────────────────
print("\n Feature importance (RandomForest):")
importance = pd.DataFrame({
    "feature":   features,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)
print(importance.to_string(index=False))

# ── ACTUAL VS PREDICTED ───────────────────────────────
print("\n Actual vs Predicted (test set):")
comparison = pd.DataFrame({
    "date":      test["date"].values,
    "actual":    y_test.values,
    "predicted": rf_pred.astype(int),
    "error":     (rf_pred - y_test.values).astype(int)
})
print(comparison.head(10).to_string(index=False))

# ── SAVE MODEL ────────────────────────────────────────
os.makedirs("data/gold", exist_ok=True)

with open("data/gold/model_randomforest.pkl", "wb") as f:
    pickle.dump(rf, f)

with open("data/gold/model_features.pkl", "wb") as f:
    pickle.dump(features, f)

# Save predictions
comparison.to_csv("data/gold/predictions.csv", index=False)

print(f"\n{'=' * 55}")
print(f" ML MODEL DONE!")
print(f"   Best model : {best_name}")
print(f"   MAE        : {min(rf_mae, lr_mae):,.0f} arrivals")
print(f"   R²         : {max(rf_r2, lr_r2):.4f}")
print(f"   Model saved: data/gold/model_randomforest.pkl")
print(f"   Predictions: data/gold/predictions.csv")
