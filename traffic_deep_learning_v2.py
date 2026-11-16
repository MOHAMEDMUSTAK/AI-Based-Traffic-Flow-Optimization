"""
Metro Interstate Traffic Volume Prediction
Improved ANN experiment + unchanged Random Forest baseline.

Run:
    pip install -r requirements_ann.txt
    python traffic_deep_learning_v2.py

Place Metro_Interstate_Traffic_Volume.csv in this script's folder.
"""

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

HERE = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    HERE / "Metro_Interstate_Traffic_Volume.csv",
    HERE / "Metro_Interstate_Traffic_Volume(20261102-131815).csv",
]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        "Dataset not found. Put Metro_Interstate_Traffic_Volume.csv "
        "in the same folder as this script."
    )

OUT = HERE / "traffic_model_outputs_v2"
OUT.mkdir(exist_ok=True)

# 1. LOAD DATA AND ENGINEER FEATURES
df = pd.read_csv(DATA_PATH)
df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
df = df.dropna(subset=["date_time", "traffic_volume"]).copy()

df["hour"] = df["date_time"].dt.hour
df["day"] = df["date_time"].dt.dayofweek

# Cyclical features help the ANN understand that the end and start
# of a time cycle are adjacent (23:00 is close to 00:00).
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["day_sin"] = np.sin(2 * np.pi * df["day"] / 7)
df["day_cos"] = np.cos(2 * np.pi * df["day"] / 7)

# Encode weather once so both models use the same weather categories.
df = pd.get_dummies(
    df, columns=["weather_main"], drop_first=True, dtype=int
)
weather_features = [
    c for c in df.columns if c.startswith("weather_main_")
]

# 2. RANDOM FOREST BASELINE — SAME FEATURES/SETTINGS AS ORIGINAL CODE
rf_features = [
    "hour", "day", "temp", "rain_1h", "snow_1h", "clouds_all"
] + weather_features

X_rf = df[rf_features].astype("float32")
y = df["traffic_volume"].astype("float32")

X_rf_train, X_rf_test, y_train, y_test = train_test_split(
    X_rf, y, test_size=0.2, random_state=42
)

print("Training original Random Forest baseline...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_rf_train, y_train)
rf_pred = rf_model.predict(X_rf_test)

# 3. ANN INPUTS: ORIGINAL FEATURES + CYCLICAL TIME FEATURES
ann_features = [
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "temp", "rain_1h", "snow_1h", "clouds_all"
] + weather_features

X_ann = df[ann_features].astype("float32")
X_ann_train, X_ann_test, y_ann_train, y_ann_test = train_test_split(
    X_ann, y, test_size=0.2, random_state=42
)

# Fit scalers on training data only.
x_scaler = StandardScaler()
X_ann_train_scaled = x_scaler.fit_transform(X_ann_train).astype("float32")
X_ann_test_scaled = x_scaler.transform(X_ann_test).astype("float32")

# Scale target for stable neural-network optimization.
y_scaler = StandardScaler()
y_ann_train_scaled = y_scaler.fit_transform(
    np.asarray(y_ann_train).reshape(-1, 1)
).astype("float32")

# 4. IMPROVED ANN
ann_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_ann_train_scaled.shape[1],)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.10),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.10),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="linear"),
])

ann_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="mse",
    metrics=["mae"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        min_delta=0.0001,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=6,
        min_lr=1e-6,
        verbose=1
    )
]

print("\nTraining improved ANN...")
history = ann_model.fit(
    X_ann_train_scaled,
    y_ann_train_scaled,
    validation_split=0.20,
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Convert ANN predictions back to original vehicle-count units.
ann_pred_scaled = ann_model.predict(X_ann_test_scaled, verbose=0)
ann_pred = y_scaler.inverse_transform(ann_pred_scaled).ravel()

# 5. EVALUATION
def metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2 Score": r2_score(actual, predicted),
    }

comparison = pd.DataFrame({
    "Random Forest": metrics(y_test, rf_pred),
    "Improved ANN": metrics(y_ann_test, ann_pred),
}).T

print("\nMODEL COMPARISON")
print(comparison.round(3).to_string())
comparison.to_csv(OUT / "model_comparison.csv")

# Choose best model by R² for shared diagnostic plots.
best_name = (
    "Improved ANN"
    if comparison.loc["Improved ANN", "R2 Score"]
       >= comparison.loc["Random Forest", "R2 Score"]
    else "Random Forest"
)
best_pred = ann_pred if best_name == "Improved ANN" else rf_pred
actual = np.asarray(y_test)
errors = actual - best_pred

# 6. ACTUAL VS PREDICTED
plt.figure(figsize=(8, 6))
plt.scatter(actual, best_pred, alpha=0.35, s=10)
lo = min(float(actual.min()), float(np.min(best_pred)))
hi = max(float(actual.max()), float(np.max(best_pred)))
plt.plot([lo, hi], [lo, hi], "r--", label="Perfect prediction")
plt.xlabel("Actual Traffic Volume")
plt.ylabel("Predicted Traffic Volume")
plt.title(f"Actual vs Predicted Traffic ({best_name})")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "actual_vs_predicted.png", dpi=160)
plt.close()

# 7. ERROR DISTRIBUTION
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=50)
plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title(f"Error Distribution ({best_name})")
plt.tight_layout()
plt.savefig(OUT / "error_distribution.png", dpi=160)
plt.close()

# 8. ANN TRAINING CURVES
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Scaled MSE loss")
plt.title("ANN Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "ann_training_loss.png", dpi=160)
plt.close()

# 9. METRIC COMPARISON CHART
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, metric_name in zip(axes, ["MAE", "RMSE", "R2 Score"]):
    comparison[metric_name].plot(kind="bar", ax=ax, rot=0)
    ax.set_title(metric_name)
    ax.set_ylabel(metric_name)
plt.suptitle("Random Forest vs Improved ANN")
plt.tight_layout()
plt.savefig(OUT / "model_comparison.png", dpi=160)
plt.close()

# 10. SAVE MODELS AND PREPROCESSING
ann_model.save(OUT / "traffic_ann_model.keras")
joblib.dump(
    {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_columns": ann_features,
    },
    OUT / "ann_preprocessing.joblib"
)
joblib.dump(rf_model, OUT / "random_forest_model.joblib")

print(f"\nBest model by R²: {best_name}")
print(f"Outputs saved in: {OUT}")
print("Files: model_comparison.csv, actual_vs_predicted.png,")
print("error_distribution.png, ann_training_loss.png, model_comparison.png")
