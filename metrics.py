import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
df["date_time"] = pd.to_datetime(df["date_time"])

# Feature engineering
df["hour"] = df["date_time"].dt.hour
df["day"] = df["date_time"].dt.dayofweek

# One-hot encoding
df = pd.get_dummies(df, columns=["weather_main"], drop_first=True)

features = ["hour", "day", "temp", "rain_1h", "snow_1h", "clouds_all"] + \
           [c for c in df.columns if "weather_main_" in c]

X = df[features]
y = df["traffic_volume"]

# -------------------------------
# TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL
# -------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -------------------------------
# METRICS
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# -------------------------------
# 📈 1. ACTUAL vs PREDICTED PLOT
# -------------------------------
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Traffic")
plt.ylabel("Predicted Traffic")
plt.title("Actual vs Predicted Traffic")

# Line (perfect prediction)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle="--")

plt.savefig("actual_vs_predicted.png")
plt.close()

# -------------------------------
# 📉 2. ERROR DISTRIBUTION
# -------------------------------
errors = y_test - y_pred

plt.figure()
plt.hist(errors, bins=50)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Distribution")

plt.savefig("error_distribution.png")
plt.close()

# -------------------------------
# 📊 3. METRICS IMAGE
# -------------------------------
plt.figure()
plt.axis("off")

text = f"""
Model Evaluation Metrics

MAE: {mae:.2f}
MSE: {mse:.2f}
R² Score: {r2:.2f}
"""

plt.text(0.1, 0.5, text, fontsize=14)

plt.savefig("metrics_summary.png")
plt.close()

print("✅ Images saved:")
print("- actual_vs_predicted.png")
print("- error_distribution.png")
print("- metrics_summary.png")