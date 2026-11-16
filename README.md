# AI-Based Traffic Flow Optimization

> Deep learning-driven traffic volume prediction and adaptive signal allocation using an Artificial Neural Network trained on the Metro Interstate Traffic Volume dataset.

---

## Project Overview

This project applies a Multi-Layer Perceptron (ANN) to predict hourly traffic volume at road intersections using temporal and meteorological features. The predicted volumes are then used by an adaptive signal allocation module to assign green-signal durations proportional to demand — replacing static fixed-time signal cycles with data-driven, dynamic control.

A Random Forest model is trained in parallel as a baseline, enabling direct benchmarking of deep learning performance against a classical ensemble method.

A separate Streamlit simulation (`app.py`) demonstrates the adaptive vs. fixed-time controller comparison in a live interactive dashboard.

---

## Problem Statement

Urban intersections still predominantly use fixed-time traffic signals that allocate equal green time to all lanes regardless of actual traffic demand. This causes:

- Unnecessary waiting at low-traffic lanes while high-traffic lanes are under-served
- Increased fuel consumption and emissions during peak hours
- Reduced overall intersection throughput

**Goal:** Predict hourly traffic volume using deep learning and allocate green-signal time proportionally to predicted demand.

---

## ANN Architecture

```
Input (7+ features)
    │
    ▼
Dense(128, activation=ReLU) → BatchNormalization → Dropout(0.10)
    │
    ▼
Dense(64, activation=ReLU) → BatchNormalization → Dropout(0.10)
    │
    ▼
Dense(32, activation=ReLU)
    │
    ▼
Output(1, activation=Linear)  →  Predicted Traffic Volume
```

| Component        | Detail                                      |
|------------------|---------------------------------------------|
| Framework        | TensorFlow 2.x / Keras                      |
| Optimizer        | Adam (lr=0.0005)                            |
| Loss Function    | Mean Squared Error (MSE)                    |
| Batch Size       | 32                                          |
| Max Epochs       | 150 (Early Stopping, patience=15)           |
| Regularization   | Dropout(0.10), BatchNormalization           |
| Callbacks        | EarlyStopping, ReduceLROnPlateau            |

---

## Dataset

**Metro Interstate Traffic Volume**  
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)

- Hourly westbound traffic volume on I-94 ATR 301, Minnesota, USA (2012–2018)
- 48,204 records

### Input Features

| Feature        | Description                                      |
|----------------|--------------------------------------------------|
| `hour_sin`     | Cyclical sine encoding of hour of day            |
| `hour_cos`     | Cyclical cosine encoding of hour of day          |
| `day_sin`      | Cyclical sine encoding of day of week            |
| `day_cos`      | Cyclical cosine encoding of day of week          |
| `temp`         | Average temperature (Kelvin)                     |
| `rain_1h`      | Rainfall in mm over the last hour                |
| `snow_1h`      | Snowfall in mm over the last hour                |
| `clouds_all`   | Cloud cover percentage                           |
| `weather_main_*` | One-hot encoded weather category columns       |

**Target:** `traffic_volume` — hourly vehicle count

---

## Data Preprocessing

1. Parse `date_time` column; drop null/duplicate rows
2. Extract `hour` and `day` (day-of-week); encode as cyclical sine/cosine features
3. One-hot encode `weather_main` categorical column
4. Apply `StandardScaler` to ANN input features (fit on training set only)
5. Scale target variable with a separate `StandardScaler` for stable ANN optimization
6. 80/20 train-test split (`random_state=42`)

---

## Adaptive Signal Allocation

After prediction, the road with the highest predicted volume receives the longest green phase:

```
Green_Time(road_i) = (Volume_i / Total_Volume) × Total_Cycle_Time
```

The Streamlit dashboard (`app.py`) simulates this in real time across four lanes (N, S, E, W), comparing the adaptive controller against a fixed-time baseline.

---

## Evaluation Metrics

Results are computed on the held-out 20% test set. Run `traffic_deep_learning_v2.py` to reproduce:

| Metric    | Random Forest | Improved ANN |
|-----------|---------------|--------------|
| MAE       | see CSV       | see CSV      |
| RMSE      | see CSV       | see CSV      |
| R² Score  | see CSV       | see CSV      |

> Exact metric values are written to `traffic_model_outputs_v2/model_comparison.csv` after training. They are not hard-coded here to avoid misrepresentation across different runs or environments.

---

## Project Structure

```
AI-Based-Traffic-Flow-Optimization/
│
├── traffic_deep_learning_v2.py        # ANN + Random Forest training pipeline
├── app.py                             # Streamlit adaptive signal simulator
├── metrics.py                         # Standalone Random Forest metrics script
├── Metro_Interstate_Traffic_Volume.csv # Dataset (UCI)
│
├── traffic_model_outputs_v2/          # Generated after running training script
│   ├── model_comparison.csv           # MAE / RMSE / R² for both models
│   ├── actual_vs_predicted.png        # Scatter plot: actual vs predicted
│   ├── ann_training_loss.png          # Training & validation loss curves
│   ├── error_distribution.png         # Prediction error histogram
│   └── model_comparison.png           # Bar chart: ANN vs Random Forest
│
├── screenshots/                       # Dashboard screenshots
├── Neural_AI_Based_DL_Traffic_Opt.pptx  # Project presentation
├── AI_Based_Traffic_Flow_Optimization_NNDL_Report.docx  # Full project report
│
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** Trained model files (`*.keras`, `*.joblib`) are excluded from version control due to size. Re-run `traffic_deep_learning_v2.py` to regenerate them locally.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/MOHAMEDMUSTAK/AI-Based-Traffic-Flow-Optimization.git
cd AI-Based-Traffic-Flow-Optimization

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Train the ANN and generate evaluation outputs

```bash
python traffic_deep_learning_v2.py
```

Outputs saved to `traffic_model_outputs_v2/`:
- `model_comparison.csv` — metric comparison table
- `actual_vs_predicted.png` — scatter plot
- `ann_training_loss.png` — loss curves
- `error_distribution.png` — error histogram
- `model_comparison.png` — bar chart

### Run the adaptive signal simulator

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` — select traffic mode (Normal / Rush Hour / Night Mode) and observe adaptive vs. fixed controller performance in real time.

### Run the standalone metrics script

```bash
python metrics.py
```

---

## Contributors

| Name               |
|--------------------|
| Mohamed Mustak M   |
| Charukesh J        |
| Surya Narayana     |
---

## Research Relevance

- Intelligent Transportation Systems (ITS)
- Adaptive Signal Control Technology (ASCT)
- Urban Congestion Management
- Smart City Infrastructure

---

## Future Work

- Reinforcement Learning (DQN/PPO) signal controller trained in SUMO simulation
- LSTM / Transformer models for temporal sequence prediction
- Multi-intersection coordination using Graph Neural Networks
- Integration with real-time IoT sensor data streams
- Emergency vehicle preemption logic
