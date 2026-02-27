"""
Train and evaluate short-term travel time prediction models.

Problem formulation:
  Given the travel times of a link at the past LOOKBACK time-steps,
  predict the travel time at the next time-step.

Methods compared:
  0. Naive Baseline  (predict TT_next = TT_current)
  1. Linear Regression
  2. Dense Neural Network (Multi-Layer Perceptron)
  3. LSTM (Long Short-Term Memory)

Reads the synthetic traffic data produced by generate_data.py,
splits into train/test sets by scenario, trains each model,
evaluates performance, and saves results and figures.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
DATA_PATH = "data/traffic_data.csv"
RESULTS_DIR = "results"
LOOKBACK = 5     # number of past time-steps used as input
EPOCHS_DENSE = 80
EPOCHS_LSTM = 80
BATCH_SIZE = 256
TEST_SCENARIO_FRAC = 0.3  # fraction of scenarios for testing

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ===== helpers =============================================================

def build_sequences(df, lookback):
    """Build (X, y) arrays for short-term TT prediction.

    For each (scenario, link) group sorted by time, create sliding
    windows of length ``lookback`` from the travel_time column.

    X shape: (N, lookback)  — past travel times
    y shape: (N,)           — next travel time
    """
    X_all, y_all = [], []
    for (sc, lk), grp in df.groupby(["scenario", "link"]):
        tt = grp.sort_values("time")["travel_time"].values.astype(np.float32)
        for i in range(lookback, len(tt)):
            X_all.append(tt[i - lookback:i])
            y_all.append(tt[i])
    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.float32)


def build_dense_model(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_lstm_model(lookback, n_features=1):
    model = keras.Sequential([
        keras.layers.LSTM(64, input_shape=(lookback, n_features)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:25s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return {"method": label, "MAE": mae, "RMSE": rmse, "R2": r2}


# ===== main ================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records from {DATA_PATH}")

    ff_tt = df["free_flow_tt"].iloc[0]
    pct_cong = (df["travel_time"] > ff_tt * 1.1).mean() * 100
    print(f"Free-flow TT: {ff_tt:.1f}s, congested records: {pct_cong:.1f}%")

    # ------------------------------------------------------------------
    # 2. Train / test split by scenario
    # ------------------------------------------------------------------
    scenarios = df["scenario"].unique()
    np.random.shuffle(scenarios)
    n_test = max(1, int(len(scenarios) * TEST_SCENARIO_FRAC))
    test_scenarios = set(scenarios[:n_test])
    train_scenarios = set(scenarios[n_test:])

    df_train = df[df["scenario"].isin(train_scenarios)].copy()
    df_test = df[df["scenario"].isin(test_scenarios)].copy()

    print(f"Train scenarios: {sorted(train_scenarios)}")
    print(f"Test  scenarios: {sorted(test_scenarios)}")
    print(f"Train records : {len(df_train)}")
    print(f"Test  records : {len(df_test)}")

    # ------------------------------------------------------------------
    # 3. Build sliding-window sequences
    # ------------------------------------------------------------------
    print("\nBuilding sequences ...")
    X_train, y_train = build_sequences(df_train, LOOKBACK)
    X_test, y_test = build_sequences(df_test, LOOKBACK)
    print(f"  Train sequences: {X_train.shape}")
    print(f"  Test  sequences: {X_test.shape}")

    results = []

    # ------------------------------------------------------------------
    # 4a. Naive Baseline: predict TT_next = TT_current
    # ------------------------------------------------------------------
    print("\n--- Naive Baseline (TT_next = TT_current) ---")
    y_pred_naive = X_test[:, -1]  # last element of lookback window
    results.append(evaluate(y_test, y_pred_naive, "Naive Baseline"))

    # ------------------------------------------------------------------
    # 4b. Linear Regression
    # ------------------------------------------------------------------
    print("\n--- Linear Regression ---")
    scaler_lr = StandardScaler()
    X_train_sc = scaler_lr.fit_transform(X_train)
    X_test_sc = scaler_lr.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)
    results.append(evaluate(y_test, y_pred_lr, "Linear Regression"))

    # ------------------------------------------------------------------
    # 4c. Dense Neural Network
    # ------------------------------------------------------------------
    print("\n--- Dense Neural Network ---")
    model_dense = build_dense_model(LOOKBACK)
    history_dense = model_dense.fit(
        X_train_sc, y_train,
        validation_split=0.1,
        epochs=EPOCHS_DENSE,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    y_pred_dense = model_dense.predict(X_test_sc, verbose=0).flatten()
    results.append(evaluate(y_test, y_pred_dense, "Dense NN"))

    # ------------------------------------------------------------------
    # 4d. LSTM
    # ------------------------------------------------------------------
    print("\n--- LSTM ---")
    # LSTM expects (samples, timesteps, features)
    scaler_lstm = StandardScaler()
    X_train_flat = scaler_lstm.fit_transform(X_train)
    X_test_flat = scaler_lstm.transform(X_test)
    X_train_lstm = X_train_flat.reshape(-1, LOOKBACK, 1)
    X_test_lstm = X_test_flat.reshape(-1, LOOKBACK, 1)

    model_lstm = build_lstm_model(LOOKBACK, n_features=1)
    history_lstm = model_lstm.fit(
        X_train_lstm, y_train,
        validation_split=0.1,
        epochs=EPOCHS_LSTM,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    y_pred_lstm = model_lstm.predict(X_test_lstm, verbose=0).flatten()
    results.append(evaluate(y_test, y_pred_lstm, "LSTM"))

    # ------------------------------------------------------------------
    # 5. Save results table
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
    print("\n" + results_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    # 6a. Bar chart of metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    methods = results_df["method"]
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        bars = ax.bar(methods, results_df[metric])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_ha("right")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "metrics_comparison.png"), dpi=150)
    plt.close(fig)

    # 6b. Scatter: predicted vs actual for each method
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    preds = [
        ("Naive Baseline", y_test, y_pred_naive),
        ("Linear Regression", y_test, y_pred_lr),
        ("Dense NN", y_test, y_pred_dense),
        ("LSTM", y_test, y_pred_lstm),
    ]
    for ax, (name, yt, yp) in zip(axes, preds):
        ax.scatter(yt, yp, alpha=0.05, s=4)
        mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
        ax.set_xlabel("Actual travel time (s)")
        ax.set_ylabel("Predicted travel time (s)")
        ax.set_title(name)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "scatter_pred_vs_actual.png"),
                dpi=150)
    plt.close(fig)

    # 6c. Training loss curves for NN models
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, hist, name in [
        (axes[0], history_dense, "Dense NN"),
        (axes[1], history_lstm, "LSTM"),
    ]:
        ax.plot(hist.history["loss"], label="Train")
        ax.plot(hist.history["val_loss"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"{name} Training Loss")
        ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "training_loss.png"), dpi=150)
    plt.close(fig)

    # 6d. Travel time distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["travel_time"], bins=80, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Travel time (s)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Travel Times in Dataset")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "travel_time_distribution.png"),
                dpi=150)
    plt.close(fig)

    # 6e. Example time-series for a single link in one test scenario
    fig, ax = plt.subplots(figsize=(10, 4))
    test_sc = sorted(test_scenarios)[0]
    example = df[df["scenario"] == test_sc]
    link_name = example["link"].value_counts().index[0]
    ts = example[example["link"] == link_name].sort_values("time")
    ax.plot(ts["time"], ts["travel_time"], "b-", linewidth=1,
            label="Actual TT")
    ax.axhline(ff_tt, color="gray", linestyle="--", linewidth=0.8,
               label=f"Free-flow ({ff_tt:.0f}s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Travel time (s)")
    ax.set_title(f"Travel Time Profile — Scenario {test_sc}, Link {link_name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "example_time_series.png"), dpi=150)
    plt.close(fig)

    print(f"\nResults and figures saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
