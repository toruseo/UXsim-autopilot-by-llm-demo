"""
Train and evaluate travel time prediction models.

Methods compared:
  1. Linear Regression
  2. Dense Neural Network (Multi-Layer Perceptron)
  3. LSTM (Long Short-Term Memory)

Reads the synthetic traffic data produced by generate_data.py,
splits into train/test sets, trains each model, evaluates performance,
and saves results and figures.
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
LOOKBACK = 5  # number of past time-steps for LSTM
EPOCHS_DENSE = 50
EPOCHS_LSTM = 50
BATCH_SIZE = 256
TEST_SCENARIO_FRAC = 0.3  # fraction of scenarios for testing

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ===== helpers =============================================================

def prepare_features(df):
    """Return feature matrix and target vector for non-sequential models."""
    feature_cols = ["time", "density", "num_vehicles", "demand_scale"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["travel_time"].values.astype(np.float32)
    return X, y


def create_sequences(X_link, y_link, lookback):
    """Create (lookback, n_features) sequences for a single link time-series."""
    Xs, ys = [], []
    for i in range(lookback, len(X_link)):
        Xs.append(X_link[i - lookback:i])
        ys.append(y_link[i])
    if len(Xs) == 0:
        return np.empty((0, lookback, X_link.shape[1])), np.empty((0,))
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def build_dense_model(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_lstm_model(lookback, n_features):
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
    # 3. Feature engineering
    # ------------------------------------------------------------------
    X_train, y_train = prepare_features(df_train)
    X_test, y_test = prepare_features(df_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    results = []

    # ------------------------------------------------------------------
    # 4a. Linear Regression
    # ------------------------------------------------------------------
    print("\n--- Linear Regression ---")
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)
    results.append(evaluate(y_test, y_pred_lr, "Linear Regression"))

    # ------------------------------------------------------------------
    # 4b. Dense Neural Network
    # ------------------------------------------------------------------
    print("\n--- Dense Neural Network ---")
    model_dense = build_dense_model(X_train_sc.shape[1])
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
    # 4c. LSTM
    # ------------------------------------------------------------------
    print("\n--- LSTM ---")
    feature_cols_seq = ["time", "density", "num_vehicles", "demand_scale"]

    # Build sequences per (scenario, link)
    scaler_lstm = StandardScaler()
    scaler_lstm.fit(df_train[feature_cols_seq].values.astype(np.float32))

    def make_seq_data(df_part):
        X_all, y_all = [], []
        for (sc, lk), grp in df_part.groupby(["scenario", "link"]):
            grp = grp.sort_values("time")
            Xf = scaler_lstm.transform(
                grp[feature_cols_seq].values.astype(np.float32))
            yf = grp["travel_time"].values.astype(np.float32)
            Xs, ys = create_sequences(Xf, yf, LOOKBACK)
            if len(Xs) > 0:
                X_all.append(Xs)
                y_all.append(ys)
        return np.concatenate(X_all), np.concatenate(y_all)

    X_train_seq, y_train_seq = make_seq_data(df_train)
    X_test_seq, y_test_seq = make_seq_data(df_test)
    print(f"  LSTM train sequences: {X_train_seq.shape}")
    print(f"  LSTM test  sequences: {X_test_seq.shape}")

    model_lstm = build_lstm_model(LOOKBACK, X_train_seq.shape[2])
    history_lstm = model_lstm.fit(
        X_train_seq, y_train_seq,
        validation_split=0.1,
        epochs=EPOCHS_LSTM,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    y_pred_lstm = model_lstm.predict(X_test_seq, verbose=0).flatten()
    results.append(evaluate(y_test_seq, y_pred_lstm, "LSTM"))

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
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    methods = results_df["method"]
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        ax.bar(methods, results_df[metric])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        for tick in ax.get_xticklabels():
            tick.set_rotation(15)
            tick.set_ha("right")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "metrics_comparison.png"), dpi=150)
    plt.close(fig)

    # 6b. Scatter: predicted vs actual for each method
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    preds = [
        ("Linear Regression", y_test, y_pred_lr),
        ("Dense NN", y_test, y_pred_dense),
        ("LSTM", y_test_seq, y_pred_lstm),
    ]
    for ax, (name, yt, yp) in zip(axes, preds):
        ax.scatter(yt, yp, alpha=0.1, s=4)
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
    ax.hist(df["travel_time"], bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Travel time (s)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Travel Times in Dataset")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "travel_time_distribution.png"),
                dpi=150)
    plt.close(fig)

    print(f"\nResults and figures saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
