# Short-Term Travel Time Prediction: A Numerical Experiment

## 1. Introduction

This report presents a numerical experiment on **short-term link-level travel time prediction** in a simulated traffic network.
We generate synthetic traffic data with substantial congestion using the mesoscopic traffic simulator [UXsim](https://github.com/toruseo/UXsim), then compare four prediction methods:

0. **Naive Baseline** — predicts that the next travel time equals the current travel time
1. **Linear Regression**
2. **Dense Neural Network** (Multi-Layer Perceptron)
3. **Long Short-Term Memory (LSTM)** Recurrent Neural Network

The prediction task is: **given the travel times of a link at the past 5 time-steps (each 30 seconds apart), predict the travel time at the next time-step.**

## 2. Mathematical Formulation of Methods

Let $\text{TT}_t$ denote the travel time at time-step $t$.  The input is the vector of past observations:

$$\mathbf{x}_t = (\text{TT}_{t-L}, \text{TT}_{t-L+1}, \dots, \text{TT}_{t-1})$$

where $L = 5$ is the lookback window.  The target is $\hat{y}_t \approx \text{TT}_t$.

### 2.0 Naive Baseline

The simplest possible predictor assumes travel time does not change:

$$\hat{y}_t = \text{TT}_{t-1}$$

This is a strong baseline because traffic conditions are often persistent over short intervals.

### 2.1 Linear Regression

Linear regression models the next travel time as a weighted sum of past travel times:

$$\hat{y}_t = \sum_{k=1}^{L} w_k \, \text{TT}_{t-k} + b$$

The parameters $\{w_k, b\}$ are found by minimising the ordinary least-squares objective:

$$\min_{\mathbf{w}, b} \sum_{i=1}^{N} \left( y_i - \mathbf{w}^\top \mathbf{x}_i - b \right)^2$$

### 2.2 Dense Neural Network (MLP)

A multi-layer perceptron applies successive non-linear transformations:

$$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \quad \in \mathbb{R}^{64}$$

$$\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2) \quad \in \mathbb{R}^{32}$$

$$\hat{y} = \mathbf{w}_3^\top \mathbf{h}_2 + b_3$$

where $\text{ReLU}(z) = \max(0, z)$.
The network is trained by minimising the mean squared error (MSE) loss via the Adam optimiser.

### 2.3 LSTM

The LSTM processes the sequence of $L$ past travel times $(\text{TT}_{t-L}, \dots, \text{TT}_{t-1})$.
At each step $k$, the LSTM cell updates its hidden state $\mathbf{h}_k$ and cell state $\mathbf{c}_k$:

$$\mathbf{f}_k = \sigma(\mathbf{W}_f [\mathbf{h}_{k-1}, x_k] + \mathbf{b}_f)$$

$$\mathbf{i}_k = \sigma(\mathbf{W}_i [\mathbf{h}_{k-1}, x_k] + \mathbf{b}_i)$$

$$\tilde{\mathbf{c}}_k = \tanh(\mathbf{W}_c [\mathbf{h}_{k-1}, x_k] + \mathbf{b}_c)$$

$$\mathbf{c}_k = \mathbf{f}_k \odot \mathbf{c}_{k-1} + \mathbf{i}_k \odot \tilde{\mathbf{c}}_k$$

$$\mathbf{o}_k = \sigma(\mathbf{W}_o [\mathbf{h}_{k-1}, x_k] + \mathbf{b}_o)$$

$$\mathbf{h}_k = \mathbf{o}_k \odot \tanh(\mathbf{c}_k)$$

where $\sigma$ is the sigmoid function and $\odot$ denotes element-wise multiplication.
The final hidden state $\mathbf{h}_L$ is passed through a dense layer to produce $\hat{y}$.

## 3. Experiment Design

### 3.1 Network Topology

The previous grid-based approach produced too much free-flow data (~72%) because the grid provides many alternative routes, spreading traffic thinly across peripheral links.

To address this, we adopt a **corridor-with-bottleneck** design: 10 parallel one-directional corridors, each containing a capacity bottleneck that forces queuing upstream.

| Parameter | Value |
|---|---|
| Corridors | 10 parallel one-directional corridors |
| Nodes per corridor | 8 (7 links each) |
| Total links | 70 |
| Link length | 300 m |
| Free-flow speed | 50 km/h (13.9 m/s) |
| Normal jam density | 0.2 veh/m |
| Bottleneck jam density | 0.05 veh/m (25% of normal capacity) |
| Free-flow travel time | 21.6 s per link |

Each corridor has **one bottleneck link** placed near the downstream end (positions 4–6, randomised per corridor).  Because traffic has no alternative route around the bottleneck, sustained demand above the bottleneck capacity creates queues that propagate upstream through most of the corridor.

### 3.2 Demand Generation

- **30 simulation scenarios** are run, each with a different `demand_scale` sampled uniformly from [1.5, 3.0].
- Each corridor receives one-directional demand from its origin to its destination node.
- A **trapezoidal demand profile** (ramp-up 0–10%, plateau 10–85%, wind-down 85–100%) maintains high demand for most of the simulation, ensuring sustained congestion rather than a brief peak.
- Base flow rates are randomised per corridor within [0.15, 0.35] veh/s (scaled by `demand_scale`).
- The simulation uses a platoon size (`deltan`) of 5 vehicles and runs for **3600 seconds (1 hour)**.

### 3.3 Prediction Task

This is a **short-term prediction** problem:

- **Input**: Travel times at the past $L = 5$ time-steps (covering 2.5 minutes at 30 s intervals)
- **Output**: Travel time at the next time-step (30 s ahead)
- Sequences are built per (scenario, link) group, sorted by time

### 3.4 Train/Test Split

- Data is split by **scenario**: 21 scenarios for training (~70%) and 9 scenarios for testing (~30%).
- This ensures the models generalise to unseen traffic conditions rather than memorising specific scenarios.

### 3.5 Model Configuration

| Hyperparameter | Linear Reg. | Dense NN | LSTM |
|---|---|---|---|
| Input features | 5 past TTs | 5 past TTs | 5 past TTs |
| Hidden layers | — | 2 (64, 32 units) | 1 LSTM (64 units) + 1 Dense (32) |
| Activation | — | ReLU | tanh/sigmoid (LSTM) + ReLU |
| Optimiser | — | Adam | Adam |
| Loss function | OLS | MSE | MSE |
| Epochs | — | 80 | 80 |
| Batch size | — | 256 | 256 |

All features are standardised (zero mean, unit variance) before training.

## 4. Results

### 4.1 Dataset Statistics

| Statistic | Value |
|---|---|
| Total records | 245,016 |
| Training records | 171,671 |
| Test records | 73,345 |
| Train sequences | 164,321 |
| Test sequences | 70,195 |
| At free-flow (TT ≤ 1.01× free-flow) | 45.9% |
| Congested records (TT > 1.1× free-flow) | 49.3% |
| Heavily congested (TT > 1.5× free-flow) | 40.2% |
| Travel time range | 21.6 – 105.0 s |
| Mean travel time | 45.4 s |
| Travel time std. dev. | 30.9 s |

Compared to the earlier grid-based approach (72% free-flow, 24% congested), the corridor-with-bottleneck design produces roughly **equal amounts of free-flow and congested data**, making the prediction problem substantially more meaningful.

### 4.2 Performance Metrics

| Method | MAE (s) | RMSE (s) | R² |
|---|---|---|---|
| Naive Baseline | **2.524** | 6.475 | 0.958 |
| Linear Regression | 3.050 | 6.359 | 0.960 |
| Dense NN | 2.952 | 5.649 | 0.968 |
| LSTM | 2.701 | **5.292** | **0.972** |

- **MAE** (Mean Absolute Error): average absolute prediction error.
- **RMSE** (Root Mean Squared Error): penalises large errors more heavily.
- **R²** (Coefficient of Determination): proportion of variance explained (1.0 = perfect).

**Key findings:**

1. **LSTM achieves the best RMSE and R²**, outperforming the naive baseline by 18% on RMSE (5.29 vs 6.47) and achieving R² = 0.972 vs 0.958.  The Dense NN also substantially outperforms the baseline (RMSE 5.65 vs 6.47, R² = 0.968).

2. **The naive baseline has the lowest MAE** (2.52 s) because it is exactly correct whenever travel time does not change between consecutive time-steps — which is the majority of observations during stable conditions.  However, its high RMSE reveals that it makes **large errors during congestion transitions** (onset and dissipation), which is precisely when accurate prediction matters most.

3. **RMSE is the more informative metric** for this problem because it penalises the large errors during congestion transitions that the naive baseline cannot handle.  The LSTM's 18% RMSE reduction means substantially better predictions during the critical periods.

4. **Linear Regression** improves over the naive baseline on RMSE and R² but is limited by its inability to capture the non-linear dynamics of queue formation and dissipation.

5. **All R² values are very high** (≥ 0.958), reflecting that the corridor-with-bottleneck design produces a wide range of travel times with clear temporal patterns that all methods can leverage.

### 4.3 Visualisations

#### Performance Comparison

![Metrics comparison bar chart](results/metrics_comparison.png)

#### Predicted vs. Actual Travel Time

![Scatter plots of predicted vs actual travel times](results/scatter_pred_vs_actual.png)

The diagonal red dashed line represents perfect prediction.  The naive baseline shows a characteristic pattern of echoing the previous value, which works well during stable periods but creates scatter during transitions.  The LSTM and Dense NN produce tighter clustering around the diagonal.

#### Training Loss Curves

![Training loss curves for Dense NN and LSTM](results/training_loss.png)

Both models converge within the first 20 epochs, with the gap between training and validation loss indicating reasonable generalisation.

#### Travel Time Distribution

![Histogram of travel times in the dataset](results/travel_time_distribution.png)

The distribution is bimodal, reflecting the two traffic states in the kinematic wave model: free-flow (≈21.6 s) and queued (≈80–95 s).  The corridor-with-bottleneck design ensures that roughly half the data represents congested conditions.

#### Example Time-Series

![Example time-series of travel time](results/example_time_series.png)

An example of travel time evolution on a single link during one test scenario, showing the congestion build-up driven by the bottleneck and its eventual dissipation.

## 5. Conclusion

This experiment demonstrates short-term travel time prediction using past travel time observations on a congestion-rich corridor network:

1. **The corridor-with-bottleneck network design** resolves the problem of excessive free-flow data that plagued the grid-based approach.  By eliminating alternative routes, the bottleneck creates sustained congestion on upstream links, yielding roughly 50/50 free-flow and congested observations.

2. **LSTM and Dense NN outperform the naive baseline** on RMSE (18% and 13% reduction respectively) and R², demonstrating that learned models capture temporal dynamics of congestion transitions that a simple persistence forecast cannot.

3. **The naive baseline is competitive on MAE** because it is perfect during stable conditions, but its high RMSE reveals critical failures during congestion onset and dissipation — the periods when accurate forecasts are most valuable.

4. The **bimodal travel time distribution** is a physical property of the kinematic wave traffic model, where links transition sharply between free-flow and queued states.

## Reproducibility

All code and data are included in this repository:

```
├── generate_data.py          # UXsim simulation and data generation
├── train_and_evaluate.py     # Model training, evaluation, and plotting
├── requirements.txt          # Python dependencies
├── data/
│   └── traffic_data.csv      # Generated synthetic traffic data
├── results/
│   ├── metrics.csv           # Numerical results
│   ├── metrics_comparison.png
│   ├── scatter_pred_vs_actual.png
│   ├── training_loss.png
│   ├── travel_time_distribution.png
│   └── example_time_series.png
└── report.md                 # This report
```

To reproduce:

```bash
pip install -r requirements.txt
python generate_data.py
python train_and_evaluate.py
```
