# Travel Time Prediction Using Various Methods: A Numerical Experiment

## 1. Introduction

This report presents a numerical experiment on link-level travel time prediction in a simulated urban traffic network.
We generate synthetic traffic data using the mesoscopic traffic simulator [UXsim](https://github.com/toruseo/UXsim), then compare three prediction methods:

1. **Linear Regression**
2. **Dense Neural Network** (Multi-Layer Perceptron)
3. **Long Short-Term Memory (LSTM)** Recurrent Neural Network

## 2. Mathematical Formulation of Methods

### 2.1 Linear Regression

Linear regression models the travel time $\hat{y}$ as a linear combination of input features:

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b$$

where $\mathbf{x} \in \mathbb{R}^d$ is the feature vector, $\mathbf{w} \in \mathbb{R}^d$ is the weight vector, and $b$ is the bias.
The parameters are found by minimising the ordinary least-squares objective:

$$\min_{\mathbf{w}, b} \sum_{i=1}^{N} \left( y_i - \mathbf{w}^\top \mathbf{x}_i - b \right)^2$$

### 2.2 Dense Neural Network (MLP)

A multi-layer perceptron applies successive non-linear transformations:

$$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \quad \in \mathbb{R}^{64}$$

$$\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2) \quad \in \mathbb{R}^{32}$$

$$\hat{y} = \mathbf{w}_3^\top \mathbf{h}_2 + b_3$$

where $\text{ReLU}(z) = \max(0, z)$.
The network is trained by minimising the mean squared error (MSE) loss via the Adam optimiser.

### 2.3 LSTM

The LSTM processes a sequence of $T$ past observations $(\mathbf{x}_{t-T+1}, \dots, \mathbf{x}_t)$.
At each step $k$, the LSTM cell updates its hidden state $\mathbf{h}_k$ and cell state $\mathbf{c}_k$:

$$\mathbf{f}_k = \sigma(\mathbf{W}_f [\mathbf{h}_{k-1}, \mathbf{x}_k] + \mathbf{b}_f)$$

$$\mathbf{i}_k = \sigma(\mathbf{W}_i [\mathbf{h}_{k-1}, \mathbf{x}_k] + \mathbf{b}_i)$$

$$\tilde{\mathbf{c}}_k = \tanh(\mathbf{W}_c [\mathbf{h}_{k-1}, \mathbf{x}_k] + \mathbf{b}_c)$$

$$\mathbf{c}_k = \mathbf{f}_k \odot \mathbf{c}_{k-1} + \mathbf{i}_k \odot \tilde{\mathbf{c}}_k$$

$$\mathbf{o}_k = \sigma(\mathbf{W}_o [\mathbf{h}_{k-1}, \mathbf{x}_k] + \mathbf{b}_o)$$

$$\mathbf{h}_k = \mathbf{o}_k \odot \tanh(\mathbf{c}_k)$$

where $\sigma$ is the sigmoid function and $\odot$ denotes element-wise multiplication.
The final hidden state $\mathbf{h}_T$ is passed through a dense layer to produce $\hat{y}$.

## 3. Experiment Design

### 3.1 Network Topology

A **5 × 5 grid network** is constructed with:

| Parameter | Value |
|---|---|
| Grid size | 5 × 5 (25 nodes) |
| Number of links | 80 (bi-directional) |
| Link length | 500 m |
| Free-flow speed | 50 km/h (13.9 m/s) |
| Jam density | 0.2 veh/m |
| Free-flow travel time | 36.0 s per link |

### 3.2 Demand Generation

- **30 simulation scenarios** are run, each with a different `demand_scale` sampled uniformly from [0.5, 1.8].
- Origin-destination (OD) pairs are selected between boundary nodes with a Manhattan distance ≥ 3.
- For each OD pair, demand is generated in 300-second intervals with a base flow rate randomised within [0.01, 0.03] veh/s per OD pair (scaled by `demand_scale`), and further perturbed by a factor drawn from [0.7, 1.3] at each interval.
- The simulation uses a platoon size (`deltan`) of 5 vehicles and runs for 3600 seconds (1 hour).
- This setup produces **moderate congestion** on many links without causing network-wide gridlock.

### 3.3 Data Collected

For each link at 30-second intervals, the following are recorded:

| Feature | Description |
|---|---|
| `time` | Simulation time (s) |
| `density` | Estimated link density (veh/m) |
| `num_vehicles` | Number of vehicles on the link |
| `demand_scale` | Scenario-level demand multiplier |
| `travel_time` | Instantaneous link travel time (s) — **prediction target** |

Travel times exceeding 5× the free-flow travel time are filtered out as unrealistic.

### 3.4 Train/Test Split

- Data is split by **scenario**: 21 scenarios for training (~70%) and 9 scenarios for testing (~30%).
- This ensures the models generalise to unseen traffic conditions rather than memorising specific scenarios.

### 3.5 Model Configuration

| Hyperparameter | Dense NN | LSTM |
|---|---|---|
| Hidden layers | 2 (64, 32 units) | 1 LSTM (64 units) + 1 Dense (32) |
| Activation | ReLU | tanh/sigmoid (LSTM) + ReLU |
| Optimiser | Adam | Adam |
| Loss function | MSE | MSE |
| Epochs | 50 | 50 |
| Batch size | 256 | 256 |
| Lookback (LSTM only) | — | 5 time-steps |

All features are standardised (zero mean, unit variance) before training.

## 4. Results

### 4.1 Dataset Statistics

| Statistic | Value |
|---|---|
| Total records | 285,061 |
| Training records | 199,705 |
| Test records | 85,356 |
| Congested records (TT > 1.1× free-flow) | 10.4% |
| Travel time range | 36.0 – 180.0 s |
| Mean travel time | 38.0 s |
| Travel time std. dev. | 8.2 s |

### 4.2 Performance Metrics

| Method | MAE (s) | RMSE (s) | R² |
|---|---|---|---|
| Linear Regression | 3.729 | 8.396 | 0.028 |
| Dense NN | 4.009 | 8.162 | 0.082 |
| LSTM | **3.592** | **8.078** | **0.136** |

- **MAE** (Mean Absolute Error): average absolute prediction error.
- **RMSE** (Root Mean Squared Error): penalises large errors more heavily.
- **R²** (Coefficient of Determination): proportion of variance explained (1.0 = perfect).

**Key findings:**

- The **LSTM achieves the best performance** across all three metrics, benefiting from its ability to capture temporal dependencies in the traffic state sequences.
- The **Dense NN** outperforms Linear Regression on RMSE and R², demonstrating that non-linear relationships improve fit.
- **Linear Regression** has the lowest MAE among the simpler models but limited capacity to capture complex congestion dynamics.
- Overall R² values are moderate because the majority of observations (~90%) are near free-flow conditions where travel time has little variation. The models differentiate most in their ability to predict congested conditions.

### 4.3 Visualisations

#### Performance Comparison

![Metrics comparison bar chart](results/metrics_comparison.png)

#### Predicted vs. Actual Travel Time

![Scatter plots of predicted vs actual travel times](results/scatter_pred_vs_actual.png)

The diagonal red dashed line represents perfect prediction. Points above the line indicate over-prediction; points below indicate under-prediction.

#### Training Loss Curves

![Training loss curves for Dense NN and LSTM](results/training_loss.png)

Both models converge within the first 10 epochs. The gap between training and validation loss indicates moderate generalisation ability.

#### Travel Time Distribution

![Histogram of travel times in the dataset](results/travel_time_distribution.png)

The distribution is heavily right-skewed: most observations are at the free-flow travel time (36 s), with a long tail representing congested conditions.

## 5. Conclusion

This experiment demonstrates the comparative performance of three travel time prediction methods on synthetic traffic data generated by UXsim:

1. **LSTM** provides the best predictive accuracy by leveraging temporal patterns in traffic state evolution.
2. **Dense Neural Networks** capture non-linear relationships but do not exploit temporal ordering.
3. **Linear Regression** serves as a simple baseline with competitive MAE performance.

The relatively low R² values across all methods reflect the challenge of predicting travel time when most observations are at free-flow conditions. Future work could explore richer feature engineering (e.g., upstream/downstream link states), longer lookback windows, or attention-based architectures to further improve prediction under congested conditions.

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
│   └── travel_time_distribution.png
└── report.md                 # This report
```

To reproduce:

```bash
pip install -r requirements.txt
python generate_data.py
python train_and_evaluate.py
```
