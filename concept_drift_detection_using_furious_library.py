import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from frouros.detectors.concept_drift import DDM, DDMConfig
from frouros.metrics import PrequentialError
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(seed=31)

# Load and preprocess the data
data = pd.read_csv('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')
data = data[['close']]
data = np.log(data / data.shift(1))
data = data.dropna()
scaler = StandardScaler()
scaled_result = scaler.fit_transform(data.values.reshape(-1, 1))
data['close'] = pd.Series(scaled_result.flatten())
data['target'] = (data['close'].diff() > 0).astype(int).shift(-1).dropna()
data = data.dropna()

# Split the data into features and target
X = data[['close']].values
y = data['target'].values

# Split train (70%) and test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=31)

# Define and fit model
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ]
)
pipeline.fit(X=X_train, y=y_train)

# Detector configuration and instantiation
config = DDMConfig(
    warning_level=2.0,
    drift_level=3.0,
    min_num_instances=25,  
)
detector = DDM(config=config)

# Metric to compute accuracy
metric = PrequentialError(alpha=1.0) 

def stream_test(X_test, y_test, y, metric, detector):
    """Simulate data stream over X_test and y_test. y is the true label."""
    drift_points = []
    drift_flag = False
    for i, (X, y) in enumerate(zip(X_test, y_test)):
        y_pred = pipeline.predict(X.reshape(1, -1))
        error = 1 - (y_pred.item() == y.item())
        metric_error = metric(error_value=error)
        _ = detector.update(value=error)
        status = detector.status
        if status["drift"] and not drift_flag:
            drift_flag = True
            drift_points.append(i)
            print(f"Concept drift detected at step {i}. Accuracy: {1 - metric_error:.4f}")
    if not drift_flag:
        print("No concept drift detected")
    print(f"Final accuracy: {1 - metric_error:.4f}\n")
    return drift_points

# Simulate data stream (assuming test label available after each prediction)
# No concept drift is expected to occur
drift_points = stream_test(
    X_test=X_test,
    y_test=y_test,
    y=y,
    metric=metric,
    detector=detector,
)
# >> No concept drift detected
# >> Final accuracy: 0.9766

# IMPORTANT: Induce/simulate concept drift in the last part (20%)
# of y_test by modifying some labels (50% approx). Therefore, changing P(y|X))
drift_size = int(y_test.shape[0] * 0.2)
y_test_drift = y_test[-drift_size:]
modify_idx = np.random.rand(*y_test_drift.shape) <= 0.5
y_test_drift[modify_idx] = (y_test_drift[modify_idx] + 1) % len(np.unique(y_test))
y_test[-drift_size:] = y_test_drift

# Reset detector and metric
detector.reset()
metric.reset()

# Simulate data stream (assuming test label available after each prediction)
# Concept drift is expected to occur because of the label modification
drift_points = stream_test(
    X_test=X_test,
    y_test=y_test,
    y=y,
    metric=metric,
    detector=detector,
)
# >> Concept drift detected at step 142. Accuracy: 0.9510
# >> Final accuracy: 0.8480

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(y_test)), y_test, label='y_test', marker='.')
plt.scatter(drift_points, y_test[drift_points], color='red', label='Drift Points', zorder=5)
plt.xlabel('Index')
plt.ylabel('Target')
plt.title('Concept Drift Detection')
plt.legend()
plt.show()