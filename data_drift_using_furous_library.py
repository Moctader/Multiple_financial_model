import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from frouros.detectors.data_drift import KSTest
import matplotlib.pyplot as plt
import seaborn as sns

# Load the new dataset
data = pd.read_csv('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv').head(1000)
data = data[['close']]

# Apply differencing to make the data stationary
data['close_diff'] = data['close'].diff().dropna()

# Create a target variable for demonstration purposes
# Here, we create a binary target based on whether the differenced close price increased or decreased
data['target'] = (data['close_diff'].diff() > 0).astype(int).shift(-1).dropna()
data = data.dropna()

# Split the data into features and target
X = data[['close_diff']].values
y = data['target'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=31)

# Set the feature index to which detector is applied
feature_idx = 0

# Visualize the distribution before inducing data drift
plt.figure(figsize=(12, 6))
sns.histplot(X_test[:, feature_idx], kde=True, color='blue', label='Before Drift')
plt.title('Distribution of Feature Before Inducing Data Drift')
plt.legend()
plt.show()

# IMPORTANT: Induce/simulate data drift in the selected feature of y_test by
# applying some gaussian noise. Therefore, changing P(X))
X_test[:, feature_idx] += np.random.normal(
    loc=0.0,
    scale=3.0,
    size=X_test.shape[0],
)

# Visualize the distribution after inducing data drift
plt.figure(figsize=(12, 6))
sns.histplot(X_test[:, feature_idx], kde=True, color='red', label='After Drift')
plt.title('Distribution of Feature After Inducing Data Drift')
plt.legend()
plt.show()

# Define and fit model
model = RandomForestClassifier(random_state=31)
model.fit(X=X_train, y=y_train)

# Set significance level for hypothesis testing
alpha = 0.001
# Define and fit detector
detector = KSTest()
_ = detector.fit(X=X_train[:, feature_idx])

# Apply detector to the selected feature of X_test
result, _ = detector.compare(X=X_test[:, feature_idx])

# Check if drift is taking place
if result.p_value <= alpha:
    print(f"Data drift detected at feature {feature_idx}")
else:
    print(f"No data drift detected at feature {feature_idx}")
# >> Data drift detected at feature 0
# Therefore, we can reject H0 (both samples come from the same distribution).