import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from frouros.detectors.data_drift import KSTest

np.random.seed(seed=31)

# Load iris dataset
X, y = load_iris(return_X_y=True)

# Split train (70%) and test (30%)
(
    X_train,
    X_test,
    y_train,
    y_test,
) = train_test_split(X, y, train_size=0.7, random_state=31)

# Set the feature index to which detector is applied
feature_idx = 0

# IMPORTANT: Induce/simulate data drift in the selected feature of y_test by
# applying some gaussian noise. Therefore, changing P(X))
X_test[:, feature_idx] += np.random.normal(
    loc=0.0,
    scale=3.0,
    size=X_test.shape[0],
)

# Define and fit model
model = DecisionTreeClassifier(random_state=31)
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