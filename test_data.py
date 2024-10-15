import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from functools import partial
from scipy.spatial.distance import pdist
from frouros.callbacks import PermutationTestDistanceBased
from frouros.detectors.data_drift import MMD
from frouros.utils.kernels import rbf_kernel

class TestDataDriftDetection(unittest.TestCase):
    def setUp(self):
        # Load and preprocess the data
        data = pd.read_csv('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv').head(1000)
        data = data[['close']]
        data = np.log(data / data.shift(1))
        data = data.dropna()
        scaler = StandardScaler()
        scaled_result = scaler.fit_transform(data.values.reshape(-1, 1))
        data['close'] = pd.Series(scaled_result.flatten())

        # Create the target variable using the next day's close price
        data['target'] = data['close'].shift(-1).dropna()
        data = data.dropna()

        # Introduce artificial drift in the last 20% of the data
        drift_size = int(len(data) * 0.2)
        data.loc[data.index[-drift_size:], 'target'] += np.random.normal(0, 0.5, size=drift_size)

        # Split the data into features and target
        X = data[['close']].values
        y = data['target'].values

        # Split train (70%) and test (30%)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=0.7, random_state=31)

        # Interpolate to bring both datasets to the same length
        length = min(len(self.X_train), len(self.X_test))
        self.X_train_interp = np.interp(np.linspace(0, len(self.X_train) - 1, length), np.arange(len(self.X_train)), self.X_train.flatten())
        self.X_test_interp = np.interp(np.linspace(0, len(self.X_test) - 1, length), np.arange(len(self.X_test)), self.X_test.flatten())

        # MMD detector configuration
        alpha = 0.01
        sigma = np.median(pdist(X=np.vstack((self.X_train, self.X_test)), metric="euclidean")) / 2

        self.detector = MMD(
            kernel=partial(rbf_kernel, sigma=sigma),
            callbacks=[
                PermutationTestDistanceBased(
                    num_permutations=1000,
                    random_state=31,
                    num_jobs=-1,
                    method="exact",
                    name="permutation_test",
                    verbose=True,
                ),
            ],
        )

        # Fit the detector on the training data
        self.detector.fit(X=self.X_train)

    def test_drift_detection(self):
        # Compare the test data with the training data
        mmd, callbacks_log = self.detector.compare(X=self.X_test)
        p_value = callbacks_log["permutation_test"]["p_value"]

        # Print the results
        print(f"MMD statistic={round(mmd.distance, 4)}, p-value={round(p_value, 8)}")
        if p_value <= 0.01:
            print("Drift detected. We can reject H0, so both samples come from different distributions.")
        else:
            print("No drift detected. We fail to reject H0, so both samples come from the same distribution.")

        # Check if drift was detected
        self.assertTrue(p_value <= 0.01, "Drift was not detected when it was expected.")

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)