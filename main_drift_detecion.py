import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from frouros.detectors.concept_drift import PageHinkley, PageHinkleyConfig
from scipy.stats import wasserstein_distance, gaussian_kde
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Configure Page-Hinkley for drift detection
config = PageHinkleyConfig()
config.min_instances = 30
config.delta = 0.05  # Adjust delta for sensitivity
config.lambda_ = 70   # Adjust lambda for sensitivity
config.alpha = 1 - 0.0001

class DataProcessor:
    def preprocess_data_log_return(self, data):
        data = np.log(data / data.shift(1))
        data = data.dropna()
        scaler = StandardScaler()
        scaled_result = scaler.fit_transform(data.values.reshape(-1, 1))
        return pd.Series(scaled_result.flatten())

def split_data(data, reference_ratio=0.02, current_ratio=0.01):
    total_length = len(data)
    reference_data = data[:int(total_length * reference_ratio)]
    current_data_1 = data[int(total_length * reference_ratio):int(total_length * (reference_ratio + current_ratio))]
    return reference_data, current_data_1

if __name__ == "__main__":
    # Load data and preprocess
    data = pd.read_csv('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    data = data.dropna()

    reference_data, current_data_1 = split_data(data['close'])

    processor = DataProcessor()
    reference_data_log_return = processor.preprocess_data_log_return(reference_data)
    current_data_1_log_return = processor.preprocess_data_log_return(current_data_1)

    # Introduce a detectable drift in current_data_1
    drift_value = 0.5  # Drift value in the opposite direction for testing
    current_data_1_with_drift = current_data_1_log_return.copy()
    current_data_1_with_drift[int(len(current_data_1_log_return) / 2):] += drift_value

    # Concatenate reference data and current data with drift to form the stream
    stream = np.concatenate([reference_data_log_return, current_data_1_with_drift])

    ### Page-Hinkley Drift Detection ###
    detector = PageHinkley(config=config)
    drift_points = []

    for i, value in enumerate(stream):
        _ = detector.update(value=value)
        if detector.drift:
            drift_points.append(i)

    # Plot the stream and detected drift points
    plt.figure(figsize=(18, 6))
    plt.plot(stream, linestyle='None', marker='.', label='Stream Data', color='blue')
    plt.scatter(drift_points, [stream[i] for i in drift_points], color='red', label='Drift Points', zorder=5)
    plt.xlabel('Index')
    plt.ylabel('Log Return')
    plt.title('Stream Data with Detected Drift Points (Page-Hinkley)')
    plt.legend()
    plt.show()

    ### Wasserstein Distance with KDE for Distribution Comparison ###
    # Split the data into two halves
    half_index = len(stream) // 2
    first_half = stream[:half_index]
    second_half = stream[half_index:]

    # Apply KDE to each half
    kde_first_half = gaussian_kde(first_half)
    kde_second_half = gaussian_kde(second_half)

    # Generate points for plotting KDE
    x_grid = np.linspace(min(stream), max(stream), 1000)
    kde_first_half_values = kde_first_half(x_grid)
    kde_second_half_values = kde_second_half(x_grid)

    # Measure Wasserstein distance between the two distributions
    wasserstein_dist = wasserstein_distance(first_half, second_half)
    print(f"Wasserstein Distance: {wasserstein_dist}")

    # Plot the KDEs and the Wasserstein distance
    plt.figure(figsize=(12, 6))
    plt.plot(x_grid, kde_first_half_values, label='First Half KDE')
    plt.plot(x_grid, kde_second_half_values, label='Second Half KDE')
    plt.fill_between(x_grid, kde_first_half_values, kde_second_half_values, color='gray', alpha=0.5, label='Difference Area')
    plt.title(f'KDE of Data Halves with Wasserstein Distance: {wasserstein_dist:.4f}')
    plt.xlabel('Log Return')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
