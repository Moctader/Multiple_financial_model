import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

class DataProcessor:
    def preprocess_data_log_return(self, data):
        data = np.log(data / data.shift(1))
        data = data.dropna()
        scaler = StandardScaler()
        scaled_result = scaler.fit_transform(data.values.reshape(-1, 1))
        return pd.Series(scaled_result.flatten())

    def preprocess_data_differencing(self, data):
        data = data.diff().dropna()
        scaler = StandardScaler()
        scaled_result = scaler.fit_transform(data.values.reshape(-1, 1))
        return pd.Series(scaled_result.flatten())

    def preprocess_data_moving_average_deviation(self, data, window=5):
        moving_avg = data.rolling(window=window).mean()
        deviation = data - moving_avg
        deviation = deviation.dropna()
        scaler = StandardScaler()
        scaled_result = scaler.fit_transform(deviation.values.reshape(-1, 1))
        return pd.Series(scaled_result.flatten())

    def preprocess_data_decompose(self, data):
        data = data[data > 0]
        log_data = np.log(data)
        log_data = log_data.dropna()
        result = seasonal_decompose(log_data, model='additive', period=12).resid
        result = pd.Series(result).dropna()
        scaler = StandardScaler()
        scaled_result = scaler.fit_transform(result.values.reshape(-1, 1))
        return pd.Series(scaled_result.flatten())

    def consolidate(self, dataset, target_length):
        total_length = len(dataset)
        assert target_length <= total_length, 'THE TARGET LENGTH CAN ONLY BE SMALLER THAN THE DATASET'
        original_indices = np.linspace(0, total_length - 1, num=total_length)
        target_indices = np.linspace(0, total_length - 1, num=target_length)
        consolidated_data = np.interp(target_indices, original_indices, dataset)
        return consolidated_data

    def bin_data(self, data, num_bins=20):
        binned_data, bin_edges = np.histogram(data, bins=num_bins)
        bin_indices = np.digitize(data, bin_edges) - 1  # Assign each value to a bin
        return bin_indices, bin_edges

def split_data(data, reference_ratio=0.2, current_ratio=0.01):
    total_length = len(data)
    reference_data = data[:int(total_length * reference_ratio)]
    current_data_1 = data[int(total_length * reference_ratio):int(total_length * (reference_ratio + current_ratio))]
    current_data_2 = data[int(total_length * (reference_ratio + current_ratio)):]
    return reference_data, current_data_1, current_data_2

if __name__ == "__main__":
    data = pd.read_csv('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    data = data.dropna()

    reference_data, current_data_1, current_data_2 = split_data(data['close'])
    #reference_target, current_target_1, current_target_2 = split_data(data['target'])

    # Preprocess the reference data using different methods
    processor = DataProcessor()
    reference_data_log_return = processor.preprocess_data_log_return(reference_data)
    reference_data_differencing = processor.preprocess_data_differencing(reference_data)
    reference_data_moving_avg_dev = processor.preprocess_data_moving_average_deviation(reference_data)
    reference_data_decompose = processor.preprocess_data_decompose(reference_data)

    # Preprocess the current data (e.g., current_data_1)
    current_data_log_return = processor.preprocess_data_log_return(current_data_1)
    current_data_differencing = processor.preprocess_data_differencing(current_data_1)
    current_data_moving_avg_dev = processor.preprocess_data_moving_average_deviation(current_data_1)
    current_data_decompose = processor.preprocess_data_decompose(current_data_1)

    # Consolidate the reference data to a smaller length for comparison
    target_length = int(len(current_data_1))
    consolidated_reference_data_log_return = processor.consolidate(reference_data_log_return, target_length)
    consolidated_reference_data_differencing = processor.consolidate(reference_data_differencing, target_length)
    consolidated_reference_data_moving_avg_dev = processor.consolidate(reference_data_moving_avg_dev, target_length)
    consolidated_reference_data_decompose = processor.consolidate(reference_data_decompose, target_length)

    # Plot the histogram and KDE comparison between consolidated reference data and current data for each preprocessing method
    methods = [
        ('Log Return', consolidated_reference_data_log_return, current_data_log_return),
        ('Differencing', consolidated_reference_data_differencing, current_data_differencing),
        ('Moving Average Deviation', consolidated_reference_data_moving_avg_dev, current_data_moving_avg_dev),
        ('Decomposition', consolidated_reference_data_decompose, current_data_decompose)
    ]

    plt.figure(figsize=(18, 12))
    for i, (method_name, ref_data, curr_data) in enumerate(methods, start=1):
        plt.subplot(len(methods), 2, 2*i-1)
        sns.histplot(ref_data, bins=20, kde=True, label=f'Reference Data - {method_name}', color='blue')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'Histogram and KDE of Reference Data - {method_name}')
        plt.legend()

        plt.subplot(len(methods), 2, 2*i)
        sns.histplot(curr_data, bins=20, kde=True, label=f'Current Data - {method_name}', color='orange')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'Histogram and KDE of Current Data - {method_name}')
        plt.legend()

    plt.tight_layout()
    plt.show()