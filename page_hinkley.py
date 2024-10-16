import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
from frouros.detectors.concept_drift import PageHinkley, PageHinkleyConfig
from sklearn.metrics import mean_absolute_error

# Step 1: Fetch Apple Stock Data Using yfinance
data = pd.read_csv('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv').head(10000)
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)
data = data[['close']].rename(columns={'close': 'Close'})
data.index = data.index.date
data.reset_index(inplace=True)
data.rename(columns={'index': 'Date'}, inplace=True)
data.index.name = 'Date'
print(data)

# Step 2: Difference the Data to Make it Stationary
data['Differenced'] = data['Close'].diff().dropna()
print(len(data))

# Step 3: Preprocess data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Differenced'].dropna().values.reshape(-1, 1))
print(scaled_data.shape)

sequence_length = 60  # We use the last 60 days to predict the next day
x_train, y_train = [], []
for i in range(sequence_length, len(scaled_data)):
    x_train.append(scaled_data[i-sequence_length:i, 0])
    y_train.append(scaled_data[i, 0])

print(x_train)
print(y_train)
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Reshape for LSTM

# Step 4: Build the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))  # Output layer

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=32, epochs=1)

# Step 5: Make Predictions
predicted_prices = model.predict(x_train)
predicted_prices = scaler.inverse_transform(predicted_prices)  # Rescale to original prices

# Align the data with the predictions
aligned_data = data.iloc[sequence_length + 1:]  # Adjust for differencing and sequence length

# Add the differenced component back to the predicted prices to get the final predictions
final_predicted_prices = predicted_prices.flatten() + data['Close'].shift(1).iloc[sequence_length + 1:].values

# Ensure the lengths of aligned_data['Date'] and final_predicted_prices match
aligned_data = aligned_data.iloc[-len(final_predicted_prices):]

aligned_dates = aligned_data['Date'].values[-len(final_predicted_prices):]
aligned_original = aligned_data['Close'].values[-len(final_predicted_prices):]

drift_data = pd.DataFrame({
    'Date': aligned_dates,
    'Original': aligned_original,
    'Prediction': final_predicted_prices
}).dropna()

# Split the data into reference and current datasets
split_index = len(drift_data) // 2
reference_data = drift_data[:split_index]
current_data = drift_data[split_index:]

# Step 8: Detect and Visualize Drift using Page-Hinkley with Custom Config
def detect_and_visualize_drift(ref_actual, ref_predicted, curr_actual, curr_predicted, data, window_size, split_index):
    # Calculate performance metrics
    ref_mae = mean_absolute_error(ref_actual, ref_predicted)
    curr_mae = mean_absolute_error(curr_actual, curr_predicted)
    print(f"Reference MAE: {ref_mae}, Current MAE: {curr_mae}")

    # Calculate maximum deviation
    ref_max_deviation = np.max(np.abs(ref_actual - ref_predicted))
    print(f"Reference Max Deviation: {ref_max_deviation}")

    # Initialize Page-Hinkley Drift Detector with Custom Config
    config = PageHinkleyConfig(min_num_instances=30, delta=0.005, lambda_=50, alpha=0.9999)
    page_hinkley_detector = PageHinkley(config=config)

    drift_results = {'Page-Hinkley': {'dates': [], 'values': []}}
    dates = data.index[window_size:]

    for idx, (actual, predicted) in enumerate(zip(curr_actual, curr_predicted)):
        error = abs(actual - predicted)
        print(f"Step {idx}: Error = {error}")  # Debug print

        # Update the detector with the error and log drift points
        page_hinkley_detector.update(error)
        if page_hinkley_detector.drift:
            print(f"Drift detected at step {idx}")  # Debug print
            if split_index + idx < len(dates):
                drift_results['Page-Hinkley']['dates'].append(dates[split_index + idx])
                drift_results['Page-Hinkley']['values'].append(predicted)

    # Ensure the lengths of dates and actual/predicted values match
    ref_dates = dates[:split_index]
    curr_dates = dates[split_index:split_index + len(curr_actual)]

    # Adjust curr_actual and curr_predicted to match the length of curr_dates
    curr_actual = curr_actual[:len(curr_dates)]
    curr_predicted = curr_predicted[:len(curr_dates)]

    plt.figure(figsize=(12, 6))
    plt.plot(ref_dates, ref_actual, label='Actual Stock Price (Reference)', color='blue')
    plt.plot(ref_dates, ref_predicted, label='Predicted Stock Price (Reference)', color='orange')
    plt.plot(curr_dates, curr_actual, label='Actual Stock Price (Current)', color='green')
    plt.plot(curr_dates, curr_predicted, label='Predicted Stock Price (Current)', color='red')

    # Plot drift points for Page-Hinkley
    if len(drift_results['Page-Hinkley']['dates']) > 0:
        for drift_date in drift_results['Page-Hinkley']['dates']:
            plt.axvline(x=drift_date, color='purple', linestyle='--', label='Page-Hinkley Drift')

    plt.title('Drift Detection in Stock Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()


ref_actual = reference_data['Original'].values
ref_predicted = reference_data['Prediction'].values
curr_actual = current_data['Original'].values
curr_predicted = current_data['Prediction'].values

detect_and_visualize_drift(ref_actual, ref_predicted, curr_actual, curr_predicted, drift_data, window_size=60, split_index=split_index)