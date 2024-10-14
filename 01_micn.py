import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from main.funcs import create_cassandra_instance

class IngestionService:
    def __init__(self):
        self.cassandra = create_cassandra_instance()

    def get_data(self, symbol, ascending, limit):
        order_by = 'ASC' if ascending else 'DESC'
        query = f"SELECT * FROM john.refined_stock_data WHERE symbol = '{symbol}' ORDER BY timestamp {order_by} LIMIT {limit};"
        results = self.cassandra.query(query)
        rows = list(results)
        df = pd.DataFrame(rows) 
        return df

# Data Preparation
ingestion_service = IngestionService()
df = ingestion_service.get_data(symbol='AAPL', ascending=True, limit=200000)
print(df['close'])
# Load your time series data
time_series = df['close']  
# Decompose the time series
decomposition = seasonal_decompose(time_series, model='additive', period=12)
seasonal = decomposition.seasonal
trend = decomposition.trend
residual = decomposition.resid

# Prepare the seasonal data for training
sequence_length = 10  # Adjust the sequence length to be greater than the kernel size
seasonal_data = seasonal.dropna().values
seasonal_target = seasonal.shift(-1).dropna().values

# Ensure the lengths of seasonal_data and seasonal_target match
min_length = min(len(seasonal_data), len(seasonal_target))
seasonal_data = seasonal_data[:min_length]
seasonal_target = seasonal_target[:min_length]

# Reshape data into sequences
num_sequences = len(seasonal_data) // sequence_length
seasonal_data = seasonal_data[:num_sequences * sequence_length].reshape(num_sequences, sequence_length, 1)
seasonal_target = seasonal_target[:num_sequences * sequence_length].reshape(num_sequences, sequence_length, 1)

# Use the last value of each sequence as the target
seasonal_target = seasonal_target[:, -1, :]

# Print shapes to debug size mismatch
print(f'Adjusted seasonal_data shape: {seasonal_data.shape}')
print(f'Adjusted seasonal_target shape: {seasonal_target.shape}')

# Define the Local-Global Module
class LocalGlobalModule(nn.Module):
    def __init__(self):
        super(LocalGlobalModule, self).__init__()
        self.downsampling_conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2)
        self.isometric_conv = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.downsampling_conv(x)
        x = torch.relu(x)
        x = self.isometric_conv(x)
        x = torch.relu(x)
        x = x.mean(dim=2)  # Global average pooling
        x = self.fc(x)
        return x

# Create DataLoader
dataset = TensorDataset(torch.tensor(seasonal_data, dtype=torch.float32), torch.tensor(seasonal_target, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize and train the model
model = LocalGlobalModule()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0  # Initialize epoch loss
    for batch_data, batch_target in dataloader:
        print("in")
        print(batch_data.shape)
        batch_data = batch_data.permute(0, 2, 1)  # Ensure the shape is (batch_size, 1, sequence_length)
        optimizer.zero_grad()
        outputs = model(batch_data)
        print(f"Outputs: {outputs}")  # Debugging statement
        loss = criterion(outputs, batch_target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()  # Accumulate loss for the epoch
    if epoch_loss > 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    else:
        print(f'Epoch [{epoch+1}/{num_epochs}], No valid batches processed.')

# Prepare the trend data for training
trend_data = trend.dropna().index.factorize()[0].reshape(-1, 1)
trend_target = trend.dropna().values

# Train the linear regression model
regressor = LinearRegression()
regressor.fit(trend_data, trend_target)

# Predict the trend
trend_predictions = regressor.predict(trend_data)

# Predict the seasonal part
seasonal_predictions = []
model.eval()
with torch.no_grad():
    for batch_data, _ in dataloader:
        batch_data = batch_data.permute(0, 2, 1)  # Ensure the shape is (batch_size, 1, sequence_length)
        outputs = model(batch_data)
        seasonal_predictions.extend(outputs.numpy().flatten())

# Convert seasonal_predictions to a numpy array
seasonal_predictions = np.array(seasonal_predictions)

# Ensure the lengths of seasonal_predictions and trend_predictions match
min_length = min(len(seasonal_predictions), len(trend_predictions))
seasonal_predictions = seasonal_predictions[:min_length]
trend_predictions = trend_predictions[:min_length]

# Combine the predictions
combined_predictions = seasonal_predictions + trend_predictions

# Align the combined predictions with the original time series
combined_index = time_series.index[-len(combined_predictions):]

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(time_series.index, time_series.values, label='Original')
plt.plot(combined_index, combined_predictions, label='Predicted')
plt.legend()
plt.show()