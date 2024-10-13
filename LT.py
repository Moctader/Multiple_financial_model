import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from LTBoost import Model

from funcs import create_cassandra_instance

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
df = ingestion_service.get_data(symbol='AAPL', ascending=True, limit=20000)

# Extract features and target
features = df[['open', 'high', 'low', 'volume']]
target = df['close']

# Scale the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

features_scaled = scaler_X.fit_transform(features)
target_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))

# Define Dataset
class FinancialDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

# TimeSeriesSplit for time-series data
tscv = TimeSeriesSplit(n_splits=5)
X_train, X_test, y_train, y_test = None, None, None, None

for train_index, test_index in tscv.split(features_scaled):
    X_train, X_test = features_scaled[train_index], features_scaled[test_index]
    y_train, y_test = target_scaled[train_index], target_scaled[test_index]
    break  # Return the first split

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = FinancialDataset(X_train_tensor, y_train_tensor)
test_dataset = FinancialDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model Definition
model = Model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for batch_features, batch_target in train_loader:
        batch_features = batch_features.unsqueeze(-1).to(device)  # Add channel dimension
        batch_target = batch_target.unsqueeze(-1).to(device)  # Add channel dimension

        # Forward pass
        outputs = model(batch_features)
        loss = criterion(outputs.squeeze(), batch_target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Validation loop
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_features, batch_target in test_loader:
            batch_features = batch_features.unsqueeze(-1).to(device)
            batch_target = batch_target.unsqueeze(-1).to(device)

            outputs = model(batch_features)
            loss = criterion(outputs.squeeze(), batch_target)
            val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        print(f'Validation Loss: {avg_val_loss:.4f}')

# Test and plot predictions
model.eval()
X_batch, y_batch = next(iter(test_loader))

# Reshape the batch to include the channel dimension
X_batch = X_batch.unsqueeze(-1).float().to(device)
y_batch = y_batch.unsqueeze(-1).to(device)

with torch.no_grad():
    y_pred_scaled = model(X_batch)

# Inverse transform the predictions and actual values
y_pred = scaler_y.inverse_transform(y_pred_scaled.squeeze().cpu().numpy().reshape(-1, 1))
y_test_actual = scaler_y.inverse_transform(y_batch.squeeze().cpu().numpy().reshape(-1, 1))

# Calculate regression metrics
mae_error = mean_absolute_error(y_test_actual, y_pred)
mse_error = mean_squared_error(y_test_actual, y_pred)
rmse_error = np.sqrt(mse_error)
r2_error = r2_score(y_test_actual, y_pred)

print(f"Mean Absolute Error (MAE): {mae_error:.4f}")
print(f"Mean Squared Error (MSE): {mse_error:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_error:.4f}")
print(f"R-squared (R²): {r2_error:.4f}")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Values')
plt.show()