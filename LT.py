import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

ingestion_service = IngestionService()
df = ingestion_service.get_data(symbol='AAPL', ascending=True, limit=2000)

# Extract features and target
features = df[['open', 'high', 'low', 'volume']].values
target = df['close'].values

# Convert to tensors
features_tensor = torch.tensor(features, dtype=torch.float32)
target_tensor = torch.tensor(target, dtype=torch.float32)

# Create a Dataset
class FinancialDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

dataset = FinancialDataset(features_tensor, target_tensor)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define the RevIN class
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

# Define the Model class
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.seq_len = 4  # Number of features
        self.pred_len = 1  # Predicting one value (close price)
        self.channels = 1  # Single channel
        self.individual = False

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.normalize = True
        self.use_RevIN = True

        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

        if self.use_RevIN:
            self.revin = RevIN(self.channels)

    def forward(self, X):
        seq_last = X[:, -1:, :]
        X = X - seq_last

        if self.normalize:
            seq_last = X[:, -1:, :].detach()
            X = X - seq_last
        if self.use_RevIN:
            X = self.revin(X, 'norm')

        output = torch.zeros([X.size(0), self.pred_len, X.size(2)], dtype=X.dtype).to(X.device)
        if self.individual:
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](X[:, :, i])
        else:
            output = self.Linear(X.permute(0, 2, 1)).permute(0, 2, 1)

        if self.use_RevIN:
            output = self.revin(output, 'denorm')
        if self.normalize:
            output = output + seq_last

        return output

# Instantiate the model and move it to the appropriate device
model = Model()
model = model.to(model.device)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch_features, batch_target in dataloader:
        batch_features = batch_features.unsqueeze(-1).to(model.device)  # Add channel dimension
        batch_target = batch_target.unsqueeze(-1).to(model.device)  # Add channel dimension

        # Forward pass
        outputs = model(batch_features)
        loss = criterion(outputs.squeeze(), batch_target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test and plot predictions
X_batch, y_batch = next(iter(dataloader))

# Reshape the batch to include the channel dimension
X_batch = X_batch.unsqueeze(-1).float().to(model.device)
y_batch = y_batch.unsqueeze(-1).to(model.device)

with torch.no_grad():
    lin_preds = model(X_batch)
    lt_preds = model(X_batch)

# Calculate regression metrics
y_true = y_batch.squeeze().cpu().numpy()
y_pred = lin_preds.squeeze().cpu().numpy()

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'R-squared (R²): {r2:.4f}')

# Fix ground truth concatenation to ensure both tensors have same dimensions
gt = np.concatenate((X_batch[0, :, -1].cpu().numpy(), y_batch[0].cpu().numpy()), axis=0)

# Plotting
plt.figure()
plt.plot(list(range(-model.seq_len, model.pred_len)), gt, label='GroundTruth', linewidth=1.5)
plt.plot(list(range(model.pred_len)), lin_preds[0, :, -1].cpu().numpy(), label='Linear part', color='orange', linewidth=1)
plt.plot(list(range(model.pred_len)), lt_preds[0, :, -1].cpu().numpy(), label='LTBoost', color='red', linewidth=1)
plt.axvline(x=0, color="k")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.title('Sample prediction')
plt.legend()
plt.show()