import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Sample data
data = {
    'timestamp': ['2019-01-02 00:25:00', '2019-01-02 00:26:00', '2019-01-02 00:27:00', '2019-01-02 00:28:00', '2019-01-02 00:29:00'],
    'open': [1.14573, 1.14565, 1.14567, 1.14577, 1.14575],
    'high': [1.14574, 1.14571, 1.14581, 1.14579, 1.14581],
    'low': [1.14565, 1.14564, 1.14562, 1.14572, 1.14573],
    'close': [1.14565, 1.14568, 1.14578, 1.14575, 1.14579],
    'volume': [128, 97, 88, 52, 71]
}

df = pd.DataFrame(data)

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

# Correct instantiation of the Model class
model=Model()
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