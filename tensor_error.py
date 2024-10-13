# Sample data
data = {
    'timestamp': ['2019-01-02 00:25:00', '2019-01-02 00:26:00', '2019-01-02 00:27:00', '2019-01-02 00:28:00', '2019-01-02 00:29:00'],
    'open': [1.14573, 1.14565, 1.14567, 1.14577, 1.14575],
    'high': [1.14574, 1.14571, 1.14581, 1.14579, 1.14581],
    'low': [1.14565, 1.14564, 1.14562, 1.14572, 1.14573],
    'close': [1.14565, 1.14568, 1.14578, 1.14575, 1.14579],
    'volume': [128, 97, 88, 52, 71]
}



import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

# Create a DataFrame
df = pd.DataFrame(data)

# Extract features and target
features = df[['open', 'high', 'low', 'volume']].values
target = df['close'].values

# Convert to tensors
features_tensor = torch.tensor(features, dtype=torch.float32)
target_tensor = torch.tensor(target, dtype=torch.float32)



# 3. Create a Dataset
class FinancialDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

dataset = FinancialDataset(features_tensor, target_tensor)


# 4. Create a DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


#5. Define the Model
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel(input_dim=4)


#6. Train the Model
# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch_features, batch_target in dataloader:
        # Forward pass
        outputs = model(batch_features)
        loss = criterion(outputs.squeeze(), batch_target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')