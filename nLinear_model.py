import torch
import torch.nn as nn
import numpy as np
from LTBoost.layers.RevIN import RevIN

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, seq_len, pred_len, enc_in, individual=False, use_revin=False):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual
        self.use_RevIN = use_revin
        
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
                # Use this line if you want to visualize the weights
                # self.Linear[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            # Use this line if you want to visualize the weights
            self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))

        if self.use_RevIN:
            self.revin = RevIN(self.channels)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        if self.use_RevIN:
            x = self.revin(x, 'norm')

        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        if self.use_RevIN:
            x = self.revin(x, 'denorm')
        
        x = x + seq_last
        return x  # [Batch, Output length, Channel]
    





from sklearn.base import BaseEstimator, RegressorMixin
import torch.optim as optim

class PyTorchModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model, criterion, optimizer, epochs=10, batch_size=32, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model.to(self.device)

    def fit(self, X, y):
        self.model.train()
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X)
        return outputs.cpu().numpy()

    def score(self, X, y):
        from sklearn.metrics import mean_squared_error
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)
    





from funcs import create_cassandra_instance
import pandas as pd

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
    


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Define model parameters directly
seq_len = 30
pred_len = 10
enc_in = 5
individual = False
use_revin = True

model = Model(seq_len, pred_len, enc_in, individual, use_revin)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create the PyTorch model wrapper
pytorch_model = PyTorchModelWrapper(model=model, criterion=criterion, optimizer=optimizer, epochs=10, batch_size=32, device='cpu')

# Create the sklearn pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pytorch_model', pytorch_model)
])

# Example usage with IngestionService
ingestion_service = IngestionService()
df = ingestion_service.get_data(symbol='AAPL', ascending=True, limit=1000)

# Assuming 'close' is the target column and the rest are features
X = df.drop(columns=['close']).values
y = df['close'].values

# Train the pipeline
pipeline.fit(X, y)

# Predict using the pipeline
predictions = pipeline.predict(X)