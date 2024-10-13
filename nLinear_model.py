import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from funcs import create_cassandra_instance
from LTBoost.layers.RevIN import RevIN
import matplotlib.pyplot as plt

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if 'log_return' in [f['name'] for f in self.features]:
            X['log_return'] = np.log(X['close'] / X['close'].shift(1))

        for feature in self.features:
            if feature['name'] == 'volatility' and 'window' in feature:
                X['volatility'] = X['log_return'].rolling(window=feature['window']).std()

        X = X.dropna()
        return X

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
        # Ensure y has the correct shape [Batch, Output length, Channel]
        y = torch.tensor(y, dtype=torch.float32)
        num_samples = y.shape[0]
        if num_samples % self.model.pred_len != 0:
            y = y[:-(num_samples % self.model.pred_len)]
        y = y.reshape(-1, self.model.pred_len, 1)
        # Ensure X has the correct shape [Batch, Input length, Channel]
        num_samples = X.shape[0]
        num_features = X.shape[1]
        if num_samples % self.model.seq_len != 0:
            X = X[:-(num_samples % self.model.seq_len)]
        X = X.reshape(-1, self.model.seq_len, num_features)
        # Ensure the number of samples in X matches the number of samples in y
        min_samples = min(X.shape[0], y.shape[0])
        X = X[:min_samples]
        y = y[:min_samples]
        print(X.shape)
        print(y.shape)
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), y)
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
            # Ensure X has the correct shape [Batch, Input length, Channel]
            num_samples = X.shape[0]
            num_features = X.shape[1]
            if num_samples % self.model.seq_len != 0:
                X = X[:-(num_samples % self.model.seq_len)]
            X = X.reshape(-1, self.model.seq_len, num_features)
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X)
        return outputs.cpu().numpy()

    def score(self, X, y):
        from sklearn.metrics import mean_squared_error
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)

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

class StockPipeline:
    def __init__(self, run_params, ingestion_service):
        self.run_params = run_params
        self.ingestion_service = ingestion_service
        self.df = None

    # Step 1: Data Ingestion
    def load_data(self):
        ingestion_params = self.run_params['ingestion']
        symbol = ingestion_params['symbol']
        num_rows = ingestion_params['num_rows']
        ascending = ingestion_params['ascending']
        self.df = self.ingestion_service.get_data(symbol=symbol, ascending=ascending, limit=num_rows)
        required_columns = ['timestamp', 'close', 'high', 'low', 'open', 'volume']
        self.df = self.df[required_columns]
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df.set_index('timestamp', inplace=True)
        self.df = self.df.dropna()

    # Step 2: Feature Engineering
    def apply_feature_engineering(self):
        features = self.run_params['features']
        feature_transformer = FeatureEngineeringTransformer(features)
        self.df = feature_transformer.transform(self.df)

    # Step 3: Data Splitting using TimeSeriesSplit
    def split_data(self):
        target_column = self.run_params['target_column']
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        data_segmentation = self.run_params['data_segmentation']
        
        tscv = TimeSeriesSplit(n_splits=data_segmentation['n_splits'])
        X_train, X_test, y_train, y_test = None, None, None, None
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            break
        
        return X_train, X_test, y_train, y_test

    # Step 4: Model Training
    def train_model(self, X_train, y_train):
        seq_len = self.run_params['model']['seq_len']
        pred_len = self.run_params['model']['pred_len']
        enc_in = X_train.shape[1]  # Set enc_in to the number of features in X_train
        individual = self.run_params['model']['individual']
        use_revin = self.run_params['model']['use_revin']
        
        model = Model(seq_len, pred_len, enc_in, individual, use_revin)
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.9)
        
        pytorch_model = PyTorchModelWrapper(model=model, criterion=criterion, optimizer=optimizer, epochs=50, batch_size=16, device='cpu')
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pytorch_model', pytorch_model)
        ])
        
        # Flatten X_train for the pipeline
        X_train_flat = X_train.values.reshape(X_train.shape[0], -1)

        pipeline.fit(X_train_flat, y_train)
        return pipeline

    # Step 5: Model Evaluation
    def evaluate_model(self, pipeline, X_test, y_test):
        # Flatten X_test for the pipeline
        X_test_flat = X_test.values.reshape(X_test.shape[0], -1)
        predictions = pipeline.predict(X_test_flat)
        # Flatten y_test to match the shape of predictions
        y_test_flat = y_test.values.reshape(-1, 1)
        predictions_flat = predictions.reshape(-1, 1)
        # Ensure the number of samples in y_test_flat matches predictions_flat
        min_samples = min(y_test_flat.shape[0], predictions_flat.shape[0])
        y_test_flat = y_test_flat[:min_samples]
        predictions_flat = predictions_flat[:min_samples]
        mae = mean_absolute_error(y_test_flat, predictions_flat)
        mse = mean_squared_error(y_test_flat, predictions_flat)
        r2 = r2_score(y_test_flat, predictions_flat)
        return {'mae': mae, 'mse': mse, 'r2': r2}, predictions_flat

    # Method to plot the data
    def plot_data(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.df.index, self.df['close'], label='Close Price')
        # if 'log_return' in self.df.columns:
        #     plt.plot(self.df.index, self.df['log_return'], label='Log Return')
        # if 'volatility' in self.df.columns:
        #     plt.plot(self.df.index, self.df['volatility'], label='Volatility')
        # plt.xlabel('Date')
        # plt.ylabel('Value')
        # plt.title('Stock Data')
        # plt.legend()
        #plt.show()

    # Method to plot the predictions
    def plot_predictions(self, X_train, y_train, X_test, y_test, predictions):
        plt.figure(figsize=(14, 7))
        plt.plot(X_train.index, y_train, label='Train Data')
        plt.plot(X_test.index, y_test, label='Test Data')
        plt.plot(X_test.index[:len(predictions)], predictions, label='Predictions')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title('Train, Test and Predictions')
        plt.legend()
        plt.show()

    def run_pipeline(self):
        self.load_data()
        self.apply_feature_engineering()
        self.plot_data()  # Plot the data for inspection
        X_train, X_test, y_train, y_test = self.split_data()
        pipeline = self.train_model(X_train, y_train)
        metrics, predictions = self.evaluate_model(pipeline, X_test, y_test)
        print(metrics)
        self.plot_predictions(X_train, y_train, X_test, y_test, predictions)  # Plot the predictions
        return metrics

def run_pipeline(ingestion_service):
    row_sizes = [5000]  # Add more dataset sizes for comparison
    results = []

    for num_rows in row_sizes:
        run_params = {
            'ingestion': {
                'symbol': 'AAPL',
                'num_rows': num_rows,
                'ascending': True,
            },
            'target_column': 'close',
            'features': [
                {'name': 'log_return'},
                {'name': 'volatility', 'window': 14}
            ],
            'data_segmentation': {  
                'n_splits': 5
            },
            'model': {
                'seq_len': 30,
                'pred_len': 10,
                'enc_in': 5,  # This will be overridden in train_model
                'individual': False,
                'use_revin': True
            }
        }

        print(f"Running stock prediction pipeline for dataset size: {num_rows}")
        
        pipeline = StockPipeline(run_params, ingestion_service)
        metrics = pipeline.run_pipeline()
        results.append((num_rows, metrics))

    # Print comparison results
    for num_rows, metrics in results:
        print(f"Dataset size: {num_rows}")
        print(f"Metrics: {metrics}")

ingestion_service = IngestionService()
run_pipeline(ingestion_service)