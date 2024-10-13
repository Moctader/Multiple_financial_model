import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Define experimental variables
SEQ_LEN = 336
LABEL_LEN = 18
PRED_LEN = 96
ENC_IN = 7
INDIVIDUAL = False
LR = 0.005
BATCH_SIZE = 32
TREE_LR = 0.01
TREE_LOSS = 'MSE'
TREE_LB = 7
TREE_ITER = 200
NUM_LEAVES = 7
PSMOOTH = 0
LB_DATA = 'N'
NUM_JOBS = -1
USE_GPU = False
DEVICE = 'cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'
NORMALIZE = True
USE_REVIN = True
DS_NAME = 'path_to_your_dataset.csv'  # Update this with the actual path to your dataset

# Define Dataset
class Dataset_Custom(Dataset):
    def __init__(self, flag='train', data_path=DS_NAME, scale=True, train_only=False):
        # size [seq_len, label_len, pred_len]
        self.seq_len = SEQ_LEN
        self.label_len = LABEL_LEN
        self.pred_len = PRED_LEN
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.train_only = train_only

        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def validate(model, vali_loader, criterion):
    errs = []
    with torch.no_grad():
        for batch_x, batch_y in vali_loader:
            batch_x = batch_x.float().to(DEVICE)
            batch_y = batch_y[:, -PRED_LEN:, :].float().to(DEVICE)
            outputs = model(batch_x)
            outputs = outputs[:, -PRED_LEN:, :]
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            loss = criterion(pred, true)
            errs.extend(np.abs(pred - true))

    return np.mean(np.array(errs) ** 2), np.mean(errs)

# RevIN class definition
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

# Model definition
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.seq_len = SEQ_LEN  # Number of features
        self.pred_len = PRED_LEN  # Predicting one value (close price)
        self.channels = 1  # Single channel
        self.individual = INDIVIDUAL

        self.device = DEVICE
        self.normalize = NORMALIZE
        self.use_RevIN = USE_REVIN

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

if __name__ == '__main__':
    # Load Model and Data
    ltboost = Model().to(DEVICE)

    train_loader = DataLoader(
        Dataset_Custom(flag='train'),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        drop_last=True)

    vali_loader = DataLoader(
        Dataset_Custom(flag='val'),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        drop_last=True)

    test_loader = DataLoader(
        Dataset_Custom(flag='test'),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    # Debug prints
    print(f"Length of training dataset: {len(train_loader.dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches in train_loader: {len(train_loader)}")

    # Check if train_loader is empty
    if len(train_loader) == 0:
        raise ValueError("The training dataset is empty. Please check the dataset and try again.")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ltboost.parameters(), lr=LR)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_features, batch_target in train_loader:
            batch_features = batch_features.unsqueeze(-1).to(DEVICE)  # Add channel dimension
            batch_target = batch_target.unsqueeze(-1).to(DEVICE)  # Add channel dimension

            # Forward pass
            outputs = ltboost(batch_features)
            loss = criterion(outputs.squeeze(), batch_target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

        # Validation
        val_loss, val_err = validate(ltboost, vali_loader, criterion)
        print(f'Validation Loss: {val_loss:.4f}, Validation Error: {val_err:.4f}')

    # Visualization and prediction
    X_batch, y_batch = next(iter(test_loader))

    # Get predictions using the model
    with torch.no_grad():
        lin_preds = ltboost(X_batch.unsqueeze(-1).float().to(DEVICE))  # Use the model for predictions
    lt_preds = ltboost(X_batch.unsqueeze(-1).float().to(DEVICE))  # LTBoost model predictions

    # Plot results
    SEQ_LEN = 4
    PRED_LEN = 1

    # Ground truth concatenation: Adjust indexing to match 2D tensors
    gt = np.concatenate((X_batch[0, :].cpu().numpy(), np.expand_dims(y_batch[0].cpu().numpy(), axis=0)), axis=0)

    plt.figure()
    plt.plot(list(range(-SEQ_LEN, PRED_LEN)), gt, label='GroundTruth', linewidth=1.5)
    plt.plot(list(range(PRED_LEN)), lin_preds[0, :, -1].cpu().numpy(), label='Linear part', color='orange', linewidth=1)
    plt.plot(list(range(PRED_LEN)), lt_preds[0, :, -1].cpu().numpy(), label='LTBoost', color='red', linewidth=1)
    plt.axvline(x=0, color="k")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.title('Sample prediction')
    plt.legend()
    plt.show()