import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import lightgbm as lgb

# Define constants
SEQ_LEN = 21
PRED_LEN = 1
ENC_IN = 1
INDIVIDUAL = False
TREE_LR = 0.1
TREE_LOSS = 'MSE'
TREE_LB = 10
LB_DATA = 'N'
NUM_LEAVES = 31
TREE_ITER = 100
PSMOOTH = 0.1
NUM_JOBS = 4
USE_GPU = torch.cuda.is_available()
NORMALIZE = True
USE_REVIN = True

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
        self.seq_len = SEQ_LEN
        self.pred_len = PRED_LEN
        self.channels = ENC_IN
        self.individual = INDIVIDUAL

        self.Tree = []
        self.tree_lr = TREE_LR
        self.tree_loss = TREE_LOSS
        self.treelb = min(TREE_LB, SEQ_LEN)
        self.lb_data = LB_DATA
        self.num_leaves = NUM_LEAVES
        self.tree_iter = TREE_ITER
        self.psmooth = PSMOOTH
        self.num_jobs = NUM_JOBS

        self.device = 'cuda' if USE_GPU else 'cpu'
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

    def train_model(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        if self.normalize:
            seq_last = X[:, -1:, :]
            X = X - seq_last
            y = y - seq_last

        output = torch.zeros([X.size(0), self.pred_len, X.size(2)], dtype=X.dtype).to(X.device)
        with torch.no_grad():
            if self.use_RevIN:
                l_in = self.revin(X, 'norm')

            lin_in = X if not self.use_RevIN else l_in
            if self.individual:
                for i in range(self.channels):
                    output[:, :, i] = self.Linear[i](lin_in[:, :, i].float())
            else:
                output = self.Linear(lin_in.permute(0, 2, 1).float()).permute(0, 2, 1)

            if self.use_RevIN:
                output = self.revin(output, 'denorm')

        y = y - output

        if self.treelb > 0:
            if self.lb_data == '0':
                if self.normalize:
                    X += seq_last
                X = torch.cat((X[:, -self.treelb:, :], output), dim=1)
            elif self.lb_data == 'N':
                X = torch.cat((X[:, -self.treelb:, :], output), dim=1)
            else:
                X = torch.cat((lin_in[:, -self.treelb:, :], output), dim=1)
        else:
            X = output

        X, y = X.cpu().detach().numpy(), y.cpu().detach().numpy()

        self.Tree = []
        for i in range(self.channels):
            dtrain = lgb.Dataset(X[:, :, i])
            def multi_mse(y_hat, dtrain):
                y_true = y[:, :, i]
                grad = y_hat - y_true
                hess = np.ones_like(y_true)
                return grad.flatten("F"), hess.flatten("F")

            def pseudo_huber(y_hat, dtrain):
                y_true = y[:, :, i]
                d = (y_hat - y_true)
                h = 1  # h is delta, 1 = huber loss
                scale = 1 + (d / h) ** 2
                scale_sqrt = np.sqrt(scale)
                grad = d / scale_sqrt
                hess = 1 / scale / scale_sqrt
                return grad, hess

            def mixed_loss(y_hat, dtrain):
                y_true = y[:, :, i]
                grad1 = y_hat - y_true
                hess1 = np.ones_like(y_true)

                scale = 1 + grad1 ** 2
                scale_sqrt = np.sqrt(scale)
                grad2 = grad1 / scale_sqrt
                hess2 = 1 / scale / scale_sqrt
                return 0.5 * (grad1 + grad2), 0.5 * (hess1 + hess2)

            if self.tree_loss == 'Huber':
                loss_func = pseudo_huber
            elif self.tree_loss == 'Mixed':
                loss_func = mixed_loss
            else:
                loss_func = multi_mse

            self.Tree.append(
                lgb.train(
                    train_set=dtrain,
                    params={
                        "boosting": "gbdt",
                        "objective": loss_func,
                        "num_class": self.pred_len,
                        "num_threads": self.num_jobs,
                        "num_leaves": self.num_leaves,
                        "learning_rate": self.tree_lr,
                        "num_iterations": self.tree_iter,
                        "force_col_wise": True,
                        "data_sample_strategy": "goss",
                        "path_smooth": self.psmooth,
                        "random_seed": 7,
                        "verbose": 1
                    },
                )
            )

    def predict(self, X):
        X = X.to(self.device)
        if self.normalize:
            seq_last = X[:, -1:, :]
            X = X - seq_last

        X = X.to(self.device)
        output = torch.zeros([X.size(0), self.pred_len, X.size(2)], dtype=X.dtype).to(X.device)
        with torch.no_grad():
            if self.use_RevIN:
                l_in = self.revin(X, 'norm')

            lin_in = X if not self.use_RevIN else l_in
            if self.individual:
                for i in range(self.channels):
                    output[:, :, i] = self.Linear[i](lin_in[:, :, i].float())
            else:
                output = self.Linear(lin_in.permute(0, 2, 1).float()).permute(0, 2, 1)

            if self.use_RevIN:
                output = self.revin(output, 'denorm')

        if self.treelb > 0:
            if self.lb_data == '0':
                if self.normalize:
                    X += seq_last
                X = torch.cat((X[:, -self.treelb:, :], output), dim=1)
            elif self.lb_data == 'N':
                X = torch.cat((X[:, -self.treelb:, :], output), dim=1)
            else:
                X = torch.cat((lin_in[:, -self.treelb:, :], output), dim=1)
        else:
            X = output

        X, output = X.cpu().detach().numpy(), output.cpu().detach().numpy()

        output2 = torch.zeros([output.shape[0], self.pred_len, output.shape[2]], dtype=output.dtype).to(output.device)
        for i in range(self.channels):
            dtest = X[:, :, i]
            output2[:, :, i] = torch.tensor(self.Tree[i].predict(dtest, num_threads=10), dtype=torch.double)

        if self.normalize:
            seq_last = seq_last.cpu()
            output = output + seq_last

        return output + output2

# # Example usage
# if __name__ == "__main__":
#     # Generate some example data
#     X = torch.randn(100, SEQ_LEN, ENC_IN)
#     y = torch.randn(100, PRED_LEN, ENC_IN)

#     model = Model()
#     model.train_model(X, y)

#     # Make predictions
#     predictions = model.predict(X)
#     print(predictions)