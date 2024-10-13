import pandas as pd
df = pd.read_pickle("features.pickle")
df.info()




# Check for duplicates in the index
duplicates = df.index.duplicated(keep=False)
if duplicates.any():
  print("Found duplicate timestamps:")
  print(df.index[duplicates])
  df = df[~df.index.duplicated(keep='first')]
  print("Duplicate timestamps are now removed")
else:
  print("No duplicate timestamps found.")


from darts import TimeSeries

def get_all_input_and_target_timeseries(df, target_column):
  """
  Creates TimeSeries objects for all inputs and target from df. Assumes the DataFrame index contains the datetime information.

  Parameters:
  - df: pandas DataFrame containing all your data.
  - target_column: the name of the target (label) column to exclude from inputs.

  Returns:
  - input_series: a TimeSeries object containing all features except the target.
  - target_series: a TimeSeries object for the target column.
  """
  # Ensure the DataFrame index is a DatetimeIndex
  df = df.copy()
  if not isinstance(df.index, pd.DatetimeIndex):
      df.index = pd.to_datetime(df.index)
  
  # Exclude the target column from the list of input columns
  input_columns = [col for col in df.columns if col != target_column]
  
  # Create the input TimeSeries using the input columns
  input_series = TimeSeries.from_dataframe(df, value_cols=input_columns, fill_missing_dates=False, freq='min')
  
  # Create the target TimeSeries using the target column
  if target_column not in df.columns:
      raise ValueError(f"The target column '{target_column}' is missing from the DataFrame.")
  target_series = TimeSeries.from_dataframe(df, value_cols=target_column, fill_missing_dates=False, freq='min')
  
  return input_series, target_series

def get_input_and_target_timeseries(df, input_columns, target_column):
  """
  Creates Darts TimeSeries objects for inputs and target from df using specified columns.
  Assumes the DataFrame index contains the datetime information.

  Parameters:
  - df: pandas DataFrame containing your data.
  - input_columns: list of column names to include in the input TimeSeries.
  - target_column: the name of the target (label) column.

  Returns:
  - input_series: a TimeSeries object containing the specified input features.
  - target_series: a TimeSeries object for the target column.
  """
  # Ensure the DataFrame index is a DatetimeIndex
  df = df.copy()
  if not isinstance(df.index, pd.DatetimeIndex):
      df.index = pd.to_datetime(df.index)
  
  # Verify that input_columns and target_column exist in the DataFrame
  missing_input_cols = [col for col in input_columns if col not in df.columns]
  if missing_input_cols:
      raise ValueError(f"The following input columns are missing from the DataFrame: {missing_input_cols}")
  if target_column not in df.columns:
      raise ValueError(f"The target column '{target_column}' is missing from the DataFrame.")
  
  # Create the input TimeSeries using the specified input columns
  input_series = TimeSeries.from_dataframe(df, value_cols=input_columns, fill_missing_dates=False)
  
  # Create the target TimeSeries using the target column
  target_series = TimeSeries.from_dataframe(df, value_cols=target_column, fill_missing_dates=False)
  
  return input_series, target_series


target_column = 'label'
input_ts, target_ts = get_all_input_and_target_timeseries(df, target_column)

# Using specified input columns
#input_columns = ["open", "close", "ema", "rsi", "macd", "bollinger_hband"]
#input_ts, target_ts = get_input_and_target_timeseries(df, input_columns, target_column)


def split_train_test(input_series, target_series, split_date):

  # Ensure split_date is a pandas Timestamp
  if not isinstance(split_date, pd.Timestamp):
      split_date = pd.Timestamp(split_date)
  input_train, input_test = input_series.split_before(split_date)
  target_train, target_test = target_series.split_before(split_date)
  
  return input_train, input_test, target_train, target_test


X_train, X_test, y_train, y_test = split_train_test(input_ts, target_ts, split_date = '2023-01-01')





from darts import TimeSeries
from darts.models import NLinearModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mse
import torch
import matplotlib.pyplot as plt

input_chunk_length = 24
output_chunk_length = 12
optimizer_cls = torch.optim.Adam
optimizer_kwargs = {'lr': 1e-3}
loss_fn = torch.nn.MSELoss()
batch_size = 32
n_epochs = 100
random_state = 42

scaler_X = Scaler()
scaler_y = Scaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

model = NLinearModel(
  input_chunk_length=input_chunk_length,
  output_chunk_length=output_chunk_length,
  optimizer_cls=optimizer_cls,
  optimizer_kwargs=optimizer_kwargs,
  loss_fn=loss_fn,
  batch_size=batch_size,
  n_epochs=n_epochs,
  random_state=random_state,
)

model.fit(
  series=y_train_scaled,
  past_covariates=X_train_scaled,
  verbose=True,
)

forecast_horizon = len(y_test_scaled)
y_pred_scaled = model.predict(
  n=forecast_horizon,
  past_covariates=X_test_scaled,
  verbose=True,
)

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

mae_error = mae(y_test_actual, y_pred)
print(f"Mean Absolute Error: {mae_error}")
mse_error = mse(y_test_actual, y_pred)
print(f"Mean Squared Error: {mse_error}")

plt.figure(figsize=(12, 6))
y_test_actual.plot(label='Actual')
y_pred.plot(label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Values')
plt.show()