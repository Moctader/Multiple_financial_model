import ta
import numpy as np
from datetime import datetime
from summarytools import dfSummary
import pandas as pd


def get_technical_indicators(df, indicators):
  # Ensure the DataFrame has the required columns
  required_columns = ['open', 'high', 'low', 'close', 'volume']
  if not all(col in df.columns for col in required_columns):
      raise ValueError("DataFrame must have 'open', 'high', 'low', 'close', and 'volume' columns")
  
  # Create a dictionary to map indicator names to their respective functions
  indicator_functions = {
      'sma': lambda: ta.trend.sma_indicator(df['close'], window=14),
      'ema': lambda: ta.trend.ema_indicator(df['close'], window=14),
      'rsi': lambda: ta.momentum.rsi(df['close'], window=14),
      'macd': lambda: ta.trend.macd(df['close']),
      'macd_signal': lambda: ta.trend.macd_signal(df['close']),
      'macd_diff': lambda: ta.trend.macd_diff(df['close']),
      'bollinger_hband': lambda: ta.volatility.bollinger_hband(df['close']),
      'bollinger_lband': lambda: ta.volatility.bollinger_lband(df['close']),
      'bollinger_mavg': lambda: ta.volatility.bollinger_mavg(df['close']),
      'stoch': lambda: ta.momentum.stoch(df['high'], df['low'], df['close']),
      'stoch_signal': lambda: ta.momentum.stoch_signal(df['high'], df['low'], df['close']),
      'adx': lambda: ta.trend.adx(df['high'], df['low'], df['close']),
      'cci': lambda: ta.trend.cci(df['high'], df['low'], df['close']),
      'obv': lambda: ta.volume.on_balance_volume(df['close'], df['volume']),
      'atr': lambda: ta.volatility.average_true_range(df['high'], df['low'], df['close'])
  }
  
  # Add requested indicators
  for indicator in indicators:
      if indicator in indicator_functions:
          df[indicator] = indicator_functions[indicator]()
      elif indicator == 'bollinger_bands':
          df['bollinger_hband'] = ta.volatility.bollinger_hband(df['close'])
          df['bollinger_lband'] = ta.volatility.bollinger_lband(df['close'])
          df['bollinger_mavg'] = ta.volatility.bollinger_mavg(df['close'])
      else:
          print(f"Warning: Indicator '{indicator}' not implemented")
  
  return df

def get_cyclical_features(df):
  # Ensure timestamp is in datetime format
  df['timestamp'] = pd.to_datetime(df['timestamp'])
  
  # Basic time features
  df['hour'] = df['timestamp'].dt.hour
  df['day_of_week'] = df['timestamp'].dt.dayofweek
  df['day_of_month'] = df['timestamp'].dt.day
  df['month'] = df['timestamp'].dt.month
  df['quarter'] = df['timestamp'].dt.quarter
  df['week_of_year'] = df['timestamp'].dt.isocalendar().week
  df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
  df['is_business_day'] = df['timestamp'].dt.dayofweek.isin(range(5)).astype(int)
  

  # Cyclical encoding for some features
  df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
  df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
  df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
  df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
  df['month_sin'] = np.sin((df['month'] - 1) * (2 * np.pi / 12))
  df['month_cos'] = np.cos((df['month'] - 1) * (2 * np.pi / 12))
  
  return df

def calculate_label(df, window_size):
    df['moving_average'] = df['close'].rolling(window=window_size).mean()
    return (df['close'].shift(-1) - df['moving_average'])



