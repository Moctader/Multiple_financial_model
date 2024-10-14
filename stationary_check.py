import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from main.funcs import create_cassandra_instance

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

def test_stationarity(timeseries):
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=True)

    # Perform Dickey-Fuller test  
    print("Results of Dickey-Fuller Test")
    adft = adfuller(timeseries, autolag='AIC')
    output = pd.Series(adft[0:4], index=['Test Statistic', 'p-value', 'No. of lags used', 'Number of observations used'])
    for key, values in adft[4].items():
        output[f'critical value ({key})'] = values
    print(output)

if __name__ == "__main__":
    ingestion_service = IngestionService()
    data = ingestion_service.get_data(symbol='AAPL', ascending=True, limit=1000)
    data['data_diff']=data['close'].diff().dropna()
    #data=pd.read_csv('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')
    data = data.dropna(subset=['close'])
    plt.plot(data['close'])
    plt.show()    
    data['close_log'] = np.log(data['close'])    
    moving_avg = data['close_log'].rolling(12).mean()    
    data['close_log_moving_avg_diff'] = data['close_log'] - moving_avg     
    data = data.iloc[11:]
    print(data['close'])
    test_stationarity(data['data_diff'])