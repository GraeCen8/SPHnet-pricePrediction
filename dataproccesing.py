import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import talib as ta 
import numpy as np


def load_data(file_path, date_column='Date'):
    df = pd.read_csv(file_path, parse_dates=[date_column])
    df = df.sort_values(by=date_column).reset_index(drop=True)
    return df

def split_data(df, test_size=0.1, val_size=0.2, date_column='Date'):
    split_index = int(len(df) * (1 - test_size))
    val_index = int(len(df) * (1 - test_size - val_size))
    train_df = df.iloc[:val_index]
    val_df = df.iloc[val_index:split_index]
    test_df = df.iloc[split_index:]
    return train_df, val_df, test_df

def FeatureEngineering(df):
    # Example feature engineering using TA-Lib
    df['SMA_10'] = ta.SMA(df['close'], timeperiod=200)
    df['EMA_10'] = ta.EMA(df['close'], timeperiod=10)
    df['RSI_14'] = ta.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['ATR_14'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['range'] = df['high'] - df['low']
    df['vol_moving_avg_10'] = df['volume'].rolling(window=10).mean()
    df = df.dropna()
    return df

def normalize_data(train_data, test_data, method='minmax'):
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Normalization method not recognized. Use 'standard' or 'minmax'.")

    scaler.fit(train_data)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)

    return train_scaled, test_scaled, scaler

def unnormalize_data(scaled_data, scaler):
    return scaler.inverse_transform(scaled_data)

def sliding_window(data, window_size, target_column): #using np funcs for speed
    X, y = [], []
    data_array = data.to_numpy()
    target_idx = data.columns.get_loc(target_column)
    
    for i in range(len(data) - window_size):
        X.append(data_array[i:i + window_size])
        y.append(data_array[i + window_size, target_idx])
    
    return np.array(X), np.array(y)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # use float32 to match model parameter dtype and avoid mixed-precision issues
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def process(file_path = 'data/stock_data.csv',  
        date_column = 'timestamp',
        target_column = 'close',
        window_size = 64,
        batch_size = 128,
        val_size = 0.2,
        test_size = 0.1
    ): 



    df = load_data(file_path, date_column)
    df = FeatureEngineering(df)

    train_df, val_df, test_df = split_data(df,val_size=val_size, test_size=test_size, date_column=date_column)

    train_scaled, val_scaled, scaler = normalize_data(train_df.drop(columns=[date_column]), val_df.drop(columns=[date_column]), method='minmax')
    test_scaled, _, _ = normalize_data(train_df.drop(columns=[date_column]), test_df.drop(columns=[date_column]), method='minmax')

    train_scaled_df = pd.DataFrame(train_scaled, columns=train_df.drop(columns=[date_column]).columns)
    val_scaled_df = pd.DataFrame(val_scaled, columns=val_df.drop(columns=[date_column]).columns)
    test_scaled_df = pd.DataFrame(test_scaled, columns=test_df.drop(columns=[date_column]).columns)

    X_train, y_train = sliding_window(train_scaled_df, window_size, target_column)
    X_val, y_val = sliding_window(val_scaled_df, window_size, target_column)
    X_test, y_test = sliding_window(test_scaled_df, window_size, target_column)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler


if __name__ == "__main__":
    dataPARAMS = {
    "file_path": "data/btc15m.csv",
    "date_column": "timestamp",
    "target_column": "close",
    "window_size": 64,
    "batch_size": 128,
    "val_size": 0.2,
    "test_size": 0.1,
    }
    trainLoader, valLoader, testLoader, scaler = process(**dataPARAMS)