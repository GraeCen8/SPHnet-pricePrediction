import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
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


def FeatureEngineering(df, ema_n=20):
    df = df.copy()

    eps = 1e-8  # numerical safety

    # --------------------------------------------------
    # 1. Core price dynamics (NO multi-lag returns)
    # --------------------------------------------------
    df['logRet'] = np.log(df['close'] / df['close'].shift(1))
    df['absLogRet'] = df['logRet'].abs()

    # Velocity (return acceleration)
    df['velocity'] = df['logRet'] - df['logRet'].shift(1)

    # --------------------------------------------------
    # 2. Range & volatility (window-light)
    # --------------------------------------------------
    df['range'] = df['high'] - df['low']
    df['logRange'] = np.log(df['range'] + eps)

    # Signed range
    df['signedLogRange'] = np.sign(df['logRet']) * df['logRange']

    # True Range
    prev_close = df['close'].shift(1)
    df['trueRange'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        )
    )

    # EMA-normalized TR (volatility surprise)
    df['emaTR'] = df['trueRange'].ewm(span=ema_n, adjust=False).mean()
    df['trRatio'] = df['trueRange'] / (df['emaTR'] + eps)

    # --------------------------------------------------
    # 3. Volatility regime (transformer friendly)
    # --------------------------------------------------
    df['emaAbsRet'] = df['absLogRet'].ewm(span=ema_n, adjust=False).mean()

    df['volShock'] = df['absLogRet'] - df['emaAbsRet']
    df['volPersist'] = (df['absLogRet'] > df['emaAbsRet']).astype(int)

    # --------------------------------------------------
    # 4. Mean-reversion state (NOT a signal)
    # --------------------------------------------------
    df['emaClose'] = df['close'].ewm(span=ema_n, adjust=False).mean()
    df['emaDist'] = (df['close'] - df['emaClose']) / (df['emaAbsRet'] + eps)

    # Return reversal proxy
    df['reversal'] = -np.sign(df['logRet'].shift(1)) * df['logRet']

    # --------------------------------------------------
    # 5. Volume (optional but useful if available)
    # --------------------------------------------------
    if 'volume' in df.columns:
        df['emaVol'] = df['volume'].ewm(span=ema_n, adjust=False).mean()
        df['volSurprise'] = np.log((df['volume'] + eps) / (df['emaVol'] + eps))
        df['volPriceInteraction'] = df['logRet'] * df['volSurprise']

    # --------------------------------------------------
    # 6. Cleanup
    # --------------------------------------------------
    df = df.replace([np.inf, -np.inf], np.nan)

    return df

def normalize_data(train_data, val_data, test_data, method='minmax'):
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Normalization method not recognized. Use 'standard' or 'minmax'.")

    scaler.fit(train_data)
    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)
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
    
    
def process(
    file_path,
    date_column='timestamp',
    target_column='close',
    window_size=64,
    batch_size=128,
    val_size=0.2,
    test_size=0.1,
    scaler_method='standard',
    ema_n=20
):
    # --------------------------------------------------
    # 1. Load & sort raw data
    # --------------------------------------------------
    df = load_data(file_path, date_column)

    # --------------------------------------------------
    # 2. Temporal split on RAW data (NO FEATURES YET)
    # --------------------------------------------------
    train_df, val_df, test_df = split_data(
        df,
        val_size=val_size,
        test_size=test_size,
        date_column=date_column
    )

    # --------------------------------------------------
    # 3. Feature engineering (per split, no leakage)
    # --------------------------------------------------
    train_df = FeatureEngineering(train_df, ema_n=ema_n)
    val_df   = FeatureEngineering(val_df,   ema_n=ema_n)
    test_df  = FeatureEngineering(test_df,  ema_n=ema_n)

    # Drop rows with incomplete EMA state
    train_df = train_df.dropna().reset_index(drop=True)
    val_df   = val_df.dropna().reset_index(drop=True)
    test_df  = test_df.dropna().reset_index(drop=True)

    # --------------------------------------------------
    # 4. Define target (NEXT-step return)
    # --------------------------------------------------
    for df_ in [train_df, val_df, test_df]:
        df_['target'] = df_['logRet'].shift(-1)

    train_df = train_df.dropna()
    val_df   = val_df.dropna()
    test_df  = test_df.dropna()

    # --------------------------------------------------
    # 5. Select features (EXCLUDE price + target)
    # --------------------------------------------------
    exclude_cols = [
        date_column,
        'open', 'high', 'low', 'close',
        'target'
    ]

    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    # Binary / categorical features (DO NOT SCALE)
    binary_cols = ['volPersist']

    scale_cols = [c for c in feature_cols if c not in binary_cols]

    # --------------------------------------------------
    # 6. Scale features (TRAIN FIT ONLY)
    # --------------------------------------------------
    if scaler_method == 'standard':
        scaler = StandardScaler()
    elif scaler_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_method must be 'standard' or 'minmax'")

    train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
    val_df[scale_cols]   = scaler.transform(val_df[scale_cols])
    test_df[scale_cols]  = scaler.transform(test_df[scale_cols])


    #
    # ext. print
    #
    #print(train_df.head())

    # --------------------------------------------------
    # 7. Windowing (AFTER scaling)
    # --------------------------------------------------
    X_train, y_train = sliding_window(
        train_df[feature_cols + ['target']],
        window_size,
        target_column=target_column
    )

    X_val, y_val = sliding_window(
        val_df[feature_cols + ['target']],
        window_size,
        target_column='target'
    )

    X_test, y_test = sliding_window(
        test_df[feature_cols + ['target']],
        window_size,
        target_column='target'
    )

    # --------------------------------------------------
    # 8. Torch datasets & loaders
    # --------------------------------------------------
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset   = TimeSeriesDataset(X_val, y_val)
    test_dataset  = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, scaler, feature_cols



if __name__ == "__main__":
    dataPARAMS = {
    "file_path": "data/btc15m.csv",
    "date_column": "timestamp",
    "target_column": "logRet",
    "window_size": 64,
    "batch_size": 128,
    "val_size": 0.2,
    "test_size": 0.1,
    }
    trainLoader, valLoader, testLoader, scaler, feature_cols = process(**dataPARAMS)
    
    #output shapes for verification
    for X_batch, y_batch in trainLoader:
        print("Train batch X shape:", X_batch.shape)
        print("Train batch y shape:", y_batch.shape)
        break