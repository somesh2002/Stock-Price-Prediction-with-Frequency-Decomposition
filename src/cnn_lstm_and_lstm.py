
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import datetime
import matplotlib.pyplot as plt
import os

symbols = ["^GSPC", "^DJI", "^GDAXI", "^N225"]
start, end = "2010-01-01", "2019-10-30"
start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2019, 10, 30)
window = 250
batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, device=None):
        if device is None:
            device = torch.device("cpu")
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class CNN_LSTM(nn.Module):
    def __init__(self, input_size=1, cnn_hidden_output = 256, lstm_hidden_size=50, num_layers=1):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_hidden_output, kernel_size=3, padding=1),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(cnn_hidden_output, lstm_hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1])
        return x
    
class LSTM(nn.Module):
    def __init__(self, input_size=1, lstm_hidden_size=256, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1])
        return x

def predict_model(model, test_dl):
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    preds = np.concatenate(preds)
    return preds
    

def train_model(model, train_dataloader, validation_dataloader, epochs=20, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    for _ in range(epochs):
        model.train()
        for features, val in train_dataloader:
            features, val = features.to(device), val.to(device)
            opt.zero_grad()
            loss = loss_fn(model(features), val)
            loss.backward()
            opt.step()
        model.eval()
        losses = []
        for xb, yb in validation_dataloader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                val_loss = loss_fn(model(xb), yb)
                losses.append(val_loss.item())
        val_loss = np.mean(losses)
        
        # if val_loss < best_val:
        #     best_val = val_loss
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    return model

# Evaluation function
def evaluate(model, test_dl, scaler = None):
    preds, actuals = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            # yb = yb.to(device)
            preds.append(model(xb).cpu().numpy())
            actuals.append(yb.numpy())
    preds = scaler.inverse_transform(np.concatenate(preds))
    actuals = scaler.inverse_transform(np.concatenate(actuals))
    rmse = mean_squared_error(actuals, preds, squared=False)
    mae = mean_absolute_error(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
    return rmse, mae, mape
    
def fetch_or_load_stock_data(symbol, start_date, end_date, data_dir='stock_data'):
    os.makedirs(data_dir, exist_ok=True)
    filename = f"{data_dir}/{symbol}.csv"

    if os.path.exists(filename):
        print(f"Reading data from {filename}")
        series = pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        print(f"Downloading data for {symbol}")
        stock = yf.Ticker(symbol)
        series = stock.history(start=start_date, end=end_date)["Close"].dropna().to_frame(name="close")
        series.to_csv(filename)

    return series
input_size=1 # for univariate time series

# CNN-LSTM and LSTM Model Parameters
cnn_hidden_output = 512
lstm_hidden_size=250
epochs = 350

current_dir = "/kaggle/working/"
save_dir = os.path.join(current_dir,"cnn_lstm")
os.makedirs(save_dir,exist_ok=True)

lstm_save_dir = os.path.join(current_dir,"lstm")
os.makedirs(lstm_save_dir,exist_ok=True)

for lstm_hidden_size in [200,250,280]:
    print(f"Training with LSTM hidden size: {lstm_hidden_size}")
    results = {}
    stats = {}

    for symbol in symbols:
        series = fetch_or_load_stock_data(symbol, start, end)
        # stock = yf.Ticker(symbol)
        # series = stock.history(start=start_date, end=end_date)["Close"].dropna().to_frame(name="close")
        # stock = yf.Ticker(symbol)
        # series = stock.history(start=start_date, end=end_date)["Close"].dropna().to_frame(name="close")

        stats[symbol] = series.describe()
        scaler = MinMaxScaler()
        scaler.fit(series[["close"]].values.reshape(-1, 1)[:-150])
        series['scaled'] = scaler.transform(series[["close"]])  # Scale the
        values = series["scaled"].values

    # Create sequences
        X, y = [], []
        for i in range(len(values) - window):
            X.append(values[i:i+window])
            y.append(values[i+window])
        X, y = np.array(X), np.array(y)

        X_test, y_test = X[-150:], y[-150:]
        X_train_val, y_train_val = X[:-150], y[:-150]

        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)

        # Loaders
        train_dl = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=batch_size)
        test_dl = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=batch_size)

        cnn_lstm_model = CNN_LSTM(input_size=input_size, cnn_hidden_output=cnn_hidden_output, lstm_hidden_size=lstm_hidden_size)
        cnn_lstm_model = cnn_lstm_model.to(device)
        lstm_model = LSTM(input_size=input_size, lstm_hidden_size=lstm_hidden_size)
        lstm_model = lstm_model.to(device)

        cnn_lstm_model = train_model(cnn_lstm_model, train_dl, val_dl, epochs=epochs, lr = 1e-5)
        lstm_model = train_model(lstm_model, train_dl, val_dl, epochs=epochs, lr=1e-5)

        pred_cnn_lstm = predict_model(cnn_lstm_model, test_dl)
        pred_cnn_lstm = scaler.inverse_transform(pred_cnn_lstm.reshape(-1, 1)).flatten()

        pred_lstm = predict_model(lstm_model, test_dl)
        pred_lstm = scaler.inverse_transform(pred_lstm.reshape(-1, 1)).flatten()

        rmse1, mae1, mape1 = evaluate(cnn_lstm_model, test_dl, scaler)
        rmse2, mae2, mape2 = evaluate(lstm_model, test_dl, scaler)

        results[symbol] = {
            "CNN-LSTM": {"RMSE": rmse1, "MAE": mae1, "MAPE": mape1},
            "LSTM": {"RMSE": rmse2, "MAE": mae2, "MAPE": mape2}
        }
        pred_cnn_lstm = np.array(pred_cnn_lstm)
        pred_lstm = np.array(pred_lstm)
        true_values = series[["close"]].values.reshape(-1)[-150:]
        plt.figure(figsize=(12, 5))
        plt.plot(true_values, label="Original Values")
        plt.plot(pred_cnn_lstm, label="Predicted Values", linestyle='--')
        plt.xlabel("Time Steps")
        plt.ylabel("Stock Price")
        plt.legend()
        save_figure = os.path.join(save_dir,f"{symbol}_cnn_lstm_predictions.png")
        plt.savefig(save_figure)

        plt.figure(figsize=(12, 5))
        plt.plot(true_values, label="Original Values")
        plt.plot(pred_lstm, label="Predicted Values", linestyle='--')
        plt.xlabel("Time Steps")
        plt.ylabel("Stock Price")
        plt.legend()
        save_figure = os.path.join(lstm_save_dir,f"{symbol}_lstm_predictions.png")
        plt.savefig(save_figure)

    print("Results:")
    for symbol, metrics in results.items():
        print(f"{symbol}:")
        for model, metrics in metrics.items():
            print(f"  {model}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.4f}%")
        print(stats[symbol])

    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(f"results_{lstm_hidden_size}_cnn_lstm.csv")
