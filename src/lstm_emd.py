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
from PyEMD.EMD import EMD
from PyEMD.CEEMDAN import CEEMDAN
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

def apply_decomposition(signal, method='emd'):
    if method == 'emd':
        decomposer = EMD()
    elif method == 'ceemd':
        decomposer = CEEMDAN()
    else:
        raise ValueError("Invalid method. Choose from 'emd' or 'ceemd'.")
    return decomposer(signal)

def generate_sequences(data, window=window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

def predict_model(X_test, model):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        return model(X_test).squeeze().cpu().numpy()

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
    for _ in tqdm(range(epochs)):
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
            preds.append(model(xb).cpu().numpy())
            actuals.append(yb.numpy())
    preds = scaler.inverse_transform(np.concatenate(preds))
    actuals = scaler.inverse_transform(np.concatenate(actuals))
    rmse = mean_squared_error(actuals, preds, squared=False)
    mae = mean_absolute_error(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
    return rmse, mae, mape



input_size=1 # for univariate time series

# CNN-LSTM and LSTM Model Parameters
cnn_hidden_output = 512
lstm_hidden_size=200
window_size = 250
train_imf_epochs = [300, 300, 270, 240, 220, 160, 130, 100]

results = {}
current_dir = "/kaggle/working/"
save_dir = os.path.join(current_dir,"emd_lstm")
os.makedirs(save_dir,exist_ok=False)

for symbol in symbols:
    stock = yf.Ticker(symbol)
    series = stock.history(start=start_date, end=end_date)["Close"].dropna().to_frame(name="close")
    values = series[["close"]].values.reshape(-1)
    ceemd_imfs = apply_decomposition(values, method='emd')
    ceemd_imfs = np.array(ceemd_imfs)
    train_imfs = ceemd_imfs[:,:-150]
    test_imfs = ceemd_imfs[:,-150-window_size:]
    final_preds = []

    for i in tqdm(range(ceemd_imfs.shape[0])):
        print(f"Training model for IMF {i+1}/{ceemd_imfs.shape[0]}")
        scaler = MinMaxScaler()
        scaler.fit(train_imfs[i].reshape(-1, 1))  # Fit scaler on the current IMF
        train_scaled = scaler.transform(train_imfs[i].reshape(-1, 1)).flatten()  # Scale the current IMF
        test_scaled = scaler.transform(test_imfs[i].reshape(-1, 1)).flatten()  # Scale the current IMF
        X_train_i, y_train_i = generate_sequences(train_scaled, window_size)
        X_test_i, y_test_i = generate_sequences(test_scaled, window_size)

        train_dl = DataLoader(TimeSeriesDataset(X_train_i, y_train_i,device=device), batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(TimeSeriesDataset(X_test_i, y_test_i,device = device), batch_size=batch_size)

        cnn_lstm_model = LSTM(input_size=1,lstm_hidden_size=lstm_hidden_size).to(device)
        train_model(cnn_lstm_model, train_dl, test_dl, epochs=train_imf_epochs[i], lr=1e-4)
        save_final_path = os.path.join(save_dir,f"lstm_imf_{i+1}.pth")
        model_path = save_final_path
        torch.save(cnn_lstm_model.state_dict(), model_path)

        # Predict
        pred_i = predict_model(cnn_lstm_model, test_dl)
        pred_i = scaler.inverse_transform(pred_i.reshape(-1, 1)).flatten()  # Inverse transform the predictions
        final_preds.append(pred_i)    

    final_preds = np.array(final_preds)
    reconstructed = np.sum(final_preds, axis=0)
    true_values = values[-150:]
    plt.figure(figsize=(12, 5))
    plt.plot(true_values, label="Original Values")
    plt.plot(reconstructed, label="Predicted Values", linestyle='--')
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.legend()
    save_figure = os.path.join(save_dir,f"{symbol}_predictions.png")
    plt.savefig(save_figure)

    rmse = mean_squared_error(true_values, reconstructed, squared=False)
    mae = mean_absolute_error(true_values, reconstructed)
    mape = np.mean(np.abs((true_values - reconstructed) / true_values)) * 100
    print(f"Results for {symbol}:")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}%")


    results[symbol] = {
        'CNN_LSTM': {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    }

# Save results to CSV
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.to_csv("results_emd_lstm.csv")
