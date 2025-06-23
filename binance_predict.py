import argparse
import datetime
import numpy as np
import requests
import torch
from models.conv_lstm import ConvLSTM

# feature normalization statistics computed from the 5S dataset
FEATURE_MEAN = {
    'std_rush_order': -0.000129,
    'avg_rush_order': -0.000271,
    'std_trades': 0.000206,
    'std_volume': 0.000656,
    'avg_volume': -0.000176,
    'std_price': -0.000287,
    'avg_price': -0.000008,
    'avg_price_max': -0.000003,
    'hour_sin': -0.121124,
    'hour_cos': 0.042832,
    'minute_sin': -0.015121,
    'minute_cos': 0.022793,
    'delta_minutes': 1.107115,
}

FEATURE_STD = {
    'std_rush_order': 0.015513,
    'avg_rush_order': 0.007438,
    'std_trades': 0.048085,
    'std_volume': 0.099605,
    'avg_volume': 0.014145,
    'std_price': 0.003314,
    'avg_price': 0.001104,
    'avg_price_max': 0.001192,
    'hour_sin': 0.690740,
    'hour_cos': 0.711686,
    'minute_sin': 0.698909,
    'minute_cos': 0.714685,
    'delta_minutes': 3.665306,
}

FEATURE_ORDER = [
    'std_rush_order',
    'avg_rush_order',
    'std_trades',
    'std_volume',
    'avg_volume',
    'std_price',
    'avg_price',
    'avg_price_max',
    'hour_sin',
    'hour_cos',
    'minute_sin',
    'minute_cos',
    'delta_minutes',
]

BINANCE_URL = "https://api.binance.com/api/v3/klines"


def fetch_klines(symbol: str, interval: str, limit: int):
    """Fetch klines from Binance"""
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def compute_features(klines, window=5):
    features = []
    for i in range(window - 1, len(klines)):
        win = klines[i - window + 1 : i + 1]
        volumes = [float(k[5]) for k in win]
        trades = [int(k[8]) for k in win]
        closes = [float(k[4]) for k in win]
        highs = [float(k[2]) for k in win]
        dt = datetime.datetime.utcfromtimestamp(int(klines[i][0]) // 1000)
        if i == 0:
            delta = 0.0
        else:
            delta = (int(klines[i][0]) - int(klines[i-1][0])) / 60000
        feat = {
            'std_rush_order': 0.0,
            'avg_rush_order': 0.0,
            'std_trades': float(np.std(trades)),
            'std_volume': float(np.std(volumes)),
            'avg_volume': float(np.mean(volumes)),
            'std_price': float(np.std(closes)),
            'avg_price': float(np.mean(closes)),
            'avg_price_max': float(np.mean(highs)),
            'hour_sin': np.sin(2 * np.pi * dt.hour / 24),
            'hour_cos': np.cos(2 * np.pi * dt.hour / 24),
            'minute_sin': np.sin(2 * np.pi * dt.minute / 60),
            'minute_cos': np.cos(2 * np.pi * dt.minute / 60),
            'delta_minutes': float(delta),
        }
        features.append([feat[k] for k in FEATURE_ORDER])
    return np.array(features, dtype=np.float32)


def normalize(features: np.ndarray) -> np.ndarray:
    mean = np.array([FEATURE_MEAN[k] for k in FEATURE_ORDER], dtype=np.float32)
    std = np.array([FEATURE_STD[k] for k in FEATURE_ORDER], dtype=np.float32)
    return (features - mean) / std


def load_model(path: str, n_feats: int, segment_length: int):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        model = ConvLSTM(n_feats, conv_kernel_size=3, embedding_size=350, num_layers=1)
        model.load_state_dict(obj)
    else:
        model = obj
    model.eval()
    return model


def predict(model, segment: np.ndarray):
    tensor = torch.tensor(segment[None, :, :], dtype=torch.float32)
    with torch.no_grad():
        preds = model(tensor)
    return float(preds[:, -1].item())


def main():
    parser = argparse.ArgumentParser(description="Check for pump using Binance data")
    parser.add_argument("--symbol", required=True, help="Trading pair, e.g. BTCUSDT")
    parser.add_argument("--model", required=True, help="Path to saved model weights")
    parser.add_argument("--interval", default="1m", help="Binance kline interval")
    parser.add_argument("--segment_length", type=int, default=15, help="Length of model segment")
    parser.add_argument("--window", type=int, default=5, help="Window size for feature statistics")
    args = parser.parse_args()

    need = args.segment_length + args.window - 1
    klines = fetch_klines(args.symbol, args.interval, need)
    feats = compute_features(klines, window=args.window)
    feats = normalize(feats)
    segment = feats[-args.segment_length :]

    model = load_model(args.model, n_feats=len(FEATURE_ORDER), segment_length=args.segment_length)
    prob = predict(model, segment)
    print(f"Pump probability for {args.symbol}: {prob:.4f}")


if __name__ == "__main__":
    main()
