import argparse
import datetime
import time
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


def fetch_klines(symbol: str, interval: str, limit: int = None,
                 start_ms: int | None = None, end_ms: int | None = None):
    """Fetch klines from Binance"""
    params = {"symbol": symbol.upper(), "interval": interval}
    if limit is not None:
        params["limit"] = limit
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms
    resp = requests.get(BINANCE_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def interval_to_millis(interval: str) -> int:
    unit = interval[-1]
    qty = int(interval[:-1])
    if unit == 'm':
        return qty * 60_000
    if unit == 'h':
        return qty * 60 * 60_000
    if unit == 'd':
        return qty * 24 * 60 * 60_000
    if unit == 'w':
        return qty * 7 * 24 * 60 * 60_000
    raise ValueError(f"Unsupported interval {interval}")


def fetch_klines_range(symbol: str, interval: str, start_ms: int, end_ms: int):
    out = []
    while start_ms < end_ms:
        resp = fetch_klines(symbol, interval, limit=1000, start_ms=start_ms, end_ms=end_ms)
        if not resp:
            break
        out.extend(resp)
        start_ms = resp[-1][0] + interval_to_millis(interval)
        if len(resp) < 1000:
            break
    return out


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
    parser.add_argument("--start", help="Start datetime UTC YYYY-MM-DD HH:MM")
    parser.add_argument("--end", help="End datetime UTC YYYY-MM-DD HH:MM")
    parser.add_argument("--mode", default="single", choices=["single", "live"],
                        help="Set to 'live' to continuously fetch data")
    args = parser.parse_args()

    if args.start and args.end:
        start_dt = datetime.datetime.strptime(args.start, "%Y-%m-%d %H:%M")
        end_dt = datetime.datetime.strptime(args.end, "%Y-%m-%d %H:%M")
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        klines = fetch_klines_range(args.symbol, args.interval, start_ms, end_ms)
    else:
        need = args.segment_length + args.window - 1
        klines = fetch_klines(args.symbol, args.interval, need)

    model = load_model(args.model, n_feats=len(FEATURE_ORDER), segment_length=args.segment_length)

    if args.mode == "live":
        interval_sec = interval_to_millis(args.interval) / 1000
        need = args.segment_length + args.window - 1
        while True:
            feats = compute_features(klines, window=args.window)
            feats = normalize(feats)
            segment = feats[-args.segment_length :]
            prob = predict(model, segment)
            dt = datetime.datetime.utcfromtimestamp(int(klines[-1][0]) // 1000)
            print(f"{dt} pump probability: {prob:.4f}")
            time.sleep(interval_sec)
            try:
                new = fetch_klines(args.symbol, args.interval, 1)
                if new and new[-1][0] != klines[-1][0]:
                    klines.append(new[-1])
                    if len(klines) > need:
                        klines = klines[-need:]
            except Exception as e:
                print("Error fetching data:", e)
    else:
        feats = compute_features(klines, window=args.window)
        feats = normalize(feats)
        if args.start and args.end:
            for i in range(args.segment_length, len(feats) + 1):
                segment = feats[i - args.segment_length : i]
                prob = predict(model, segment)
                dt = datetime.datetime.utcfromtimestamp(int(klines[i - 1][0]) // 1000)
                print(f"{dt} pump probability: {prob:.4f}")
        else:
            segment = feats[-args.segment_length :]
            prob = predict(model, segment)
            print(f"Pump probability for {args.symbol}: {prob:.4f}")


if __name__ == "__main__":
    main()
