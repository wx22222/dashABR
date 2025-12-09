import os
import glob
import json
import math
import time
import argparse
import numpy as np

def load_trace(path, to_kbps=False):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                dt = float(parts[0])
                v = float(parts[1])
            except Exception:
                continue
            if to_kbps:
                v *= 1000.0
            if dt > 0 and math.isfinite(v):
                out.append((dt, v))
    return out

def to_uniform_series(trace, dt):
    series = []
    acc_t = 0.0
    acc_v = 0.0
    for dur, val in trace:
        rem = dur
        while rem > 1e-12:
            take = min(rem, dt - acc_t)
            acc_v += val * take
            acc_t += take
            rem -= take
            if acc_t >= dt - 1e-12:
                series.append(acc_v / dt)
                acc_t = 0.0
                acc_v = 0.0
    return series

def build_xy(series, hist_len, pred_horizon):
    X = []
    y = []
    n = len(series)
    for i in range(hist_len, n - pred_horizon + 1):
        X.append(series[i - hist_len:i])
        y.append(sum(series[i:i + pred_horizon]) / float(pred_horizon))
    if not X:
        return None, None
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def collect_dataset(fcc_dir, dt, hist_len, pred_horizon, max_traces, to_kbps=False):
    paths = sorted(glob.glob(os.path.join(fcc_dir, "test_fcc_trace_*")))
    if max_traces is not None and max_traces > 0:
        paths = paths[:max_traces]
    Xs = []
    ys = []
    for p in paths:
        tr = load_trace(p, to_kbps=to_kbps)
        if not tr:
            continue
        s = to_uniform_series(tr, dt)
        if not s or len(s) < hist_len + pred_horizon + 1:
            continue
        Xi, yi = build_xy(s, hist_len, pred_horizon)
        if Xi is None:
            continue
        Xs.append(Xi)
        ys.append(yi)
    if not Xs:
        return None, None
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y

def split_train_val(X, y, val_frac):
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_frac))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

def zscore_fit(arr):
    m = float(arr.mean())
    s = float(arr.std() + 1e-8)
    return m, s

def zscore_apply(arr, m, s):
    return (arr - m) / s

def train_torch(Xtr, ytr, Xval, yval, hist_len, hidden, layers, lr, epochs, batch_size, save_path, cfg_path, onnx_path=None, dropout=0.2, huber_beta=1.0):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    xm, xs = zscore_fit(Xtr)
    ym, ys = zscore_fit(ytr)
    Xtr_n = zscore_apply(Xtr, xm, xs)
    ytr_n = zscore_apply(ytr, ym, ys)
    Xval_n = zscore_apply(Xval, xm, xs)
    yval_n = zscore_apply(yval, ym, ys)
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers, batch_first=True, dropout=float(dropout))
            self.dropout = nn.Dropout(p=float(dropout))
            self.fc = nn.Linear(hidden, 1)
        def forward(self, x):
            x = x.unsqueeze(-1)
            o, _ = self.lstm(x)
            h = o[:, -1, :]
            h = self.dropout(h)
            y = self.fc(h)
            return y.squeeze(-1)
    model = M()
    opt = optim.Adam(model.parameters(), lr=lr)
    try:
        loss_fn = nn.SmoothL1Loss(beta=float(huber_beta))
    except TypeError:
        loss_fn = nn.SmoothL1Loss()
    tr_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr_n), torch.from_numpy(ytr_n))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xval_n), torch.from_numpy(yval_n))
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    best = None
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tl = 0.0
        tc = 0
        for xb, yb in tr_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tl += float(loss.item()) * xb.size(0)
            tc += xb.size(0)
        tl = tl / max(1, tc)
        model.eval()
        vl = 0.0
        vc = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                vl += float(loss.item()) * xb.size(0)
                vc += xb.size(0)
        vl = vl / max(1, vc)
        print(f"epoch={ep+1} train_loss={tl:.6f} val_loss={vl:.6f} time_s={time.time()-t0:.3f}")
        if best is None or vl < best:
            best = vl
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            except Exception:
                pass
            try:
                os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
            except Exception:
                pass
            torch.save({
                "state_dict": model.state_dict(),
                "hist_len": hist_len,
                "hidden": hidden,
                "layers": layers,
                "xm": xm,
                "xs": xs,
                "ym": ym,
                "ys": ys
            }, save_path)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({
                    "hist_len": hist_len,
                    "hidden": hidden,
                    "layers": layers,
                    "xm": xm,
                    "xs": xs,
                    "ym": ym,
                    "ys": ys
                }, f)
            if onnx_path:
                try:
                    model.eval()
                    d = torch.from_numpy(np.zeros((1, hist_len), dtype=np.float32))
                    torch.onnx.export(model, d, onnx_path, input_names=["hist"], output_names=["pred"], opset_version=17)
                except Exception as e:
                    print(f"onnx_export_failed {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fcc_dir", default=os.path.join(os.getcwd(), "datasets", "fcc"))
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--hist_len", type=int, default=30)
    ap.add_argument("--pred_horizon", type=int, default=1)
    ap.add_argument("--max_traces", type=int, default=200)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--save_path", default=os.path.join(os.getcwd(), "assets", "models", "fcc_lstm.pt"))
    ap.add_argument("--cfg_path", default=os.path.join(os.getcwd(), "assets", "models", "fcc_lstm_cfg.json"))
    ap.add_argument("--onnx_path", default=os.path.join(os.getcwd(), "assets", "models", "fcc_lstm.onnx"))
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--to_kbps", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--huber_beta", type=float, default=1.0)
    args = ap.parse_args()
    X, y = collect_dataset(args.fcc_dir, args.dt, args.hist_len, args.pred_horizon, args.max_traces, to_kbps=args.to_kbps)
    if X is None or y is None:
        print("dataset_empty")
        return
    print(f"dataset_shapes X={X.shape} y={y.shape}")
    Xtr, ytr, Xval, yval = split_train_val(X, y, args.val_frac)
    print(f"train_val_shapes Xtr={Xtr.shape} Xval={Xval.shape}")
    if args.dry_run:
        return
    try:
        import torch
    except Exception as e:
        print(f"torch_missing {e}")
        return
    train_torch(Xtr, ytr, Xval, yval, args.hist_len, args.hidden, args.layers, args.lr, args.epochs, args.batch_size, args.save_path, args.cfg_path, args.onnx_path, dropout=args.dropout, huber_beta=args.huber_beta)

if __name__ == "__main__":
    main()
