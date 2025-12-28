import os
import glob
import json
import math
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

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

def ewma_predict_windows(X, alpha):
    t0 = time.perf_counter()
    preds = []
    for i in range(X.shape[0]):
        xh = X[i]
        ew = None
        for v in xh:
            vv = float(v)
            ew = vv if ew is None else (alpha * vv + (1 - alpha) * ew)
        preds.append(float(ew) if ew is not None else float(xh[-1]))
    t1 = time.perf_counter()
    per_step = (t1 - t0) / float(X.shape[0]) if X.shape[0] > 0 else float('nan')
    return np.array(preds, dtype=np.float32), per_step

def lstm_predict(X, cfg_path, onnx_path):
    try:
        import onnxruntime as ort
    except Exception:
        return None, None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        xm = float(cfg.get("xm", 0.0))
        xs = float(cfg.get("xs", 1.0))
        ym = float(cfg.get("ym", 0.0))
        ys = float(cfg.get("ys", 1.0))
        hlen = int(cfg.get("hist_len", X.shape[1]))
        if hlen != X.shape[1]:
            return None, None
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        Xn = (X - xm) / (xs if xs != 0 else 1.0)
        out = []
        t_sum = 0.0
        for i in range(Xn.shape[0]):
            inp = Xn[i:i+1].astype(np.float32)
            feeds = {inp_name: inp}
            t0 = time.perf_counter()
            r = sess.run(None, feeds)
            t1 = time.perf_counter()
            t_sum += (t1 - t0)
            yhat = np.array(r[0]).reshape((-1,))
            yv = yhat * (ys if ys != 0 else 1.0) + ym
            out.append(yv.astype(np.float32))
        per_step = t_sum / float(Xn.shape[0]) if Xn.shape[0] > 0 else float('nan')
        return np.concatenate(out, axis=0), per_step
    except Exception:
        return None, None

def _mean(arr):
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))

def _rmse(err):
    if err.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(err ** 2)))

def _mae(err):
    if err.size == 0:
        return float("nan")
    return float(np.mean(np.abs(err)))

def _mape(y, yhat):
    m = []
    for i in range(len(y)):
        yi = float(y[i])
        pi = float(yhat[i])
        if yi != 0:
            m.append(abs((pi - yi) / yi))
    if not m:
        return float("nan")
    return float(np.mean(np.array(m)))

def _r2(y, yhat):
    if len(y) == 0:
        return float("nan")
    y = np.array(y, dtype=np.float64)
    yhat = np.array(yhat, dtype=np.float64)
    sst = np.sum((y - np.mean(y)) ** 2)
    sse = np.sum((y - yhat) ** 2)
    if sst == 0:
        return float("nan")
    return float(1 - sse / sst)

def _corr(y, yhat):
    if len(y) == 0:
        return float("nan")
    y = np.array(y, dtype=np.float64)
    yhat = np.array(yhat, dtype=np.float64)
    if y.size < 2:
        return float("nan")
    return float(np.corrcoef(y, yhat)[0,1])

def metrics(y, yhat):
    y = np.array(y, dtype=np.float32)
    yhat = np.array(yhat, dtype=np.float32)
    err = yhat - y
    return {
        "samples": int(y.shape[0]),
        "mae": _mae(err),
        "rmse": _rmse(err),
        "mape": _mape(y, yhat),
        "bias": _mean(err),
        "r2": _r2(y, yhat),
        "corr": _corr(y, yhat)
    }

def fmt(v):
    if v is None:
        return "NA"
    if isinstance(v, float):
        if math.isnan(v):
            return "NA"
        return f"{v:.6f}"
    return str(v)

def print_metrics(name, m):
    print(f"[{name}]")
    print(f"samples={fmt(m.get('samples'))}")
    print(f"mae={fmt(m.get('mae'))}")
    print(f"rmse={fmt(m.get('rmse'))}")
    print(f"mape={fmt(m.get('mape'))}")
    print(f"bias={fmt(m.get('bias'))}")
    print(f"r2={fmt(m.get('r2'))}")
    print(f"corr={fmt(m.get('corr'))}")
    print("")

def _setup_fonts():
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

def plot_summary_bars(m_lstm, m_ewma, out_path):
    _setup_fonts()
    names = []
    maes = []
    rmses = []
    mapes = []
    r2s = []
    def add(name, m):
        names.append(name)
        maes.append(m.get('mae'))
        rmses.append(m.get('rmse'))
        mapes.append(m.get('mape'))
        r2s.append(m.get('r2'))
    if m_lstm is not None:
        add('LSTM', m_lstm)
    if m_ewma is not None:
        add('EWMA', m_ewma)
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    axs[0][0].bar(names, maes, color=['#4C78A8','#F58518'][:len(names)])
    axs[0][0].set_title('MAE')
    axs[0][1].bar(names, rmses, color=['#4C78A8','#F58518'][:len(names)])
    axs[0][1].set_title('RMSE')
    axs[1][0].bar(names, mapes, color=['#4C78A8','#F58518'][:len(names)])
    axs[1][0].set_title('MAPE')
    axs[1][1].bar(names, r2s, color=['#4C78A8','#F58518'][:len(names)])
    axs[1][1].set_title('R²')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_error_hist(y, y_lstm, y_ewma, out_path):
    _setup_fonts()
    fig, ax = plt.subplots(figsize=(10,5))
    bins = 50
    if y_lstm is not None:
        ax.hist(y_lstm - y, bins=bins, alpha=0.5, density=True, label='LSTM', color='#4C78A8')
    if y_ewma is not None:
        ax.hist(y_ewma - y, bins=bins, alpha=0.5, density=True, label='EWMA', color='#F58518')
    ax.set_title('预测误差分布 (yhat - y)')
    ax.set_xlabel('误差')
    ax.set_ylabel('密度')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_scatter(y, y_lstm, y_ewma, out_path, sample_n=10000):
    _setup_fonts()
    idx = np.arange(len(y))
    if len(idx) > sample_n:
        rng = np.random.default_rng(123)
        idx = rng.choice(idx, size=sample_n, replace=False)
    ys = y[idx]
    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    def one(ax, yh, name, color):
        if yh is None:
            ax.axis('off')
            ax.set_title(f'{name} (NA)')
            return
        ax.scatter(ys, yh[idx], s=5, alpha=0.3, color=color)
        ax.set_xlabel('真实值 y')
        ax.set_ylabel('预测值 yhat')
        ax.set_title(name)
    one(axs[0], y_lstm, 'LSTM', '#4C78A8')
    one(axs[1], y_ewma, 'EWMA', '#F58518')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_fit_timeseries(times, y, y_lstm, y_ewma, out_path):
    _setup_fonts()
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(times, y, label='真实值', color='#000000')
    if y_lstm is not None:
        ax.plot(times, y_lstm, label='LSTM', color='#4C78A8')
    if y_ewma is not None:
        ax.plot(times, y_ewma, label='EWMA', color='#F58518')
    ax.set_title('预测拟合时序（单步）')
    ax.set_xlabel('时间(s)')
    ax.set_ylabel('吞吐(kbps)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def detect_change_points(y, dt, threshold):
    arr = np.array(y, dtype=np.float32)
    if arr.size < 2:
        return []
    d = np.diff(arr)
    s = np.abs(d) / (dt if dt > 0 else 1.0)
    idx = np.where(s >= float(threshold))[0] + 1
    return [int(i) for i in idx]

def step_response_metrics(y, yhat, cps, ks):
    out = {}
    if yhat is None or y is None or len(y) == 0:
        for k in ks:
            out[f"mae@{k}"] = float('nan')
            out[f"rmse@{k}"] = float('nan')
            out[f"overshoot@{k}"] = float('nan')
        return out
    ya = np.array(y, dtype=np.float32)
    yh = np.array(yhat, dtype=np.float32)
    for k in ks:
        errs = []
        overs = 0
        tot = 0
        for cp in cps:
            t = cp + int(k)
            if t < ya.shape[0] and t < yh.shape[0] and cp < ya.shape[0]:
                e = float(yh[t] - ya[t])
                errs.append(e)
                delta = float(ya[t] - ya[cp])
                if delta != 0 and e * delta < 0:
                    overs += 1
                tot += 1
        if errs:
            ae = np.mean(np.abs(np.array(errs, dtype=np.float32)))
            re = np.sqrt(np.mean(np.array(errs, dtype=np.float32) ** 2))
            out[f"mae@{k}"] = float(ae)
            out[f"rmse@{k}"] = float(re)
            out[f"overshoot@{k}"] = float(overs) / float(tot) if tot > 0 else float('nan')
        else:
            out[f"mae@{k}"] = float('nan')
            out[f"rmse@{k}"] = float('nan')
            out[f"overshoot@{k}"] = float('nan')
    return out

def print_step_metrics(name, d, ks):
    print(f"[{name}-StepResponse]")
    for k in ks:
        print(f"step_mae@{k}={fmt(d.get('mae@'+str(k)))}")
        print(f"step_rmse@{k}={fmt(d.get('rmse@'+str(k)))}")
        print(f"overshoot_rate@{k}={fmt(d.get('overshoot@'+str(k)))}")
    print("")

def collect_series(fcc_dir, dt, hist_len, pred_horizon, max_traces, to_kbps=False):
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

def plot_step_response(ks, d_lstm, d_ewma, out_path):
    _setup_fonts()
    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    x = np.arange(len(ks))
    w = 0.25
    def series(d, prefix):
        if d is None:
            return None
        return [d.get(prefix + str(k)) for k in ks]
    m_l = series(d_lstm, 'mae@')
    m_e = series(d_ewma, 'mae@')
    r_l = series(d_lstm, 'rmse@')
    r_e = series(d_ewma, 'rmse@')
    o_l = series(d_lstm, 'overshoot@')
    o_e = series(d_ewma, 'overshoot@')
    def bars(ax, l, e, title):
        if l is not None:
            ax.bar(x - w/2, l, width=w, label='LSTM', color='#4C78A8')
        if e is not None:
            ax.bar(x + w/2, e, width=w, label='EWMA', color='#F58518')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in ks])
        ax.legend()
    bars(axs[0], m_l, m_e, 'Step MAE')
    bars(axs[1], r_l, r_e, 'Step RMSE')
    bars(axs[2], o_l, o_e, 'Overshoot Rate')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fcc_dir", default=os.path.join(os.getcwd(), "datasets", "fcc"))
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--hist_len", type=int, default=30)
    ap.add_argument("--pred_horizon", type=int, default=1)
    ap.add_argument("--max_traces", type=int, default=200)
    ap.add_argument("--to_kbps", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--cfg_path", default=os.path.join(os.getcwd(), "assets", "models", "fcc_lstm_cfg.json"))
    ap.add_argument("--onnx_path", default=os.path.join(os.getcwd(), "assets", "models", "fcc_lstm.onnx"))
    ap.add_argument("--example_trace_idx", type=int, default=0)
    ap.add_argument("--example_trace_path", default=None)
    ap.add_argument("--example_t0", type=float, default=None)
    ap.add_argument("--example_t1", type=float, default=None)
    ap.add_argument("--cp_threshold", type=float, default=1000.0)
    ap.add_argument("--step_ks", default="1,2,3")
    args = ap.parse_args()
    t0 = time.time()
    X, y = collect_series(args.fcc_dir, args.dt, args.hist_len, args.pred_horizon, args.max_traces, to_kbps=args.to_kbps)
    if X is None or y is None:
        print("dataset_empty")
        return
    print(f"dataset_shapes X={X.shape} y={y.shape}")
    t1 = time.time()
    yhat_lstm, lstm_step = lstm_predict(X, args.cfg_path, args.onnx_path)
    t2 = time.time()
    yhat_ewma, ewma_step = ewma_predict_windows(X, args.alpha)
    t3 = time.time()
    m_lstm = None
    m_ewma = None
    if yhat_lstm is not None:
        m_lstm = metrics(y, yhat_lstm)
        print_metrics("LSTM_ONNX", m_lstm)
        print(f"time_s={fmt(t2 - t1)}")
        print(f"time_per_step_s={fmt(lstm_step)}")
    else:
        print("LSTM_ONNX=NA")
    if yhat_ewma is not None:
        m_ewma = metrics(y, yhat_ewma)
        print_metrics("EWMA", m_ewma)
        print(f"time_s={fmt(t3 - t2)}")
        print(f"time_per_step_s={fmt(ewma_step)}")
    else:
        print("EWMA=NA")
    try:
        ks = [int(x.strip()) for x in str(args.step_ks).split(",") if x.strip()]
    except Exception:
        ks = [1, 2, 3]
    cps_all = detect_change_points(y, args.dt, args.cp_threshold)
    d_lstm = step_response_metrics(y, yhat_lstm, cps_all, ks)
    print_step_metrics("LSTM_ONNX", d_lstm, ks)
    d_ewma = step_response_metrics(y, yhat_ewma, cps_all, ks)
    print_step_metrics("EWMA", d_ewma, ks)
    base_dir = os.getcwd()
    img_dir = os.path.join(base_dir, "assets", "images")
    try:
        os.makedirs(img_dir, exist_ok=True)
    except Exception:
        pass
    out_step = os.path.join(img_dir, "predict_step_response_fcc_ewma_lstm.png")
    out_summary = os.path.join(img_dir, "predict_compare_summary_fcc_ewma_lstm.png")
    out_hist = os.path.join(img_dir, "predict_compare_error_hist_fcc_ewma_lstm.png")
    out_scatter = os.path.join(img_dir, "predict_compare_scatter_fcc_ewma_lstm.png")
    try:
        plot_step_response(ks, d_lstm, d_ewma, out_step)
    except Exception:
        pass
    y_for_hist = y
    y_lstm = yhat_lstm if yhat_lstm is not None else None
    y_ewma = yhat_ewma if yhat_ewma is not None else None
    plot_summary_bars(m_lstm, m_ewma, out_summary)
    plot_error_hist(y_for_hist, y_lstm, y_ewma, out_hist)
    plot_scatter(y_for_hist, y_lstm, y_ewma, out_scatter)
    try:
        paths_all = sorted(glob.glob(os.path.join(args.fcc_dir, "test_fcc_trace_*")))
        ex_path = args.example_trace_path if args.example_trace_path else (paths_all[args.example_trace_idx] if paths_all else None)
        if ex_path:
            tr = load_trace(ex_path, to_kbps=args.to_kbps)
            s = to_uniform_series(tr, args.dt)
            Xi, yi = build_xy(s, args.hist_len, args.pred_horizon)
            if Xi is not None and yi is not None and len(yi) > 0:
                yh_lstm, _ = lstm_predict(Xi, args.cfg_path, args.onnx_path)
                yh_ewma, _ = ewma_predict_windows(Xi, args.alpha)
                ts = np.array([ (args.hist_len + k) * args.dt for k in range(len(yi)) ], dtype=np.float32)
                if args.example_t0 is not None and args.example_t1 is not None:
                    m = (ts >= float(args.example_t0)) & (ts <= float(args.example_t1))
                    if np.any(m):
                        ts = ts[m]
                        yi = yi[m]
                        if yh_lstm is not None:
                            yh_lstm = yh_lstm[m]
                        if yh_ewma is not None:
                            yh_ewma = yh_ewma[m]
                out_fit = os.path.join(img_dir, "predict_fit_example_fcc_ewma_lstm.png")
                plot_fit_timeseries(ts, yi, yh_lstm, yh_ewma, out_fit)
    except Exception:
        pass

if __name__ == "__main__":
    main()

