import csv
import sys
import os
import math
import statistics
import datetime
from collections import Counter
import matplotlib.pyplot as plt

def _to_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None

def _to_int(x):
    try:
        if x is None or x == "":
            return None
        return int(float(x))
    except Exception:
        return None

def _to_bool_rebuffer(x):
    s = (x or "").strip().lower()
    return s.startswith("rebuffer")

def _to_dt(x):
    if not x:
        return None
    s = x.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(s)
    except Exception:
        return None

def load_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ts = _to_dt(row.get("timestamp"))
            rows.append({
                "ts": ts,
                "timestamp": row.get("timestamp"),
                "bufferLevel": _to_float(row.get("bufferLevel")),
                "qualityIndex": _to_int(row.get("qualityIndex")),
                "bitrateKbps": _to_int(row.get("bitrateKbps")),
                "resolution": row.get("resolution"),
                "liveLatency": _to_float(row.get("liveLatency")),
                "predictedKbps": _to_int(row.get("predictedKbps")),
                "rebuffer": _to_bool_rebuffer(row.get("rebuffer"))
            })
    rows = [r for r in rows if r.get("ts") is not None]
    rows.sort(key=lambda x: x["ts"])
    return rows

def _rel_times(ts_list):
    t0 = None
    for t in ts_list:
        if t is not None:
            t0 = t
            break
    out = []
    for t in ts_list:
        tr = None
        if t is not None and t0 is not None:
            tr = (t - t0).total_seconds()
        out.append(tr)
    return out

def _series_at(rows, idx, fields, last_vals):
    r = rows[idx]
    for k in fields:
        v = r.get(k)
        if v is not None:
            last_vals[k] = v
    return {k: last_vals.get(k) for k in fields}

def resample_rows(rows, dt, start_t, end_t):
    if not rows:
        return [], []
    times = []
    cur = start_t
    while cur <= end_t:
        times.append(cur)
        cur += datetime.timedelta(seconds=dt)
    fields = ["bitrateKbps","bufferLevel","liveLatency","qualityIndex","predictedKbps","rebuffer"]
    out = {k: [] for k in fields}
    last_vals = {}
    p = 0
    n = len(rows)
    for t in times:
        while p+1 < n and rows[p+1]["ts"] <= t:
            p += 1
        vals = _series_at(rows, p, fields, last_vals)
        for k in fields:
            out[k].append(vals.get(k))
    return times, out

def _mean(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return statistics.mean(vals)

def _p95(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    vals.sort()
    idx = max(0, math.floor(0.95 * (len(vals) - 1)))
    return vals[idx]

def _count_edges(bools):
    prev = False
    c = 0
    for v in bools:
        v2 = bool(v)
        if v2 and not prev:
            c += 1
        prev = v2
    return c

def compute_time_metrics(times, series, dt):
    N = len(times)
    dur = N * dt
    b = series["bitrateKbps"]
    buf = series["bufferLevel"]
    lat = series["liveLatency"]
    q = series["qualityIndex"]
    pred = series["predictedKbps"]
    rb = [bool(x) for x in series["rebuffer"]]
    avg_bitrate = _mean(b)
    avg_buffer = _mean(buf)
    avg_latency = _mean(lat)
    p95_latency = _p95(lat)
    rebuffer_fraction = (sum(1 for v in rb if v) / N) if N else None
    rebuffer_seconds = rebuffer_fraction * dur if rebuffer_fraction is not None else None
    rebuffer_events_per_min = (_count_edges(rb) * 60.0 / dur) if dur > 0 else None
    switches = 0
    up = 0
    down = 0
    mags = []
    prev_q = None
    for v in q:
        if v is None:
            continue
        if prev_q is not None:
            d = v - prev_q
            if d != 0:
                switches += 1
                if d > 0:
                    up += 1
                else:
                    down += 1
                mags.append(abs(d))
        prev_q = v
    quality_switches_per_min = (switches * 60.0 / dur) if dur > 0 else None
    upswitches_per_min = (up * 60.0 / dur) if dur > 0 else None
    downswitches_per_min = (down * 60.0 / dur) if dur > 0 else None
    avg_switch_magnitude = _mean(mags)
    min_quality_time_fraction = None
    if N:
        cnt_min = sum(1 for v in q if v == 0)
        min_quality_time_fraction = cnt_min / N
    ratios = []
    for i in range(N):
        bi = b[i]
        pi = pred[i]
        if bi is not None and bi > 0 and pi is not None and pi > 0:
            ratios.append(pi / bi)
    pred_to_bitrate_ratio_mean = _mean(ratios)
    avg_recover_seconds = None
    prev_t = None
    prev_rb = False
    prev_q0 = False
    recs = []
    for i in range(N):
        if rb[i]:
            prev_rb = True
            prev_t = times[i]
            prev_q0 = (q[i] == 0)
        else:
            if prev_rb and prev_q0 and q[i] is not None and q[i] > 0 and prev_t is not None:
                delta = (times[i] - prev_t).total_seconds()
                if delta >= 0:
                    recs.append(delta)
                prev_rb = False
                prev_q0 = False
                prev_t = None
            elif q[i] is not None and q[i] > 0:
                prev_rb = False
                prev_q0 = False
                prev_t = None
    avg_recover_seconds = _mean(recs)
    qoe = []
    for i in range(N):
        bi = b[i]
        li = lat[i]
        rbi = 1 if rb[i] else 0
        bs = math.log(1 + bi) if bi is not None and bi > 0 else 0.0
        lp = li if li is not None and li >= 0 else 0.0
        qv = bs - 4.0 * rbi - 0.2 * lp
        qoe.append(qv)
    qoe_mean = _mean(qoe)
    return {
        "duration_s": dur,
        "avg_bitrate": avg_bitrate,
        "avg_buffer": avg_buffer,
        "avg_latency": avg_latency,
        "p95_latency": p95_latency,
        "rebuffer_fraction": rebuffer_fraction,
        "rebuffer_seconds": rebuffer_seconds,
        "rebuffer_events_per_min": rebuffer_events_per_min,
        "quality_switches_per_min": quality_switches_per_min,
        "upswitches_per_min": upswitches_per_min,
        "downswitches_per_min": downswitches_per_min,
        "avg_switch_magnitude": avg_switch_magnitude,
        "min_quality_time_fraction": min_quality_time_fraction,
        "pred_to_bitrate_ratio_mean": pred_to_bitrate_ratio_mean,
        "avg_recover_seconds": avg_recover_seconds,
        "qoe_mean": qoe_mean
    }

def fmt(v):
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)

def print_time_metrics(name, m):
    print(f"[{name}]")
    print(f"duration_s={fmt(m['duration_s'])}")
    print(f"avg_bitrate_kbps={fmt(m['avg_bitrate'])}")
    print(f"avg_buffer_s={fmt(m['avg_buffer'])}")
    print(f"avg_latency_s={fmt(m['avg_latency'])}")
    print(f"p95_latency_s={fmt(m['p95_latency'])}")
    print(f"rebuffer_fraction_time={fmt(m['rebuffer_fraction'])}")
    print(f"rebuffer_seconds={fmt(m['rebuffer_seconds'])}")
    print(f"rebuffer_events_per_min={fmt(m['rebuffer_events_per_min'])}")
    print(f"quality_switches_per_min={fmt(m['quality_switches_per_min'])}")
    print(f"upswitches_per_min={fmt(m['upswitches_per_min'])}")
    print(f"downswitches_per_min={fmt(m['downswitches_per_min'])}")
    print(f"avg_switch_magnitude={fmt(m['avg_switch_magnitude'])}")
    print(f"min_quality_time_fraction={fmt(m['min_quality_time_fraction'])}")
    print(f"pred_to_bitrate_ratio_mean={fmt(m['pred_to_bitrate_ratio_mean'])}")
    print(f"avg_recover_seconds={fmt(m['avg_recover_seconds'])}")
    print(f"qoe_mean={fmt(m['qoe_mean'])}")
    print("")

def compare_time(a, b, an, bn):
    def diff(k):
        va = a.get(k)
        vb = b.get(k)
        if va is None or vb is None:
            return None
        return va - vb
    print("[Comparison-TimeNormalized]")
    keys = [
        ("avg_bitrate", "avg_bitrate_kbps"),
        ("rebuffer_fraction", "rebuffer_fraction_time"),
        ("rebuffer_seconds", "rebuffer_seconds"),
        ("rebuffer_events_per_min", "rebuffer_events_per_min"),
        ("avg_latency", "avg_latency_s"),
        ("p95_latency", "p95_latency_s"),
        ("min_quality_time_fraction", "min_quality_time_fraction"),
        ("quality_switches_per_min", "quality_switches_per_min"),
        ("avg_switch_magnitude", "avg_switch_magnitude"),
        ("avg_recover_seconds", "avg_recover_seconds"),
        ("qoe_mean", "qoe_mean")
    ]
    for k, label in keys:
        d = diff(k)
        print(f"{label}: {an}={fmt(a.get(k))} | {bn}={fmt(b.get(k))} | diff({an}-{bn})={fmt(d)}")

def _rel_sec(times):
    if not times:
        return []
    t0 = times[0]
    return [(t - t0).total_seconds() for t in times]

def plot_summary(m1, m2, n1, n2, out_path):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    ax = axs[0][0]
    ax.bar([n1, n2], [m1.get("avg_bitrate"), m2.get("avg_bitrate")], color=['#4C78A8','#F58518'])
    ax.set_title("时间均值码率(kbps)")
    ax = axs[0][1]
    ax.bar([n1, n2], [m1.get("avg_latency"), m2.get("avg_latency")], color=['#4C78A8','#F58518'])
    ax.set_title("时间均值延迟(s)")
    ax = axs[0][2]
    ax.bar([n1, n2], [m1.get("p95_latency"), m2.get("p95_latency")], color=['#4C78A8','#F58518'])
    ax.set_title("时间采样95分位延迟(s)")
    ax = axs[1][0]
    ax.bar([n1, n2], [m1.get("rebuffer_fraction"), m2.get("rebuffer_fraction")], color=['#4C78A8','#F58518'])
    ax.set_title("重缓冲时间占比")
    ax = axs[1][1]
    ax.bar([n1, n2], [m1.get("quality_switches_per_min"), m2.get("quality_switches_per_min")], color=['#4C78A8','#F58518'])
    ax.set_title("质量切换/分钟")
    ax = axs[1][2]
    ax.bar([n1, n2], [m1.get("qoe_mean"), m2.get("qoe_mean")], color=['#4C78A8','#F58518'])
    ax.set_title("时间均值QoE")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_timeseries(t1, s1, t2, s2, n1, n2, out_path):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    rt1 = _rel_sec(t1)
    rt2 = _rel_sec(t2)
    b1 = s1["bitrateKbps"]
    b2 = s2["bitrateKbps"]
    buf1 = s1["bufferLevel"]
    buf2 = s2["bufferLevel"]
    lat1 = s1["liveLatency"]
    lat2 = s2["liveLatency"]
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    ax = axs[0]
    ax.plot(rt1, b1, label=n1, color="#4C78A8")
    ax.plot(rt2, b2, label=n2, color="#F58518")
    ax.set_ylabel("码率(kbps)")
    ax.set_title("时间归一化码率")
    ax.legend()
    ax = axs[1]
    ax.plot(rt1, buf1, label=n1, color="#4C78A8")
    ax.plot(rt2, buf2, label=n2, color="#F58518")
    ax.set_ylabel("缓冲(s)")
    ax.set_title("时间归一化缓冲")
    ax = axs[2]
    ax.plot(rt1, lat1, label=n1, color="#4C78A8")
    ax.plot(rt2, lat2, label=n2, color="#F58518")
    ax.set_ylabel("延时(s)")
    ax.set_title("时间归一化延时")
    ax.set_xlabel("时间(s)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_quality_dist(series1, series2, n1, n2, out_path):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    q1 = [v for v in series1["qualityIndex"] if v is not None]
    q2 = [v for v in series2["qualityIndex"] if v is not None]
    c1 = Counter(q1)
    c2 = Counter(q2)
    max_q = 0
    if c1:
        max_q = max(max_q, max(c1.keys()))
    if c2:
        max_q = max(max_q, max(c2.keys()))
    idxs = list(range(max_q+1))
    v1 = [c1.get(i,0) for i in idxs]
    v2 = [c2.get(i,0) for i in idxs]
    s1 = sum(v1) if v1 else 0
    s2 = sum(v2) if v2 else 0
    f1 = [(x/s1 if s1>0 else 0) for x in v1]
    f2 = [(x/s2 if s2>0 else 0) for x in v2]
    x = list(range(len(idxs)))
    w = 0.4
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar([i-w/2 for i in x], f1, width=w, label=n1, color="#4C78A8")
    ax.bar([i+w/2 for i in x], f2, width=w, label=n2, color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in idxs])
    ax.set_xlabel("质量档位")
    ax.set_ylabel("占比")
    ax.set_title("时间采样质量分布")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def align_duration(rows1, rows2):
    if not rows1 or not rows2:
        return None
    s1 = rows1[0]["ts"]
    e1 = rows1[-1]["ts"]
    s2 = rows2[0]["ts"]
    e2 = rows2[-1]["ts"]
    d1 = (e1 - s1).total_seconds()
    d2 = (e2 - s2).total_seconds()
    cd = min(d1, d2)
    if cd <= 0:
        return None
    return (s1, s1 + datetime.timedelta(seconds=cd), s2, s2 + datetime.timedelta(seconds=cd))

def main():
    base_dir = os.getcwd()
    p1 = os.path.join(base_dir, "assets", "data", "dash_stat_lolp_oboe.csv")
    p2 = os.path.join(base_dir, "assets", "data", "dash_stats_customrule_oboe.csv")
    dt = 0.5
    args = sys.argv[1:]
    if len(args) >= 2:
        p1 = args[0]
        p2 = args[1]
    if len(args) >= 3:
        try:
            dt = float(args[2])
        except Exception:
            dt = 0.5
    rows1 = load_rows(p1)
    rows2 = load_rows(p2)
    aligned = align_duration(rows1, rows2)
    if aligned is None:
        print("No aligned duration")
        return
    s1_t0, s1_t1, s2_t0, s2_t1 = aligned
    t1, s1 = resample_rows(rows1, dt, s1_t0, s1_t1)
    t2, s2 = resample_rows(rows2, dt, s2_t0, s2_t1)
    m1 = compute_time_metrics(t1, s1, dt)
    m2 = compute_time_metrics(t2, s2, dt)
    print_time_metrics("LoLp-Time", m1)
    print_time_metrics("CustomRule-Time", m2)
    compare_time(m2, m1, "CustomRule-Time", "LoLp-Time")
    out_summary = os.path.join(base_dir, "assets", "images", "abr_time_norm_summary.png")
    out_ts = os.path.join(base_dir, "assets", "images", "abr_time_norm_timeseries.png")
    out_qdist = os.path.join(base_dir, "assets", "images", "abr_time_norm_quality_dist.png")
    plot_summary(m1, m2, "LoLp", "CustomRule", out_summary)
    plot_timeseries(t1, s1, t2, s2, "LoLp", "CustomRule", out_ts)
    plot_quality_dist(s1, s2, "LoLp", "CustomRule", out_qdist)

if __name__ == "__main__":
    main()
