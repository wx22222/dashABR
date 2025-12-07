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
            rows.append({
                "timestamp": row.get("timestamp"),
                "bufferLevel": _to_float(row.get("bufferLevel")),
                "qualityIndex": _to_int(row.get("qualityIndex")),
                "bitrateKbps": _to_int(row.get("bitrateKbps")),
                "resolution": row.get("resolution"),
                "liveLatency": _to_float(row.get("liveLatency")),
                "predictedKbps": _to_int(row.get("predictedKbps")),
                "rebuffer": _to_bool_rebuffer(row.get("rebuffer"))
            })
    return rows

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

def _sum_bool(vals):
    return sum(1 for v in vals if v)

def compute_metrics(rows):
    n = len(rows)
    bitrates = [r.get("bitrateKbps") for r in rows]
    buffers = [r.get("bufferLevel") for r in rows]
    latencies = [r.get("liveLatency") for r in rows]
    predicted = [r.get("predictedKbps") for r in rows]
    rebuf_flags = [r.get("rebuffer") for r in rows]
    qidx = [r.get("qualityIndex") for r in rows]

    avg_bitrate = _mean(bitrates)
    avg_buffer = _mean(buffers)
    avg_latency = _mean(latencies)
    p95_latency = _p95(latencies)
    rebuffer_fraction = _sum_bool(rebuf_flags) / n if n else None

    rebuffer_events = 0
    prev = False
    for r in rebuf_flags:
        if r and not prev:
            rebuffer_events += 1
        prev = r

    quality_switches = 0
    up_switches = 0
    down_switches = 0
    stability = []
    prev_q = None
    for r in qidx:
        if r is None:
            continue
        if prev_q is not None:
            d = r - prev_q
            if d != 0:
                quality_switches += 1
                if d > 0:
                    up_switches += 1
                else:
                    down_switches += 1
            stability.append(abs(d))
        prev_q = r
    avg_switch_magnitude = _mean(stability)

    min_quality_fraction = None
    if n:
        cnt_min = sum(1 for r in qidx if r == 0)
        min_quality_fraction = cnt_min / n

    pred_to_bitrate_ratio_mean = None
    ratios = []
    for i in range(n):
        b = bitrates[i]
        p = predicted[i]
        if b is not None and b > 0 and p is not None and p > 0:
            ratios.append(p / b)
    pred_to_bitrate_ratio_mean = _mean(ratios)

    recover_times = []
    prev_ts = None
    prev_rebuf = False
    prev_q0 = False
    for i in range(n):
        ts = _to_dt(rows[i].get("timestamp"))
        q = qidx[i]
        rb = rebuf_flags[i]
        if rb:
            prev_rebuf = True
            prev_ts = ts
            prev_q0 = (q == 0)
        elif prev_rebuf and prev_q0 and q is not None and q > 0 and prev_ts is not None and ts is not None:
            delta = (ts - prev_ts).total_seconds()
            if delta >= 0:
                recover_times.append(delta)
            prev_rebuf = False
            prev_q0 = False
            prev_ts = None
        else:
            if q is not None and q > 0:
                prev_rebuf = False
                prev_q0 = False
                prev_ts = None

    avg_recover_seconds = _mean(recover_times)

    qoe_samples = []
    for i in range(n):
        b = bitrates[i]
        l = latencies[i]
        rb = 1 if rebuf_flags[i] else 0
        bs = math.log(1 + b) if b is not None and b > 0 else 0.0
        lp = l if l is not None and l >= 0 else 0.0
        q = bs - 4.0 * rb - 0.2 * lp
        qoe_samples.append(q)
    qoe_mean = _mean(qoe_samples)

    return {
        "samples": n,
        "avg_bitrate": avg_bitrate,
        "avg_buffer": avg_buffer,
        "avg_latency": avg_latency,
        "p95_latency": p95_latency,
        "rebuffer_fraction": rebuffer_fraction,
        "rebuffer_events": rebuffer_events,
        "quality_switches": quality_switches,
        "up_switches": up_switches,
        "down_switches": down_switches,
        "avg_switch_magnitude": avg_switch_magnitude,
        "min_quality_fraction": min_quality_fraction,
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

def print_metrics(name, m):
    print(f"[{name}]")
    print(f"samples={fmt(m['samples'])}")
    print(f"avg_bitrate_kbps={fmt(m['avg_bitrate'])}")
    print(f"avg_buffer_s={fmt(m['avg_buffer'])}")
    print(f"avg_latency_s={fmt(m['avg_latency'])}")
    print(f"p95_latency_s={fmt(m['p95_latency'])}")
    print(f"rebuffer_fraction={fmt(m['rebuffer_fraction'])}")
    print(f"rebuffer_events={fmt(m['rebuffer_events'])}")
    print(f"quality_switches={fmt(m['quality_switches'])}")
    print(f"upswitches={fmt(m['up_switches'])}")
    print(f"downswitches={fmt(m['down_switches'])}")
    print(f"avg_switch_magnitude={fmt(m['avg_switch_magnitude'])}")
    print(f"min_quality_fraction={fmt(m['min_quality_fraction'])}")
    print(f"pred_to_bitrate_ratio_mean={fmt(m['pred_to_bitrate_ratio_mean'])}")
    print(f"avg_recover_seconds={fmt(m['avg_recover_seconds'])}")
    print(f"qoe_mean={fmt(m['qoe_mean'])}")
    print("")

def compare(a, b, an, bn):
    def diff(k):
        va = a.get(k)
        vb = b.get(k)
        if va is None or vb is None:
            return None
        return va - vb
    print("[Comparison]")
    keys = [
        ("avg_bitrate", "avg_bitrate_kbps"),
        ("rebuffer_fraction", "rebuffer_fraction"),
        ("rebuffer_events", "rebuffer_events"),
        ("avg_latency", "avg_latency_s"),
        ("p95_latency", "p95_latency_s"),
        ("min_quality_fraction", "min_quality_fraction"),
        ("quality_switches", "quality_switches"),
        ("avg_switch_magnitude", "avg_switch_magnitude"),
        ("avg_recover_seconds", "avg_recover_seconds"),
        ("qoe_mean", "qoe_mean")
    ]
    for k, label in keys:
        d = diff(k)
        print(f"{label}: {an}={fmt(a.get(k))} | {bn}={fmt(b.get(k))} | diff({an}-{bn})={fmt(d)}")

def _rel_times(rows):
    ts = [ _to_dt(r.get("timestamp")) for r in rows ]
    t0 = None
    for t in ts:
        if t is not None:
            t0 = t
            break
    out = []
    for i in range(len(rows)):
        t = ts[i]
        tr = None
        if t is not None and t0 is not None:
            tr = (t - t0).total_seconds()
        out.append(tr)
    return out

def plot_summary(m1, m2, n1, n2, out_path):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    ax = axs[0][0]
    ax.bar([n1, n2], [m1.get("avg_bitrate"), m2.get("avg_bitrate")], color=['#4C78A8','#F58518'])
    ax.set_title("平均码率(kbps)")
    ax = axs[0][1]
    ax.bar([n1, n2], [m1.get("avg_latency"), m2.get("avg_latency")], color=['#4C78A8','#F58518'])
    ax.set_title("平均延迟(s)")
    ax = axs[0][2]
    ax.bar([n1, n2], [m1.get("p95_latency"), m2.get("p95_latency")], color=['#4C78A8','#F58518'])
    ax.set_title("95分位延迟(s)")
    ax = axs[1][0]
    ax.bar([n1, n2], [m1.get("rebuffer_fraction"), m2.get("rebuffer_fraction")], color=['#4C78A8','#F58518'])
    ax.set_title("重缓冲比例")
    ax = axs[1][1]
    ax.bar([n1, n2], [m1.get("quality_switches"), m2.get("quality_switches")], color=['#4C78A8','#F58518'])
    ax.set_title("质量切换次数")
    ax = axs[1][2]
    ax.bar([n1, n2], [m1.get("qoe_mean"), m2.get("qoe_mean")], color=['#4C78A8','#F58518'])
    ax.set_title("QoE均值")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_timeseries(rows1, rows2, n1, n2, out_path):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    t1 = _rel_times(rows1)
    t2 = _rel_times(rows2)
    b1 = [ r.get("bitrateKbps") for r in rows1 ]
    b2 = [ r.get("bitrateKbps") for r in rows2 ]
    buf1 = [ r.get("bufferLevel") for r in rows1 ]
    buf2 = [ r.get("bufferLevel") for r in rows2 ]
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    ax = axs[0]
    ax.plot([x for x in t1 if x is not None], [b1[i] for i,x in enumerate(t1) if x is not None], label=n1, color="#4C78A8")
    ax.plot([x for x in t2 if x is not None], [b2[i] for i,x in enumerate(t2) if x is not None], label=n2, color="#F58518")
    ax.set_ylabel("码率(kbps)")
    ax.set_title("码率随时间")
    ax.legend()
    ax = axs[1]
    ax.plot([x for x in t1 if x is not None], [buf1[i] for i,x in enumerate(t1) if x is not None], label=n1, color="#4C78A8")
    ax.plot([x for x in t2 if x is not None], [buf2[i] for i,x in enumerate(t2) if x is not None], label=n2, color="#F58518")
    ax.set_ylabel("缓冲(s)")
    ax.set_title("缓冲随时间")
    ax.set_xlabel("时间(s)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_quality_dist(rows1, rows2, n1, n2, out_path):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    q1 = [ r.get("qualityIndex") for r in rows1 if r.get("qualityIndex") is not None ]
    q2 = [ r.get("qualityIndex") for r in rows2 if r.get("qualityIndex") is not None ]
    c1 = Counter(q1)
    c2 = Counter(q2)
    max_q = 0
    if c1:
        max_q = max(max_q, max(c1.keys()))
    if c2:
        max_q = max(max_q, max(c2.keys()))
    idxs = list(range(max_q+1))
    v1 = [ c1.get(i,0) for i in idxs ]
    v2 = [ c2.get(i,0) for i in idxs ]
    s1 = sum(v1) if v1 else 0
    s2 = sum(v2) if v2 else 0
    f1 = [ (x/s1 if s1>0 else 0) for x in v1 ]
    f2 = [ (x/s2 if s2>0 else 0) for x in v2 ]
    x = list(range(len(idxs)))
    w = 0.4
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar([i-w/2 for i in x], f1, width=w, label=n1, color="#4C78A8")
    ax.bar([i+w/2 for i in x], f2, width=w, label=n2, color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in idxs])
    ax.set_xlabel("质量档位")
    ax.set_ylabel("占比")
    ax.set_title("质量分布对比")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    base_dir = os.getcwd()
    p1 = os.path.join(base_dir, "assets", "data", "dash_stat_LoLp_oboe.csv")
    p2 = os.path.join(base_dir, "assets", "data", "dash_stats_customrule_oboe.csv")
    args = sys.argv[1:]
    if len(args) >= 2:
        p1 = args[0]
        p2 = args[1]
    rows1 = load_rows(p1)
    rows2 = load_rows(p2)
    m1 = compute_metrics(rows1)
    m2 = compute_metrics(rows2)
    print_metrics("LoLp", m1)
    print_metrics("CustomRule", m2)
    compare(m2, m1, "CustomRule", "LoLp")
    out_summary = os.path.join(base_dir, "assets", "images", "abr_compare_summary.png")
    out_ts = os.path.join(base_dir, "assets", "images", "abr_compare_timeseries.png")
    out_qdist = os.path.join(base_dir, "assets", "images", "abr_compare_quality_dist.png")
    plot_summary(m1, m2, "LoLp", "CustomRule", out_summary)
    plot_timeseries(rows1, rows2, "LoLp", "CustomRule", out_ts)
    plot_quality_dist(rows1, rows2, "LoLp", "CustomRule", out_qdist)

if __name__ == "__main__":
    main()

