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
    prev_b_for_switch = None
    for i in range(N):
        bi = b[i]
        li = lat[i]
        rbi = 1 if rb[i] else 0
        bs = math.log(1 + bi) if bi is not None and bi > 0 else 0.0
        lp = li if li is not None and li >= 0 else 0.0
        switch_penalty = 0.0
        if prev_b_for_switch is not None and bi is not None and bi > 0 and prev_b_for_switch > 0 and bi != prev_b_for_switch:
            max_b = max(bi, prev_b_for_switch)
            if max_b > 0:
                rel_change = abs(bi - prev_b_for_switch) / max_b
                switch_penalty = 0.5 * rel_change
        if bi is not None and bi > 0:
            prev_b_for_switch = bi
        qv = bs - 3.5 * rbi - 0.15 * lp - switch_penalty
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

def plot_summary(metrics_list, names, out_path):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    colors = ['#4C78A8','#F58518','#54A24B','#E45756']
    x = names
    ax = axs[0][0]
    ax.bar(x, [m.get("avg_bitrate") for m in metrics_list], color=colors[:len(names)])
    ax.set_title("时间均值码率(kbps)")
    ax = axs[0][1]
    ax.bar(x, [m.get("avg_latency") for m in metrics_list], color=colors[:len(names)])
    ax.set_title("时间均值延迟(s)")
    ax = axs[0][2]
    ax.bar(x, [m.get("p95_latency") for m in metrics_list], color=colors[:len(names)])
    ax.set_title("时间采样95分位延迟(s)")
    ax = axs[1][0]
    ax.bar(x, [m.get("rebuffer_fraction") for m in metrics_list], color=colors[:len(names)])
    ax.set_title("重缓冲时间占比")
    ax = axs[1][1]
    ax.bar(x, [m.get("quality_switches_per_min") for m in metrics_list], color=colors[:len(names)])
    ax.set_title("质量切换/分钟")
    ax = axs[1][2]
    ax.bar(x, [m.get("qoe_mean") for m in metrics_list], color=colors[:len(names)])
    ax.set_title("时间均值QoE")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_timeseries(times_list, series_list, names, out_path):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    colors = ["#4C78A8","#F58518","#54A24B","#E45756"]
    rel_times = [_rel_sec(t) for t in times_list]
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    ax = axs[0]
    for i in range(len(names)):
        ax.plot(rel_times[i], series_list[i]["bitrateKbps"], label=names[i], color=colors[i % len(colors)])
    ax.set_ylabel("码率(kbps)")
    ax.set_title("时间归一化码率")
    ax.legend()
    ax = axs[1]
    for i in range(len(names)):
        ax.plot(rel_times[i], series_list[i]["bufferLevel"], label=names[i], color=colors[i % len(colors)])
    ax.set_ylabel("缓冲(s)")
    ax.set_title("时间归一化缓冲")
    ax = axs[2]
    for i in range(len(names)):
        ax.plot(rel_times[i], series_list[i]["liveLatency"], label=names[i], color=colors[i % len(colors)])
    ax.set_ylabel("延时(s)")
    ax.set_title("时间归一化延时")
    ax.set_xlabel("时间(s)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_quality_dist(series_list, names, out_path):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    q_list = [[v for v in s["qualityIndex"] if v is not None] for s in series_list]
    counters = [Counter(q) for q in q_list]
    max_q = 0
    for c in counters:
        if c:
            max_q = max(max_q, max(c.keys()))
    idxs = list(range(max_q+1))
    fracs = []
    for c in counters:
        v = [c.get(i,0) for i in idxs]
        s = sum(v) if v else 0
        fracs.append([(x/s if s>0 else 0) for x in v])
    x = list(range(len(idxs)))
    w = 0.8 / max(1, len(names))
    colors = ["#4C78A8","#F58518","#54A24B","#E45756"]
    fig, ax = plt.subplots(figsize=(12,5))
    for j in range(len(names)):
        ax.bar([i - 0.4 + w*j for i in x], fracs[j], width=w, label=names[j], color=colors[j % len(colors)])
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

def align_duration_multi(rows_list):
    if not rows_list:
        return None
    durs = []
    for rows in rows_list:
        if not rows:
            return None
        s = rows[0]["ts"]
        e = rows[-1]["ts"]
        durs.append((s, e))
    cd = min((e - s).total_seconds() for s, e in durs)
    if cd <= 0:
        return None
    return [(s, s + datetime.timedelta(seconds=cd)) for s, e in durs]

def main():
    base_dir = os.getcwd()
    p_lolp = os.path.join(base_dir, "assets", "data", "dash_stats_lolp_fcc.csv")
    p_l2a = os.path.join(base_dir, "assets", "data", "dash_stats_l2a_fcc.csv")
    p_custom = os.path.join(base_dir, "assets", "data", "dash_stats_customrule_fcc.csv")
    dt = 0.5
    args = sys.argv[1:]
    if len(args) >= 3:
        p_lolp = args[0]
        p_l2a = args[1]
        p_custom = args[2]
    if len(args) >= 4:
        try:
            dt = float(args[3])
        except Exception:
            dt = 0.5
    rows_lolp = load_rows(p_lolp)
    rows_l2a = load_rows(p_l2a)
    rows_custom = load_rows(p_custom)
    datasets = [("LoLp", rows_lolp), ("L2A", rows_l2a), ("CustomRule", rows_custom)]
    datasets = [(n, r) for (n, r) in datasets if r]
    if len(datasets) < 2:
        print("No aligned duration")
        return
    rows_list = [r for (_, r) in datasets]
    names = [n for (n, _) in datasets]
    aligned_multi = align_duration_multi(rows_list)
    if aligned_multi is None:
        print("No aligned duration")
        return
    times_list = []
    series_list = []
    for i in range(len(rows_list)):
        t0, t1 = aligned_multi[i]
        ti, si = resample_rows(rows_list[i], dt, t0, t1)
        times_list.append(ti)
        series_list.append(si)
    metrics_list = [compute_time_metrics(times_list[i], series_list[i], dt) for i in range(len(names))]
    for i in range(len(names)):
        print_time_metrics(names[i] + "-Time", metrics_list[i])
    baseline_idx = 0
    for i in range(len(names)):
        if names[i].lower().startswith("lolp"):
            baseline_idx = i
            break
    for i in range(len(names)):
        if i == baseline_idx:
            continue
        compare_time(metrics_list[i], metrics_list[baseline_idx], names[i] + "-Time", names[baseline_idx] + "-Time")
    out_summary = os.path.join(base_dir, "assets", "images", "abr_time_norm_summary.png")
    out_ts = os.path.join(base_dir, "assets", "images", "abr_time_norm_timeseries.png")
    out_qdist = os.path.join(base_dir, "assets", "images", "abr_time_norm_quality_dist.png")
    plot_summary(metrics_list, names, out_summary)
    plot_timeseries(times_list, series_list, names, out_ts)
    plot_quality_dist(series_list, names, out_qdist)

if __name__ == "__main__":
    main()
