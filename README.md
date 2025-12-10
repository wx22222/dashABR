# Dash.js 自定义播放器（ABR 研究）

## 项目概览
- 基于 Dash.js 的前端播放器，支持直播/点播 MPD 播放、可自定义流地址、接入自定义 ABR 规则。
- 集成 ONNX Runtime Web 的 LSTM 吞吐预测（模型与配置位于 `assets/models/`）。
- 提供实时指标面板（Stats）：缓冲、码率、分辨率、直播延时、预测吞吐、重缓冲提示，并支持导出为 CSV。
- 采样采用“事件驱动”模式：每当一个视频分片下载完成时采样更新，避免时间采样的遗漏。

## 目录结构
- `index.html` 页面入口
- `styles/` 前端样式
  - `styles/style.css`
- `scripts/` 前端脚本
  - `scripts/main.js` 播放器与指标逻辑
  - `scripts/CustomAbrRule.js` 自定义 ABR 规则与 LSTM 推理
- `assets/models/` 模型与配置
  - `assets/models/fcc_lstm.onnx`、`assets/models/fcc_lstm_cfg.json`
  - `assets/models/oboe_lstm.onnx`、`assets/models/oboe_lstm_cfg.json`
- `assets/data/` 指标 CSV（用于分析脚本）
- `assets/images/` 分析输出图片（汇总、时序、质量分布、步响应）
- `datasets/` 评测数据集
  - `datasets/fcc/` FCC 轨迹
  - `datasets/oboe/` Oboe 轨迹（可选）
- `python/train/` 训练脚本（可选）
  - `python/train/train_lstm_oboe.py`
- `python/analysis/` 分析脚本
  - `python/analysis/compare_abr_time_norm.py`
  - `python/analysis/compare_predictors_fcc.py`
  - `python/analysis/compare_predictors_oboe.py`

## 功能
- 自定义 MPD URL 输入并手动加载，不自动播放。
- 注册自定义 ABR 规则，默认 ABR 可按需启/停，支持自定义规则与默认策略协同。
- Stats 指标展示：
  - Buffer Level（秒）
  - Current Quality Index（当前质量索引）
  - Bitrate（kbps）
  - Resolution（如 1920x1080）
  - Live Latency（直播延时，秒）
  - Predicted Throughput（预测吞吐）
  - Rebuffer 提示（发生卡顿时显示“rebuffer happen”）
- 导出 CSV（含时间戳与上述指标），一键下载。

## 快速开始
1. 启动本地静态服务器（示例端口 `8081`）：
   - `python -m http.server 8081`
2. 浏览器访问：
   - `http://localhost:8081/`
3. 在输入框填入 MPD 地址，点击 `Load Stream` 开始播放。
4. 若需启用自定义 ABR，点击 `Enable Custom ABR Rule` 按钮。
5. 在播放过程中，可点击 `Export CSV` 将当前采样数据下载为 `.csv` 并导入 Excel。

> 提示：请在项目根目录运行静态服务器，以便 `assets/`、`scripts/`、`styles/` 等路径可直接访问。

## 指标与采样
- 采样触发：
  - 每个视频分片下载完成时进行一次采样。
  - 发生缓冲耗尽或浏览器等待事件时，也会立即采样。
- 直播延时计算：
  - 使用 `player.getDvrWindow().end - player.time()` 估算直播延时。
- CSV 字段：
  - `timestamp, bufferLevel, qualityIndex, bitrateKbps, resolution, liveLatency, predictedKbps, rebuffer`

## 自定义 ABR（核心思路）
- 吞吐估计融合：EWMA 快/慢与 LSTM 预测融合，生成保守吞吐 `predictedKbps * safetyFactor`。
- 升档门限：缓冲阈值、下载时长预算、码率裕量、最大步进；缓冲安全时放宽，延时偏高时收紧。
- 降档门限：新增下行业务门限 `downswitchMargin`，仅在预测显著低于当前档位码率时考虑降档，降低“轻微波动即降档”的频率。
- 目标函数：综合 `Bitrate↑`、`Rebuffer↓`、`Latency↓` 进行评分，重缓冲与延时显式纳入惩罚，紧急低缓冲时强制最低档。

## 指标分析
- 时间归一化对比（LoLp / L2A / CustomRule / Bola）：
  - `python python/analysis/compare_abr_time_norm.py assets/data/dash_stats_lolp_fcc.csv assets/data/dash_stats_l2a_fcc.csv assets/data/dash_stats_customrule_fcc.csv assets/data/dash_stats_bola_fcc.csv 0.5`
  - 输出图片：`assets/images/abr_time_norm_summary.png`、`abr_time_norm_timeseries.png`、`abr_time_norm_quality_dist.png`

### 时间归一化分析脚本说明
- 脚本位置：`python/analysis/compare_abr_time_norm.py`
- 作用：对多种 ABR 策略的运行结果 CSV 进行时间对齐与统一采样，计算关键指标并生成汇总图、时序图与质量分布图。

- 输入与用法：
  - 基本调用：`python python/analysis/compare_abr_time_norm.py <lolp.csv> <l2a.csv> <custom.csv> <bola.csv> [dt]`
  - 参数含义：
    - `<lolp.csv> <l2a.csv> <custom.csv> <bola.csv>` 按顺序对应策略标签：`LoLp/L2A/CustomRule/Bola`（标签固定对应位置）
    - `[dt]` 为可选的统一采样间隔（秒），默认 `0.5`（`python/analysis/compare_abr_time_norm.py:409,418–420`）
  - 至少需要两个非空数据集，脚本会自动按最短可重叠时长对齐（不足时输出 `No aligned duration`，`python/analysis/compare_abr_time_norm.py:427–435`）。

- CSV 字段要求（与前端导出一致）：
  - `timestamp, bufferLevel, qualityIndex, bitrateKbps, resolution, liveLatency, predictedKbps, rebuffer`
  - 时间按 ISO 格式（含 `Z` 时区标志），脚本会解析与排序（`python/analysis/compare_abr_time_norm.py:41–60`）。

- 对齐与重采样：
  - 多数据集对齐至共同最短持续时间（`align_duration_multi`，`python/analysis/compare_abr_time_norm.py:388–402`）。
  - 统一时间栅格上进行“最后值保持”的重采样（`resample_rows`，`python/analysis/compare_abr_time_norm.py:84–103`）。

- 计算指标（时间归一化）：
  - 包含：
    - `duration_s, avg_bitrate, avg_buffer, avg_latency, p95_latency, rebuffer_fraction, rebuffer_seconds, rebuffer_events_per_min, quality_switches_per_min, upswitches_per_min, downswitches_per_min, avg_switch_magnitude, min_quality_time_fraction, pred_to_bitrate_ratio_mean, avg_recover_seconds, qoe_mean`（`python/analysis/compare_abr_time_norm.py:129–228`）
  - 额外说明：
    - 预测/码率均值比：当 `bitrateKbps>0` 且 `predictedKbps>0` 时才计入（`python/analysis/compare_abr_time_norm.py:172–177`）。
    - QoE 评分：`log(1+bitrateKbps) - 4*rebuffer - 0.2*liveLatency`（`python/analysis/compare_abr_time_norm.py:202–209`）。

- 输出与基线比较：
  - 控制台打印各策略的时间归一化指标（`print_time_metrics`，`python/analysis/compare_abr_time_norm.py:237–255`）。
  - 基线比较：默认以 `LoLp` 为基线，输出差异（`compare_time`，`python/analysis/compare_abr_time_norm.py:257–281`；基线选择逻辑见 `python/analysis/compare_abr_time_norm.py:446–454`）。
  - 生成图片：
    - 汇总条形图：`assets/images/abr_time_norm_summary.png`（`python/analysis/compare_abr_time_norm.py:455,288–314`）
    - 时序对比图：`assets/images/abr_time_norm_timeseries.png`（`python/analysis/compare_abr_time_norm.py:456,316–341`）
    - 质量分布图：`assets/images/abr_time_norm_quality_dist.png`（`python/analysis/compare_abr_time_norm.py:457,343–372`）

- 运行示例：
  - `python python/analysis/compare_abr_time_norm.py assets/data/dash_stats_lolp_fcc.csv assets/data/dash_stats_l2a_fcc.csv assets/data/dash_stats_customrule_fcc.csv assets/data/dash_stats_bola_fcc.csv 0.5`
  - 若只比较两种策略（例如 LoLp vs CustomRule）：
    - `python python/analysis/compare_abr_time_norm.py assets/data/dash_stats_lolp_fcc.csv assets/data/dash_stats_customrule_fcc.csv assets/data/dash_stats_customrule_fcc.csv assets/data/dash_stats_customrule_fcc.csv 0.5`
    - 按位置标签固定，未使用的文件可用某一策略 CSV 进行占位。

- 常见问题与提示：
  - `No aligned duration`：确保各 CSV 存在重叠时间段；导出应来自同一播放会话或相近时段。
  - `pred_to_bitrate_ratio_mean` 为 `NA`：可能缺少 `predictedKbps`，仅在启用自定义规则并接入 LSTM/EWMA 预测时有该字段。
  - 字体显示问题：脚本已设置中文字体回退（`matplotlib`），若仍异常，请检查本机字体与 `matplotlib` 配置。
- 预测方法对比（FCC 数据集）：
  - 基本评估：
    - `python python/analysis/compare_predictors_fcc.py --fcc_dir datasets/fcc --dt 0.5 --hist_len 30 --pred_horizon 1 --max_traces 200 --to_kbps --cp_threshold 1000 --step_ks 1,2,3`
  - 仅输出窗口拟合（示例 100–200s）：
    - `python python/analysis/compare_predictors_fcc.py --fcc_dir datasets/fcc --dt 0.5 --hist_len 30 --pred_horizon 1 --max_traces 200 --to_kbps --example_t0 100 --example_t1 200`
  - 可选参数：`--alpha`（EWMA）、`--arima_p --arima_d --arima_q`（ARIMA 阶数）、`--cfg_path --onnx_path`（LSTM 模型与配置）、`--example_trace_idx/--example_trace_path`（指定轨迹）、`--cp_threshold`（变点阈值，`--to_kbps` 时单位为 kbps/s）、`--step_ks`（评估的步数集合）
  - 输出图片：
    - `assets/images/predict_compare_summary_fcc.png`、`predict_compare_error_hist_fcc.png`、`predict_compare_scatter_fcc.png`、`predict_fit_example_fcc.png`
    - 新增步响应图：`assets/images/predict_step_response_fcc.png`（展示变点后 `k=1,2,3` 的 MAE/RMSE/Overshoot 对比）

- 预测方法对比（Oboe 数据集）：
  - 基本评估：
    - `python python/analysis/compare_predictors_oboe.py --oboe_dir datasets/oboe --dt 0.5 --hist_len 30 --pred_horizon 1 --max_traces 428 --to_kbps --cp_threshold 1000 --step_ks 1,2,3`
  - 仅输出窗口拟合（支持区间裁剪）：
    - `python python/analysis/compare_predictors_oboe.py --oboe_dir datasets/oboe --dt 0.5 --hist_len 30 --pred_horizon 1 --max_traces 428 --to_kbps --example_t0 100 --example_t1 200`
  - 可选参数同 FCC（目录与模型路径为 Oboe 对应项）
  - 输出图片：
    - `assets/images/predict_compare_summary_oboe.png`、`predict_compare_error_hist_oboe.png`、`predict_compare_scatter_oboe.png`、`predict_fit_example_oboe.png`
    - 新增步响应图：`assets/images/predict_step_response_oboe.png`

## 依赖
- Dash.js（CDN 引入）
- ONNX Runtime Web（CDN 引入）
- 现代浏览器（支持 MSE）
- 任意静态服务器用于本地开发（示例 `python -m http.server 8081`）
- Python 分析脚本依赖：`numpy matplotlib statsmodels onnxruntime`

## 训练与导出（可选）
- 安装依赖：`python -m pip install numpy torch onnx`
- 示例（Oboe 数据集）：`python python/train/train_lstm_oboe.py --max_traces 428 --epochs 15 --batch_size 128 --dt 0.5 --hist_len 30 --pred_horizon 1 --to_kbps`
- 输出：`assets/models/oboe_lstm.onnx`、`assets/models/oboe_lstm_cfg.json`

### 训练脚本与模型结构
- FCC 训练脚本：`python/train/train_lstm_fcc.py`
  - 模型结构：LSTM（含 `dropout`）+ `Dropout` + 全连接
  - 损失函数：`SmoothL1Loss`（Huber），参数可用 `--huber_beta` 调整
  - 常用参数：`--dropout --huber_beta --hist_len --hidden --layers --lr --epochs --batch_size --to_kbps`
  - 标签：`pred_horizon=1`（下一窗口均值）
- Oboe 训练脚本：`python/train/train_lstm_oboe.py`
  - 模型结构与损失已与 FCC 对齐：支持 `dropout` 与 `SmoothL1Loss`
  - 默认标签：`pred_horizon=1`
  - 新增参数：`--dropout --huber_beta`
 - 示例命令：
    - `python python/train/train_lstm_oboe.py --oboe_dir datasets/oboe --dt 0.5 --hist_len 30 --pred_horizon 1 --max_traces 428 --val_frac 0.1 --hidden 64 --layers 2 --lr 1e-3 --epochs 5 --batch_size 128 --to_kbps --dropout 0.2 --huber_beta 1.0`

### FCC 训练脚本命令说明
- 脚本位置：`python/train/train_lstm_fcc.py`
- 作用：基于 FCC 轨迹训练单步预测 LSTM，并导出 `PT/ONNX` 与归一化配置。

- 快速入门（推荐参数）：
  - 基本训练（含 ONNX 导出、单位统一为 kbps）：
    - `python python/train/train_lstm_fcc.py --fcc_dir datasets/fcc --dt 0.5 --hist_len 30 --pred_horizon 1 --max_traces 200 --val_frac 0.1 --hidden 64 --layers 2 --lr 1e-3 --epochs 5 --batch_size 128 --to_kbps --dropout 0.2 --huber_beta 1.0`
  - 只查看数据切分与形状（不训练）：
    - `python python/train/train_lstm_fcc.py --fcc_dir datasets/fcc --dt 0.5 --hist_len 30 --pred_horizon 1 --max_traces 50 --val_frac 0.1 --dry_run`
  - 增加训练规模与轮数（更好拟合）：
    - `python python/train/train_lstm_fcc.py --fcc_dir datasets/fcc --max_traces 428 --epochs 15 --batch_size 128 --dt 0.5 --hist_len 30 --pred_horizon 1 --to_kbps --dropout 0.2 --huber_beta 1.0`

- 输出文件：
  - `assets/models/fcc_lstm.pt`：权重与元信息（`hist_len, hidden, layers, xm, xs, ym, ys`）
  - `assets/models/fcc_lstm_cfg.json`：前端推理用归一化配置（同上字段）
  - `assets/models/fcc_lstm.onnx`：ONNX 推理模型，输入 `hist` 形状为 `[1, hist_len]`，输出 `pred`

- 数据目录与格式：
  - 期望目录：`datasets/fcc/`
  - 文件匹配：`test_fcc_trace_*`（每行形如 `dt value`；若源单位为 Mbps，使用 `--to_kbps` 转为 kbps）
  - 采样步长：`--dt` 控制窗口平均的采样间隔秒数；序列不足时会跳过该轨迹。

- 全部参数（含默认值）：
  - `--fcc_dir` 默认 `datasets/fcc`，FCC 轨迹目录（`python/train/train_lstm_fcc.py:199`）
  - `--dt` 默认 `0.5`，统一序列的采样间隔秒（`python/train/train_lstm_fcc.py:200`）
  - `--hist_len` 默认 `30`，历史窗口长度（`python/train/train_lstm_fcc.py:201`）
  - `--pred_horizon` 默认 `1`，标签为后续 `horizon` 步均值（`python/train/train_lstm_fcc.py:202`）
  - `--max_traces` 默认 `200`，最大读取轨迹数（`python/train/train_lstm_fcc.py:203`）
  - `--val_frac` 默认 `0.1`，验证集比例（`python/train/train_lstm_fcc.py:204`）
  - `--hidden` 默认 `64`，LSTM 隐层维度（`python/train/train_lstm_fcc.py:205`）
  - `--layers` 默认 `2`，LSTM 层数（`python/train/train_lstm_fcc.py:206`）
  - `--lr` 默认 `1e-3`，学习率（`python/train/train_lstm_fcc.py:207`）
  - `--epochs` 默认 `5`，训练轮数（`python/train/train_lstm_fcc.py:208`）
  - `--batch_size` 默认 `128`，批大小（`python/train/train_lstm_fcc.py:209`）
  - `--save_path` 默认 `assets/models/fcc_lstm.pt`（`python/train/train_lstm_fcc.py:210`）
  - `--cfg_path` 默认 `assets/models/fcc_lstm_cfg.json`（`python/train/train_lstm_fcc.py:211`）
  - `--onnx_path` 默认 `assets/models/fcc_lstm.onnx`（`python/train/train_lstm_fcc.py:212`）
  - `--dry_run` 无参数值，启用后只做数据与切分检查（`python/train/train_lstm_fcc.py:213`）
  - `--to_kbps` 无参数值，启用后将值乘以 1000 转为 kbps（`python/train/train_lstm_fcc.py:214`）
  - `--dropout` 默认 `0.2`，LSTM 层间与末端 `Dropout` 比例（`python/train/train_lstm_fcc.py:215`）
  - `--huber_beta` 默认 `1.0`，`SmoothL1Loss` 的 beta（`python/train/train_lstm_fcc.py:216`）

- 日志与诊断：
  - 数据集形状与切分：训练前打印 `dataset_shapes` 与 `train_val_shapes`（`python/train/train_lstm_fcc.py:222–224`）
  - 每轮输出训练/验证损失与耗时：`epoch=... train_loss=... val_loss=... time_s=...`（`python/train/train_lstm_fcc.py:158`）
  - 数据集为空时打印 `dataset_empty`（`python/train/train_lstm_fcc.py:219–221`）
  - ONNX 导出失败时打印 `onnx_export_failed ...`，可检查 `opset_version=17` 或环境（`python/train/train_lstm_fcc.py:193–195`）

- 使用建议：
  - 与前端一致的单位：启用 `--to_kbps` 以匹配 `kbps` 单位，避免与 ABR 规则中的吞吐单位不一致。
  - 训练规模与稳健性：增大 `--max_traces --epochs` 提升泛化；通过 `--dropout` 与 `--huber_beta` 抗过拟合与异常值。
  - 推理一致性：前端加载 ONNX 时需同时加载对应 `fcc_lstm_cfg.json` 的归一化参数（`xm xs ym ys`）。

## 常见问题
- 不能播放或跨域错误：确保 MPD 与分片服务端开启 CORS。
- Stats 未显示或 CSV 为空：确认分片完成与缓冲事件是否触发；源无数据时不会采样。
- ONNX 加载失败：检查 `assets/models/fcc_lstm.onnx` 与 `assets/models/fcc_lstm_cfg.json` 是否存在，静态服务器根目录是否为项目根。
