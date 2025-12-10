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
    - `python python/analysis/compare_predictors_oboe.py --oboe_dir datasets/oboe --dt 0.5 --hist_len 30 --pred_horizon 1 --max_traces 200 --to_kbps --cp_threshold 1000 --step_ks 1,2,3`
  - 仅输出窗口拟合（支持区间裁剪）：
    - `python python/analysis/compare_predictors_oboe.py --oboe_dir datasets/oboe --dt 0.5 --hist_len 30 --pred_horizon 1 --max_traces 200 --to_kbps --example_t0 100 --example_t1 200`
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
- 示例（Oboe 数据集）：`python python/train/train_lstm_oboe.py --max_traces 400 --epochs 15 --batch_size 128 --dt 0.5 --hist_len 30 --pred_horizon 1 --to_kbps`
- 输出：`assets/models/oboe_lstm.onnx`、`assets/models/oboe_lstm_cfg.json`

## 常见问题
- 不能播放或跨域错误：确保 MPD 与分片服务端开启 CORS。
- Stats 未显示或 CSV 为空：确认分片完成与缓冲事件是否触发；源无数据时不会采样。
- ONNX 加载失败：检查 `assets/models/fcc_lstm.onnx` 与 `assets/models/fcc_lstm_cfg.json` 是否存在，静态服务器根目录是否为项目根。
