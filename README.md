# Dash.js 自定义播放器（工程化结构）

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
  - `assets/models/oboe_lstm.onnx`
  - `assets/models/oboe_lstm_cfg.json`
- `assets/data/` 指标 CSV（用于分析脚本）
- `assets/images/` 分析输出图片（汇总、时序、质量分布）
- `datasets/` 训练/评测数据集
  - `datasets/oboe/` Oboe 轨迹
  - `datasets/fcc/` FCC 轨迹
- `python/train/` 训练脚本
  - `python/train/train_lstm_oboe.py`
- `python/analysis/` 分析脚本
  - `python/analysis/compare_abr.py`
  - `python/analysis/compare_abr_time_norm.py`

## 功能
- 自定义 MPD URL 输入并手动加载，不自动播放（`index.html:15–18`，`main.js:45–53`）。
- 注册自定义 ABR 规则，和默认 ABR 并行工作（`main.js:175–205`，`CustomAbrRule.js:7–69`）。
- Stats 指标展示（`main.js:120–127`）：
  - Buffer Level（秒）
  - Current Quality Index（当前质量索引）
  - Bitrate（kbps）
  - Resolution（如 1920x1080）
  - Live Latency（直播延时，秒）
  - Rebuffer 提示（发生卡顿时显示“rebuffer happen”）
- 导出 CSV（含时间戳与上述指标），一键下载（`index.html:24–30`，`main.js:152–172`）。

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
  - 每个视频分片下载完成时（`FRAGMENT_LOADING_COMPLETED`）进行一次采样（`main.js:35–41`）。
  - 发生缓冲耗尽或浏览器等待事件时，也会立即采样（`BUFFER_EMPTY`、`PLAYBACK_WAITING`，`main.js:28–41`）。
- 直播延时计算：
  - 使用 `player.getDvrWindow().end - player.time()` 估算直播延时（`main.js:104–115`）。
- 重缓冲提示：
  - 当缓冲耗尽或进入等待，Stats 显示 `rebuffer happen`（`main.js:28–41`, `main.js:120–127`）。
- CSV 字段：
  - `timestamp, bufferLevel, qualityIndex, bitrateKbps, resolution, liveLatency, predictedKbps, rebuffer`（`scripts/main.js:156–168`）。

## 自定义 ABR
- 自定义规则在 `scripts/CustomAbrRule.js` 中定义，并通过 `player.addABRCustomRule('qualitySwitchRules', 'CustomAbrRule', CustomAbrRule)` 注册（`scripts/main.js:175–188`）。
- LSTM 预测集成：
  - 模型文件：`assets/models/oboe_lstm.onnx`
  - 配置文件：`assets/models/oboe_lstm_cfg.json`（含 `hist_len, xm, xs, ym, ys`）
  - 规则侧聚合：固定时间步 `dt=0.5s`，单位为 Kbps，历史长度默认 `hist_len=30`，预测未来 `1` 步（0.5s）的均值作为吞吐预测（`scripts/CustomAbrRule.js:155–193`）。
  - 若模型或配置缺失，规则会自动降级为 EWMA 吞吐估计。

## 代码入口与关键位置
- 页面与样式：
  - `index.html`（结构与按钮，`index.html:15–30`）
  - `styles/style.css`（基础样式）
- 播放器逻辑：
  - 初始化与事件绑定：`scripts/main.js:15–69`
  - 加载 MPD：`scripts/main.js:71–82`
  - 单次采样与 UI 更新：`scripts/main.js:91–129`
  - 分片完成事件采样：`scripts/main.js:64–68`
  - CSV 导出：`scripts/main.js:149–169`
  - 自定义 ABR 注入：`scripts/main.js:171–208`

## 常见问题
- 不能播放或跨域错误：确保 MPD 与分片服务端开启 CORS（至少 `Access-Control-Allow-Origin: *` 或允许你的域）。
- Stats 未显示或 CSV 为空：确认事件已触发（分片完成、缓冲变化）；若源无数据或中断，将不会有新的采样。
- ONNX 404：请确认 `assets/models/oboe_lstm.onnx` 与 `assets/models/oboe_lstm_cfg.json` 已存在，并且静态服务器根目录为项目根（`index.html` 所在目录）。
- 直播播放“结束”后仍更新：直播通常不会触发 `PLAYBACK_ENDED`；如需在停止流或错误时停止 Stats，可在逻辑中加入播放状态门控（例如根据 `MANAGED_MEDIA_SOURCE_END_STREAMING`、`PLAYBACK_ERROR` 关闭更新）。

## 依赖
- Dash.js（CDN 引入，`index.html:9`）
- ONNX Runtime Web（CDN 引入，`index.html:10`）
- 现代浏览器（支持 MSE）
- 任意静态服务器用于本地开发（示例 `python -m http.server 8081`）

## 训练与导出（可选）
- 安装依赖：`python -m pip install numpy torch onnx`
- 训练并导出模型：
  - `python python/train/train_lstm_oboe.py --max_traces 400 --epochs 15 --batch_size 128 --dt 0.5 --hist_len 30 --pred_horizon 1 --to_kbps`
- 输出：
  - 模型：`assets/models/oboe_lstm.onnx`
  - 配置：`assets/models/oboe_lstm_cfg.json`

## 指标分析（可选）
- 将导出的 CSV 放入 `assets/data/`，或以参数形式传给脚本。
- 直接运行（默认读取 `assets/data/dash_stat_LoLp_oboe.csv` 与 `assets/data/dash_stats_customrule_oboe.csv`）：
  - `python python/analysis/compare_abr.py`
  - `python python/analysis/compare_abr_time_norm.py`
- 自定义输入：
  - `python python/analysis/compare_abr.py <LoLp.csv> <CustomRule.csv>`
  - `python python/analysis/compare_abr_time_norm.py <LoLp.csv> <CustomRule.csv> [dt]`
