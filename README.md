# Dash.js 自定义播放器

## 项目概览
- 基于 Dash.js 的前端播放器，支持直播/点播 MPD 播放、可自定义流地址、接入自定义 ABR 规则。
- 提供实时指标面板（Stats）：缓冲、码率、分辨率、直播延时、重缓冲提示，并支持导出为 CSV。
- 采样采用“事件驱动”模式：每当一个视频分片下载完成时采样更新，避免时间采样的遗漏。

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
   - `python3 -m http.server 8081`
2. 浏览器访问：
   - `http://localhost:8081/`
3. 在输入框填入 MPD 地址，点击 `Load Stream` 开始播放。
4. 若需启用自定义 ABR，点击 `Enable Custom ABR Rule` 按钮。
5. 在播放过程中，可点击 `Export CSV` 将当前采样数据下载为 `.csv` 并导入 Excel。

## 指标与采样
- 采样触发：
  - 每个视频分片下载完成时（`FRAGMENT_LOADING_COMPLETED`）进行一次采样（`main.js:35–41`）。
  - 发生缓冲耗尽或浏览器等待事件时，也会立即采样（`BUFFER_EMPTY`、`PLAYBACK_WAITING`，`main.js:28–41`）。
- 直播延时计算：
  - 使用 `player.getDvrWindow().end - player.time()` 估算直播延时（`main.js:104–115`）。
- 重缓冲提示：
  - 当缓冲耗尽或进入等待，Stats 显示 `rebuffer happen`（`main.js:28–41`, `main.js:120–127`）。
- CSV 字段：
  - `timestamp, bufferLevel, qualityIndex, bitrateKbps, resolution, liveLatency, rebuffer`（`main.js:160–167`）。

## 自定义 ABR
- 默认 ABR：Dash.js 动态模式（吞吐量规则 + BOLA），由 `useDefaultABRRules: true` 启用（`main.js:197–202`）。
- 自定义规则：在 `CustomAbrRule.js` 定义并通过 `player.addABRCustomRule('qualitySwitchRules', 'CustomAbrRule', CustomAbrRule)` 注册（`main.js:175–182`）。
- 你可以在 `CustomAbrRule.js:20–66` 的 `checkIndex` 中实现自己的码率选择逻辑。

## 代码入口与关键位置
- 页面与样式：
  - `index.html`（结构与按钮，`index.html:15–30`）
  - `style.css`（基础样式）
- 播放器逻辑：
  - 初始化与事件绑定：`main.js:15–42`
  - 加载 MPD：`main.js:45–53`
  - 单次采样与 UI 更新：`main.js:94–131`
  - 分片完成事件采样：`main.js:35–41`
  - CSV 导出：`main.js:152–172`
  - 自定义 ABR 注入：`main.js:175–205`

## 常见问题
- 不能播放或跨域错误：确保 MPD 与分片服务端开启 CORS（至少 `Access-Control-Allow-Origin: *` 或允许你的域）。
- Stats 未显示或 CSV 为空：确认事件已触发（分片完成、缓冲变化）；若源无数据或中断，将不会有新的采样。
- 直播播放“结束”后仍更新：直播通常不会触发 `PLAYBACK_ENDED`；如需在停止流或错误时停止 Stats，可在逻辑中加入播放状态门控（例如根据 `MANAGED_MEDIA_SOURCE_END_STREAMING`、`PLAYBACK_ERROR` 关闭更新）。

## 依赖
- Dash.js（CDN 引入，`index.html:9`）
- 现代浏览器（支持 MSE）
- 任意静态服务器用于本地开发（示例 `python3 -m http.server`）

