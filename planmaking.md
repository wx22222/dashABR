**目标与指标**
- 主要目标：在不显著降低画质的前提下，稳定把直播端到端延时压到目标范围，并显著降低重缓冲率。
- 成功指标（上线验收用）：
  - 95 分位直播延时 ≤ 2.5s，均值/中位数接近 2.0s 目标
  - 重缓冲率 ≤ 0.5%，每小时卡顿次数 ≤ 1 次
  - 平均码率较默认 ABR 降幅 ≤ 10%，质量切换频率 ≤ 3 次/分钟
  - 预测误差（MAPE）≤ 20%，预测覆盖率（预算满足率）≥ 85%

**总体架构**
- 预测层：分片级一步带宽预测，提供均值与不确定性（保守分位值）
- 决策层：基于“下载时间预算”的风险约束筛选候选表示，QoE 最大化选择
- 控制层：低延时控制（live catch-up、延时门控）、紧急状态降级、平滑与跃迁限制
- 集成层：在 `CustomAbrRule.js` 实现规则，通过 `main.js` 注册并与 Dash.js 运行时交互
- 监控层：实时采样与日志，CSV 导出对比分析

**里程碑与时间线**
- M1（第 1 周）：建立离线仿真与快速基线预测（EWMA），实现风险约束 ABR，完成 A/B 实验框架
- M2（第 2 周）：训练并集成 LSTM 一步预测，加入不确定性与保守分位，完善 QoE 与延时控制
- M3（第 3 周）：线上灰度与调参，性能对比与验收，固化回退与告警

**实施步骤**

- 数据与仿真基线
  - 解析 `dataset/oboe/trace_*.txt` 为分片级吞吐量序列，统一单位为 `kbps`
  - 构建离线仿真器：给定序列、分片大小、ABR 策略，输出延时/缓冲/码率/切换等指标
  - 指标与可视化：与默认 ABR（LoLP+BOLA/Throughput）对比，生成报表

- 带宽预测模块
  - 快速基线（优先落地）：EWMA/EMA+调和平均/截尾平均；输出点预测与滑窗方差
  - LSTM 一步预测：输入差分或归一化差分，序列长度 L=10–20，输出下一分片吞吐量
  - 不确定性与保守化：维护残差分布，采用保守分位（如 15% 分位值）或动态安全系数（稳定 0.8，不稳定 0.6–0.7）
  - TFJS 集成：Keras→TFJS 转换，静态服务 `http://localhost:8080/model.json`；加载失败或样本不足时回退 EWMA

- 风险约束与可行性筛选
  - 预算定义：`T_budget ≈ buffer_level + (target_latency − current_latency)`
  - 下载时长估算：`T_download ≈ segment_size_kbits / predicted_throughput_kbps`
  - 可行集合：筛选满足 `T_download ≤ margin × T_budget`（margin=0.8–0.9）的表示
  - 空集合处理：强降级（最低或次低），进入紧急态

- QoE 模型设计
  - QoE：`QoE = α·log(b) − β·rebuffer − γ·|Δb| − δ·latency`，权重满足 `β > δ > γ > α`
  - 在可行集合中最大化 QoE；高方差时增加平滑惩罚，限制升级
  - 平滑与跃迁：正常态最大跃迁 ±1；紧急态允许更大幅度降级

- 低延时控制策略
  - 目标延时：2.0s；高于目标时保守选码并启用 catch-up，低于目标时适度放宽
  - `liveCatchup.playbackRate.max` 建议调整到 `0.15–0.20`（现代码为 0.3）
  - 中途放弃策略：启用/复用 Dash.js `abandonRequestsRule`，分片下载过慢时弃单重试低码率

- ABR 规则接口与集成
  - 在 `CustomAbrRule.js` 实现 `getSwitchRequest`，使用：
    - `rulesContext.getThroughputController().getRawThroughputData('video')`
    - `abrController.getPossibleVoRepresentationsFilteredBySettings(mediaInfo, true)`
    - `scheduleController.getPlaybackController().getCurrentLiveLatency()`
  - 模型加载与回退：初始化时尝试加载 TFJS；失败或高方差时退回 EWMA
  - 事件驱动：与 `FRAGMENT_LOADING_COMPLETED` 对齐进行预测/决策，避免中途震荡
  - 与现有设置协调：
    - 初始阶段保留/启用 `abandonRequestsRule` 与 `loLPRule` 的关键逻辑，避免关闭所有默认保护
    - 当前 `main.js:33–41` 关闭了多种默认规则，建议阶段性只关闭与选择冲突的规则

- 监控与日志
  - 运行时指标：`bufferLevel, bitrateKbps, qualityIndex, liveLatency, rebuffer`（已在 `main.js:108–115`）
  - 新增字段：预测吞吐量、保守分位值、预算、选码 QoE 分数、网络方差与稳定性标记
  - CSV 导出与对比：保留现有导出（`main.js:153–165`），对比实验时标注策略与参数版本

- 实验设计与评估
  - 离线：轨迹分层抽样，训练/验证/测试拆分；关注预算满足率与 QoE 提升
  - 在线：A/B 测试（默认 ABR vs 新 ABR），统计延时/重缓冲/码率/切换；高方差场景重点观察
  - 调参：安全系数、margin、权重 αβγδ、最大跃迁、缓冲目标、自适应门控阈值

- 上线与回滚策略
  - 灰度：逐步扩大流量占比，实时告警（延时暴涨、卡顿率飙升时自动回退默认 ABR）
  - 回退路径：模型加载失败/不稳定→EWMA；风险约束过严导致画质过低→调整 margin 与权重；异常时恢复默认 ABR

**资源与依赖**
- 浏览器端：Dash.js（CDN），TFJS（模型加载），静态服务器用于 TFJS 模型与配置
- Windows 环境准备（示例命令）：
  - 启动静态服务：```powershell\npython -m http.server 8080\n```
  - TFJS 转换工具安装：```powershell\npip install tensorflowjs\n```
  - 模型转换：```powershell\ntensorflowjs_converter --input_format keras C:\\model\\lstm.h5 C:\\model\\tfjs\\\n```

**风险与规避**
- 预测漂移：采用保守分位与动态安全系数，方差高时禁止升级、提高缓冲目标
- 模型负载：控制 LSTM 规模与输入长度，异步预加载；预测失败立即回退
- 规则冲突：保留 `abandonRequestsRule` 与 LoL+ 的关键保护逻辑，避免全部关闭默认规则
- 单位一致性：统一吞吐量单位为 kbps、时间为秒；训练与推理一致的 scaler

**验收标准**
- 离线仿真：在代表性轨迹上达到指标目标（延时、重缓冲、预算满足率）
- 线上 A/B：在实际流上稳定达到或接近目标，且用户体验（码率与切换）不劣于默认 ABR
- 稳定性：异常情况下自动回退，日志与监控完整，可复现与审计

如果你确认这份计划，我可以在下一步给出“实现清单”与每个任务的具体产出物模板（例如预测模块接口规范、ABR 决策伪代码、参数表、实验报表字段等），让你按项执行并快速迭代。