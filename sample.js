/**
 * The copyright in this software is being made available under the BSD License,
 * included below. This software may be subject to other third party and contributor
 * rights, including patent rights, and no such rights are granted under this license.
 *
 * Copyright (c) 2013, Dash Industry Forum.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *  * Redistributions of source code must retain the above copyright notice, this
 *  list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation and/or
 *  other materials provided with the distribution.
 *  * Neither the name of Dash Industry Forum nor the names of its
 *  contributors may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS AND ANY
 *  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 *  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 *  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 *  NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 *  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */
import FactoryMaker from '../../../core/FactoryMaker.js';
import SwitchRequest from '../SwitchRequest.js';
//import MetricsConstants from '../../constants/MetricsConstants.js';
import * as tf from '@tensorflow/tfjs';
import Debug from '../../../core/Debug.js';
// 简化状态机：只保留两个基本状态
const LIVE_STATE_NORMAL = 'LIVE_STATE_NORMAL';       // 正常状态
const LIVE_STATE_EMERGENCY = 'LIVE_STATE_EMERGENCY'; // 紧急状态

// 直播场景特定常量
const TARGET_LATENCY = 2.0;           // 目标延迟（秒）
const MAX_ACCEPTABLE_LATENCY = 4.0;   // 最大可接受延迟（秒）
const MIN_BUFFER_TARGET = 1.5;        // 最小缓冲区目标（秒）
const MAX_BUFFER_TARGET = 4.0;        // 最大缓冲区目标（秒）
const EMERGENCY_BUFFER_THRESHOLD = 0.5; // 紧急缓冲区阈值（秒）

// 网络评估参数
const THROUGHPUT_SAFETY_FACTOR = 0.8; // 吞吐量安全系数
const LATENCY_WEIGHT = 0.2;           // 延迟权重
const BUFFER_WEIGHT = 0.6;            // 缓冲区权重
const THROUGHPUT_WEIGHT = 0.2;        // 吞吐量权重

// 质量切换参数
const MAX_QUALITY_JUMP = 1;           // 最大质量跳跃级别
const DOWNGRADE_THRESHOLD = 0.9;      // 降级阈值
const UPGRADE_THRESHOLD = 1.5;        // 升级阈值

// 网络波动检测
const NETWORK_VARIANCE_WINDOW = 5;    // 网络方差检测窗口
const HIGH_VARIANCE_THRESHOLD = 0.3;  // 高方差阈值





function ThroughputRule(config) {

    config = config || {};
    const context = this.context;
    const dashMetrics = config.dashMetrics;

    // 默认配置
    const defaultConfig = {
        modelBaseUrl: 'http://127.0.0.1:8080',
        enableTensorFlow: true,
        enableLogging: true,
        safetyFactor: THROUGHPUT_SAFETY_FACTOR,
        maxQualityJump: MAX_QUALITY_JUMP
    };
    
    // 合并配置
    config = Object.assign({}, defaultConfig, config);

    let instance,
        model,
        liveState,
        bufferHistory,
        logger,
        lastSwitchTime,
        networkVariance,
        adaptiveBufferTarget,
        scalerInfo,
        stateInfo;

    function setup() {
            logger = Debug(context).getInstance().getLogger(instance);
            reset();
            // 设置ensorFlow后端
                tf.setBackend('cpu');
            // 加载模型并优化内存管理
            loadModel()
            .then(loadedModel => {
                // 如果已有模型，先释放旧模型
                if (model) {
                    model.dispose();
                }
                model = loadedModel;
            })
    }


    

    function reset() {
        try {
            liveState = {
                state: LIVE_STATE_NORMAL,
                currentRepresentation: null,
                lastQualityIndex: 0,
                consecutiveDowngrades: 0,
                emergencyModeStartTime: null
            };
            
            bufferHistory = [];
            lastSwitchTime = 0;
            networkVariance = 0;
            adaptiveBufferTarget = MIN_BUFFER_TARGET;
            
            // 清理模型资源
            if (model) {
                try {
                    if (typeof model.dispose === 'function') {
                        model.dispose();
                    }
                } catch (disposeError) {
                    if (logger) {
                        logger.warn('[ThroughputRule] 释放模型资源失败:', disposeError);
                    } else {
                        console.warn('[ThroughputRule] 释放模型资源失败:', disposeError);
                    }
                } finally {
                    model = null;
                }
            }
            
            // 清理TensorFlow内存
            try {
                if (typeof tf !== 'undefined' && tf.disposeVariables) {
                    tf.disposeVariables();
                }
            } catch (tfError) {
                if (logger) {
                    logger.warn('[ThroughputRule] 清理TensorFlow变量失败:', tfError);
                }
            }
            
            logger.debug('[ThroughputRule] 状态已重置');
        } catch (error) {
            const errorMessage = '[ThroughputRule] 重置过程出错: ' + (error.message || error);
            if (logger) {
                logger.error(errorMessage);
            } else {
                console.error(errorMessage);
            }
        }
    }
    function loadModel() {
        return new Promise(async (resolve, reject) => {
            try {
                const base = config.modelBaseUrl || 'http://127.0.0.1:8080';
                const modelUrl = base.endsWith('/model.json') ? base : base + '/model.json';
                const loadedModel = await tf.loadLayersModel(modelUrl);
                const scalerResp = await fetch(base + '/scaler.json');
                const stateResp = await fetch(base + '/state.json');
                const sc = await scalerResp.json();
                const st = await stateResp.json();
                scalerInfo = { mean: sc.mean ?? 0, scale: sc.scale ?? 1 };
                stateInfo = st;
                if (!loadedModel || typeof loadedModel.predict !== 'function') {
                    throw new Error('加载的模型无效或不完整');
                }
                if (logger) {
                    logger.info('ResidualLSTM 模型与配置加载成功');
                }
                resolve(loadedModel);
            } catch (error) {
                const msg = `ResidualLSTM 加载失败: ${error.message || error}`;
                if (logger) logger.error(msg);
                reject(new Error(msg));
            }
        });
    }
    
    /**
     * 计算网络稳定性指标
     */
    function calculateNetworkStability(throughputController, mediaType) {
        try {
            const rawThroughputData = throughputController.getRawThroughputData(mediaType);
            
            if (!rawThroughputData || !Array.isArray(rawThroughputData)) {
                if (logger) {
                    logger.warn('[ThroughputRule] 网络稳定性计算: 无效的吞吐量数据');
                }
                return { variance: 1, isStable: false, mean: 0 };
            }
            
            if (rawThroughputData.length < 3) {
                return { variance: 0, isStable: false, mean: 0 };
            }

            const recentThroughput = rawThroughputData.slice(-NETWORK_VARIANCE_WINDOW)
                .map(item => {
                    if (!item || typeof item.value !== 'number') {
                        return null;
                    }
                    return item.value;
                })
                .filter(val => val !== null && val > 0 && !isNaN(val)); // 过滤无效值
            
            if (recentThroughput.length === 0) {
                return { variance: 1, isStable: false, mean: 0 };
            }
            
            const mean = recentThroughput.reduce((a, b) => a + b, 0) / recentThroughput.length;
            
            // 防止除零错误和异常值
            if (mean === 0 || mean < 0.001 || !isFinite(mean)) {
                return { variance: 1, isStable: false, mean: 0 };
            }
            
            const variance = recentThroughput.reduce((acc, val) => {
                const diff = val - mean;
                return acc + (diff * diff);
            }, 0) / recentThroughput.length;
            
            // 确保方差计算结果有效
            if (!isFinite(variance) || variance < 0) {
                return { variance: 1, isStable: false, mean: mean };
            }
            
            const normalizedVariance = Math.min(1.0, variance / (mean * mean)); // 限制最大值
            
            return {
                variance: normalizedVariance,
                isStable: normalizedVariance < HIGH_VARIANCE_THRESHOLD,
                mean: mean
            };
        } catch (error) {
            if (logger) {
                logger.warn('[ThroughputRule] 计算网络稳定性失败:', error);
            } else {
                console.warn('[ThroughputRule] 计算网络稳定性失败:', error);
            }
            return { variance: 1, isStable: false, mean: 0 };
        }
    }

    /**
     * 动态调整缓冲区目标
     */
    function updateAdaptiveBufferTarget(networkStability, currentLatency) {
        let targetAdjustment = 0;
        
        // 基于网络稳定性调整
        if (!networkStability.isStable) {
            targetAdjustment += 0.5; // 网络不稳定时增加缓冲
        }
        
        // 基于延迟调整
        if (currentLatency > TARGET_LATENCY) {
            targetAdjustment -= 0.3; // 延迟过高时减少缓冲
        }
        
        adaptiveBufferTarget = Math.max(
            MIN_BUFFER_TARGET,
            Math.min(MAX_BUFFER_TARGET, MIN_BUFFER_TARGET + targetAdjustment)
        );
        
        return adaptiveBufferTarget;
    }

    /**
     * 计算质量评分
     */
    function calculateQualityScore(representation, throughput, latency, bufferLevel, networkStability) {
        const bitrate = representation.bitrateInKbit || 0;
        
        // 边界条件检查
        const safeThroughput = Math.max(0, throughput || 0) * THROUGHPUT_SAFETY_FACTOR;
        const safeLatency = Math.max(0, latency || 0);
        const safeBufferLevel = Math.max(0, bufferLevel || 0);
        const safeAdaptiveTarget = Math.max(0.1, adaptiveBufferTarget);
        
        // 吞吐量评分（考虑安全系数）
        let throughputScore = 0;
        if (bitrate > 0 && safeThroughput > 0) {
            throughputScore = Math.min(1.0, safeThroughput / bitrate);
            
            // 应用升降级阈值 - 更保守的评分调整
            if (throughputScore > UPGRADE_THRESHOLD) {
                throughputScore = Math.min(1.0, throughputScore * 1.02); // 从1.1降至1.02，减少升级激励
            } else if (throughputScore < DOWNGRADE_THRESHOLD) {
                throughputScore = throughputScore * 0.80; // 从0.9降至0.80，增加降级激励
            }
        }
        
        // 延迟评分（延迟越低分数越高）
        let latencyScore = 1.0;
        if (safeLatency > 0) {
            const latencyRange = MAX_ACCEPTABLE_LATENCY - TARGET_LATENCY;
            if (latencyRange > 0) {
                latencyScore = Math.max(0, 1.0 - (safeLatency - TARGET_LATENCY) / latencyRange);
            }
        }
        
        // 缓冲区评分
        const bufferScore = Math.min(1.0, safeBufferLevel / safeAdaptiveTarget);
        
        // 网络稳定性调整
        const stabilityMultiplier = networkStability.isStable ? 1.0 : 0.8;
        
        // 综合评分
        const totalScore = Math.max(0, Math.min(1.0, (
            throughputScore * THROUGHPUT_WEIGHT +
            latencyScore * LATENCY_WEIGHT +
            bufferScore * BUFFER_WEIGHT
        ) * stabilityMultiplier));
        
        return {
            totalScore,
            throughputScore,
            latencyScore,
            bufferScore,
            stabilityMultiplier
        };
    }

    /**
     * 选择最优质量级别
     */
    function selectOptimalQuality(possibleRepresentations, throughput, latency, bufferLevel, networkStability) {
        let bestRepresentation = possibleRepresentations[0];
        let bestScore = -1;
        let scores = [];
        
        for (let i = 0; i < possibleRepresentations.length; i++) {
            const representation = possibleRepresentations[i];
            const score = calculateQualityScore(representation, throughput, latency, bufferLevel, networkStability);
            
            scores.push({
                index: i,
                representation,
                score: score.totalScore,
                details: score
            });
            // console.log('score',score.totalScore);
            // console.log('representation',representation);
            if (score.totalScore >= bestScore) {
                bestScore = score.totalScore;
                bestRepresentation = representation;
            }
        }
        
        // 智能质量跳跃限制和平滑过渡策略
        const currentIndex = liveState.currentRepresentation ? 
            possibleRepresentations.findIndex(rep => rep.id === liveState.currentRepresentation.id) : 0;
        
        const bestIndex = possibleRepresentations.findIndex(rep => rep.id === bestRepresentation.id);
        const qualityJump = Math.abs(bestIndex - currentIndex);
        
        if (qualityJump > MAX_QUALITY_JUMP) {
            const direction = bestIndex > currentIndex ? 1 : -1;
            
            // 动态调整最大跳跃限制
            let maxJump = MAX_QUALITY_JUMP;
            
            // 在紧急状态下允许更大的降级跳跃
            if (liveState.state === LIVE_STATE_EMERGENCY && bestIndex < currentIndex) {
                maxJump = Math.max(MAX_QUALITY_JUMP, Math.floor(possibleRepresentations.length * 0.5));
            }
            
            // 网络不稳定时限制升级跳跃
            if (!networkStability.isStable && bestIndex > currentIndex) {
                maxJump = Math.min(MAX_QUALITY_JUMP, 1);
            }
            
            // 缓冲区充足时允许更大的升级跳跃
            if (bufferLevel > adaptiveBufferTarget * 1.5 && bestIndex > currentIndex) {
                maxJump = Math.min(possibleRepresentations.length - 1, MAX_QUALITY_JUMP + 1);
            }
            
            let limitedIndex = Math.max(0, Math.min(
                possibleRepresentations.length - 1,
                currentIndex + direction * maxJump
            ));
            
            // 在限制范围内选择最佳质量（平滑过渡策略）
            let limitedBestScore = -1;
            let limitedBestRepresentation = possibleRepresentations[limitedIndex];
            
            const searchStart = Math.min(currentIndex, limitedIndex);
            const searchEnd = Math.max(currentIndex, limitedIndex);
            
            for (let i = searchStart; i <= searchEnd; i++) {
                const score = scores[i];
                if (score && score.score > limitedBestScore) {
                    limitedBestScore = score.score;
                    limitedBestRepresentation = score.representation;
                    limitedIndex = i;
                }
            }
            
            bestRepresentation = limitedBestRepresentation;
            
            logger.debug(`[ThroughputRule] 智能质量跳跃限制: ${currentIndex} -> ${bestIndex} 限制为 ${limitedIndex}, 最大跳跃: ${maxJump}`);
        }
        
        return {
            representation: bestRepresentation,
            scores,
            networkStability
        };
    }

    /**
     * 简化的紧急状态处理
     */
    function handleEmergencyState(possibleRepresentations, bufferLevel, currentRepresentation) {
        //logger.warn(`[ThroughputRule] 进入紧急模式，缓冲区: ${bufferLevel}s`);
        
        liveState.emergencyModeStartTime = Date.now();
        liveState.consecutiveDowngrades++;
        
        // 简化降级策略：直接选择较低质量
        const currentIndex = currentRepresentation ? 
            possibleRepresentations.findIndex(rep => rep.id === currentRepresentation.id) : 0;
        
        // 根据缓冲区严重程度选择降级幅度
        let targetIndex;
        if (bufferLevel < EMERGENCY_BUFFER_THRESHOLD * 0.5) {
            // 极度紧急：选择最低质量
            targetIndex = 0;
        } else {
            // 一般紧急：降低2-3级
            targetIndex = Math.max(0, currentIndex - 3);
        }
        
        // logger.debug(`[ThroughputRule] 紧急降级: ${currentIndex} -> ${targetIndex}`);
        return possibleRepresentations[targetIndex];
    }



    function getSwitchRequest(rulesContext) {
        try {
            const switchRequest = SwitchRequest(context).create();
            const abrController = rulesContext.getAbrController();
            const mediaInfo = rulesContext.getMediaInfo();
            const mediaType = rulesContext.getMediaType();
            const streamInfo = rulesContext.getStreamInfo();
            const scheduleController = rulesContext.getScheduleController();
            const playbackController = scheduleController.getPlaybackController();
            
            const isDynamic = streamInfo && streamInfo.manifestInfo && streamInfo.manifestInfo.isDynamic;
            
            // 只处理直播流
            if (!isDynamic) {
                logger.debug('[ThroughputRule] 非直播流，跳过处理');
                return switchRequest;
            }
            
            // dashMetrics 通过构造函数的 config 参数获取，而不是从 rulesContext
            // const dashMetrics = rulesContext.getDashMetrics(); // 这行代码是错误的
            const throughputController = rulesContext.getThroughputController();
            const possibleRepresentations = abrController.getPossibleVoRepresentationsFilteredBySettings(mediaInfo, true);
            
            if (!possibleRepresentations || possibleRepresentations.length === 0) {
                logger.warn('[ThroughputRule] 没有可用的表示');
                return switchRequest;
            }
            
            // 获取当前指标（增强错误处理）
            const currentRepresentation = rulesContext.getRepresentation();
            
            let bufferLevel = 0;
            let throughput = 0;
            let httpRequest = null;
            try {
                bufferLevel = dashMetrics.getCurrentBufferLevel(mediaType,true) || 0;
                bufferLevel = Math.max(0, bufferLevel); // 确保非负
            } catch (error) {
                logger.warn('[ThroughputRule] 获取缓冲区级别失败:', error);
                bufferLevel = 0;
            }
            const rawThroughputData = throughputController.getRawThroughputData(mediaType);
            //console.log('rawThroughputData',rawThroughputData);
            try {
                const L = stateInfo && stateInfo.sequence_length ? stateInfo.sequence_length : 20;
                const need = L + 1;
                const values = rawThroughputData ? rawThroughputData.map(item => item && item.value).filter(v => typeof v === 'number' && isFinite(v)) : [];
                if (model && values.length >= 1) {
                    const lastVal = values.length ? values[values.length - 1] : 0;
                    const tail = values.slice(-need);
                    const padded = tail.length >= need ? tail : Array(need - tail.length).fill(lastVal).concat(tail);
                    const mode = stateInfo && stateInfo.residual_mode ? stateInfo.residual_mode : 'raw';
                    const eps = stateInfo && typeof stateInfo.eps === 'number' ? stateInfo.eps : 1e-3;
                    const shift = stateInfo && typeof stateInfo.value_shift === 'number' ? stateInfo.value_shift : 0;
                    const r = (() => {
                        if (mode === 'log') {
                            const u = padded.map(x => Math.log1p(x + shift));
                            const d = []; for (let i = 1; i < u.length; i++) d.push(u[i] - u[i - 1]);
                            return d;
                        }
                        if (mode === 'normalized') {
                            const d = []; for (let i = 1; i < padded.length; i++) d.push((padded[i] - padded[i - 1]) / (padded[i - 1] + eps));
                            return d;
                        }
                        const d = []; for (let i = 1; i < padded.length; i++) d.push(padded[i] - padded[i - 1]);
                        return d;
                    })();
                    if (!r || r.length < L) {
                        throughput = throughputController.getAverageThroughput(mediaType) || lastVal;
                    } else {
                        const mean = scalerInfo && typeof scalerInfo.mean === 'number' ? scalerInfo.mean : 0;
                        const scale = scalerInfo && typeof scalerInfo.scale === 'number' && scalerInfo.scale !== 0 ? scalerInfo.scale : 1;
                        const rs = r.map(x => (x - mean) / scale);
                        const seq = rs.slice(rs.length - L);
                        const X = tf.tensor3d([seq.map(v => [v])]);
                        const yhat = model.predict(X);
                        const data = yhat.dataSync();
                        const predScaled = data[0];
                        X.dispose();
                        yhat.dispose();
                        const predRes = predScaled * scale + mean;
                        const last = padded[padded.length - 1];
                        let next;
                        if (mode === 'normalized') next = last + predRes * (last + eps);
                        else if (mode === 'log') next = Math.expm1(Math.log1p(last + shift) + predRes) - shift;
                        else next = last + predRes;
                        throughput = Math.max(0, next);
                        console.log('throughput',throughput);
                    }
                } else {
                    throughput = throughputController.getAverageThroughput(mediaType) || 0;
                }
            } catch (error) {
                throughput = throughputController.getAverageThroughput(mediaType) || 0;
            }
            
            
            try {
                httpRequest = dashMetrics.getCurrentHttpRequest(mediaType, true);
            } catch (error) {
                logger.warn('[ThroughputRule] 获取HTTP请求信息失败:', error);
                httpRequest = null;
            }
            
            // 简化延迟计算：专注于端到端延迟
            let latency = 0;
            
            try {
                // 主要使用HTTP请求延迟作为端到端延迟指标
                latency = playbackController.getCurrentLiveLatency();
                
                // 确保延迟值在合理范围内
                latency = Math.max(0, Math.min(latency, MAX_ACCEPTABLE_LATENCY));
                
            } catch (error) {
                logger.warn('[ThroughputRule] 延迟计算出错:', error);
                latency = TARGET_LATENCY; // 使用默认值
            }
            
            // 更新缓冲区历史数据
            const validBufferLevel = isFinite(bufferLevel) ? bufferLevel : 0;
            bufferHistory.push(validBufferLevel);
            if (bufferHistory.length > 10) {
                bufferHistory.shift();
            }
            
            // 验证关键参数
            if (!possibleRepresentations || possibleRepresentations.length === 0) {
                logger.error('[ThroughputRule] 没有可用的表示');
                return SwitchRequest(context).create();
            }
            
            // 计算网络稳定性
            const networkStability = calculateNetworkStability(throughputController, mediaType);
            
            // 更新自适应缓冲区目标
            updateAdaptiveBufferTarget(networkStability, latency);
            
            // 简化状态管理：直接处理紧急情况
            if (bufferLevel < EMERGENCY_BUFFER_THRESHOLD) {
                liveState.state = LIVE_STATE_EMERGENCY;
                const emergencyRepresentation = handleEmergencyState(possibleRepresentations, bufferLevel, currentRepresentation);
                switchRequest.representation = emergencyRepresentation;
                switchRequest.reason = {
                    rule: 'ThroughputRule',
                    state: 'EMERGENCY',
                    bufferLevel,
                    threshold: EMERGENCY_BUFFER_THRESHOLD
                };
                switchRequest.priority = SwitchRequest.PRIORITY.STRONG;
                
                liveState.currentRepresentation = emergencyRepresentation;
                return switchRequest;
            } else if (liveState.state === LIVE_STATE_EMERGENCY && bufferLevel > EMERGENCY_BUFFER_THRESHOLD * 1.5) {
                // 简单的紧急状态退出条件
                liveState.state = LIVE_STATE_NORMAL;
                liveState.consecutiveDowngrades = 0;
                // logger.info('[ThroughputRule] 退出紧急模式');
            }
            
            // 选择最优质量
            const selectionResult = selectOptimalQuality(
                possibleRepresentations,
                throughput,
                latency,
                bufferLevel,
                networkStability
            );
            
            const selectedRepresentation = selectionResult.representation;
            
            // 简化状态更新：移除不必要的状态转换
            
            // 设置切换请求
            switchRequest.representation = selectedRepresentation;
            
            // 简化切换原因信息
            const switchReason = {
                rule: 'ThroughputRule',
                state: liveState.state,
                throughput: Math.round(throughput),
                latency: Math.round(latency * 1000) / 1000,
                bufferLevel: Math.round(bufferLevel * 1000) / 1000,
                selectedBitrate: selectedRepresentation.bitrateInKbit,
                previousBitrate: currentRepresentation ? currentRepresentation.bitrateInKbit : 0
            };
            
            switchRequest.reason = switchReason;
            
            // 简化优先级设置
            if (liveState.state === LIVE_STATE_EMERGENCY) {
                switchRequest.priority = SwitchRequest.PRIORITY.STRONG;
            } else if (!networkStability.isStable) {
                switchRequest.priority = SwitchRequest.PRIORITY.DEFAULT;
            } else {
                switchRequest.priority = SwitchRequest.PRIORITY.WEAK;
            }
            
            // 性能监控日志
            // const bitrateChange = currentRepresentation ? 
                // selectedRepresentation.bitrateInKbit - currentRepresentation.bitrateInKbit : 0;
            
            // if (Math.abs(bitrateChange) > 0) {
            //     const changeType = bitrateChange > 0 ? '升级' : '降级';
            //     logger.info(`[ThroughputRule] 质量${changeType}: ${currentRepresentation?.bitrateInKbit || 0} -> ${selectedRepresentation.bitrateInKbit}kbps`);
            //     // logger.debug(`[ThroughputRule] 切换详情:`, switchReason);
            // }
            
            liveState.currentRepresentation = selectedRepresentation;
            lastSwitchTime = Date.now();
            
            // logger.debug(`[ThroughputRule] 选择质量: ${selectedRepresentation.bitrateInKbit}kbps, 状态: ${liveState.state}`);
            
            return switchRequest;
            
        } catch (error) {
            logger.error('[ThroughputRule] 错误:', error);
            const switchRequest = SwitchRequest(context).create();
            return switchRequest;
        }
    }

    function getClassName() {
        return 'ThroughputRule';
    }

    instance = {
        getSwitchRequest: getSwitchRequest,
        getClassName: getClassName,
        reset: reset,
        setup: setup,
        getLogStats: () => LogManager.getLogStats(),
        configureLogging: (config) => LogManager.configure(config),
        configureRemoteLogging: (config) => LogManager.configureRemoteLogging(config),
        cleanOldLogs: () => LogManager.cleanOldLogs()
    };

    setup();

    return instance;
}

ThroughputRule.__dashjs_factory_name = 'ThroughputRule';

export default FactoryMaker.getClassFactory(ThroughputRule);
