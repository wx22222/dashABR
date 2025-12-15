/**
 * Custom ABR Rule for Dash.js
 * This is a template for a custom Adaptive Bitrate Rule.
 */

// Define the rule factory function
function CustomAbrRule(context) {
    const factory = dashjs.FactoryMaker;
    const SwitchRequestFactory = factory.getClassFactoryByName('SwitchRequest');
    return {
        create: function (config) {
            let ewmaFast = null;
            let ewmaSlow = null;
            let lastIndex = null;
            let goodCount = 0;
            let lastSwitchTimeMs = null;
            let lstmSession = null;
            let lstmCfg = null;
            let lstmLoading = false;
            let histSeries = [];
            let predKbpsLstm = null;
            let lastMs = null;
            let bucketTime = 0.0;
            let ruleCfg = config || {};
            const aggDt = 0.5;
            const alphaFast = 0.6;
            const alphaSlow = 0.25;
            const safetyFactor = 1.0;
            const maxStepUp = 2;
            const upswitchBufferThreshold = 0.75;
            const upswitchMargin = 1.12;
            const betaBudget = 1.03;
            const defaultSegDuration = 0.5;
            const wBitrate = 1.0;
            const wRebuffer = 3.4;
            const wLatency = 0.2;
            const fastDownloadFrac = 0.88;
            const targetLatency = 2.0;
            const downswitchMargin = 0.95;
            const minSwitchInterval = 3.4;
            const throughputGuardLow = 0.7;
            const throughputGuardDefault = 0.85;
            const throughputGuardHigh = 1.05;
            return {
                getClassName: function () { return 'CustomAbrRule'; },
                getSwitchRequest: function (rulesContext) { return this.checkIndex(rulesContext); },
                checkIndex: function (rulesContext) {
                    const mediaInfo = rulesContext.getMediaInfo();
                    const mediaType = mediaInfo.type;
                    const streamInfo = rulesContext.getStreamInfo();
                    const abrController = rulesContext.getAbrController();
                    const scheduleController = rulesContext.getScheduleController();
                    let dashMetrics = ruleCfg && ruleCfg.dashMetrics ? ruleCfg.dashMetrics : null;

                   
                    const throughputController = rulesContext.getThroughputController ? rulesContext.getThroughputController() : null;
                    const playbackController = scheduleController && typeof scheduleController.getPlaybackController === 'function' ? scheduleController.getPlaybackController() : null;

                    const switchRequest = SwitchRequestFactory(context).create();
                    let bufferLevel = null;  
                    bufferLevel = dashMetrics.getCurrentBufferLevel(mediaType)
                    let effSafety = safetyFactor;
                    let effMargin = upswitchMargin;
                    let effBufThr = upswitchBufferThreshold;
                    let effMaxUp = maxStepUp;
                    let effBeta = betaBudget;
                    let effWRebuffer = wRebuffer;
                    let wFast = 0.5;
                    let liveLatency = null;
                    if (playbackController) {
                        if (typeof playbackController.getCurrentLiveLatency === 'function') {
                            liveLatency = playbackController.getCurrentLiveLatency();
                        } else if (typeof playbackController.getLiveLatency === 'function') {
                            liveLatency = playbackController.getLiveLatency();
                        }
                    }
                    let effWLatency = wLatency;
                    let effFastFrac = fastDownloadFrac;
                    const latHigh = typeof liveLatency === 'number' && liveLatency > targetLatency;
                    let effThroughputGuard = throughputGuardDefault;
                    if (typeof bufferLevel === 'number') {
                        if (bufferLevel < 1.0) {
                            effThroughputGuard = throughputGuardLow;
                        } else if (bufferLevel > 4.0) {
                            effThroughputGuard = throughputGuardHigh;
                        }
                    }

                    const isDynamic = !!(streamInfo && streamInfo.manifestInfo && streamInfo.manifestInfo.isDynamic);

                    const reps = abrController && typeof abrController.getPossibleVoRepresentationsFilteredBySettings === 'function'
                        ? abrController.getPossibleVoRepresentationsFilteredBySettings(mediaInfo, true)
                        : [];
                    if (!reps || reps.length === 0) {
                        return switchRequest;
                    }

                    const sorted = reps.slice().sort(function (a, b) {
                        const ba = (a.bitrateInKbit || a.bitrate || 0);
                        const bb = (b.bitrateInKbit || b.bitrate || 0);
                        return ba - bb;
                    });

                    let measurementKbps = null;
                    if (throughputController && typeof throughputController.getSafeAverageThroughput === 'function') {
                        measurementKbps = throughputController.getSafeAverageThroughput(mediaType);
                    } else if (abrController && typeof abrController.getThroughputHistory === 'function') {
                        try {
                            const th = abrController.getThroughputHistory();
                            if (th && typeof th.getAverageThroughput === 'function') {
                                measurementKbps = th.getAverageThroughput(mediaType, isDynamic);
                            }
                        } catch (e) {}
                    }

                    if (!measurementKbps || !isFinite(measurementKbps) || measurementKbps <= 0) {
                        return switchRequest;
                    }

                    if (!lstmSession && !lstmLoading) {
                        lstmLoading = true;
                        try {
                            fetch('assets/models/fcc_lstm_cfg.json').then(function(r){return r.json();}).then(function(cfg){
                                lstmCfg = cfg || {};
                                const hp = (lstmCfg && typeof lstmCfg.hist_len==='number') ? lstmCfg.hist_len : 30;
                                lstmCfg = {
                                    hist_len: hp,
                                    xm: typeof cfg.xm==='number'?cfg.xm:0.0,
                                    xs: typeof cfg.xs==='number'?cfg.xs:1.0,
                                    ym: typeof cfg.ym==='number'?cfg.ym:0.0,
                                    ys: typeof cfg.ys==='number'?cfg.ys:1.0
                                };
                                if (typeof ort !== 'undefined' && ort && typeof ort.InferenceSession !== 'undefined') {
                                    ort.InferenceSession.create('assets/models/fcc_lstm.onnx').then(function(s){
                                        lstmSession = s;
                                        lstmLoading = false;
                                    }).catch(function(){lstmLoading=false;});
                                } else {
                                    lstmLoading = false;
                                }
                            }).catch(function(){lstmLoading=false;});
                        } catch(e){
                            lstmLoading = false;
                        }
                    }

                    try {
                        const nowMs = (typeof performance !== 'undefined' && typeof performance.now === 'function') ? performance.now() : Date.now();
                        if (lastMs === null) {
                            lastMs = nowMs;
                        }
                        const dtSec = Math.max(0, (nowMs - lastMs) / 1000.0);
                        lastMs = nowMs;
                        bucketTime += dtSec;
                        while (bucketTime >= aggDt) {
                            histSeries.push(measurementKbps);
                            bucketTime -= aggDt;
                        }
                        const hlen = (lstmCfg && lstmCfg.hist_len) ? lstmCfg.hist_len : 30;
                        if (histSeries.length > hlen) histSeries = histSeries.slice(histSeries.length - hlen);
                        const ready = lstmSession && histSeries.length >= hlen;
                  
                        if (ready) {
                            const xm = lstmCfg.xm || 0.0;
                            const xs = lstmCfg.xs || 1.0;
                            const ys = lstmCfg.ys || 1.0;
                            const ym = lstmCfg.ym || 0.0;
                            const inp = new Float32Array(hlen);
                            for (let i = 0; i < hlen; i++) {
                                const v = histSeries[histSeries.length - hlen + i];
                                inp[i] = (v - xm) / (xs || 1.0);
                            }
                            const feeds = { hist: new ort.Tensor('float32', inp, [1, hlen]) };
                        
                            lstmSession.run(feeds).then(function(o){
                                const pred = o && o.pred && o.pred.data && o.pred.data[0];
                              
                                if (typeof pred === 'number' && isFinite(pred)) {
                                    const kbps = pred * (ys || 1.0) + (ym || 0.0);
                                  
                                    if (kbps > 0 && isFinite(kbps)) {
                                        predKbpsLstm = kbps;
                                        if (typeof window !== 'undefined') {
                                            try { window.dashPredictedKbps = Math.round(predKbpsLstm); } catch(e){}
                                        }
                                    }
                                }
                            }).catch(function(){});
                        }
                    } catch(e){}

                    ewmaFast = ewmaFast === null ? measurementKbps : (alphaFast * measurementKbps + (1 - alphaFast) * ewmaFast);
                    ewmaSlow = ewmaSlow === null ? measurementKbps : (alphaSlow * measurementKbps + (1 - alphaSlow) * ewmaSlow);
                    const ewmaCombined = (wFast * ewmaFast + (1 - wFast) * ewmaSlow);
                    let predictedKbpsBase = ewmaCombined;
                    if (predKbpsLstm && isFinite(predKbpsLstm) && predKbpsLstm > 0) {
                        predictedKbpsBase = predKbpsLstm;
                        console.log('predictedKbp',predKbpsLstm);
                    }
                    let predictedKbps = predictedKbpsBase * effSafety;
                    if (typeof window !== 'undefined') {
                        try {
                            window.dashPredictedKbps = Math.round(predictedKbps);
                        } catch (e) {}
                    }
                    if (!predictedKbps || !isFinite(predictedKbps) || predictedKbps <= 0) {
                        predictedKbps = measurementKbps;
                    }

                    let allowBypass = false;
                    if (lastIndex === 0 && typeof bufferLevel === 'number' && bufferLevel < 0.2) {
                        const nextIdxProbe = Math.min(sorted.length - 1, 1);
                        const nextKbProbe = (sorted[nextIdxProbe].bitrateInKbit || sorted[nextIdxProbe].bitrate || 0);
                        if (nextKbProbe > 0 && predictedKbps >= nextKbProbe * 1.18) {
                            allowBypass = true;
                        }
                    }

                    let maxAllowedUpIndex = lastIndex !== null ? Math.min(sorted.length - 1, lastIndex + effMaxUp) : sorted.length - 1;
                    if (lastIndex !== null) {
                        const nextIdx = Math.min(sorted.length - 1, lastIndex + 1);
                        const nextKb = (sorted[nextIdx].bitrateInKbit || sorted[nextIdx].bitrate || 0);
                        const segDur2 = typeof mediaInfo.fragmentDuration === 'number' ? mediaInfo.fragmentDuration : (typeof mediaInfo.segmentDuration === 'number' ? mediaInfo.segmentDuration : defaultSegDuration);
                        const downloadTimeNext = nextKb > 0 && predictedKbps > 0 ? (nextKb * segDur2) / predictedKbps : Infinity;
                        const budgetFloorNext = 0.35 * segDur2;
                        const budgetOkNext = typeof bufferLevel === 'number' ? (downloadTimeNext <= Math.max(bufferLevel * effBeta, budgetFloorNext)) : true;
                        const gateOkNext = (typeof bufferLevel === 'number' && bufferLevel >= effBufThr) || (downloadTimeNext <= segDur2 * effFastFrac) || allowBypass;
                        const marginOkNext = predictedKbps >= nextKb * effMargin;
                        if (budgetOkNext && gateOkNext && marginOkNext) {
                            goodCount = Math.min(goodCount + 1, 10);
                            maxAllowedUpIndex = Math.min(sorted.length - 1, lastIndex + Math.max(effMaxUp, 2));
                        } else {
                            goodCount = 0;
                        }
                    }
                    let bestIndex = lastIndex !== null ? lastIndex : 0;
                    let bestScore = -Infinity;
                    const segDur = typeof mediaInfo.fragmentDuration === 'number' ? mediaInfo.fragmentDuration : (typeof mediaInfo.segmentDuration === 'number' ? mediaInfo.segmentDuration : defaultSegDuration);
                    const budgetFloor = 0.35 * segDur;
                    for (let i = 0; i < sorted.length; i++) {
                        const kb = (sorted[i].bitrateInKbit || sorted[i].bitrate || 0);
                        if (i > 0 && kb > predictedKbps * effThroughputGuard) {
                            continue;
                        }
                        const downloadTime = kb > 0 && predictedKbps > 0 ? (kb * segDur) / predictedKbps : Infinity;
                        const buf = typeof bufferLevel === 'number' ? bufferLevel : 0;
                        const rebuffer = downloadTime > buf ? (downloadTime - buf) : 0;
                        const budgetOk = typeof bufferLevel === 'number' ? (downloadTime <= Math.max(bufferLevel * effBeta, budgetFloor)) : true;

                        if (lastIndex !== null && i > lastIndex) {
                            if (i > maxAllowedUpIndex) {
                                continue;
                            }
                            const bufferOk = typeof bufferLevel === 'number' && bufferLevel >= effBufThr;
                        const fastOk = downloadTime <= segDur * effFastFrac;
                        let marginOk = predictedKbps >= kb * effMargin;
                        if (goodCount >= 1 && i === lastIndex + 1) {
                            marginOk = true;
                        }
                        if (!((bufferOk || fastOk || allowBypass) && marginOk)) {
                            continue;
                        }
                    }

                        if (!budgetOk) {
                            continue;
                        }

                        const bitrateScore = Math.log(1 + kb);
                        const latencyPenaltyCandidate = latHigh ? Math.max(0, downloadTime - segDur) : 0;
                        const score = wBitrate * bitrateScore - effWRebuffer * rebuffer - effWLatency * latencyPenaltyCandidate;
                        if (score > bestScore) {
                            bestScore = score;
                            bestIndex = i;
                        }
                    }

                    if (bestScore === -Infinity) {
                        let fallback = lastIndex !== null ? lastIndex : 0;
                        for (let i = 0; i < sorted.length; i++) {
                            const kb = (sorted[i].bitrateInKbit || sorted[i].bitrate || 0);
                            if (kb <= predictedKbps) {
                                fallback = i;
                            } else {
                                break;
                            }
                        }
                        bestIndex = fallback;
                    }

                    const nowSwitchMs = (typeof performance !== 'undefined' && typeof performance.now === 'function') ? performance.now() : Date.now();
                    if (lastIndex !== null && bestIndex !== lastIndex) {
                        const sinceLast = lastSwitchTimeMs !== null ? (nowSwitchMs - lastSwitchTimeMs) / 1000.0 : Infinity;
                        if (bestIndex > lastIndex) {
                            if (sinceLast < minSwitchInterval) {
                                bestIndex = lastIndex;
                            } else {
                                lastSwitchTimeMs = nowSwitchMs;
                            }
                        } else {
                            const currentKb = (sorted[lastIndex].bitrateInKbit || sorted[lastIndex].bitrate || 0);
                            if (!(predictedKbps < currentKb * downswitchMargin) && sinceLast < minSwitchInterval) {
                                bestIndex = lastIndex;
                            } else {
                                lastSwitchTimeMs = nowSwitchMs;
                            }
                        }
                    } else if (lastIndex === null && bestIndex !== null) {
                        lastSwitchTimeMs = nowSwitchMs;
                    }

                    switchRequest.representation = sorted[bestIndex];
                    switchRequest.priority = 0.5;
                    switchRequest.reason = { predictedKbps: Math.round(predictedKbps) };
                    lastIndex = bestIndex;

                    return switchRequest;
                }
            };
        }
    };
}

// We need to register this class factory in main.js or expose it
// Since this is a script tag, `CustomAbrRule` is now in global scope.
