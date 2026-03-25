function MPCRule(context) {
    const factory = dashjs.FactoryMaker;
    const SwitchRequestFactory = factory.getClassFactoryByName('SwitchRequest');
    return {
        create: function (config) {
            let lastIndex = null;
            let lastSwitchTimeMs = null;
            let ruleCfg = config || {};
            const horizon = 5;
            const defaultSegDuration = 0.5;
            const wBitrate = 1.0;
            const wRebuffer = 3.5;
            const wSmooth = 0.7;
            const minSwitchInterval = 3.4;
            const safetyFactor = 1.03;
            const throughputGuardLow = 0.7;
            const throughputGuardDefault = 1.0;
            const throughputGuardHigh = 1.15;
            let historyKbps = [];
            return {
                getClassName: function () { return 'MPCRule'; },
                getSwitchRequest: function (rulesContext) { return this.checkIndex(rulesContext); },
                checkIndex: function (rulesContext) {
                    console.log('MPCRULE');
                    const mediaInfo = rulesContext.getMediaInfo();
                    const mediaType = mediaInfo.type;
                    const abrController = rulesContext.getAbrController();
                    const scheduleController = rulesContext.getScheduleController();
                    const dashMetrics = ruleCfg && ruleCfg.dashMetrics ? ruleCfg.dashMetrics : null;
                    const throughputController = rulesContext.getThroughputController ? rulesContext.getThroughputController() : null;
                    const playbackController = scheduleController && typeof scheduleController.getPlaybackController === 'function' ? scheduleController.getPlaybackController() : null;
                    const switchRequest = SwitchRequestFactory(context).create();
                    let bufferLevel = null;
                    if (dashMetrics) {
                        bufferLevel = dashMetrics.getCurrentBufferLevel(mediaType);
                    }
                    let effThroughputGuard = throughputGuardDefault;
                    if (typeof bufferLevel === 'number') {
                        if (bufferLevel < 1.0) {
                            effThroughputGuard = throughputGuardLow;
                        } else if (bufferLevel > 4.0) {
                            effThroughputGuard = throughputGuardHigh;
                        }
                    }
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
                    }
                    if (!measurementKbps || !isFinite(measurementKbps) || measurementKbps <= 0) {
                        return switchRequest;
                    }
                    historyKbps.push(measurementKbps);
                    if (historyKbps.length > 5) {
                        historyKbps = historyKbps.slice(historyKbps.length - 5);
                    }
                    let harmonic = null;
                    let sumInv = 0, n = 0;
                    for (let v of historyKbps) {
                        if (v > 0 && isFinite(v)) {
                            sumInv += 1.0 / v;
                            n++;
                        }
                    }
                    if (n > 0) {
                        harmonic = n / sumInv;
                    }
                    let predictedKbpsBase = harmonic || measurementKbps;
                    let predictedKbps = predictedKbpsBase * safetyFactor;
                    if (!predictedKbps || !isFinite(predictedKbps) || predictedKbps <= 0) {
                        predictedKbps = measurementKbps;
                    }
                    if (typeof window !== 'undefined') {
                        try {
                            window.dashPredictedKbps = Math.round(predictedKbps);
                        } catch (e) {}
                    }
                    const segDur = typeof mediaInfo.fragmentDuration === 'number' ? mediaInfo.fragmentDuration : (typeof mediaInfo.segmentDuration === 'number' ? mediaInfo.segmentDuration : defaultSegDuration);
                    let bestIndex = lastIndex !== null ? lastIndex : 0;
                    let bestScore = -Infinity;
                    const lastKb = (lastIndex !== null) ? (sorted[lastIndex].bitrateInKbit || sorted[lastIndex].bitrate || 0) : 0;
                    for (let i = 0; i < sorted.length; i++) {
                        const kb = (sorted[i].bitrateInKbit || sorted[i].bitrate || 0);
                        if (i > 0 && kb > predictedKbps * effThroughputGuard) {
                            continue;
                        }
                        let currBuffer = typeof bufferLevel === 'number' ? bufferLevel : 0;
                        let rebufSum = 0;
                        let bitrateSum = 0;
                        for (let k = 0; k < horizon; k++) {
                            const downloadTime = kb > 0 && predictedKbps > 0 ? (kb * segDur) / predictedKbps : Infinity;
                            if (currBuffer < downloadTime) {
                                rebufSum += (downloadTime - currBuffer);
                                currBuffer = 0;
                            } else {
                                currBuffer -= downloadTime;
                            }
                            currBuffer += segDur;
                            bitrateSum += Math.log(1 + kb);
                        }
                        let smoothPenalty = 0;
                        if (lastKb > 0) {
                            const maxKb = Math.max(kb, lastKb);
                            const relDelta = Math.abs(kb - lastKb) / maxKb;
                            smoothPenalty = wSmooth * relDelta;
                        }
                        const score = wBitrate * bitrateSum - wRebuffer * rebufSum - smoothPenalty;
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
                            if (!(predictedKbps < currentKb * 0.9) && sinceLast < minSwitchInterval) {
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
