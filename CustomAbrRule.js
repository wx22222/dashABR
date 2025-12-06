/**
 * Custom ABR Rule for Dash.js
 * This is a template for a custom Adaptive Bitrate Rule.
 */

// Define the rule factory function
function CustomAbrRule(context) {
    const factory = dashjs.FactoryMaker;
    const SwitchRequestFactory = factory.getClassFactoryByName('SwitchRequest');
    return {
        create: function () {
            let ewmaFast = null;
            let ewmaSlow = null;
            let lastIndex = null;
            const alphaFast = 0.6;
            const alphaSlow = 0.25;
            const safetyFactor = 0.85;
            const maxStepUp = 1;
            const upswitchBufferThreshold = 1.0;
            const upswitchMargin = 1.3;
            const betaBudget = 0.9;
            const defaultSegDuration = 0.5;
            const wBitrate = 1.0;
            const wRebuffer = 4.0;
            const wLatency = 0.2;
            return {
                getClassName: function () { return 'CustomAbrRule'; },
                getSwitchRequest: function (rulesContext) { return this.checkIndex(rulesContext); },
                checkIndex: function (rulesContext) {
                    const mediaInfo = rulesContext.getMediaInfo();
                    const mediaType = mediaInfo.type;
                    const streamInfo = rulesContext.getStreamInfo();
                    const abrController = rulesContext.getAbrController();
                    const scheduleController = rulesContext.getScheduleController();
                    const dashMetrics = rulesContext.getDashMetrics ? rulesContext.getDashMetrics() : null;
                    const throughputController = rulesContext.getThroughputController ? rulesContext.getThroughputController() : null;
                    const playbackController = scheduleController && typeof scheduleController.getPlaybackController === 'function' ? scheduleController.getPlaybackController() : null;

                    const switchRequest = SwitchRequestFactory(context).create();
                    let bufferLevel = null;
                    if (dashMetrics && typeof dashMetrics.getCurrentBufferLevel === 'function') {
                        bufferLevel = dashMetrics.getCurrentBufferLevel(mediaType);
                    } else if (scheduleController && typeof scheduleController.getBufferController === 'function') {
                        const bc = scheduleController.getBufferController();
                        if (bc && typeof bc.getBufferLevel === 'function') {
                            bufferLevel = bc.getBufferLevel();
                        }
                    }
                    let effSafety = safetyFactor;
                    let effMargin = upswitchMargin;
                    let effBufThr = upswitchBufferThreshold;
                    let effMaxUp = maxStepUp;
                    let effBeta = betaBudget;
                    if (typeof bufferLevel === 'number') {
                        if (bufferLevel >= 1.5) {
                            effSafety = 0.95;
                            effMargin = 1.1;
                            effBufThr = 0.5;
                            effMaxUp = 2;
                            effBeta = 1.0;
                        } else if (bufferLevel >= 1.0) {
                            effSafety = 0.9;
                            effMargin = 1.2;
                            effBufThr = 1.0;
                            effMaxUp = 1;
                            effBeta = 0.95;
                        }
                    }
                    let liveLatency = null;
                    if (playbackController && typeof playbackController.getLiveLatency === 'function') {
                        liveLatency = playbackController.getLiveLatency();
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

                    const emergency = typeof bufferLevel === 'number' && bufferLevel < 0.3;
                    if (emergency) {
                        switchRequest.representation = sorted[0];
                        switchRequest.priority = 1;
                        switchRequest.reason = { bufferLevel: bufferLevel };
                        lastIndex = 0;
                        return switchRequest;
                    }

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

                    ewmaFast = ewmaFast === null ? measurementKbps : (alphaFast * measurementKbps + (1 - alphaFast) * ewmaFast);
                    ewmaSlow = ewmaSlow === null ? measurementKbps : (alphaSlow * measurementKbps + (1 - alphaSlow) * ewmaSlow);
                    let predictedKbps = Math.min(ewmaFast, ewmaSlow) * effSafety;
                    if (!predictedKbps || !isFinite(predictedKbps) || predictedKbps <= 0) {
                        predictedKbps = measurementKbps;
                    }

                    let maxAllowedUpIndex = lastIndex !== null ? Math.min(sorted.length - 1, lastIndex + effMaxUp) : sorted.length - 1;
                    let bestIndex = lastIndex !== null ? lastIndex : 0;
                    let bestScore = -Infinity;
                    const segDur = typeof mediaInfo.fragmentDuration === 'number' ? mediaInfo.fragmentDuration : (typeof mediaInfo.segmentDuration === 'number' ? mediaInfo.segmentDuration : defaultSegDuration);
                    for (let i = 0; i < sorted.length; i++) {
                        const kb = (sorted[i].bitrateInKbit || sorted[i].bitrate || 0);
                        const downloadTime = kb > 0 && predictedKbps > 0 ? (kb * segDur) / predictedKbps : Infinity;
                        const buf = typeof bufferLevel === 'number' ? bufferLevel : 0;
                        const rebuffer = downloadTime > buf ? (downloadTime - buf) : 0;
                        const budgetOk = typeof bufferLevel === 'number' ? (downloadTime <= bufferLevel * effBeta) : true;

                        if (lastIndex !== null && i > lastIndex) {
                            if (i > maxAllowedUpIndex) {
                                continue;
                            }
                            const bufferOk = typeof bufferLevel === 'number' && bufferLevel >= effBufThr;
                            const marginOk = predictedKbps >= kb * effMargin;
                            if (!(bufferOk && marginOk)) {
                                continue;
                            }
                        }

                        if (!budgetOk) {
                            continue;
                        }

                        const bitrateScore = Math.log(1 + kb);
                        const latencyPenalty = typeof liveLatency === 'number' ? liveLatency : 0;
                        const score = wBitrate * bitrateScore - wRebuffer * rebuffer - wLatency * latencyPenalty;
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

                    switchRequest.representation = sorted[bestIndex];
                    switchRequest.priority = 0.5;
                    switchRequest.reason = { predictedKbps: Math.round(predictedKbps), bufferLevel: bufferLevel };
                    lastIndex = bestIndex;

                    return switchRequest;
                }
            };
        }
    };
}

// We need to register this class factory in main.js or expose it
// Since this is a script tag, `CustomAbrRule` is now in global scope.
