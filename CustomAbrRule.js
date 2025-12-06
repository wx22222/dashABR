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
                        return switchRequest;
                    }

                    let safeThroughputKbps = null;
                    if (throughputController && typeof throughputController.getSafeAverageThroughput === 'function') {
                        safeThroughputKbps = throughputController.getSafeAverageThroughput(mediaType);
                    } else if (abrController && typeof abrController.getThroughputHistory === 'function') {
                        try {
                            const th = abrController.getThroughputHistory();
                            if (th && typeof th.getAverageThroughput === 'function') {
                                safeThroughputKbps = th.getAverageThroughput(mediaType, isDynamic);
                            }
                        } catch (e) {}
                    }

                    if (!safeThroughputKbps || !isFinite(safeThroughputKbps) || safeThroughputKbps <= 0) {
                        return switchRequest;
                    }

                    let optimalRep = null;
                    if (typeof abrController.getOptimalRepresentationForBitrate === 'function') {
                        optimalRep = abrController.getOptimalRepresentationForBitrate(mediaInfo, safeThroughputKbps, true);
                    }

                    if (optimalRep) {
                        switchRequest.representation = optimalRep;
                        switchRequest.priority = 0.5;
                        switchRequest.reason = { throughputKbps: Math.round(safeThroughputKbps) };
                    }

                    return switchRequest;
                }
            };
        }
    };
}

// We need to register this class factory in main.js or expose it
// Since this is a script tag, `CustomAbrRule` is now in global scope.
