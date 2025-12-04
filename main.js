document.addEventListener('DOMContentLoaded', function () {
    const video = document.querySelector('#videoPlayer');
    const urlInput = document.querySelector('#stream-url');
    const loadBtn = document.querySelector('#load-btn');
    const statsOutput = document.querySelector('#stats-output');
    const exportCsvBtn = document.querySelector('#export-csv-btn');
    const enableAbrBtn = document.querySelector('#enable-custom-abr');
    const abrStatus = document.querySelector('#abr-status');

    let player = dashjs.MediaPlayer().create();
    let useCustomAbr = false;
    let rebufferFlag = false;
    let statsTimer = null;

    // Initialize player (but don't auto play yet)
    function initPlayer() {
        // Just initialize the player instance, but don't attach source yet
        player.initialize(video, null, true);
        
        // Enable console logging for Dash.js
        player.updateSettings({
            'debug': {
                'logLevel': dashjs.Debug.LOG_LEVEL_INFO
            }
        });

        setupStatsUpdate();

        player.on(dashjs.MediaPlayer.events.BUFFER_EMPTY, function () {
            rebufferFlag = true;
            addSample(true);
        });
        player.on(dashjs.MediaPlayer.events.BUFFER_LOADED, function () {
            setTimeout(() => { rebufferFlag = false; }, 1500);
        });
        player.on(dashjs.MediaPlayer.events.PLAYBACK_WAITING, function () {
            rebufferFlag = true;
            addSample(true);
        });
        player.on(dashjs.MediaPlayer.events.PLAYBACK_PLAYING, function () {
            setTimeout(() => { rebufferFlag = false; }, 1500);
        });
    }

    // Load stream on button click
    loadBtn.addEventListener('click', function () {
        const url = urlInput.value;
        if (url) {
             // Check if player is initialized with a source, if not or we are changing, attach
             if (!player.isReady()) {
                 player.attachSource(url);
             } else {
                 player.attachSource(url);
             }
        }
    });

    // ... (rest of the code)

    // Start with player initialized but empty
    initPlayer();
    // Stats sampling buffer
    const statsSamples = [];

    function addSample(rebufferTag) {
        if (!(player && player.isReady())) return;
        const dashMetrics = player.getDashMetrics();
        const bufferLevel = dashMetrics.getCurrentBufferLevel('video');
        const rep = player.getCurrentRepresentationForType('video');
        const reps = player.getRepresentationsByType('video') || [];
        const qualityIndex = rep ? reps.findIndex(r => r.id === rep.id) : -1;
        let liveLatency = null;
        try {
            const dvr = player.getDvrWindow();
            if (dvr && typeof dvr.end === 'number') {
                const nowPT = player.time();
                liveLatency = dvr.end - nowPT;
            }
        } catch (e) {
            liveLatency = null;
        }
        statsSamples.push({
            timestamp: new Date().toISOString(),
            bufferLevel: typeof bufferLevel === 'number' ? bufferLevel : null,
            qualityIndex: qualityIndex,
            bitrateKbps: rep && rep.bitrateInKbit ? Math.round(rep.bitrateInKbit) : null,
            resolution: rep && rep.width && rep.height ? `${rep.width}x${rep.height}` : null,
            liveLatency: typeof liveLatency === 'number' ? liveLatency : null,
            rebuffer: rebufferTag ? 'rebuffer happen' : ''
        });
    }

    function setupStatsUpdate() {
        // Clear any previous interval
        if (statsTimer) {
            clearInterval(statsTimer);
            statsTimer = null;
        }
        statsTimer = setInterval(() => {
            if (player && player.isReady()) {
                const dashMetrics = player.getDashMetrics();
                const bufferLevel = dashMetrics.getCurrentBufferLevel('video');
                const rep = player.getCurrentRepresentationForType('video');
                const reps = player.getRepresentationsByType('video') || [];
                const qualityIndex = rep ? reps.findIndex(r => r.id === rep.id) : -1;

                // Live latency estimation (seconds): difference between DVR window end and current presentation time
                // Works for dynamic streams (live). For VoD this value will equal duration and is not meaningful.
                let liveLatency = null;
                try {
                    const dvr = player.getDvrWindow();
                    if (dvr && typeof dvr.end === 'number') {
                        const nowPT = player.time();
                        liveLatency = dvr.end - nowPT;
                    }
                } catch (e) {
                    liveLatency = null;
                }

                let statsHtml = `
                    <div>Buffer Level: ${bufferLevel ? bufferLevel.toFixed(2) : 0} s</div>
                    <div>Current Quality Index: ${qualityIndex >= 0 ? qualityIndex : 'N/A'}</div>
                    <div>Bitrate: ${rep && rep.bitrateInKbit ? Math.round(rep.bitrateInKbit) + ' kbps' : 'N/A'}</div>
                    <div>Resolution: ${rep && rep.width && rep.height ? rep.width + 'x' + rep.height : 'N/A'}</div>
                    <div>Live Latency: ${liveLatency !== null ? liveLatency.toFixed(2) + ' s' : 'N/A'}</div>
                    <div>Rebuffer: ${rebufferFlag ? 'rebuffer happen' : '—'}</div>
                `;
                statsOutput.innerHTML = statsHtml;

                // Push sample
                statsSamples.push({
                    timestamp: new Date().toISOString(),
                    bufferLevel: typeof bufferLevel === 'number' ? bufferLevel : null,
                    qualityIndex: qualityIndex,
                    bitrateKbps: rep && rep.bitrateInKbit ? Math.round(rep.bitrateInKbit) : null,
                    resolution: rep && rep.width && rep.height ? `${rep.width}x${rep.height}` : null,
                    liveLatency: typeof liveLatency === 'number' ? liveLatency : null,
                    rebuffer: rebufferFlag ? 'rebuffer happen' : ''
                });
            }
        }, 1000);
    }

    // Stop stats updates when playback ends
    player.on(dashjs.MediaPlayer.events.PLAYBACK_ENDED, function () {
        if (statsTimer) {
            clearInterval(statsTimer);
            statsTimer = null;
        }
    });

    // Export CSV button
    if (exportCsvBtn) {
        exportCsvBtn.addEventListener('click', function () {
            if (!statsSamples.length) {
                alert('No samples collected yet.');
                return;
            }

            const header = ['timestamp','bufferLevel','qualityIndex','bitrateKbps','resolution','liveLatency','rebuffer'];
            const rows = statsSamples.map(s => header.map(h => s[h] !== null && s[h] !== undefined ? s[h] : ''));
            const csv = [header.join(','), ...rows.map(r => r.join(','))].join('\n');
            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `dash_stats_${Date.now()}.csv`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
    }

    // Toggle Custom ABR
    enableAbrBtn.addEventListener('click', function() {
        useCustomAbr = !useCustomAbr;
        
        if (useCustomAbr) {
            // Register Custom Rule
            // In Dash.js 3.x/4.x, we add custom rules via addABRCustomRule
            
            // We assume CustomAbrRule function is available globally from CustomAbrRule.js
            if (typeof CustomAbrRule !== 'function') {
                alert("CustomAbrRule not found!");
                return;
            }

            // Add the rule
            // 'QualitySwitchRules' is the collection of rules for quality switching
            player.addABRCustomRule('qualitySwitchRules', 'CustomAbrRule', CustomAbrRule);
            
            // Usually we need to update settings to prioritize or enable custom rules if needed,
            // but addABRCustomRule should inject it. 
            // We might need to force a reload or just let it run.
            
            player.updateSettings({
                'streaming': {
                    'abr': {
                        'useDefaultABRRules': true // Keep default rules, ours will run alongside or override depending on implementation
                    }
                }
            });

            abrStatus.textContent = "Status: Custom ABR Enabled";
            enableAbrBtn.textContent = "Disable Custom ABR (Reload required to fully reset)";
            
            console.log("Custom ABR Rule Added");
        } else {
            // Removing rules is trickier, often requires removing all and re-adding defaults or just reloading the player.
            // For simplicity, we will just notify user to reload.
            player.removeAllABRCustomRules();
            abrStatus.textContent = "Status: Default ABR";
            enableAbrBtn.textContent = "Enable Custom ABR Rule";
            console.log("Custom ABR Rule Removed");
        }
    });

    // ... (rest of the code)
});
