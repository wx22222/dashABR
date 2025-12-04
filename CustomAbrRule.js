/**
 * Custom ABR Rule for Dash.js
 * This is a template for a custom Adaptive Bitrate Rule.
 */

// Define the rule factory function
function CustomAbrRule() {
    
    let context = this.context;
    let log = function(msg) {
        console.log("[CustomAbrRule] " + msg);
    };

    // The metrics constants are usually available in dash.js, 
    // but for a standalone rule file we might need to access them via the context or just use raw strings if needed,
    // or rely on Dash.js being loaded globally.
    
    let instance = {
        // The name of the rule - useful for debugging
        checkIndex: function (rulesContext) {
            // This method is called by Dash.js to determine the best quality index.
            
            // Get available information
            const mediaType = rulesContext.getMediaInfo().type;
            const streamInfo = rulesContext.getStreamInfo();
            const abrController = rulesContext.getAbrController();
            const throughputHistory = abrController.getThroughputHistory();
            
            // Example: Get current buffer level
            const bufferLevel =  rulesContext.getDashMetrics().getCurrentBufferLevel(mediaType);
            
            log("Checking index for " + mediaType + ". Buffer: " + bufferLevel);

            // --- YOUR CUSTOM ALGORITHM GOES HERE ---
            
            // Example Logic:
            // For now, let's just return a switch request.
            // Depending on the Dash.js version, the return type might differ.
            // In modern Dash.js, we return a SwitchRequest object.
            
            const switchRequest = dashjs.FactoryMaker.getClassFactoryByName('SwitchRequest')();
            
            // Simple logic: If buffer is low (< 10s), pick lowest quality (index 0).
            // If buffer is healthy, let's try to be aggressive or just return a specific index.
            // Note: Ideally you would calculate the best index based on throughput, buffer, etc.
            
            // To make this "Custom ABR" noticeable, let's force a specific behavior or log it.
            // For demonstration, we will NOT force an index here (returning empty switchRequest usually means "no opinion" or "keep current"),
            // OR we can return a specific index.
            
            // Let's implement a very dummy rule: 
            // If buffer < 5, quality 0. 
            // If buffer > 20, max quality.
            // Otherwise, let other rules decide (priority matters).
            
            // For this example, we will return an empty request to not break things immediately,
            // but we will log that we are running.
            
            // To truly take over, you might want to set a specific quality.
            // switchRequest.quality = 0;
            // switchRequest.reason = 'Custom ABR Rule Decision';
            // switchRequest.priority = SwitchRequest.PRIORITY.STRONG;
            
            return switchRequest;
        }
    };

    return instance;
}

// We need to register this class factory in main.js or expose it
// Since this is a script tag, `CustomAbrRule` is now in global scope.
