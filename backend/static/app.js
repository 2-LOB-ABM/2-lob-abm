// WebSocket connection
const socket = io({
    transports: ['websocket', 'polling'],
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionAttempts: 5
});

// Socket event handlers
socket.on('connect', () => {
    updateStatus('connected', false);
});

socket.on('disconnect', () => {
    updateStatus('disconnected', false);
});

socket.on('connect_error', (error) => {
    console.error('Socket connection error:', error);
});

socket.on('connected', () => {
    // Connection confirmed
});

// Register simulation_step handler
socket.on('simulation_step', (data) => {
    if (!data) {
        console.error('Received empty data!');
        return;
    }
    
    // Update metrics immediately
    try {
        if (typeof updateMetrics === 'function') {
            updateMetrics(data);
        }
    } catch (e) {
        console.error('Error updating metrics:', e);
    }
    
    // Update charts incrementally
    try {
        if (typeof updateCharts === 'function') {
            updateCharts(data);
        }
    } catch (e) {
        console.error('Error updating charts:', e);
    }
    
    // Update options and hyperparameters less frequently
    if (data.step % 10 === 0) {
        try {
            if (typeof updateOptions === 'function') {
                updateOptions(data);
            }
            if (typeof updateHyperparameters === 'function') {
                updateHyperparameters(data);
            }
        } catch (e) {
            console.error('Error updating options/hyperparams:', e);
        }
    }
    
    // Update trades tables
    try {
        if (typeof updateTrades === 'function') {
            updateTrades(data);
        }
    } catch (e) {
        console.error('Error updating trades:', e);
    }
});

// Chart instances
let charts = {};
let dataBuffers = {
    prices: [],
    spreads: [],
    regime: [],
    volume: [],
    modelDistBS: [],
    modelDistTFBS: [],
    modelDistHESTON: [],
    hedgeErrorBS: [],
    hedgeErrorTFBS: [],
    hedgeErrorHESTON: [],
    optionSpreads: {},
    optionDepths: {},
};

const MAX_DATA_POINTS = 500;

// Data storage for incremental updates
let chartData = {
    prices: [],
    spreads: [],
    regime: [],
    volume: [],
    modelDistBS: [],
    modelDistTFBS: [],
    modelDistHESTON: [],
    hedgeErrorBS: [],
    hedgeErrorTFBS: [],
    hedgeErrorHESTON: [],
};

// Trades storage - keep last N trades
const MAX_TRADES_DISPLAY = 100;
let spotTrades = [];
let optionsTrades = [];

// Throttling for chart updates - update every N steps or every X ms
let lastChartUpdate = 0;
let chartUpdateCounter = 0;
const CHART_UPDATE_INTERVAL = 16; // ~60 FPS (16ms)
const CHART_UPDATE_STEPS = 1; // Update every step (but throttle by time)
let chartUpdateScheduled = false;

// Initialize charts
function initCharts() {
    // Check if Chart.js is loaded
    if (typeof Chart === 'undefined') {
        console.error('Chart.js is not loaded!');
        setTimeout(initCharts, 100); // Retry after 100ms
        return;
    }
    
    // Register zoom plugin if available
    try {
        if (typeof Chart !== 'undefined') {
            // Try multiple ways to access zoom plugin
            if (typeof zoomPlugin !== 'undefined') {
                Chart.register(zoomPlugin);
                console.log('Zoom plugin registered via zoomPlugin');
            } else if (window.zoomPlugin) {
                Chart.register(window.zoomPlugin);
                console.log('Zoom plugin registered via window.zoomPlugin');
            } else if (typeof ChartZoom !== 'undefined') {
                Chart.register(ChartZoom);
                console.log('Zoom plugin registered via ChartZoom');
            } else {
                console.warn('Zoom plugin not found, panning will not work');
            }
        }
    } catch (e) {
        console.error('Plugin registration failed:', e);
    }
    
    // Check if zoom plugin is registered
    const zoomEnabled = window.zoomPluginRegistered !== false;
    
    // Create base plugins object
    const basePlugins = {
        legend: {
            labels: {
                color: 'rgba(255, 255, 255, 0.7)',
                font: {
                    size: 11
                },
                usePointStyle: true,
                padding: 12
            }
        },
        tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: 'rgba(15, 20, 25, 0.95)',
            titleColor: 'rgba(255, 255, 255, 0.9)',
            bodyColor: 'rgba(255, 255, 255, 0.7)',
            borderColor: 'rgba(255, 255, 255, 0.1)',
            borderWidth: 1,
            padding: 12,
            cornerRadius: 8
        }
    };
    
    // Add zoom plugin if available
    if (zoomEnabled) {
        basePlugins.zoom = {
            pan: {
                enabled: true,
                mode: 'x',
                modifierKey: null,
                threshold: 0,
                speed: 1,
                onPanStart: function({chart, event}) {
                    return true;
                },
                onPan: function({chart}) {
                    // Panning in progress
                },
                onPanComplete: function({chart}) {
                    // Pan completed
                }
            },
            zoom: {
                wheel: { enabled: false },
                pinch: { enabled: false },
                drag: {
                    enabled: false,
                    mode: 'x',
                    modifierKey: null,
                    threshold: 5
                },
                mode: 'x',
                limits: {
                    x: { min: 0, max: undefined }  // Allow max to be updated dynamically
                }
            }
        };
    }
    
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        layout: {
            padding: {
                left: 0,
                right: 0,
                top: 0,
                bottom: 0
            }
        },
        plugins: basePlugins,
        scales: {
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.03)'
                },
                ticks: {
                    color: 'rgba(255, 255, 255, 0.4)',
                    font: {
                        size: 11
                    },
                    maxRotation: 45,
                    minRotation: 0
                },
                border: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                min: 0
            },
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.03)'
                },
                ticks: {
                    color: 'rgba(255, 255, 255, 0.4)',
                    font: {
                        size: 11
                    }
                },
                border: {
                    color: 'rgba(255, 255, 255, 0.1)'
                }
            }
        },
        interaction: {
            intersect: false,
            mode: 'index',
            includeInvisible: false
        },
        onHover: function(event, activeElements) {
            // Allow panning even when hovering
        }
    };

    // Price chart
    const priceCtx = document.getElementById('priceChart');
    if (!priceCtx) {
        console.error('priceChart element not found!');
        return;
    }
    
    charts.price = new Chart(priceCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Spot Price',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 1.5,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 4
            }]
        },
        options: chartOptions
    });
    
    // Price chart initialized silently

    // Regime chart
    const regimeCtx = document.getElementById('regimeChart');
    if (!regimeCtx) {
        console.error('regimeChart element not found!');
        return;
    }
    
    charts.regime = new Chart(regimeCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Regime (0=Calm, 1=Stress)',
                data: [],
                borderColor: 'rgba(255, 255, 255, 0.6)',
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 1.5,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 4
            }]
        },
        options: chartOptions
    });
    
    // Regime chart initialized silently

    // Spread chart
    const spreadCtx = document.getElementById('spreadChart');
    if (!spreadCtx) {
        console.error('spreadChart element not found!');
        return;
    }
    
    charts.spread = new Chart(spreadCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Bid-Ask Spread',
                data: [],
                borderColor: 'rgba(59, 130, 246, 0.7)',
                backgroundColor: 'rgba(59, 130, 246, 0.08)',
                borderWidth: 1.5,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 4
            }]
        },
        options: chartOptions
    });
    
    // Spread chart initialized silently

    // Volume chart
    const volumeCtx = document.getElementById('volumeChart');
    if (!volumeCtx) {
        console.error('volumeChart element not found!');
        return;
    }
    
    charts.volume = new Chart(volumeCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Volume',
                data: [],
                backgroundColor: 'rgba(59, 130, 246, 0.4)',
                borderColor: 'rgba(59, 130, 246, 0.6)',
                borderWidth: 0
            }]
        },
        options: chartOptions
    });
    
    // Volume chart initialized silently

    // Model distribution chart
    const modelDistCtx = document.getElementById('modelDistChart');
    if (!modelDistCtx) {
        console.error('modelDistChart element not found!');
        return;
    }
    
    charts.modelDist = new Chart(modelDistCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'BS',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 1.5,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4
                },
                {
                    label: 'TFBS',
                    data: [],
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.08)',
                    borderWidth: 1.5,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4
                },
                {
                    label: 'HESTON',
                    data: [],
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.06)',
                    borderWidth: 1.5,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4
                }
            ]
        },
        options: chartOptions
    });

    // Model distribution chart initialized silently

    // Hedging errors chart
    const hedgeErrorCtx = document.getElementById('hedgeErrorChart');
    if (!hedgeErrorCtx) {
        console.error('hedgeErrorChart element not found!');
        return;
    }
    
    charts.hedgeError = new Chart(hedgeErrorCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'BS Error',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 1.5,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4
                },
                {
                    label: 'TFBS Error',
                    data: [],
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.08)',
                    borderWidth: 1.5,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4
                },
                {
                    label: 'HESTON Error',
                    data: [],
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.06)',
                    borderWidth: 1.5,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4
                }
            ]
        },
        options: {
            ...chartOptions,
            scales: {
                ...chartOptions.scales,
                y: {
                    ...chartOptions.scales.y,
                    type: 'logarithmic'
                }
            }
        }
    });
    
    // All charts initialized silently
    
    // Add scroll synchronization for touchpad gestures
    if (zoomEnabled) {
        const chartContainers = document.querySelectorAll('.chart-container');
        chartContainers.forEach((container, index) => {
            const chart = Object.values(charts)[index];
            if (!chart) return;
            
            // Handle scroll events (touchpad gestures)
            container.addEventListener('wheel', function(e) {
                // Check if horizontal scroll
                if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
                    e.preventDefault();
                    const xScale = chart.scales.x;
                    if (!xScale) return;
                    
                    const currentMin = xScale.min;
                    const currentMax = xScale.max;
                    const range = currentMax - currentMin;
                    const scrollRatio = -e.deltaX / container.offsetWidth;
                    const newRange = range * (1 + scrollRatio * 0.1);
                    const center = (currentMin + currentMax) / 2;
                    
                    const newMin = Math.max(0, center - newRange / 2);
                    const newMax = Math.min(chart.data.labels.length, center + newRange / 2);
                    
                    xScale.options.min = newMin;
                    xScale.options.max = newMax;
                    chart.update('none');
                }
            }, { passive: false });
        });
    }
}

// Update metrics display
function updateMetrics(data) {
    const metricsContainer = document.getElementById('metrics');
    const spot = data.spot;
    const regime = data.regime;
    
    metricsContainer.innerHTML = `
        <div class="metric">
            <div class="metric-label">Step</div>
            <div class="metric-value">${data.step}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Price</div>
            <div class="metric-value">${spot.price.toFixed(2)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Spread</div>
            <div class="metric-value">${spot.spread.toFixed(4)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Regime</div>
            <div class="metric-value ${regime.regime === 1 ? 'regime-stress' : 'regime-calm'}">${regime.regime_label}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Volatility</div>
            <div class="metric-value">${(spot.volatility * 100).toFixed(2)}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">Volume</div>
            <div class="metric-value">${spot.volume.toFixed(0)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Trades</div>
            <div class="metric-value">${spot.trades}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Depth (Bid/Ask)</div>
            <div class="metric-value">${spot.depth_bid.toFixed(0)} / ${spot.depth_ask.toFixed(0)}</div>
        </div>
    `;
}

// Fast chart update function - optimized
function _updateChartsFast() {
    if (!charts.price || chartData.prices.length === 0) return;
    
    try {
        const len = chartData.prices.length;
        
        // Update all charts at once - faster than individual updates
        // Price chart
        charts.price.data.labels = Array.from({length: len}, (_, i) => i);
        charts.price.data.datasets[0].data = chartData.prices;
        // Ensure x-axis shows all steps - always update max to show all data
        charts.price.options.scales.x.max = Math.max(len - 1, 0);
        
        // Regime chart
        const regimeLen = chartData.regime.length;
        charts.regime.data.labels = Array.from({length: regimeLen}, (_, i) => i);
        charts.regime.data.datasets[0].data = chartData.regime;
        charts.regime.options.scales.x.max = Math.max(regimeLen - 1, 0);
        
        // Spread chart
        const spreadLen = chartData.spreads.length;
        charts.spread.data.labels = Array.from({length: spreadLen}, (_, i) => i);
        charts.spread.data.datasets[0].data = chartData.spreads;
        charts.spread.options.scales.x.max = Math.max(spreadLen - 1, 0);
        
        // Volume chart
        const volumeLen = chartData.volume.length;
        charts.volume.data.labels = Array.from({length: volumeLen}, (_, i) => i);
        charts.volume.data.datasets[0].data = chartData.volume;
        charts.volume.options.scales.x.max = Math.max(volumeLen - 1, 0);
        
        // Model distribution chart
        if (chartData.modelDistBS.length > 0) {
            const modelDistLen = chartData.modelDistBS.length;
            charts.modelDist.data.labels = Array.from({length: modelDistLen}, (_, i) => i);
            charts.modelDist.data.datasets[0].data = chartData.modelDistBS;
            charts.modelDist.data.datasets[1].data = chartData.modelDistTFBS;
            charts.modelDist.data.datasets[2].data = chartData.modelDistHESTON;
            charts.modelDist.options.scales.x.max = Math.max(modelDistLen - 1, 0);
        }
        
        // Hedging errors chart
        if (chartData.hedgeErrorBS.length > 0) {
            const hedgeErrorLen = chartData.hedgeErrorBS.length;
            charts.hedgeError.data.labels = Array.from({length: hedgeErrorLen}, (_, i) => i);
            charts.hedgeError.data.datasets[0].data = chartData.hedgeErrorBS;
            charts.hedgeError.data.datasets[1].data = chartData.hedgeErrorTFBS;
            charts.hedgeError.data.datasets[2].data = chartData.hedgeErrorHESTON;
            charts.hedgeError.options.scales.x.max = Math.max(hedgeErrorLen - 1, 0);
        }
        
        // Update all charts in one batch
        // Note: Chart.js should automatically apply scale changes
        charts.price.update('none');
        charts.regime.update('none');
        charts.spread.update('none');
        charts.volume.update('none');
        if (chartData.modelDistBS.length > 0) charts.modelDist.update('none');
        if (chartData.hedgeErrorBS.length > 0) charts.hedgeError.update('none');
        
    } catch (error) {
        console.error('Error updating charts:', error);
    }
    
    chartUpdateScheduled = false;
}

// Update charts incrementally - add one point at a time
function updateCharts(data) {
    const ts = data.time_series;
    
    // Check if charts are initialized
    if (!charts.price || !charts.regime || !charts.spread || !charts.volume || 
        !charts.modelDist || !charts.hedgeError) {
        return; // Charts not ready yet
    }
    
    // Add new data points immediately (fast)
    if (ts.price !== undefined) {
        chartData.prices.push(ts.price);
        if (chartData.prices.length > MAX_DATA_POINTS) chartData.prices.shift();
    }
    
    if (ts.spread !== undefined) {
        chartData.spreads.push(ts.spread);
        if (chartData.spreads.length > MAX_DATA_POINTS) chartData.spreads.shift();
    }
    
    if (ts.regime !== undefined) {
        chartData.regime.push(ts.regime);
        if (chartData.regime.length > MAX_DATA_POINTS) chartData.regime.shift();
    }
    
    if (ts.volume !== undefined) {
        chartData.volume.push(ts.volume);
        if (chartData.volume.length > MAX_DATA_POINTS) chartData.volume.shift();
    }
    
    if (ts.model_dist_BS !== undefined) {
        chartData.modelDistBS.push(ts.model_dist_BS);
        chartData.modelDistTFBS.push(ts.model_dist_TFBS || 0);
        chartData.modelDistHESTON.push(ts.model_dist_HESTON || 0);
        if (chartData.modelDistBS.length > MAX_DATA_POINTS) {
            chartData.modelDistBS.shift();
            chartData.modelDistTFBS.shift();
            chartData.modelDistHESTON.shift();
        }
    }
    
    if (ts.hedge_error_BS !== undefined) {
        chartData.hedgeErrorBS.push(ts.hedge_error_BS || 0);
        chartData.hedgeErrorTFBS.push(ts.hedge_error_TFBS || 0);
        chartData.hedgeErrorHESTON.push(ts.hedge_error_HESTON || 0);
        if (chartData.hedgeErrorBS.length > MAX_DATA_POINTS) {
            chartData.hedgeErrorBS.shift();
            chartData.hedgeErrorTFBS.shift();
            chartData.hedgeErrorHESTON.shift();
        }
    }
    
    // Throttle chart updates - update max 60 FPS
    const now = Date.now();
    chartUpdateCounter++;
    
    const shouldUpdate = (now - lastChartUpdate) >= CHART_UPDATE_INTERVAL || 
                         chartUpdateCounter >= CHART_UPDATE_STEPS;
    
    if (shouldUpdate && !chartUpdateScheduled) {
        chartUpdateScheduled = true;
        lastChartUpdate = now;
        chartUpdateCounter = 0;
        requestAnimationFrame(_updateChartsFast);
    }
}

// Update options display
function updateOptions(data) {
    const container = document.getElementById('optionsContainer');
    const options = data.options || {};
    
    if (Object.keys(options).length === 0) {
        container.innerHTML = '<p style="color: rgba(255, 255, 255, 0.4); text-align: center; padding: 20px;">No options data available</p>';
        return;
    }
    
    let html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px;">';
    
    for (const [key, opt] of Object.entries(options)) {
        html += `
            <div style="background: rgba(255, 255, 255, 0.02); padding: 16px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.06);">
                <div style="font-weight: 500; margin-bottom: 12px; color: rgba(255, 255, 255, 0.9); font-size: 14px;">
                    Contract ${opt.contract_id} - ${opt.type.toUpperCase()} @ ${opt.strike.toFixed(2)}
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 13px;">
                    <div>
                        <div style="color: rgba(255, 255, 255, 0.5); margin-bottom: 4px;">Mid Price</div>
                        <div style="color: rgba(255, 255, 255, 0.9); font-weight: 500;">${opt.mid_price ? opt.mid_price.toFixed(4) : 'N/A'}</div>
                    </div>
                    <div>
                        <div style="color: rgba(255, 255, 255, 0.5); margin-bottom: 4px;">Spread</div>
                        <div style="color: rgba(255, 255, 255, 0.9); font-weight: 500;">${opt.spread ? opt.spread.toFixed(4) : 'N/A'}</div>
                    </div>
                    <div>
                        <div style="color: rgba(255, 255, 255, 0.5); margin-bottom: 4px;">Depth Bid</div>
                        <div style="color: rgba(255, 255, 255, 0.9); font-weight: 500;">${opt.depth_bid.toFixed(0)}</div>
                    </div>
                    <div>
                        <div style="color: rgba(255, 255, 255, 0.5); margin-bottom: 4px;">Depth Ask</div>
                        <div style="color: rgba(255, 255, 255, 0.9); font-weight: 500;">${opt.depth_ask.toFixed(0)}</div>
                    </div>
                    <div>
                        <div style="color: rgba(255, 255, 255, 0.5); margin-bottom: 4px;">Maturity</div>
                        <div style="color: rgba(255, 255, 255, 0.9); font-weight: 500;">${opt.maturity.toFixed(4)}</div>
                    </div>
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    container.innerHTML = html;
}

// Update hyperparameters display
function updateHyperparameters(data) {
    const container = document.getElementById('hyperparameters');
    const params = data.hyperparameters || {};
    
    let html = '';
    for (const [key, value] of Object.entries(params)) {
        html += `
            <div class="param">
                <span class="param-label">${key}</span>
                <span class="param-value">${typeof value === 'number' ? value.toFixed(4) : value}</span>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

// Update trades tables
function updateTrades(data) {
    // Helper function to check if trade already exists
    const tradeExists = (trades, newTrade) => {
        return trades.some(t => 
            t.step === newTrade.step && 
            t.price === newTrade.price && 
            t.qty === newTrade.qty &&
            t.passive_id === newTrade.passive_id &&
            t.aggressive_id === newTrade.aggressive_id
        );
    };
    
    // Add new spot trades (avoid duplicates)
    if (data.spot_trades && Array.isArray(data.spot_trades)) {
        for (const trade of data.spot_trades) {
            if (!tradeExists(spotTrades, trade)) {
                spotTrades.unshift(trade); // Add to beginning
                if (spotTrades.length > MAX_TRADES_DISPLAY) {
                    spotTrades.pop(); // Remove oldest
                }
            }
        }
    }
    
    // Add new options trades (avoid duplicates)
    if (data.options_trades && Array.isArray(data.options_trades)) {
        for (const trade of data.options_trades) {
            const tradeWithContract = {
                ...trade,
                key: `${trade.step}-${trade.contract_id}-${trade.price}-${trade.qty}-${trade.passive_id}-${trade.aggressive_id}`
            };
            const exists = optionsTrades.some(t => 
                t.step === trade.step && 
                t.contract_id === trade.contract_id &&
                t.price === trade.price && 
                t.qty === trade.qty &&
                t.passive_id === trade.passive_id &&
                t.aggressive_id === trade.aggressive_id
            );
            if (!exists) {
                optionsTrades.unshift(trade); // Add to beginning
                if (optionsTrades.length > MAX_TRADES_DISPLAY) {
                    optionsTrades.pop(); // Remove oldest
                }
            }
        }
    }
    
    // Update spot trades table
    const spotTradesBody = document.getElementById('spotTradesBody');
    if (spotTradesBody) {
        if (spotTrades.length === 0) {
            spotTradesBody.innerHTML = `
                <tr>
                    <td colspan="5" style="text-align: center; color: rgba(255, 255, 255, 0.4); padding: 20px;">
                        No trades yet
                    </td>
                </tr>
            `;
        } else {
            let html = '';
            for (const trade of spotTrades) {
                html += `
                    <tr>
                        <td>${trade.step}</td>
                        <td class="trade-price">${trade.price.toFixed(4)}</td>
                        <td class="trade-qty">${trade.qty}</td>
                        <td>${trade.passive_id}</td>
                        <td>${trade.aggressive_id}</td>
                    </tr>
                `;
            }
            spotTradesBody.innerHTML = html;
        }
    }
    
    // Update options trades table
    const optionsTradesBody = document.getElementById('optionsTradesBody');
    if (optionsTradesBody) {
        if (optionsTrades.length === 0) {
            optionsTradesBody.innerHTML = `
                <tr>
                    <td colspan="6" style="text-align: center; color: rgba(255, 255, 255, 0.4); padding: 20px;">
                        No trades yet
                    </td>
                </tr>
            `;
        } else {
            let html = '';
            for (const trade of optionsTrades) {
                html += `
                    <tr>
                        <td>${trade.step}</td>
                        <td>${trade.contract_id}</td>
                        <td class="trade-price">${trade.price.toFixed(4)}</td>
                        <td class="trade-qty">${trade.qty}</td>
                        <td>${trade.passive_id}</td>
                        <td>${trade.aggressive_id}</td>
                    </tr>
                `;
            }
            optionsTradesBody.innerHTML = html;
        }
    }
}

// WebSocket event handlers (duplicate removed - handler registered at top of file)

socket.on('simulation_started', () => {
    updateStatus('running', true);
    const startBtn = document.getElementById('startBtn');
    if (startBtn) startBtn.disabled = false;
});

socket.on('simulation_stopped', () => {
    updateStatus('stopped', false);
    const startBtn = document.getElementById('startBtn');
    if (startBtn) startBtn.disabled = false;
});

socket.on('simulation_paused', (data) => {
    updateStatus(data.status === 'paused' ? 'paused' : 'running', data.status !== 'paused');
});

socket.on('simulation_complete', () => {
    updateStatus('completed', false);
    const startBtn = document.getElementById('startBtn');
    if (startBtn) startBtn.disabled = false;
});

socket.on('error', (data) => {
    console.error('Error:', data.message);
    alert('Error: ' + data.message);
    // Re-enable buttons on error
    const startBtn = document.getElementById('startBtn');
    if (startBtn) startBtn.disabled = false;
});

socket.on('simulation_error', (data) => {
    console.error('Simulation error:', data.error);
    alert('Simulation error: ' + data.error);
    // Re-enable buttons on error
    const startBtn = document.getElementById('startBtn');
    if (startBtn) startBtn.disabled = false;
});

// Update status indicator
function updateStatus(status, running) {
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    
    dot.className = 'status-dot';
    if (running) {
        dot.classList.add('running');
        text.textContent = 'Running';
    } else if (status === 'paused') {
        text.textContent = 'Paused';
    } else if (status === 'completed') {
        text.textContent = 'Completed';
    } else {
        text.textContent = 'Stopped';
    }
}

// Control functions
function startSimulation(event) {
    // Prevent multiple starts
    if (socket.connected === false) {
        alert('Not connected to server. Please wait...');
        return;
    }
    
    // Reset chart data and update counters
    chartData = {
        prices: [],
        spreads: [],
        regime: [],
        volume: [],
        modelDistBS: [],
        modelDistTFBS: [],
        modelDistHESTON: [],
        hedgeErrorBS: [],
        hedgeErrorTFBS: [],
        hedgeErrorHESTON: [],
    };
    spotTrades = [];
    optionsTrades = [];
    lastChartUpdate = 0;
    chartUpdateCounter = 0;
    chartUpdateScheduled = false;
    
    // Reset x-axis max for all charts to show all steps
    // Also reset zoom if zoom plugin is available
    const resetChartAxis = (chart) => {
        if (!chart) return;
        chart.options.scales.x.max = undefined;
        chart.options.scales.x.min = 0;
        // Reset zoom if zoom plugin is available
        if (chart.resetZoom && typeof chart.resetZoom === 'function') {
            try {
                chart.resetZoom();
            } catch (e) {
                // Zoom reset failed, continue anyway
            }
        }
    };
    
    resetChartAxis(charts.price);
    resetChartAxis(charts.regime);
    resetChartAxis(charts.spread);
    resetChartAxis(charts.volume);
    resetChartAxis(charts.modelDist);
    resetChartAxis(charts.hedgeError);
    
    // Disable button to prevent double-click
    const startBtn = event?.target || document.querySelector('button.btn-primary');
    if (startBtn) {
        startBtn.disabled = true;
        setTimeout(() => { 
            if (startBtn) startBtn.disabled = false; 
        }, 1000);
    }
    
    const config = {
        S0: 100.0,
        dt: 0.001,
        steps: 1000,
        n_fund: 10,
        n_chart: 10,
        n_mm: 3,
        n_noise: 50,
        fundamental_price: 100.0,
        tick_size: 0.1,
        enable_options: true,
        n_option_dealers: 5,
        n_option_contracts: 3,
        n_option_takers: 10,
        enable_model_switching: true,
        dealer_model_distribution: {
            BS: 0.33,
            TFBS: 0.33,
            HESTON: 0.34
        },
        p01: 0.005,
        p10: 0.03,
        shock_rate: 0.003,
    };
    
    socket.emit('start_simulation', config);
}

function stopSimulation(event) {
    // Reset chart update counter
    chartUpdateCounter = 0;
    socket.emit('stop_simulation');
}

function pauseSimulation(event) {
    socket.emit('pause_simulation');
}

// Register simulation_step handler BEFORE DOM loads
// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    try {
        initCharts();
    } catch (error) {
        console.error('Error initializing charts:', error);
    }
});

