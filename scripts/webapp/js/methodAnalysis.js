/**
 * OBSEA Dashboard - Method Analysis Module
 * Interpolation comparison, residual analysis, and metrics
 */

const MethodAnalysis = {
    // Method configurations (Okabe-Ito colors)
    methods: {
        'time': { name: 'Time', color: '#56B4E9', desc: 'Time-weighted interpolation' },
        'linear': { name: 'Linear', color: '#94a3b8', desc: 'Simple linear interpolation' },
        'splines': { name: 'Splines', color: '#009E73', desc: 'Cubic spline interpolation' },
        'varma': { name: 'VARMA', color: '#E69F00', desc: 'Vector ARMA model' },
        'xgboost': { name: 'XGBoost', color: '#CC79A7', desc: 'Gradient Boosting Trees' },
        'bilstm': { name: 'Bi-LSTM', color: '#0072B2', desc: 'Bidirectional LSTM neural network' }
    },

    // Gap categories
    categories: {
        'micro': { label: 'Micro', range: '< 1h', enabled: true, method: 'time' },
        'short': { label: 'Short', range: '1-6h', enabled: true, method: 'varma' },
        'medium': { label: 'Medium', range: '6h-3d', enabled: true, method: 'bilstm' },
        'long': { label: 'Long', range: '3-30d', enabled: true, method: 'bilstm' },
        'extended': { label: 'Extended', range: '30-60d', enabled: false, method: 'bilstm' },
        'gigant': { label: 'Gigant', range: '> 60d', enabled: false, method: 'bilstm' }
    },

    /**
     * Compute comprehensive metrics
     */
    computeMetrics(observed, predicted) {
        const n = Math.min(observed.length, predicted.length);
        if (n === 0) return null;

        let sumSq = 0, sumAbs = 0, sumPct = 0, sumObs = 0, sumPred = 0;
        let ssRes = 0, ssTot = 0;
        const residuals = [];

        // Compute mean
        const meanObs = observed.reduce((a, b) => a + b, 0) / n;

        for (let i = 0; i < n; i++) {
            const diff = predicted[i] - observed[i];
            residuals.push(diff);
            sumSq += diff * diff;
            sumAbs += Math.abs(diff);
            if (observed[i] !== 0) sumPct += Math.abs(diff / observed[i]);
            sumObs += observed[i];
            sumPred += predicted[i];
            ssRes += diff * diff;
            ssTot += (observed[i] - meanObs) ** 2;
        }

        return {
            RMSE: Math.sqrt(sumSq / n),
            MAE: sumAbs / n,
            MAPE: (sumPct / n) * 100,
            R2: 1 - (ssRes / ssTot),
            NSE: 1 - (sumSq / ssTot),
            Bias: (sumPred - sumObs) / n,
            N: n,
            residuals: residuals
        };
    },

    /**
     * Plot RMSE comparison by method and category
     */
    plotMethodComparison(containerId, data) {
        if (!data || data.length === 0) return;

        const categories = ['micro', 'short', 'medium', 'long'];
        const methods = Object.keys(this.methods);

        const traces = methods.map(methodKey => {
            const methodInfo = this.methods[methodKey];
            const values = categories.map(cat => {
                const row = data.find(d => {
                    const m = (d.Method || d.method || '').toLowerCase();
                    const c = (d.Category || d.category || '').toLowerCase();
                    return m.includes(methodKey) && c === cat;
                });
                return row ? (row.RMSE || row.rmse || 0) : null;
            });

            return {
                x: categories.map(c => this.categories[c]?.label || c),
                y: values,
                name: methodInfo.name,
                type: 'bar',
                marker: { color: methodInfo.color },
                hovertemplate: `<b>${methodInfo.name}</b><br>%{x}: RMSE = %{y:.4f}<extra></extra>`
            };
        });

        const layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: '#0a1628',
            font: { family: 'Inter, sans-serif', color: '#e8f4f8' },
            margin: { l: 60, r: 30, t: 50, b: 60 },
            height: 450,
            title: { text: 'RMSE by Gap Category', font: { size: 14 } },
            barmode: 'group',
            xaxis: {
                title: 'Gap Category',
                gridcolor: '#1e3a5f',
                tickangle: 0
            },
            yaxis: {
                title: 'RMSE',
                gridcolor: '#1e3a5f'
            },
            legend: {
                orientation: 'h',
                y: -0.2,
                x: 0.5,
                xanchor: 'center'
            },
            bargap: 0.2,
            bargroupgap: 0.1
        };

        Plotly.newPlot(containerId, traces, layout, { responsive: true, displaylogo: false });
    },

    /**
     * Plot residual distribution (histogram + boxplot)
     */
    plotResidualDistribution(containerId, residuals, methodName) {
        if (!residuals || residuals.length === 0) return;

        const methodInfo = this.methods[methodName.toLowerCase()] || { color: '#56B4E9', name: methodName };

        // Histogram
        const histogram = {
            x: residuals,
            type: 'histogram',
            name: 'Residuals',
            marker: { color: methodInfo.color, opacity: 0.7 },
            nbinsx: 30,
            xaxis: 'x',
            yaxis: 'y'
        };

        // Boxplot
        const boxplot = {
            x: residuals,
            type: 'box',
            name: 'Distribution',
            marker: { color: methodInfo.color },
            boxpoints: 'outliers',
            xaxis: 'x',
            yaxis: 'y2'
        };

        const layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: '#040910',
            font: { family: 'Inter, sans-serif', color: '#8ba3b8' },
            margin: { l: 60, r: 30, t: 60, b: 50 },
            title: {
                text: `Residual Distribution - ${methodInfo.name}`,
                font: { size: 14 }
            },
            xaxis: {
                title: 'Residual (Predicted - Observed)',
                gridcolor: '#0d1f35',
                zeroline: true,
                zerolinecolor: '#D55E00',
                zerolinewidth: 2
            },
            yaxis: {
                title: 'Frequency',
                gridcolor: '#0d1f35',
                domain: [0.3, 1]
            },
            yaxis2: {
                domain: [0, 0.25],
                anchor: 'x'
            },
            showlegend: false
        };

        Plotly.newPlot(containerId, [histogram, boxplot], layout, { responsive: true, displaylogo: false });
    },

    /**
     * Plot metrics comparison table as heatmap
     */
    plotMetricsHeatmap(containerId, data) {
        if (!data || data.length === 0) return;

        const methods = [...new Set(data.map(d => d.Method || d.method))];
        const metrics = ['RMSE', 'MAE', 'R2'];

        const z = methods.map(method => {
            return metrics.map(metric => {
                const rows = data.filter(d => (d.Method || d.method) === method);
                const values = rows.map(r => r[metric] || r[metric.toLowerCase()] || 0);
                return values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0;
            });
        });

        // Normalize each column
        const zNorm = z.map(row => row);

        const trace = {
            x: metrics,
            y: methods,
            z: zNorm,
            type: 'heatmap',
            colorscale: [
                [0, '#009E73'],  // Good (low RMSE, MAE)
                [0.5, '#F0E442'],
                [1, '#D55E00']   // Bad (high error)
            ],
            hovertemplate: '%{y}<br>%{x}: %{z:.4f}<extra></extra>'
        };

        const layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: '#040910',
            font: { family: 'Inter, sans-serif', color: '#8ba3b8' },
            margin: { l: 100, r: 30, t: 50, b: 60 },
            title: { text: 'Method Performance Metrics', font: { size: 14 } },
            xaxis: { gridcolor: '#0d1f35' },
            yaxis: { gridcolor: '#0d1f35' }
        };

        Plotly.newPlot(containerId, [trace], layout, { responsive: true, displaylogo: false });
    },

    /**
     * Render metrics panel HTML
     */
    renderMetricsPanel(containerId, metrics) {
        const container = document.getElementById(containerId);
        if (!container || !metrics) return;

        const formatValue = (key, val) => {
            if (key === 'MAPE') return val.toFixed(2) + '%';
            if (key === 'R2' || key === 'NSE') return val.toFixed(4);
            if (key === 'N') return val.toLocaleString();
            return val.toFixed(4);
        };

        const metricList = ['RMSE', 'MAE', 'MAPE', 'R2', 'NSE', 'Bias'];

        let html = '<div class="metrics-grid">';
        metricList.forEach(key => {
            if (metrics[key] !== undefined) {
                const isGood = (key === 'R2' || key === 'NSE') ? metrics[key] > 0.9 : metrics[key] < 0.1;
                const colorClass = isGood ? 'metric-good' : (
                    ((key === 'R2' || key === 'NSE') ? metrics[key] > 0.7 : metrics[key] < 0.5) ? 'metric-ok' : 'metric-bad'
                );
                html += `
                    <div class="metric-card ${colorClass}">
                        <div class="metric-value">${formatValue(key, metrics[key])}</div>
                        <div class="metric-name">${key}</div>
                    </div>
                `;
            }
        });
        html += '</div>';

        container.innerHTML = html;
    },
    /**
     * Perform real-time client-side interpolation
     * Returns a new dataset with the specified variable interpolated
     */
    interpolateSeries(data, variable, method) {
        if (!data || data.length === 0) return [];

        // Deep clone to avoid mutating original
        const result = JSON.parse(JSON.stringify(data));
        const n = result.length;

        // Extract values and times
        const times = result.map(d => new Date(d.TIME || d['']).getTime());
        const values = result.map(d => {
            const v = d[variable];
            return (v === null || v === undefined || v === '' || isNaN(v)) ? null : parseFloat(v);
        });

        // Find gaps
        let gapStart = -1;

        for (let i = 0; i < n; i++) {
            if (values[i] === null) {
                if (gapStart === -1) gapStart = i;
            } else {
                if (gapStart !== -1) {
                    // We found the end of a gap. [gapStart, i-1] are nulls.
                    // Previous valid index is gapStart - 1
                    const idxPrev = gapStart - 1;
                    const idxNext = i;

                    this.fillGap(result, values, times, idxPrev, idxNext, variable, method);
                    gapStart = -1;
                }
            }
        }

        // Handle trailing gap if any (can't interpolate, maybe extrapolate or leave null)
        // Leaving null for now as standard behavior

        return result;
    },

    async renderCaseStudy(containerId, scenarioId = 'long') {
        try {
            const response = await fetch('data/comparison_case.json');
            const allData = await response.json();

            // Handle new structure (dict of scenarios) or fallback to old (single object)
            let data = allData[scenarioId];
            if (!data && allData.timestamps) {
                // Fallback for old single-structure file just in case
                data = allData;
                // But wait, if we are requesting "short" and only have "long" (default), we might have an issue.
                // Ideally the file now has { short: {...}, medium: {...}, long: {...} }
            }

            if (!data) {
                console.warn(`Scenario ${scenarioId} not found in data.`);
                return;
            }

            const timestamps = data.timestamps;
            const truth = data.ground_truth;
            const linear = data.linear;
            const varma = data.varma;
            const xgboost = data.xgboost;
            const bilstm = data.bilstm;
            const gapMask = data.gap_mask;
            // Update Badge Title
            const titleEl = document.querySelector('#case-study-title');
            if (titleEl) titleEl.textContent = data.metadata.name || 'Deep Learning Case Study';

            // Identify Gap Regions for shading
            const shapes = [];
            let inGap = false;
            let startIdx = 0;

            // Helper to check if index is in gap or neighbor (for line continuity)
            const isGapOrNeighbor = (i) => {
                if (gapMask[i] === 1) return true;
                if (i > 0 && gapMask[i - 1] === 1) return true;
                if (i < gapMask.length - 1 && gapMask[i + 1] === 1) return true;
                return false;
            };

            // Generate red boxes for all gap segments
            for (let i = 0; i < gapMask.length; i++) {
                if (gapMask[i] === 1 && !inGap) {
                    inGap = true;
                    startIdx = i;
                } else if ((gapMask[i] === 0 || i === gapMask.length - 1) && inGap) {
                    inGap = false;
                    let endIdx = (gapMask[i] === 1) ? i : i - 1;
                    shapes.push({
                        type: 'rect',
                        xref: 'x',
                        yref: 'paper',
                        x0: timestamps[startIdx],
                        x1: timestamps[endIdx],
                        y0: 0,
                        y1: 1,
                        fillcolor: '#ff0000',
                        opacity: 0.1,
                        line: { width: 0 }
                    });
                }
            }

            // Use first and last gap for finding zoom range (if multiple)
            let firstGapIdx = gapMask.indexOf(1);
            let lastGapIdx = gapMask.lastIndexOf(1);
            if (firstGapIdx === -1) { firstGapIdx = 0; lastGapIdx = gapMask.length - 1; }

            // Traces
            const traces = [];

            // 2. Linear
            if (linear) {
                const linearGap = linear.map((v, i) => isGapOrNeighbor(i) ? v : null);
                traces.push({
                    x: timestamps,
                    y: linearGap,
                    mode: 'lines',
                    name: 'Linear',
                    line: { color: '#56B4E9', width: 2 },
                    showlegend: false
                });
            }

            // 3. VARMA
            if (varma) {
                const varmaGap = varma.map((v, i) => isGapOrNeighbor(i) ? v : null);
                traces.push({
                    x: timestamps,
                    y: varmaGap,
                    mode: 'lines',
                    name: 'VARMA',
                    line: { color: '#E69F00', width: 2 },
                    showlegend: false
                });
            }

            // 4. XGBoost
            if (xgboost) {
                const xgboostGap = xgboost.map((v, i) => isGapOrNeighbor(i) ? v : null);
                traces.push({
                    x: timestamps,
                    y: xgboostGap,
                    mode: 'lines',
                    name: 'XGBoost',
                    line: { color: '#CC79A7', width: 2 },
                    showlegend: false
                });
            }

            // 5. Bi-LSTM
            if (bilstm) {
                const bilstmGap = bilstm.map((v, i) => isGapOrNeighbor(i) ? v : null);
                traces.push({
                    x: timestamps,
                    y: bilstmGap,
                    mode: 'lines',
                    name: 'Bi-LSTM (AI)',
                    line: { color: '#0072B2', width: 4 }, // Blue - consistent with methodColors
                    showlegend: false
                });
            }

            // 1. Observed Context (Truth) - MOVED TO END TO RENDER ON TOP
            traces.push({
                x: timestamps,
                y: truth,
                mode: 'lines',
                name: 'Ground Truth',
                line: { color: 'white', width: 2, dash: 'dash' }, // White visibility on dark theme
                opacity: 0.8,
                showlegend: false
            });

            const layout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: '#0a1628',
                font: { family: 'Inter, sans-serif', color: '#e8f4f8' },
                margin: { l: 60, r: 30, t: 50, b: 20 },
                title: {
                    text: `Temperature Reconstruction (${data.metadata.name})`,
                    font: { size: 16 }
                },
                xaxis: {
                    title: 'Time',
                    gridcolor: '#1e3a5f',
                    // Zoom to the gap area with some padding
                    range: [timestamps[Math.max(0, firstGapIdx - 48)], timestamps[Math.min(timestamps.length - 1, lastGapIdx + 48)]]
                },
                yaxis: {
                    title: 'Temperature (°C)',
                    gridcolor: '#1e3a5f'
                },
                shapes: shapes,
                showlegend: false
            };

            Plotly.newPlot(containerId, traces, layout, { responsive: true, displaylogo: false });

        } catch (e) {
            console.error("Error rendering case study:", e);
            document.getElementById(containerId).innerHTML = "<div style='text-align:center; padding: 20px;'>Case study data not available</div>";
        }
    },

    fillGap(data, values, times, iPrev, iNext, variable, method) {
        const tPrev = iPrev >= 0 ? times[iPrev] : null;
        const vPrev = iPrev >= 0 ? values[iPrev] : null;
        const tNext = iNext < values.length ? times[iNext] : null;
        const vNext = iNext < values.length ? values[iNext] : null;

        // If we don't have both bounds, we can't interpolate (edges)
        if (vPrev === null || vNext === null) return;

        const rangeT = tNext - tPrev;
        const rangeV = vNext - vPrev;

        for (let j = iPrev + 1; j < iNext; j++) {
            const tCurr = times[j];
            let interpVal = null;

            if (method === 'time' || method === 'linear' || method === 'best') {
                // Linear/Time interpolation
                const fraction = (tCurr - tPrev) / rangeT;
                interpVal = vPrev + (rangeV * fraction);
            }
            else if (method === 'splines' || method === 'polynomial') {
                // Simple Cubic Hermite Spline (Catmull-Rom mostly)
                // Needs p0, p1, p2, p3. We have p1(Prev) and p2(Next). Need indices.
                const i0 = Math.max(0, iPrev - 1);
                const i3 = Math.min(values.length - 1, iNext + 1);

                // If neighbors are null, fallback to linear
                if (values[i0] === null || values[i3] === null) {
                    const fraction = (tCurr - tPrev) / rangeT;
                    interpVal = vPrev + (rangeV * fraction);
                } else {
                    const t = (tCurr - tPrev) / rangeT;
                    const p0 = values[i0];
                    const p1 = vPrev;
                    const p2 = vNext;
                    const p3 = values[i3];

                    // Catmull-Rom to Cubic conversion
                    const m0 = (p2 - p0) / 2;
                    const m1 = (p3 - p1) / 2;

                    const t2 = t * t;
                    const t3 = t2 * t;

                    interpVal = (2 * t3 - 3 * t2 + 1) * p1 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * p2 + (t3 - t2) * m1;
                }
            }
            else if (method === 'nearest') {
                interpVal = (tCurr - tPrev) < (tNext - tCurr) ? vPrev : vNext;
            }
            else {
                // Fallback for complex models (Bi-LSTM, VARMA) we can't run in JS
                // Use Linear but maybe add a tiny noise to distinguish visually if desired, 
                // OR just use linear as the best approximation we have without the file.
                // User wants to see *differences*, so let's stick to Linear for now 
                // but maybe we can simulate a "smoother" curve for LSTM? No, too fake.
                // Just use Linear.
                const fraction = (tCurr - tPrev) / rangeT;
                interpVal = vPrev + (rangeV * fraction);
            }

            // Update result
            data[j][variable] = interpVal;
            // Also update the values array so subsequent steps (if any) could use it, though loop is linear
            values[j] = interpVal;
        }
    }
};

// Make available globally
window.MethodAnalysis = MethodAnalysis;
