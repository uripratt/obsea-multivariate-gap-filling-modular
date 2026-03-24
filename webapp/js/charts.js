/**
 * OBSEA Dashboard - Charts Module (Premium Version)
 * Plotly.js chart configurations with oceanographic industrial theme
 */

const Charts = {
    // Design System Colors (Synced with CSS)
    colors: {
        primary: '#56B4E9',
        secondary: '#009E73',
        accent: '#E69F00',
        danger: '#D55E00',
        highlight: '#CC79A7',
        gray: '#94a3b8',
        bg: 'rgba(13, 20, 38, 0.4)',
        grid: 'rgba(255, 255, 255, 0.05)',
        text: '#f8fafc'
    },

    // Method colors (distinct Okabe-Ito colors)
    methodColors: {
        'linear': '#94a3b8',
        'time': '#56B4E9',
        'splines': '#009E73',
        'polynomial': '#CC79A7',
        'var': '#F0E442',
        'varma': '#E69F00',
        'bilstm': '#0072B2',
        'bi-lstm': '#0072B2',
        'best': '#009E73',
        'saits': '#2CA02C',
        'brits_pro': '#D55E00',
        'brits-pro': '#D55E00'
    },

    /**
     * Get consistent color for a given interpolation method
     */
    getMethodColor(method) {
        if (!method) return this.colors.primary;
        const key = method.toLowerCase().trim();
        return this.methodColors[key] || this.colors.primary;
    },

    // Global Plotly Defaults
    getLayout(title = '', options = {}) {
        return {
            title: {
                text: title,
                font: { color: this.colors.text, size: 16, weight: '600' },
                x: 0.05,
                xanchor: 'left'
            },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                family: 'Outfit, sans-serif',
                color: this.colors.text
            },
            margin: { l: 60, r: 30, t: title ? 60 : 20, b: 60 },
            autosize: true,
            xaxis: {
                gridcolor: this.colors.grid,
                linecolor: this.colors.grid,
                zerolinecolor: 'transparent',
                tickfont: { size: 11, color: this.colors.gray },
                ...options.xaxis
            },
            yaxis: {
                gridcolor: this.colors.grid,
                linecolor: this.colors.grid,
                zerolinecolor: 'transparent',
                tickfont: { size: 11, color: this.colors.gray },
                ...options.yaxis
            },
            legend: {
                bgcolor: 'transparent',
                font: { color: this.colors.text, size: 11 },
                orientation: 'h',
                x: 0.5,
                xanchor: 'center',
                y: -0.2
            },
            hoverlabel: {
                bgcolor: '#1a2b4b',
                bordercolor: this.colors.primary,
                font: { family: 'JetBrains Mono, monospace', size: 12, color: '#fff' }
            },
            ...options
        };
    },

    /**
     * Missing Data Analysis (Overview)
     */
    plotMissingData(containerId, data) {
        const variables = data.map(d => d.Variable || d.variable);
        const missing = data.map(d => d.missing_pct || d['missing_%'] || 0);

        const trace = {
            x: variables,
            y: missing,
            type: 'bar',
            marker: {
                color: missing.map(v => v < 25 ? this.colors.accent : this.colors.danger),
                line: { color: 'rgba(255,255,255,0.1)', width: 1 }
            },
            hovertemplate: '<b>%{x}</b><br>Loss: %{y:.1f}%<extra></extra>'
        };

        const layout = this.getLayout('', {
            xaxis: { title: 'STATION VARIABLE', tickangle: -45 },
            yaxis: { title: 'MISSING (%)', gridcolor: this.colors.grid },
            margin: { l: 60, r: 20, t: 30, b: 100 }
        });

        Plotly.newPlot(containerId, [trace], layout, { responsive: true, displaylogo: false });
    },

    /**
     * Correlation Diagnostic Heatmap
     */
    plotCorrelationHeatmap(containerId, data) {
        if (!data || data.length === 0) return;

        const variables = Object.keys(data[0]).filter(k => k !== '' && k !== 'Unnamed: 0' && k !== 'Variable');
        const firstCol = data[0][''] || data[0]['Unnamed: 0'] || data[0]['Variable'];
        const rowNames = firstCol ? data.map(r => r[''] || r['Unnamed: 0'] || r['Variable']) : variables;

        const z = data.map(row => variables.map(v => parseFloat(row[v]) || 0));

        const trace = {
            x: variables,
            y: rowNames,
            z: z,
            type: 'heatmap',
            colorscale: 'RdBu',
            reversescale: true,
            zmin: -1,
            zmax: 1,
            hovertemplate: '%{x} ↔ %{y}<br>r = %{z:.2f}<extra></extra>'
        };

        const layout = this.getLayout('', {
            xaxis: { tickangle: -45 },
            yaxis: { autorange: 'reversed' },
            margin: { l: 120, r: 20, t: 20, b: 120 }
        });

        Plotly.newPlot(containerId, [trace], layout, { responsive: true, displaylogo: false });
    },

    /**
     * Multi-variable time series for instruments
     */
    plotInstrumentTimeSeries(containerId, data, variables) {
        const times = data.map(d => d[''] || d.TIME);
        const numVars = variables.length;
        const palette = [this.colors.primary, this.colors.secondary, this.colors.accent, this.colors.highlight];

        const traces = variables.map((varName, i) => ({
            x: times,
            y: data.map(d => d[varName]),
            type: 'scatter',
            mode: 'lines',
            name: varName,
            line: { color: palette[i % palette.length], width: 1.5 },
            xaxis: 'x',
            yaxis: `y${i + 1}`,
            hovertemplate: `<b>${varName}</b>: %{y:.2f}<extra></extra>`
        }));

        const layout = {
            ...this.getLayout('', {}),
            height: Math.max(400, numVars * 140),
            grid: { rows: numVars, columns: 1, pattern: 'coupled' },
            xaxis: { type: 'date', gridcolor: this.colors.grid, rangeslider: { visible: false } }
        };

        variables.forEach((_, i) => {
            const suffix = i === 0 ? '' : (i + 1);
            layout[`yaxis${suffix}`] = {
                title: variables[i],
                gridcolor: this.colors.grid,
                domain: [(numVars - i - 1) / numVars + 0.05, (numVars - i) / numVars - 0.05]
            };
        });

        Plotly.newPlot(containerId, traces, layout, { responsive: true, displaylogo: false });
    },

    /**
     * Data Availability Gantt-Heatmap
     */
    plotVariableAvailability(containerId, data, instruments) {
        if (!data || !instruments) return;
        const times = data.map(d => d[''] || d.TIME);

        // map instrument keys to indices
        const instKeys = Object.keys(instruments);
        const instColors = instKeys.map(k => instruments[k].color);

        // Define explicit colors for the heatmap
        // 0: Missing (subtle dark, similar to background)
        // 1..N: Instrument Colors
        const missingColor = '#1a2b4b'; // Dark blue-gray, blends with background theme

        // Flatten variables in reverse order (bottom to top)
        let allVars = [];
        let varToInstIndex = {};

        instKeys.forEach((key, idx) => {
            if (instruments[key].variables) {
                instruments[key].variables.forEach(v => {
                    allVars.push(v);
                    varToInstIndex[v] = idx + 1; // 1-based index
                });
            }
        });
        allVars.reverse();

        // Binning for performance
        const width = 1000;
        const binSize = Math.ceil(data.length / width);
        const binnedTimes = [];
        const z = allVars.map(() => []);

        for (let i = 0; i < data.length; i += binSize) {
            binnedTimes.push(times[i]);
            const chunk = data.slice(i, i + binSize);

            allVars.forEach((varName, rowIdx) => {
                const instIdx = varToInstIndex[varName];
                // Check if we have valid data in this chunk
                let hasData = false;
                for (let j = 0; j < chunk.length; j++) {
                    const val = chunk[j][varName];
                    if (val != null && !isNaN(val)) {
                        hasData = true;
                        break;
                    }
                }
                z[rowIdx].push(hasData ? instIdx : 0);
            });
        }

        // Construct Discrete Colorscale
        // We need discrete boundaries. 
        // 0 -> 0.5 : Color 0
        // 0.5 -> 1.5 : Color 1
        // ...
        const maxValue = instKeys.length;
        const cScale = [[0, missingColor], [1 / (maxValue + 1), missingColor]]; // First block for 0

        for (let i = 1; i <= maxValue; i++) {
            const normStart = i / (maxValue + 1);
            const normEnd = (i + 1) / (maxValue + 1);
            cScale.push([normStart, instColors[i - 1]]);
            cScale.push([normEnd, instColors[i - 1]]);
        }

        const trace = {
            x: binnedTimes,
            y: allVars,
            z: z,
            type: 'heatmap',
            colorscale: cScale,
            showscale: false,
            zmin: 0,
            zmax: maxValue,
            hovertemplate: '%{y}<br>%{x}<extra></extra>'
        };

        // Create Dummy Traces for Legend
        const legendTraces = instKeys.map((key, i) => ({
            x: [null],
            y: [null],
            type: 'scatter',
            mode: 'markers',
            name: key,
            marker: { size: 10, color: instColors[i] },
            showlegend: true
        }));

        // Add "Missing/Gap" to legend if desired, but user asked for Instrument Colors
        legendTraces.push({
            x: [null],
            y: [null],
            type: 'scatter',
            mode: 'markers',
            name: 'Data Gap',
            marker: { size: 10, color: missingColor },
            showlegend: true
        });

        const layout = this.getLayout('', {
            xaxis: { type: 'date', title: '' },
            yaxis: { tickfont: { size: 10 } },
            margin: { l: 100, r: 30, t: 60, b: 50 },
            height: Math.max(500, allVars.length * 20),
            legend: { orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center', font: { size: 11 } }
        });

        Plotly.newPlot(containerId, [trace, ...legendTraces], layout, { responsive: true, displaylogo: false });
    },

    /**
     * T-S Diagram (Stretched)
     */
    plotTSDiagram(containerId, data) {
        // Filter valid PAIRS only
        const valid = data.filter(d => d.TEMP != null && d.PSAL != null && !isNaN(d.TEMP) && !isNaN(d.PSAL));
        const temps = valid.map(d => d.TEMP);
        const sals = valid.map(d => d.PSAL);

        const scatter = {
            x: sals,
            y: temps,
            mode: 'markers',
            marker: {
                color: temps,
                colorscale: [[0, '#0072B2'], [0.5, '#56B4E9'], [1, '#D55E00']],
                size: 3,
                opacity: 0.5,
                showscale: true,
                colorbar: {
                    title: 'Temp (°C)',
                    titleside: 'right',
                    titlefont: { color: this.colors.text, size: 12 },
                    tickfont: { color: this.colors.gray, size: 10 },
                    thickness: 15,
                    len: 0.8
                }
            },
            hovertemplate: 'S: %{x:.2f}<br>T: %{y:.2f}°C<extra></extra>'
        };

        const layout = this.getLayout('', {
            xaxis: { title: 'SALINITY (PSU)', autorange: true },
            yaxis: { title: 'TEMPERATURE (°C)', autorange: true },
            margin: { l: 60, r: 20, t: 20, b: 60 }
        });

        Plotly.newPlot(containerId, [scatter], layout, { responsive: true, displaylogo: false });
    },

    /**
     * Rose Charts (Wind, Current, Waves)
     */
    _plotPolar(containerId, values, directions, vBins, colors, title) {
        const data = vBins.slice(0, -1).map((limit, i) => {
            const nextLimit = vBins[i + 1];
            const indices = values.map((v, idx) => (v >= limit && v < nextLimit) ? idx : -1).filter(idx => idx !== -1);
            const r = new Array(16).fill(0);
            const theta = new Array(16).fill(0).map((_, k) => k * 22.5);
            indices.forEach(idx => {
                if (directions[idx] != null) {
                    const binIdx = Math.floor(((directions[idx] + 11.25) % 360) / 22.5);
                    r[binIdx]++;
                }
            });
            return {
                type: 'barpolar',
                r: r,
                theta: theta,
                name: `${limit}-${nextLimit}`,
                marker: { color: colors[i % colors.length] }
            };
        });

        const layout = {
            ...this.getLayout(title, {}),
            polar: {
                bgcolor: 'transparent',
                radialaxis: { showline: false, tickfont: { color: this.colors.gray, size: 8 } },
                angularaxis: { rotation: 90, direction: 'clockwise', gridcolor: this.colors.grid }
            },
            margin: { l: 40, r: 40, t: 60, b: 40 }
        };

        Plotly.newPlot(containerId, data, layout, { responsive: true, displaylogo: false });
    },

    plotWindRose(c, s, d, t) { this._plotPolar(c, s, d, [0, 2, 4, 6, 8, 12, 100], ['#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695', '#d73027'], t); },
    plotCurrentRose(c, s, d, t) { this._plotPolar(c, s, d, [0, 0.05, 0.1, 0.2, 0.3, 0.5, 2], ['#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695', '#d73027'], t); },
    plotWaveRose(c, s, d, t) { this._plotPolar(c, s, d, [0, 0.5, 1, 1.5, 2, 3, 4, 10], ['#ffffff', '#abd9e9', '#74add1', '#4575b4', '#313695', '#d73027', '#67000d'], t); },

    /**
     * Performance Diagnostics
     */
    plotMethodComparison(containerId, data) {
        const methods = [...new Set(data.map(d => d.Method || d.method))];
        const categories = [...new Set(data.map(d => d.Category || d.category))];

        const traces = methods.map(m => ({
            x: categories,
            y: categories.map(c => data.find(d => (d.Method || d.method) === m && (d.Category || d.category) === c)?.RMSE || 0),
            name: m,
            type: 'bar',
            marker: { color: this.getMethodColor(m) }
        }));

        const layout = this.getLayout('', {
            barmode: 'group',
            yaxis: {
                title: 'RMSE (Lower is Better)',
                range: [0, 3] // Fixed range for better visibility
            }
        });
        Plotly.newPlot(containerId, traces, layout, { responsive: true, displaylogo: false });
    },

    plotInterpolatedSeries(containerId, originalData, interpolatedData, variable, method, showRaw = true, showInterp = true) {
        const times = interpolatedData.map(d => d[''] || d.TIME);

        // Dynamic Method Color
        const methodColor = this.getMethodColor(method);

        const trace = {
            x: times,
            y: interpolatedData.map(d => d[variable]),
            mode: 'lines',
            name: `Reconstructed (${method.toUpperCase()})`,
            line: { color: methodColor, width: 1.5 },
            visible: showInterp ? true : 'legendonly'
        };
        const origTrace = {
            x: originalData.map(d => d[''] || d.TIME),
            y: originalData.map(d => d[variable]),
            mode: 'markers',
            name: 'Raw Logs',
            marker: { color: 'rgba(255,255,255,0.3)', size: 2 },
            visible: showRaw ? true : 'legendonly'
        };

        const layout = this.getLayout(`PREVIEW: ${variable} [${method.toUpperCase()}]`, {
            xaxis: {
                type: 'date',
                rangeslider: { visible: true },
                rangeselector: {
                    buttons: [
                        { count: 1, label: '1m', step: 'month', stepmode: 'backward' },
                        { count: 6, label: '6m', step: 'month', stepmode: 'backward' },
                        { count: 1, label: '1y', step: 'year', stepmode: 'backward' },
                        { step: 'all' }
                    ],
                    font: { color: '#000' }
                }
            },
            legend: { orientation: 'h', y: 1.2 }
        });

        // Always pass both traces so Plotly can toggle, but set initial visibility
        Plotly.newPlot(containerId, [origTrace, trace], layout, { responsive: true, displaylogo: false });
    },

    plotGapPie(containerId, data) {
        const counts = {};
        data.forEach(d => {
            const cat = d.Category || d.category || 'Unknown';
            const val = parseInt(d.count || d.N || 1); // Sum actual events
            counts[cat] = (counts[cat] || 0) + val;
        });
        const labels = Object.keys(counts);
        const trace = {
            labels: labels,
            values: Object.values(counts),
            type: 'pie',
            hole: 0.6,
            marker: {
                colors: labels.map(l => {
                    const map = {
                        'micro': '#56B4E9',
                        'short': '#009E73',
                        'medium': '#E69F00',
                        'long': '#D55E00',
                        'extended': '#94a3b8',
                        'gigant': '#CC79A7'
                    };
                    return map[l.toLowerCase()] || this.colors.gray;
                })
            },
            textinfo: 'percent',
            textposition: 'inside',
            hovertemplate: '<b>%{label}</b><br>Total Events: %{value}<extra></extra>'
        };
        const layout = this.getLayout('', { height: 450 });
        Plotly.newPlot(containerId, [trace], layout, { responsive: true, displaylogo: false });
    }
};

window.Charts = Charts;
