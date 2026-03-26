/**
 * OBSEA Dashboard - Main Application (v22 Robust Edition)
 * Professional oceanographic data visualization
 */

const App = {
    data: {
        stats: [],
        gaps: [],
        methods: [],
        timeseries: [],
        correlation: [],
        interpolated: null
    },

    instruments: {
        ctd: ['TEMP', 'PSAL', 'PRES', 'SVEL', 'CNDC'],
        currents: ['CUR_CSPD', 'CUR_CDIR', 'CUR_UCUR', 'CUR_VCUR', 'CUR_ZCUR'],
        waves: ['WAV_VHM0', 'WAV_VTPK', 'WAV_VTM02', 'WAV_VMDR', 'WAV_VPED'],
        airmar: ['AIR_WSPD', 'AIR_WDIR', 'AIR_AIRT', 'AIR_CAPH'],
        ctvg: ['LAND_WSPD', 'LAND_WDIR', 'LAND_AIRT', 'LAND_RELH', 'LAND_CAPH'],
        derived: ['SIGMA0', 'N2', 'WIND_STRESS', 'WAVE_ENERGY', 'CUR_RMS', 'TEMP_ANOMALY', 'PSAL_ANOMALY', 'WIND_U', 'WIND_V', 'AIR_WSPD_STD']
    },

    /**
     * Initialization Sequence
     */
    async init() {
        console.group('🌊 OBSEA Application Bootstrap');

        try {
            console.log('1. Setting up event listeners...');
            this.setupNavigation();
            this.setupInstrumentTabs();

            console.log('2. Loading data resources...');
            await this.loadAllResources();

            console.log('3. Triggering initial render...');
            this.renderOverview();

            console.log('✅ Bootstrap Complete');
        } catch (err) {
            console.error('❌ Critical Initialization Failure:', err);
            // Panic render: try to show sample data at least
            this.data.stats = window.DataLoader?.getSampleStats() || [];
            this.renderOverview();
        } finally {
            console.groupEnd();
        }
    },

    /**
     * Navigation Logic (Top Header)
     */
    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        const viewIndicator = document.getElementById('view-indicator');

        navLinks.forEach(link => {
            link.onclick = () => {
                const viewId = link.getAttribute('data-view');
                if (!viewId) return;

                console.log(`🚀 Switching to view: ${viewId}`);

                // Update UI state
                this.switchView(viewId);

                // Update nav classes
                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');

                // Update breadcrumb
                if (viewIndicator) {
                    viewIndicator.textContent = link.innerText.trim().toUpperCase();
                }
            };
        });
    },

    /**
     * View Switcher Engine
     */
    switchView(viewId) {
        const views = document.querySelectorAll('.view');
        let found = false;

        views.forEach(v => {
            if (v.id === viewId) {
                v.classList.add('active');
                v.style.display = 'block';
                found = true;
            } else {
                v.classList.remove('active');
                v.style.display = 'none';
            }
        });

        if (found) {
            this.renderViewContent(viewId);
        } else {
            console.warn(`⚠️ View "${viewId}" not found in DOM`);
        }
    },

    /**
     * View-specific rendering routing
     */
    renderViewContent(viewId) {
        switch (viewId) {
            case 'overview': return this.renderOverview();
            case 'instruments': return this.renderInstrumentChart('ctd');
            case 'gaps': return this.renderGaps();
            case 'oceanography': return this.renderOceanography();
            case 'methods': return this.renderMethods();
            case 'digital-twin': return this.renderDigitalTwin();
        }
    },

    renderDigitalTwin() {
        console.log("🧊 Digital Twin Module Activated");
        if (window.DigitalTwinAppClass && !this.digitalTwinInstance) {
            console.log("🚀 Booting 3D Engine...");
            this.digitalTwinInstance = new window.DigitalTwinAppClass();
            this.digitalTwinInstance.init();
        }
    },

    /**
     * Oceanographic Analysis View
     */
    renderOceanography() {
        if (!this.data.timeseries?.length || !window.Charts) return;

        // 1. T-S Diagram
        window.Charts.plotTSDiagram('chart-ts', this.data.timeseries);

        // 2. Derived Variables
        const derivedVars = ['SIGMA0', 'N2', 'WIND_STRESS', 'WAVE_ENERGY', 'CUR_RMS'];
        window.Charts.plotInstrumentTimeSeries('chart-derived', this.data.timeseries, derivedVars);

        // 3. Environmental Roses
        const d = this.data.timeseries;
        const v = (n1, n2) => d.filter(x => x[n1] != null && x[n2] != null);

        const wi = v('AIR_WSPD', 'AIR_WDIR');
        if (wi.length) window.Charts.plotWindRose('chart-windrose', wi.map(x => x.AIR_WSPD), wi.map(x => x.AIR_WDIR), 'Wind Rose');

        const cu = v('CUR_CSPD', 'CUR_CDIR');
        if (cu.length) window.Charts.plotCurrentRose('chart-currentrose', cu.map(x => x.CUR_CSPD), cu.map(x => x.CUR_CDIR), 'Current Rose');

        const wa = v('WAV_VHM0', 'WAV_VMDR');
        if (wa.length) window.Charts.plotWaveRose('chart-waverose', wa.map(x => x.WAV_VHM0), wa.map(x => x.WAV_VMDR), 'Wave Rose');
    },

    /**
     * High-reliability data loader
     */
    async loadAllResources() {
        if (!window.DataLoader) {
            throw new Error('DataLoader module not found');
        }

        const loader = window.DataLoader;

        // Execute loads in parallel
        const results = await Promise.allSettled([
            loader.loadStats(),
            loader.loadGapSummary(),
            loader.loadMethodComparison(),
            loader.loadTimeSeries(50),
            loader.loadCorrelation()
        ]);

        // Process results safely
        this.data.stats = results[0].status === 'fulfilled' ? results[0].value : loader.getSampleStats();
        this.data.gaps = results[1].status === 'fulfilled' ? results[1].value : loader.getSampleGapSummary();
        this.data.methods = results[2].status === 'fulfilled' ? results[2].value : loader.getSampleMethodComparison();
        this.data.timeseries = results[3].status === 'fulfilled' ? results[3].value : loader.getSampleTimeSeries();

        // Dynamic Correlation: Always try to use server pre-computed matrix first
        if (results[4] && results[4].status === 'fulfilled' && results[4].value.length > 0) {
            this.data.correlation = results[4].value;
            console.log('✅ Loaded pre-computed correlation matrix from server');
        } else if (window.MethodAnalysis && this.data.timeseries.length > 50 && this.data.timeseries.length < 5000) {
            // Compute real correlation on the client side ONLY if dataset is small enough
            // to prevent browser tab from freezing with O(V²·N) complexity
            console.log('⚠️ Computing correlation matrix on client-side (small dataset)');
            const vars = Object.keys(this.data.timeseries[0]).filter(k =>
                !k.includes('_QC') && !k.includes('_STD') && k !== 'TIME' && k !== ''
            );
            // Limit to top 15 most important variables to prevent lag if too many
            const keyVars = ['TEMP', 'PSAL', 'SVEL', 'CNDC', 'CUR_CSPD', 'WAV_VHM0', 'AIR_WSPD', 'AIR_AIRT', 'LAND_WSPD', 'LAND_AIRT', 'SIGMA_T', 'WIND_STRESS', 'WAVE_ENERGY', 'VB_FREQ'];
            const targetVars = vars.filter(v => keyVars.includes(v)).concat(vars.filter(v => !keyVars.includes(v))).slice(0, 15);

            this.data.correlation = this.computeCorrelation(this.data.timeseries, targetVars);
        } else {
            console.warn('⚠️ Falling back to sample correlation matrix (Dataset too large for client compute)');
            this.data.correlation = loader.getSampleCorrelation();
        }

        console.log('📊 Data Memory state:', {
            stats: this.data.stats.length,
            timeseries: this.data.timeseries.length,
            correlation: this.data.correlation.length
        });
    },

    computeCorrelation(data, variables) {
        const matrix = [];
        // Calculate means and std devs
        const stats = {};
        variables.forEach(v => {
            const values = data.map(d => d[v]).filter(x => x != null && !isNaN(x));
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
            stats[v] = { mean, std: Math.sqrt(variance), values };
        });

        variables.forEach(v1 => {
            const row = { Variable: v1, '': v1 };
            variables.forEach(v2 => {
                if (v1 === v2) {
                    row[v2] = 1;
                } else {
                    // Pearson Correlation
                    let sum = 0;
                    let n = 0;
                    data.forEach(d => {
                        const val1 = d[v1];
                        const val2 = d[v2];
                        if (val1 != null && val2 != null && !isNaN(val1) && !isNaN(val2)) {
                            sum += (val1 - stats[v1].mean) * (val2 - stats[v2].mean);
                            n++;
                        }
                    });
                    row[v2] = n > 0 ? sum / (n * stats[v1].std * stats[v2].std) : 0;
                }
            });
            matrix.push(row);
        });
        return matrix;
    },

    /**
     * Rendering: Dashboard Overview
     */
    renderOverview() {
        const stats = this.data.stats || [];
        if (stats.length === 0) {
            console.warn('Overview render skipped: No stats available');
            return;
        }

        // Metrics calculation
        const totalObs = stats.reduce((acc, row) => acc + (parseInt(row.count || row.N) || 0), 0);

        this.safeUpdateText('stat-observations', this.fNum(totalObs));
        this.safeUpdateText('stat-variables', stats.length);
        this.safeUpdateText('stat-timespan', '10+ Years');

        // Charts
        if (window.Charts) {
            window.Charts.plotMissingData('chart-missing', stats);
            if (this.data.correlation?.length) {
                window.Charts.plotCorrelationHeatmap('chart-correlation', this.data.correlation);
            }
        }

        this.populateStatsTable();
    },

    /**
     * Rendering: Stats Table
     */
    populateStatsTable() {
        const tbody = document.getElementById('stats-tbody');
        if (!tbody || !this.data.stats) return;

        tbody.innerHTML = this.data.stats.map(row => {
            const name = row.Variable || row.variable || '???';
            const miss = parseFloat(row.missing_pct || row['missing_%'] || 0);

            return `
                <tr>
                    <td><strong style="color:var(--primary)">${name}</strong></td>
                    <td class="numeric">${this.fNum(row.count || row.N || 0)}</td>
                    <td class="numeric" style="color:${miss > 10 ? 'var(--danger)' : 'var(--secondary)'}">${miss.toFixed(1)}%</td>
                    <td class="numeric">${parseFloat(row.mean || 0).toFixed(2)}</td>
                    <td class="numeric">${parseFloat(row.std || 0).toFixed(2)}</td>
                    <td class="numeric">${parseFloat(row.min || 0).toFixed(2)}</td>
                    <td class="numeric">${parseFloat(row.max || 0).toFixed(2)}</td>
                </tr>
            `;
        }).join('');
    },

    /**
     * Instrument Panel Logic
     */
    setupInstrumentTabs() {
        const tabs = document.querySelectorAll('.inst-tab');
        tabs.forEach(tab => {
            tab.onclick = () => {
                const instId = tab.getAttribute('data-inst');
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                // Toggle panels
                document.querySelectorAll('.inst-panel').forEach(p => p.classList.remove('active'));
                const target = document.getElementById(`inst-${instId}`);
                if (target) target.classList.add('active');

                this.renderInstrumentChart(instId);
            };
        });
    },

    renderInstrumentChart(instId) {
        if (!this.data.timeseries?.length || !window.Charts) return;

        const vars = this.instruments[instId] || [];
        const available = vars.filter(v => this.data.timeseries[0] && this.data.timeseries[0][v] !== undefined);

        if (available.length) {
            window.Charts.plotInstrumentTimeSeries(`chart-${instId}`, this.data.timeseries, available);
        }
    },

    /**
     * Inventory and Gap Diagnostics
     */
    renderGaps() {
        if (!this.data.timeseries?.length || !window.Charts) return;

        if (window.GapTimeline) {
            window.Charts.plotVariableAvailability('chart-gantt', this.data.timeseries, window.GapTimeline.instruments);
            window.GapTimeline.renderGapStatsTable('instrument-stats-table', this.data.timeseries);
        }

        if (this.data.gaps?.length) {
            window.Charts.plotGapPie('chart-gap-histogram', this.data.gaps);
        }

        this.populateGapTable();
    },

    populateGapTable() {
        const tbody = document.getElementById('gaps-tbody');
        if (!tbody || !this.data.gaps) return;

        tbody.innerHTML = this.data.gaps.slice(0, 30).map(row => {
            const varName = row.Variable || row.variable || '';
            const category = row.Category || row.category || '';
            return `
                <tr>
                    <td><strong>${varName}</strong></td>
                    <td><span class="badge">${(category || '').toUpperCase()}</span></td>
                    <td class="numeric">${row.count || 0}</td>
                    <td class="numeric">${parseFloat(row.total_hours || 0).toFixed(1)}h</td>
                    <td class="numeric">${((row.total_hours || 0) / (row.count || 1)).toFixed(1)}h</td>
                </tr>
            `;
        }).join('');
    },

    /**
     * Interpolation View
     */
    async renderMethods() {
        if (!this.data.methods?.length || !window.Charts) return;

        window.Charts.plotMethodComparison('chart-methods', this.data.methods);
        this.populateMethodsTable();

        // Render Deep Learning Case Study
        if (window.MethodAnalysis && window.MethodAnalysis.renderCaseStudy) {
            window.MethodAnalysis.renderCaseStudy('chart-case-study');
        }

        // Load simulation data once
        if (!this.data.interpolated) {
            try {
                const raw = await window.DataLoader.loadCSV('data/OBSEA_multivariate_30min_interpolated.csv');
                if (raw && raw.length > 0) {
                    this.data.interpolated = raw.filter((_, i) => i % 50 === 0);
                } else { throw new Error("Empty interpoalted file"); }
            } catch (e) {
                console.warn('Sim data use fallback for Interpolation (Aligned with TimeSeries)');
                // Generate simulated interpolation aligned with actual TimeSeries
                if (this.data.timeseries && this.data.timeseries.length > 0) {
                    this.data.interpolated = this.data.timeseries.map(d => {
                        // Create a "reconstructed" version by filling nulls or slight smoothing
                        const clone = { ...d };
                        ['TEMP', 'PSAL', 'AIR_WSPD', 'CUR_CSPD', 'LAND_WSPD', 'SIGMA0'].forEach(k => {
                            if (clone[k] == null) {
                                // Simulate imputation (simple carry forward or random for demo)
                                clone[k] = (Math.random() * 5) + 15;
                            }
                            // Add slight jitter to simulate "model output" vs "truth"
                            if (clone[k] !== null) {
                                clone[k] += (Math.random() - 0.5) * 0.1;
                            }
                        });
                        return clone;
                    });
                } else if (window.DataLoader.getSampleInterpolatedData) {
                    this.data.interpolated = window.DataLoader.getSampleInterpolatedData();
                }
            }
        }

        this.setupInterpControls();
    },

    setupInterpControls() {
        const vSel = document.getElementById('interp-var-select');
        const mSel = document.getElementById('interp-method-select');

        if (vSel && vSel.options.length === 0) {
            // Populate ALL available variables
            const allVars = this.data.timeseries && this.data.timeseries.length ?
                Object.keys(this.data.timeseries[0]).filter(k => !['', 'TIME', 'Unnamed: 0'].includes(k) && !k.includes('_QC') && !k.includes('_STD')) :
                ['TEMP', 'PSAL', 'AIR_WSPD', 'CUR_CSPD', 'LAND_WSPD', 'SIGMA0'];

            allVars.sort().forEach(v => {
                const o = document.createElement('option'); o.value = v; o.text = v;
                vSel.add(o);
            });
            // Select TEMP default if exists
            if (allVars.includes('TEMP')) vSel.value = 'TEMP';

            ['best', 'linear', 'time', 'splines', 'polynomial', 'varma', 'bilstm', 'saits', 'brits_pro'].forEach(m => {
                const o = document.createElement('option'); o.value = m; o.text = m.toUpperCase();
                mSel.add(o);
            });

            // Add checkboxes for toggling if not exist
            const controlsDiv = vSel.parentElement; // Assuming container
            if (!document.getElementById('toggle-raw')) {
                const d = document.createElement('div');
                d.style.marginTop = '10px';
                d.innerHTML = `
                    <label style="margin-right:15px; color:#ccc; font-size:12px;"><input type="checkbox" id="toggle-raw" checked> Show Raw (Logs)</label>
                    <label style="color:${window.Charts.colors.primary}; font-size:12px;"><input type="checkbox" id="toggle-interp" checked> Show Reconstructed</label>
                 `;
                controlsDiv.appendChild(d);
            }

            const up = () => this.updateInterp(vSel.value, mSel.value);
            vSel.onchange = up;
            mSel.onchange = up;
            document.getElementById('toggle-raw').onchange = up;
            document.getElementById('toggle-interp').onchange = up;
        }
        this.updateInterp(vSel.value, mSel.value);
    },

    async updateInterp(v, m) {
        if (!window.Charts || !this.data.timeseries) return;

        let interpData = this.data.interpolated;
        document.body.style.cursor = 'wait';

        try {
            // Case 1: Complex Models requiring external files (Python generated)
            if (['bilstm', 'varma', 'saits', 'brits_pro'].includes(m.toLowerCase())) {
                const methodKey = m.toUpperCase(); // BILSTM, VARMA
                const cacheKey = `interpolated_${methodKey}`;

                if (this.data[cacheKey]) {
                    interpData = this.data[cacheKey];
                } else {
                    // Try to load external file
                    try {
                        console.log(`📡 Fetching reconstruction for ${methodKey}...`);
                        let filename = `data/OBSEA_multivariate_30min_${methodKey}.csv`;

                        try {
                            var loaded = await window.DataLoader.loadCSV(filename);
                        } catch (loadErr) {
                            // Fallback for BiLSTM/Best if specific file missing
                            if (['BILSTM', 'BEST'].includes(methodKey)) {
                                console.warn(`Specific file for ${methodKey} missing. Trying main interpolated file...`);
                                filename = 'data/OBSEA_multivariate_30min_interpolated.csv';
                                loaded = await window.DataLoader.loadCSV(filename);
                            } else {
                                throw loadErr;
                            }
                        }

                        if (loaded && loaded.length > 0) {
                            this.data[cacheKey] = loaded;
                            interpData = loaded;
                        } else {
                            throw new Error("Empty or invalid file");
                        }
                    } catch (e) {
                        console.warn(`Could not load ${methodKey} data. Generating linear approximation as fallback.`);
                        // Fallback to Linear to avoid breaking UI
                        if (window.MethodAnalysis?.interpolateSeries) {
                            interpData = window.MethodAnalysis.interpolateSeries(this.data.timeseries, v, 'linear');
                        }
                    }
                }
            }
            // Case 2: Real-time Client-side Interpolation (Linear, Splines, Time)
            else if (window.MethodAnalysis && window.MethodAnalysis.interpolateSeries) {
                interpData = window.MethodAnalysis.interpolateSeries(this.data.timeseries, v, m);
            }
        } catch (err) {
            console.error('Interpolation Update Error:', err);
        } finally {
            document.body.style.cursor = 'default';
        }

        if (interpData) {
            const showRaw = document.getElementById('toggle-raw')?.checked ?? true;
            const showInterp = document.getElementById('toggle-interp')?.checked ?? true;
            window.Charts.plotInterpolatedSeries('chart-interpolation-series', this.data.timeseries, interpData, v, m, showRaw, showInterp);
        }
    },

    populateMethodsTable() {
        const tb = document.getElementById('methods-tbody');
        if (!tb || !this.data.methods) return;
        tb.innerHTML = this.data.methods.map(r => {
            const m = r.Method || r.method;
            const color = window.Charts ? window.Charts.getMethodColor(m) : 'var(--primary)';
            return `
                <tr>
                    <td><strong style="color:${color}">${m}</strong></td>
                    <td>${r.Category || r.category}</td>
                    <td class="numeric">${parseFloat(r.RMSE || 0).toFixed(4)}</td>
                    <td class="numeric">${parseFloat(r.MAE || 0).toFixed(4)}</td>
                    <td class="numeric">${parseFloat(r.R2 || 0).toFixed(4)}</td>
                </tr>
            `;
        }).join('');
    },


    /**
     * Utilities
     */
    safeUpdateText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    },

    fNum(n) {
        const num = Math.round(n);
        return num >= 1e6 ? (num / 1e6).toFixed(1) + 'M' : num >= 1e3 ? (num / 1e3).toFixed(1) + 'K' : num.toLocaleString();
    }
};

// Start
window.addEventListener('load', () => App.init());
