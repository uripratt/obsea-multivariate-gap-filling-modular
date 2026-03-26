/**
 * OBSEA Dashboard - Data Loader Module
 * Handles CSV parsing and data preprocessing
 */

const DataLoader = {
    // Paths for CSV loading (using symlinks in webapp/)
    paths: {
        tables: 'tables/',
        data: 'data/'
    },

    /**
     * Load CSV file and parse it
     * Robust implementation with fallback
     */
    async loadCSV(filename) {
        console.log('📄 Loading CSV:', filename);
        return new Promise((resolve, reject) => {
            Papa.parse(filename, {
                download: true,
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    console.log('✅ Loaded:', filename, 'rows:', results.data?.length);
                    resolve(results.data);
                },
                error: (error) => {
                    console.error('❌ Error loading:', filename, error);
                    reject(error);
                }
            });
        });
    },

    /**
     * Load descriptive statistics
     */
    async loadStats() {
        if (window.OBSEA_STATS) {
            console.log('⚡ Using pre-loaded stats DB');
            return window.OBSEA_STATS;
        }
        try {
            const data = await this.loadCSV('tables/descriptive_statistics.csv');
            if (data && data.length > 0) return data;
            throw new Error('Empty data');
        } catch (e) {
            console.warn('⚠️ Using sample stats (CSV load failed)');
            return this.getSampleStats();
        }
    },

    /**
     * Load gap summary
     */
    async loadGapSummary() {
        try {
            const data = await this.loadCSV('tables/gap_summary.csv');
            if (data && data.length > 0) return data;
            throw new Error('Empty data');
        } catch (e) {
            console.warn('⚠️ Using sample gap summary');
            return this.getSampleGapSummary();
        }
    },

    /**
     * Load interpolation comparison
     */
    async loadMethodComparison() {
        try {
            const data = await this.loadCSV('tables/interpolation_comparison.csv');
            if (data && data.length > 0) return data;
            throw new Error('Empty data');
        } catch (e) {
            console.warn('⚠️ Using sample method comparison');
            return this.getSampleMethodComparison();
        }
    },

    /**
     * Load correlation matrix
     */
    async loadCorrelation() {
        try {
            const data = await this.loadCSV('tables/correlation_matrix.csv');
            if (data && data.length > 0) return data;
            return [];
        } catch (e) {
            console.warn('⚠️ Using sample correlation matrix');
            return this.getSampleCorrelation();
        }
    },

    /**
     * Load sampled time series data (for performance)
     * Falls back to sample data if CSV is too large or fails
     */
    async loadTimeSeries(sampleRate = 100) {
        if (window.OBSEA_DATA) {
            console.log('⚡ Using pre-loaded time series DB:', window.OBSEA_DATA.length);
            return window.OBSEA_DATA;
        }
        try {
            console.log('⏳ Loading time series (this may take a moment)...');
            const data = await this.loadCSV('data/OBSEA_multivariate_30min.csv');
            if (data && data.length > 0) {
                // If data is very large, ensure we sample evenly across the WHOLE dataset
                // If it's 10 years @ 30min = ~175,000 points. 
                // We want ~2000 points for performance. Step ~85.
                const total = data.length;
                const targetPoints = 2500;
                const step = Math.ceil(total / targetPoints);

                const sampled = data.filter((_, i) => i % step === 0);
                console.log(`📊 Time series sampled: ${sampled.length} points (Step: ${step})`);
                return sampled;
            }
            throw new Error('Empty data');
        } catch (e) {
            console.warn('⚠️ Using sample time series');
            return this.getSampleTimeSeries();
        }
    },

    // =========================================================================
    // Sample Data (fallback when CSVs not accessible)
    // =========================================================================

    getSampleStats() {
        return [
            { Variable: 'TEMP', count: 150000, missing_pct: 8.5, mean: 15.2, std: 3.1, min: 11.5, max: 26.8 },
            { Variable: 'PSAL', count: 150000, missing_pct: 9.2, mean: 38.1, std: 0.3, min: 36.8, max: 39.2 },
            { Variable: 'PRES', count: 148000, missing_pct: 10.1, mean: 20.5, std: 0.8, min: 18.2, max: 22.1 },
            { Variable: 'SVEL', count: 147000, missing_pct: 11.3, mean: 1512, std: 8.5, min: 1495, max: 1535 },
            { Variable: 'CUR_CSPD', count: 120000, missing_pct: 25.4, mean: 0.15, std: 0.08, min: 0, max: 0.85 },
            { Variable: 'WAV_VHM0', count: 135000, missing_pct: 18.2, mean: 0.8, std: 0.6, min: 0.1, max: 4.2 },
            { Variable: 'AIR_WSPD', count: 140000, missing_pct: 15.5, mean: 4.5, std: 3.2, min: 0, max: 22.5 },
            { Variable: 'AIR_AIRT', count: 142000, missing_pct: 14.1, mean: 18.3, std: 5.8, min: 2.1, max: 35.2 },
        ];
    },

    getSampleGapSummary() {
        return [
            { Variable: 'TEMP', Category: 'micro', count: 245, total_hours: 120 },
            { Variable: 'TEMP', Category: 'short', count: 89, total_hours: 312 },
            { Variable: 'TEMP', Category: 'medium', count: 23, total_hours: 856 },
            { Variable: 'TEMP', Category: 'long', count: 5, total_hours: 1420 },
            { Variable: 'PSAL', Category: 'micro', count: 267, total_hours: 132 },
            { Variable: 'PSAL', Category: 'short', count: 95, total_hours: 345 },
            { Variable: 'PSAL', Category: 'medium', count: 28, total_hours: 945 },
            { Variable: 'CUR_CSPD', Category: 'micro', count: 456, total_hours: 225 },
            { Variable: 'CUR_CSPD', Category: 'short', count: 178, total_hours: 623 },
            { Variable: 'CUR_CSPD', Category: 'medium', count: 67, total_hours: 2340 },
            { Variable: 'CUR_CSPD', Category: 'long', count: 12, total_hours: 4560 },
        ];
    },

    getSampleMethodComparison() {
        return [
            { Method: 'Linear', Category: 'micro', RMSE: 0.12, MAE: 0.08, R2: 0.95, Spectral_Error: 1.2 },
            { Method: 'Time', Category: 'micro', RMSE: 0.11, MAE: 0.07, R2: 0.96, Spectral_Error: 1.1 },
            { Method: 'Splines', Category: 'micro', RMSE: 0.10, MAE: 0.06, R2: 0.97, Spectral_Error: 0.8 },
            { Method: 'VARMA', Category: 'micro', RMSE: 0.09, MAE: 0.05, R2: 0.98, Spectral_Error: 0.5 },
            { Method: 'Linear', Category: 'short', RMSE: 0.35, MAE: 0.25, R2: 0.82, Spectral_Error: 2.1 },
            { Method: 'Time', Category: 'short', RMSE: 0.32, MAE: 0.22, R2: 0.84, Spectral_Error: 2.0 },
            { Method: 'Splines', Category: 'short', RMSE: 0.28, MAE: 0.19, R2: 0.87, Spectral_Error: 1.5 },
            { Method: 'VARMA', Category: 'short', RMSE: 0.22, MAE: 0.15, R2: 0.91, Spectral_Error: 0.9 },
            { Method: 'Linear', Category: 'medium', RMSE: 0.85, MAE: 0.62, R2: 0.65, Spectral_Error: 3.5 },
            { Method: 'Time', Category: 'medium', RMSE: 0.78, MAE: 0.55, R2: 0.70, Spectral_Error: 3.2 },
            { Method: 'Splines', Category: 'medium', RMSE: 0.68, MAE: 0.48, R2: 0.76, Spectral_Error: 2.5 },
            { Method: 'VARMA', Category: 'medium', RMSE: 0.52, MAE: 0.38, R2: 0.84, Spectral_Error: 1.2 },
            { Method: 'Bi-LSTM', Category: 'medium', RMSE: 0.45, MAE: 0.32, R2: 0.88, Spectral_Error: 0.6 },
        ];
    },

    getSampleTimeSeries() {
        const data = [];
        const startDate = new Date('2010-01-01T00:00:00Z');
        const totalPoints = 3000; // More points for longer timeline

        // Generate valid data: 2010 to 2024 (approx 14 years)
        // totalPoints = 3000. Interval needs to cover 14*365 days.
        const years = 14;
        const totalMs = years * 365 * 24 * 60 * 60 * 1000;
        const interval = totalMs / totalPoints;

        for (let i = 0; i < totalPoints; i++) {
            const date = new Date(startDate.getTime() + i * interval);
            const t = i / 100; // time factor for waves

            data.push({
                TIME: date.toISOString(),
                '': date.toISOString(), // Legacy support

                // CTD
                TEMP: 18 + 5 * Math.sin(t / 5) + Math.random(),
                PSAL: 37.5 + 0.5 * Math.sin(t / 10) + Math.random() * 0.2,
                PRES: 20 + Math.sin(t) * 0.5,
                SVEL: 1520 + 10 * Math.sin(t / 2),
                CNDC: 4 + 0.1 * Math.sin(t),

                // Currents (AWAC)
                CUR_UCUR: 0.2 * Math.cos(t),
                CUR_VCUR: 0.2 * Math.sin(t),
                CUR_CSPD: 0.3 + 0.1 * Math.sin(t) + Math.random() * 0.05,
                CUR_CDIR: 180 + 179 * Math.sin(t / 5),
                CUR_ZCUR: 10 + Math.sin(t),

                // Waves (AWAC)
                WAV_VHM0: 1.5 + 0.5 * Math.sin(t / 4) + Math.random() * 0.2,
                WAV_VTPK: 6 + 2 * Math.sin(t / 4),
                WAV_VPED: 90 + 45 * Math.sin(t / 10),
                WAV_VTM02: 4 + Math.sin(t / 3),
                WAV_VMDR: 135 + 45 * Math.cos(t),

                // Airmar (Met)
                AIR_WSPD: 5 + 3 * Math.sin(t / 2) + Math.random() * 2,
                AIR_WDIR: 180 + 90 * Math.sin(t),
                AIR_AIRT: 20 + 7 * Math.sin(t / 3),
                AIR_CAPH: 1013 + 5 * Math.sin(t / 10),
                AIR_WSPD_STD: 0.5 + Math.random() * 0.5,

                // CTVG (Land)
                LAND_WSPD: 4 + 2 * Math.sin(t / 2),
                LAND_WDIR: 200 + 50 * Math.sin(t),
                LAND_AIRT: 22 + 8 * Math.sin(t / 3),
                LAND_RELH: 60 + 20 * Math.cos(t / 3),
                LAND_CAPH: 1015 + 5 * Math.sin(t / 10),

                // Derived
                SIGMA0: 27 + 0.5 * Math.sin(t / 10),
                N2: 0.0001 * Math.random(),
                WIND_STRESS: 0.05 * Math.random(),
                WAVE_ENERGY: 1000 * Math.random(),
                CUR_RMS: 0.1 * Math.random(),
                WIND_U: 3 * Math.cos(t),
                WIND_V: 3 * Math.sin(t),
                TEMP_ANOMALY: (Math.random() - 0.5) * 2,
                PSAL_ANOMALY: (Math.random() - 0.5) * 0.5
            });
        }
        return data;
    },

    getSampleInterpolatedData() {
        const data = this.getSampleTimeSeries();
        return data.map(d => {
            const clone = { ...d };
            clone.TEMP = clone.TEMP ? clone.TEMP + (Math.random() - 0.5) * 0.1 : null;
            return clone;
        });
    },

    getSampleCorrelation() {
        return [
            { '': 'TEMP', TEMP: 1.0, PSAL: 0.85, PRES: -0.1, AIR_AIRT: 0.95, AIR_WSPD: -0.2 },
            { '': 'PSAL', TEMP: 0.85, PSAL: 1.0, PRES: -0.05, AIR_AIRT: 0.8, AIR_WSPD: -0.15 },
            { '': 'PRES', TEMP: -0.1, PSAL: -0.05, PRES: 1.0, AIR_AIRT: -0.1, AIR_WSPD: 0.3 },
            { '': 'AIR_AIRT', TEMP: 0.95, PSAL: 0.8, PRES: -0.1, AIR_AIRT: 1.0, AIR_WSPD: -0.25 },
            { '': 'AIR_WSPD', TEMP: -0.2, PSAL: -0.15, PRES: 0.3, AIR_AIRT: -0.25, AIR_WSPD: 1.0 }
        ];
    }
};

window.DataLoader = DataLoader;
