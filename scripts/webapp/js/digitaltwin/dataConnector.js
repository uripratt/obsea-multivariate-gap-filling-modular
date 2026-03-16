/**
 * Data Connector Module - Connects to real oceanographic data
 * Provides interface to load and process sensor data
 */

export class DataConnector {
    constructor() {
        // Data storage
        this.timeseries = null;
        this.currentIndex = 0;

        // Data paths (relative to index.html)
        this.dataPath = 'data/';

        // Column mappings
        this.columns = {
            time: 'TIME',
            temp: 'TEMP',
            psal: 'PSAL',
            currentSpeed: 'CUR_CSPD',
            currentDir: 'CUR_CDIR',
            currentU: 'CUR_UCUR',
            currentV: 'CUR_VCUR',
            waveHeight: 'WAV_VHM0',
            wavePeriod: 'WAV_VTPK',
            waveDir: 'WAV_VMDR'
        };

        // Sample data for demo/fallback
        this.sampleData = this.generateSampleData();
    }

    async init() {
        console.log('📡 DataConnector: Initializing...');

        try {
            // Try to load real data
            await this.loadData();
            console.log(`✅ DataConnector: Loaded ${this.timeseries.length} records`);
        } catch (error) {
            console.warn('⚠️ DataConnector: Using sample data -', error.message);
            this.timeseries = this.sampleData;
        }

        return Promise.resolve();
    }

    async loadData() {
        const filepath = `${this.dataPath}OBSEA_multivariate_30min.csv`;

        try {
            const response = await fetch(filepath);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const text = await response.text();
            this.timeseries = this.parseCSV(text);

        } catch (error) {
            throw new Error(`Failed to load data: ${error.message}`);
        }
    }

    parseCSV(text) {
        const lines = text.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));

        const data = [];

        // Sample every 48 rows (1 day at 30min resolution) for performance
        for (let i = 1; i < lines.length; i += 48) {
            const values = lines[i].split(',');
            const row = {};

            headers.forEach((header, idx) => {
                const val = values[idx]?.trim().replace(/"/g, '');
                row[header] = val === '' || val === 'NaN' ? null : parseFloat(val) || val;
            });

            data.push(row);
        }

        return data;
    }

    generateSampleData() {
        const data = [];
        const startDate = new Date('2015-01-01');
        const endDate = new Date('2026-02-06');
        const dayMs = 24 * 60 * 60 * 1000;

        let currentDate = new Date(startDate);
        let index = 0;

        while (currentDate < endDate) {
            // Seasonal variation
            const dayOfYear = Math.floor((currentDate - new Date(currentDate.getFullYear(), 0, 0)) / dayMs);
            const seasonalPhase = (dayOfYear / 365) * 2 * Math.PI;

            // Diurnal variation
            const hourOfDay = currentDate.getHours();
            const diurnalPhase = (hourOfDay / 24) * 2 * Math.PI;

            // Temperature: 12-26°C with seasonal cycle
            const temp = 18 + Math.sin(seasonalPhase) * 6 +
                Math.sin(diurnalPhase) * 0.5 +
                (Math.random() - 0.5) * 1;

            // Salinity: 37.5-38.5 PSU
            const psal = 38 + Math.sin(seasonalPhase * 0.5) * 0.3 +
                (Math.random() - 0.5) * 0.2;

            // Current speed: 0-0.5 m/s
            const currentSpeed = Math.abs(0.15 + Math.sin(diurnalPhase * 2) * 0.1 +
                (Math.random() - 0.5) * 0.1);

            // Current direction: varies with tides
            const currentDir = (180 + Math.sin(diurnalPhase * 2) * 90 +
                (Math.random() - 0.5) * 30 + 360) % 360;

            // Wave height: 0.3-2.5m (higher in winter)
            const waveBase = 0.8 - Math.cos(seasonalPhase) * 0.5;
            const waveHeight = Math.max(0.3, waveBase + (Math.random() - 0.3) * 0.8);

            // Wave period: 4-12 seconds
            const wavePeriod = 6 + waveHeight * 2 + (Math.random() - 0.5) * 2;

            // Wave direction: predominantly from E-SE
            const waveDir = (90 + Math.sin(seasonalPhase * 0.5) * 30 +
                (Math.random() - 0.5) * 20 + 360) % 360;

            data.push({
                TIME: currentDate.toISOString(),
                TEMP: parseFloat(temp.toFixed(2)),
                PSAL: parseFloat(psal.toFixed(2)),
                CUR_CSPD: parseFloat(currentSpeed.toFixed(3)),
                CUR_CDIR: parseFloat(currentDir.toFixed(1)),
                CUR_UCUR: parseFloat((currentSpeed * Math.sin(currentDir * Math.PI / 180)).toFixed(3)),
                CUR_VCUR: parseFloat((currentSpeed * Math.cos(currentDir * Math.PI / 180)).toFixed(3)),
                WAV_VHM0: parseFloat(waveHeight.toFixed(2)),
                WAV_VTPK: parseFloat(wavePeriod.toFixed(1)),
                WAV_VMDR: parseFloat(waveDir.toFixed(1))
            });

            // Advance by 1 day for sample data
            currentDate = new Date(currentDate.getTime() + dayMs);
            index++;
        }

        return data;
    }

    /**
     * Get data at a specific time (normalized 0-1 progress through dataset)
     */
    async getDataAtTime(progress) {
        if (!this.timeseries || this.timeseries.length === 0) {
            return this.getLatestData();
        }

        const index = Math.floor(progress * (this.timeseries.length - 1));
        const row = this.timeseries[index];

        return this.extractSensorData(row);
    }

    /**
     * Get the latest/current data point
     */
    getLatestData() {
        if (!this.timeseries || this.timeseries.length === 0) {
            return {
                time: new Date().toISOString(),
                temp: 18.5,
                psal: 38.2,
                currentSpeed: 0.15,
                currentDir: 45,
                currentU: 0.1,
                currentV: 0.1,
                waveHeight: 0.8,
                wavePeriod: 6,
                waveDir: 90
            };
        }

        const row = this.timeseries[this.timeseries.length - 1];
        return this.extractSensorData(row);
    }

    /**
     * Extract sensor data from a data row
     */
    extractSensorData(row) {
        if (!row) return null;

        return {
            time: row[this.columns.time] || new Date().toISOString(),
            temp: this.safeNumber(row[this.columns.temp], 18),
            psal: this.safeNumber(row[this.columns.psal], 38),
            currentSpeed: this.safeNumber(row[this.columns.currentSpeed], 0.1),
            currentDir: this.safeNumber(row[this.columns.currentDir], 45),
            currentU: this.safeNumber(row[this.columns.currentU], 0),
            currentV: this.safeNumber(row[this.columns.currentV], 0),
            waveHeight: this.safeNumber(row[this.columns.waveHeight], 0.8),
            wavePeriod: this.safeNumber(row[this.columns.wavePeriod], 6),
            waveDir: this.safeNumber(row[this.columns.waveDir], 90)
        };
    }

    /**
     * Get a stream of data points for animation
     */
    getDataStream(startProgress = 0, endProgress = 1, steps = 100) {
        if (!this.timeseries || this.timeseries.length === 0) {
            return [];
        }

        const stream = [];
        const startIdx = Math.floor(startProgress * (this.timeseries.length - 1));
        const endIdx = Math.floor(endProgress * (this.timeseries.length - 1));
        const stepSize = Math.max(1, Math.floor((endIdx - startIdx) / steps));

        for (let i = startIdx; i <= endIdx; i += stepSize) {
            stream.push(this.extractSensorData(this.timeseries[i]));
        }

        return stream;
    }

    /**
     * Get statistics for a variable
     */
    getStats(variable) {
        if (!this.timeseries || this.timeseries.length === 0) {
            return { min: 0, max: 0, mean: 0, std: 0 };
        }

        const values = this.timeseries
            .map(row => row[variable])
            .filter(v => v !== null && v !== undefined && !isNaN(v));

        if (values.length === 0) {
            return { min: 0, max: 0, mean: 0, std: 0 };
        }

        const min = Math.min(...values);
        const max = Math.max(...values);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / values.length;
        const std = Math.sqrt(variance);

        return { min, max, mean, std };
    }

    safeNumber(value, defaultVal) {
        if (value === null || value === undefined || isNaN(value)) {
            return defaultVal;
        }
        return parseFloat(value);
    }
}
