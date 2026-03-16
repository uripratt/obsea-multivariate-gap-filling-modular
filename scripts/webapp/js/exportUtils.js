/**
 * OBSEA Dashboard - Export Utilities
 * HTML, PNG/PDF, and config export functionality
 */

const ExportUtils = {
    /**
     * Export chart as PNG
     */
    async exportPNG(chartId, filename = 'chart') {
        const chartDiv = document.getElementById(chartId);
        if (!chartDiv) {
            console.error('Chart not found:', chartId);
            return;
        }

        try {
            await Plotly.downloadImage(chartDiv, {
                format: 'png',
                width: 1200,
                height: 800,
                filename: filename,
                scale: 2
            });
            console.log(`Exported ${filename}.png`);
        } catch (e) {
            console.error('PNG export failed:', e);
        }
    },

    /**
     * Export chart as SVG
     */
    async exportSVG(chartId, filename = 'chart') {
        const chartDiv = document.getElementById(chartId);
        if (!chartDiv) return;

        try {
            await Plotly.downloadImage(chartDiv, {
                format: 'svg',
                width: 1200,
                height: 800,
                filename: filename
            });
            console.log(`Exported ${filename}.svg`);
        } catch (e) {
            console.error('SVG export failed:', e);
        }
    },

    /**
     * Export all visible charts
     */
    async exportAllCharts() {
        const charts = document.querySelectorAll('.js-plotly-plot');
        const timestamp = new Date().toISOString().slice(0, 10);

        for (const chart of charts) {
            if (chart.id) {
                await this.exportPNG(chart.id, `OBSEA_${chart.id}_${timestamp}`);
            }
        }
    },

    /**
     * Export data table as CSV
     */
    exportTableCSV(tableId, filename = 'data') {
        const table = document.getElementById(tableId);
        if (!table) return;

        const rows = [];
        const headers = Array.from(table.querySelectorAll('th')).map(th => th.textContent.trim());
        rows.push(headers.join(','));

        table.querySelectorAll('tbody tr').forEach(tr => {
            const cells = Array.from(tr.querySelectorAll('td')).map(td => {
                let text = td.textContent.trim();
                // Escape commas and quotes
                if (text.includes(',') || text.includes('"')) {
                    text = `"${text.replace(/"/g, '""')}"`;
                }
                return text;
            });
            rows.push(cells.join(','));
        });

        const csv = rows.join('\n');
        this.downloadFile(csv, `${filename}.csv`, 'text/csv');
    },

    /**
     * Export configuration as JSON
     */
    exportConfigJSON() {
        const config = {
            version: '1.0.0',
            timestamp: new Date().toISOString(),
            palette: 'Okabe-Ito',
            categories: MethodAnalysis?.categories || {},
            methods: MethodAnalysis?.methods || {},
            instruments: GapTimeline?.instruments || {},
            settings: {
                sampleRate: 50,
                theme: 'dark'
            }
        };

        const json = JSON.stringify(config, null, 2);
        this.downloadFile(json, 'obsea_config.json', 'application/json');
    },

    /**
     * Export configuration as YAML
     */
    exportConfigYAML() {
        const config = {
            version: '1.0.0',
            timestamp: new Date().toISOString(),
            palette: 'Okabe-Ito',
            categories: MethodAnalysis?.categories || {},
            methods: MethodAnalysis?.methods || {}
        };

        // Simple YAML serialization
        const yaml = this.toYAML(config);
        this.downloadFile(yaml, 'obsea_config.yaml', 'text/yaml');
    },

    /**
     * Simple YAML serialization
     */
    toYAML(obj, indent = 0) {
        let yaml = '';
        const spaces = '  '.repeat(indent);

        for (const [key, value] of Object.entries(obj)) {
            if (value === null || value === undefined) {
                yaml += `${spaces}${key}: null\n`;
            } else if (typeof value === 'object' && !Array.isArray(value)) {
                yaml += `${spaces}${key}:\n${this.toYAML(value, indent + 1)}`;
            } else if (Array.isArray(value)) {
                yaml += `${spaces}${key}:\n`;
                value.forEach(item => {
                    yaml += `${spaces}  - ${typeof item === 'object' ? '\n' + this.toYAML(item, indent + 2) : item}\n`;
                });
            } else {
                yaml += `${spaces}${key}: ${value}\n`;
            }
        }

        return yaml;
    },

    /**
     * Download file helper
     */
    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.click();
        URL.revokeObjectURL(url);
        console.log(`Downloaded ${filename}`);
    },

    /**
     * Add export buttons to UI
     */
    addExportButtons(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="export-toolbar">
                <button class="export-btn" onclick="ExportUtils.exportAllCharts()">
                    📷 Export All Charts (PNG)
                </button>
                <button class="export-btn" onclick="ExportUtils.exportConfigJSON()">
                    📄 Export Config (JSON)
                </button>
                <button class="export-btn" onclick="ExportUtils.exportConfigYAML()">
                    📄 Export Config (YAML)
                </button>
            </div>
        `;
    }
};

// Make available globally
window.ExportUtils = ExportUtils;
