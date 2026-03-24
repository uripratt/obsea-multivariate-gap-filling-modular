/**
 * OBSEA Dashboard - Gap Timeline Module (Premium)
 * Gantt-style timeline and multisensor gap analysis
 */

const GapTimeline = {
    // Instrument configurations synced with design system
    instruments: {
        'CTD': { color: '#56B4E9', variables: ['TEMP', 'PSAL', 'PRES', 'SVEL'] },
        'AWAC Currents': { color: '#009E73', variables: ['CUR_UCUR', 'CUR_VCUR', 'CUR_CSPD'] },
        'AWAC Waves': { color: '#E69F00', variables: ['WAV_VHM0', 'WAV_VTPK', 'WAV_VPED'] },
        'Airmar': { color: '#CC79A7', variables: ['AIR_WSPD', 'AIR_WDIR', 'AIR_AIRT', 'AIR_CAPH'] },
        'CTVG (Land)': { color: '#F0E442', variables: ['LAND_WSPD', 'LAND_WDIR', 'LAND_AIRT', 'LAND_RELH', 'LAND_CAPH'] }
    },

    /**
     * Compute gap statistics per instrument
     */
    computeGapStats(data) {
        const stats = {};
        const times = data.map(d => new Date(d[''] || d.TIME));

        Object.entries(this.instruments).forEach(([instName, inst]) => {
            const available = data.map(d =>
                inst.variables.some(v => d[v] != null && !isNaN(d[v])) ? 1 : 0
            );

            const total = available.length;
            const valid = available.filter(a => a === 1).length;

            // Simplified segment finder logic for stats
            let numGaps = 0;
            let gapLengths = [];
            let currentGap = 0;

            available.forEach((a, i) => {
                if (a === 0) {
                    if (currentGap === 0) numGaps++;
                    currentGap += 0.5; // Assuming 30min resol
                } else if (currentGap > 0) {
                    gapLengths.push(currentGap);
                    currentGap = 0;
                }
            });

            stats[instName] = {
                availability: (valid / total * 100).toFixed(1) + '%',
                numGaps: numGaps,
                meanGap: gapLengths.length ? (gapLengths.reduce((a, b) => a + b, 0) / gapLengths.length).toFixed(1) + 'h' : '0h',
                maxGap: gapLengths.length ? Math.max(...gapLengths).toFixed(1) + 'h' : '0h',
                color: inst.color
            };
        });

        return stats;
    },

    /**
     * Render gap statistics table using premium layout
     */
    renderGapStatsTable(containerId, data) {
        const stats = this.computeGapStats(data);
        const container = document.getElementById(containerId);
        if (!container) return;

        let html = `
            <div class="table-wrap">
                <table class="premium-table">
                    <thead>
                        <tr>
                            <th>Instrument</th>
                            <th>Uptime</th>
                            <th>Events</th>
                            <th>Avg Gap</th>
                            <th>Max Gap</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        Object.entries(stats).forEach(([instName, s]) => {
            html += `
                <tr>
                    <td>
                        <div style="display:flex; align-items:center; gap:10px;">
                            <div style="width:10px; height:10px; border-radius:50%; background:${s.color}; box-shadow:0 0 8px ${s.color};"></div>
                            <strong>${instName}</strong>
                        </div>
                    </td>
                    <td class="numeric" style="color:#f8fafc">${s.availability}</td>
                    <td class="numeric">${s.numGaps}</td>
                    <td class="numeric">${s.meanGap}</td>
                    <td class="numeric">${s.maxGap}</td>
                </tr>
            `;
        });

        html += '</tbody></table></div>';
        container.innerHTML = html;
    }
};

window.GapTimeline = GapTimeline;
