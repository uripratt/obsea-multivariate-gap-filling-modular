/**
 * OBSEA Digital Twin - Main Application Entry Point
 * Orchestrates all 3D modules and connects to real oceanographic data
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SceneManager } from './sceneManager.js';
import { Bathymetry } from './bathymetry.js';
import { WaterSurface } from './waterSurface.js';
import { CurrentField } from './currentField.js';
import { Observatory } from './observatory.js';
import { DataConnector } from './dataConnector.js';

export class DigitalTwinApp {
    constructor() {
        this.container = document.getElementById('scene-container');
        this.loadingScreen = document.getElementById('loading-screen');
        this.loadingProgress = document.getElementById('loading-progress');

        this.sceneManager = null;
        this.bathymetry = null;
        this.water = null;
        this.currents = null;
        this.observatory = null;
        this.dataConnector = null;

        // Animation state
        this.clock = new THREE.Clock();
        this.animationSpeed = 1.0;
        this.isRunning = true;

        // Current data state
        this.currentData = {
            temp: 18.5,
            psal: 38.2,
            currentSpeed: 0.15,
            currentDir: 45,
            waveHeight: 0.8
        };
    }

    async init() {
        console.log('🌊 OBSEA Digital Twin Initializing...');

        try {
            // Step 1: Create scene manager
            this.updateProgress('Creating 3D scene...');
            this.sceneManager = new SceneManager(this.container);
            await this.sceneManager.init();

            // Step 2: Add bathymetry
            this.updateProgress('Loading bathymetry data...');
            this.bathymetry = new Bathymetry(this.sceneManager.scene);
            await this.bathymetry.init();

            // Step 3: Add water surface
            this.updateProgress('Generating water surface...');
            this.water = new WaterSurface(this.sceneManager.scene);
            await this.water.init();

            // Step 4: Add current field
            this.updateProgress('Creating current field visualization...');
            this.currents = new CurrentField(this.sceneManager.scene);
            await this.currents.init();

            // Step 5: Add observatory
            this.updateProgress('Building observatory model...');
            this.observatory = new Observatory(this.sceneManager.scene);
            await this.observatory.init();

            // Step 6: Connect to data
            this.updateProgress('Connecting to sensor data...');
            this.dataConnector = new DataConnector();
            await this.dataConnector.init();

            // Setup controls
            this.setupControls();

            // Step 7: Load XBeach simulation data
            this.updateProgress('Loading XBeach wave simulation...');
            await this.loadXBeachData();

            // Hide loading screen
            this.hideLoading();

            // Start animation loop
            this.animate();

            console.log('✅ Digital Twin Ready!');

        } catch (error) {
            console.error('❌ Digital Twin initialization failed:', error);
            this.updateProgress(`Error: ${error.message}`);
        }
    }

    async loadXBeachData() {
        if (!this.water) return;

        try {
            const success = await this.water.loadSimulationData('assets/data/xbeach_waves.bin');
            if (success) {
                console.log('🌊 XBeach simulation activated');
                // Update simulation controls UI
                this.updateSimUI();
            } else {
                console.warn('⚠️ XBeach data not available, using procedural waves');
            }
        } catch (e) {
            console.warn('⚠️ XBeach data not available:', e.message);
        }
    }

    updateProgress(message) {
        if (this.loadingProgress) {
            this.loadingProgress.textContent = message;
        }
        console.log(`📍 ${message}`);
    }

    hideLoading() {
        if (this.loadingScreen) {
            this.loadingScreen.classList.add('hidden');
            setTimeout(() => {
                this.loadingScreen.style.display = 'none';
            }, 500);
        }
    }

    setupControls() {
        // Toggle controls
        const toggleCurrents = document.getElementById('toggle-currents');
        const toggleWaves = document.getElementById('toggle-waves');
        const toggleBathymetry = document.getElementById('toggle-bathymetry');
        const animationSpeedSlider = document.getElementById('animation-speed');
        const currentScaleSlider = document.getElementById('current-scale');

        if (toggleCurrents) {
            toggleCurrents.addEventListener('change', (e) => {
                this.currents?.setVisible(e.target.checked);
            });
        }

        if (toggleWaves) {
            toggleWaves.addEventListener('change', (e) => {
                this.water?.setVisible(e.target.checked);
            });
        }

        if (toggleBathymetry) {
            toggleBathymetry.addEventListener('change', (e) => {
                this.bathymetry?.setVisible(e.target.checked);
            });
        }

        if (animationSpeedSlider) {
            animationSpeedSlider.addEventListener('input', (e) => {
                this.animationSpeed = parseFloat(e.target.value);
            });
        }

        if (currentScaleSlider) {
            currentScaleSlider.addEventListener('input', (e) => {
                this.currents?.setScale(parseFloat(e.target.value));
            });
        }

        // Time slider (original data navigation)
        const timeSlider = document.getElementById('time-slider');
        if (timeSlider) {
            timeSlider.addEventListener('input', (e) => {
                const progress = e.target.value / 100;
                this.onTimeChange(progress);
            });
        }

        // ── XBeach Simulation Controls ──────────────────────────────
        const simPlayBtn = document.getElementById('sim-play-btn');
        const simSlider = document.getElementById('sim-time-slider');
        const simSpeedSl = document.getElementById('sim-speed');
        const simToggle = document.getElementById('toggle-simulation');

        if (simPlayBtn) {
            simPlayBtn.addEventListener('click', () => {
                if (!this.water?.simLoaded) return;
                const playing = !this.water.simPlaying;
                this.water.setSimPlaying(playing);
                simPlayBtn.textContent = playing ? '⏸' : '▶';
                simPlayBtn.title = playing ? 'Pause Simulation' : 'Play Simulation';
            });
        }

        if (simSlider) {
            simSlider.addEventListener('input', (e) => {
                if (!this.water?.simLoaded) return;
                const dur = this.water.getSimDuration();
                const t = (parseFloat(e.target.value) / 1000) * dur;
                this.water.setSimTime(t);
                this.water.setSimPlaying(false);
                if (simPlayBtn) {
                    simPlayBtn.textContent = '▶';
                }
            });
        }

        if (simSpeedSl) {
            simSpeedSl.addEventListener('input', (e) => {
                const speed = parseFloat(e.target.value);
                if (this.water) this.water.setSimSpeed(speed);
                const label = document.getElementById('sim-speed-label');
                if (label) label.textContent = `${speed.toFixed(1)}×`;
            });
        }

        if (simToggle) {
            simToggle.addEventListener('change', (e) => {
                if (!this.water) return;
                if (e.target.checked) {
                    // Reload simulation if deactivated
                    if (!this.water.simLoaded) {
                        this.loadXBeachData();
                    }
                } else {
                    this.water.deactivateSimulation();
                }
            });
        }
    }

    updateSimUI() {
        const simPanel = document.getElementById('sim-controls-panel');
        if (simPanel) simPanel.style.display = 'block';

        const simToggle = document.getElementById('toggle-simulation');
        if (simToggle) simToggle.checked = true;
    }

    onTimeChange(progress) {
        // Update time display
        const startYear = 2015;
        const endYear = 2026;
        const currentYear = startYear + (endYear - startYear) * progress;

        const date = new Date(currentYear, 0, 1);
        date.setDate(date.getDate() + (365 * (currentYear % 1)));

        const timeDisplay = document.getElementById('current-time');
        if (timeDisplay) {
            timeDisplay.textContent = date.toISOString().slice(0, 19).replace('T', ' ');
        }

        // Request data for this time
        if (this.dataConnector) {
            this.dataConnector.getDataAtTime(progress).then(data => {
                this.updateVisualization(data);
            });
        }
    }

    updateVisualization(data) {
        if (!data) return;

        // Update HUD
        this.updateHUD(data);

        // Update currents
        if (this.currents && data.currentSpeed !== undefined) {
            this.currents.updateFromData(data.currentSpeed, data.currentDir);
        }

        // Update waves with height, period, and direction from AWAC
        // Amplify waveHeight for visual impact (real Hs 0.3-2.5m → visual 0.75-6.25m)
        if (this.water) {
            if (data.waveHeight !== undefined) {
                this.water.setWaveHeight(data.waveHeight * 2.5);
            }
            if (data.waveDir !== undefined) {
                this.water.setWaveDirection(data.waveDir);
            }
            if (data.wavePeriod !== undefined) {
                this.water.setWavePeriod(data.wavePeriod);
            }
        }

        // Update bathymetry caustics with wave/current data
        if (this.bathymetry && data.currentSpeed !== undefined) {
            this.bathymetry.updateFromData(
                data.currentSpeed,
                data.currentDir,
                data.waveHeight
            );
        }
    }

    updateHUD(data) {
        const elements = {
            'hud-temp': data.temp?.toFixed(1),
            'hud-psal': data.psal?.toFixed(1),
            'hud-current': data.currentSpeed?.toFixed(2),
            'hud-wave': data.waveHeight?.toFixed(1)
        };

        Object.entries(elements).forEach(([id, value]) => {
            const el = document.getElementById(id);
            if (el && value !== undefined) {
                el.textContent = value;
            }
        });
    }

    animate() {
        if (!this.isRunning) return;

        requestAnimationFrame(() => this.animate());

        const delta = this.clock.getDelta() * this.animationSpeed;
        const elapsed = this.clock.getElapsedTime();

        // Update all animated components
        if (this.water) {
            this.water.update(elapsed, delta);
        }

        if (this.currents) {
            this.currents.update(elapsed, delta);
        }

        if (this.observatory) {
            this.observatory.update(elapsed, delta);
        }

        // Animate bathymetry caustics
        if (this.bathymetry) {
            this.bathymetry.update(elapsed);
        }

        // Update simulation time display
        if (this.water?.simLoaded) {
            const simTimeDisplay = document.getElementById('sim-time-display');
            if (simTimeDisplay) {
                const t = this.water.getSimTime().toFixed(0);
                simTimeDisplay.textContent = `t = ${t}s`;
            }

            // Update slider position
            const simSlider = document.getElementById('sim-time-slider');
            if (simSlider && this.water.simPlaying) {
                simSlider.value = Math.round(this.water.getSimProgress() * 1000);
            }
        }

        // Render scene
        if (this.sceneManager) {
            this.sceneManager.render();
        }
    }

    dispose() {
        this.isRunning = false;

        this.bathymetry?.dispose();
        this.water?.dispose();
        this.currents?.dispose();
        this.observatory?.dispose();
        this.sceneManager?.dispose();
    }
}

// Auto-initialization removed for webapp integration
// The app will be initialized by app.js when the tab is active
