/**
 * Water Surface Module — Photorealistic Ocean Renderer
 *
 * Techniques:
 *   • 8-octave Gerstner waves with ocean-spectrum amplitudes
 *   • Analytical normal computation from Gerstner partial derivatives
 *   • Triple-layer parallax normal mapping
 *   • PBR Fresnel (Schlick, IOR 1.33)
 *   • GGX-inspired specular highlight (sun disc)
 *   • Subsurface scattering (light through wave crests)
 *   • Multi-layer foam: whitecaps + trailing streaks + shore foam
 *   • Depth-aware color: deep navy troughs → bright teal crests
 *   • XBeach simulation overlay mode
 */

import * as THREE from 'three';

const HEADER_BYTES = 4096;

export class WaterSurface {
    constructor(scene) {
        this.scene = scene;
        this.mesh = null;
        this.material = null;

        this.waveParams = {
            height: 1.8,
            frequency: 0.5,
            speed: 1.0,
            direction: 45,
            steepness: 0.45
        };

        this.config = {
            size: 500,
            segments: 256       // High-res for smooth wave geometry
        };

        // Simulation state
        this.simLoaded = false;
        this.simPlaying = true;
        this.simTime = 0;
        this.simSpeed = 1.0;
        this.simMeta = null;
        this.simZs = null;
        this.simH = null;
    }

    async init() {
        const geometry = this.createWaterGeometry();
        this.material = this.createWaterMaterial();

        this.mesh = new THREE.Mesh(geometry, this.material);
        this.mesh.position.y = 0;
        this.mesh.rotation.x = -Math.PI / 2;
        this.scene.add(this.mesh);
        return Promise.resolve();
    }

    createWaterGeometry() {
        return new THREE.PlaneGeometry(
            this.config.size, this.config.size,
            this.config.segments, this.config.segments
        );
    }

    createSimulationGeometry() {
        const { nx, ny, domain_width_m, domain_height_m } = this.simMeta;
        return new THREE.PlaneGeometry(domain_width_m, domain_height_m, nx - 1, ny - 1);
    }

    // ══════════════════════════════════════════════════════════════
    //  MATERIAL — Photorealistic Ocean Shader
    // ══════════════════════════════════════════════════════════════
    createWaterMaterial() {
        const loader = new THREE.TextureLoader();
        const normalMap = loader.load('assets/textures/water_normal.png');
        normalMap.wrapS = normalMap.wrapT = THREE.RepeatWrapping;

        return new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                waveHeight: { value: this.waveParams.height },
                waveFrequency: { value: this.waveParams.frequency },
                waveDirection: { value: this.waveParams.direction * Math.PI / 180 },
                waveSteepness: { value: this.waveParams.steepness },
                simMode: { value: 0.0 },

                normalMap: { value: normalMap },
                normalScale: { value: new THREE.Vector2(1.8, 1.8) },

                // Ocean color palette
                deepColor: { value: new THREE.Color(0x001828) },
                midColor: { value: new THREE.Color(0x04516e) },
                shallowColor: { value: new THREE.Color(0x1aafcc) },
                foamColor: { value: new THREE.Color(0xf5fbff) },
                skyColor: { value: new THREE.Color(0x1a3a52) },
                sunColor: { value: new THREE.Color(0xfff8ee) },
                sssColor: { value: new THREE.Color(0x0fc9a4) },

                opacity: { value: 0.93 },
                sunDirection: { value: new THREE.Vector3(0.4, 0.85, 0.35).normalize() }
            },

            vertexShader: /* glsl */ `
                uniform float time;
                uniform float waveHeight;
                uniform float waveDirection;
                uniform float waveSteepness;
                uniform float simMode;

                attribute float simElevation;
                attribute float simWaveH;

                varying vec2  vUv;
                varying float vElevation;
                varying float vFoam;
                varying vec3  vViewPosition;
                varying vec3  vWorldPosition;
                varying vec3  vWorldNormal;

                // ── Gerstner displacement + analytical normal ──
                // Returns vec3(dx, dy, dz)
                // Also accumulates partial derivatives for normal computation
                vec3 gerstnerDisp(vec2 pos, float amp, float freq, float phase,
                                  vec2 dir, float steep,
                                  inout vec3 tangent, inout vec3 binormal) {
                    float theta = dot(dir, pos) * freq + phase;
                    float s = sin(theta);
                    float c = cos(theta);
                    float qa = steep * amp;

                    // Partial derivatives for normal
                    tangent  += vec3(
                        -dir.x * dir.x * qa * freq * s,
                        -dir.x * dir.y * qa * freq * s,
                         dir.x * amp * freq * c
                    );
                    binormal += vec3(
                        -dir.x * dir.y * qa * freq * s,
                        -dir.y * dir.y * qa * freq * s,
                         dir.y * amp * freq * c
                    );

                    return vec3(
                        dir.x * qa * c,
                        dir.y * qa * c,
                        amp * s
                    );
                }

                void main() {
                    vUv = uv * 10.0;

                    vec3 pos = position;
                    float t = time;

                    if (simMode > 0.5) {
                        // ── Simulation mode ──
                        pos.z = simElevation * 350.0;
                        vFoam = simWaveH;

                        vec2 d1 = vec2(cos(0.3), sin(0.3));
                        vec2 d2 = vec2(cos(1.8), sin(1.8));
                        vec3 tang = vec3(1.0, 0.0, 0.0);
                        vec3 bino = vec3(0.0, 1.0, 0.0);
                        vec3 g1 = gerstnerDisp(position.xy, 3.0, 0.01, t*2.0, d1, 0.4, tang, bino);
                        vec3 g2 = gerstnerDisp(position.xy, 1.5, 0.02, t*1.5, d2, 0.3, tang, bino);
                        pos.z += g1.z + g2.z;

                        vElevation = pos.z / 50.0;
                        vWorldNormal = normalize(cross(binormal, tangent));

                    } else {
                        // ── Procedural mode: 8-octave Gerstner (ocean spectrum) ──
                        float dir = waveDirection;
                        float h = waveHeight;
                        float Q = waveSteepness;

                        // For analytical normal accumulation
                        vec3 tang = vec3(1.0, 0.0, 0.0);
                        vec3 bino = vec3(0.0, 1.0, 0.0);

                        // Octave 1: Primary swell (wavelength ~120m, period ~8.8s)
                        vec2 d1 = vec2(cos(dir), sin(dir));
                        vec3 g1 = gerstnerDisp(pos.xy, h*0.48, 0.018, t*0.72, d1, Q, tang, bino);

                        // Octave 2: Secondary swell shifted 25° (~70m, ~6.7s)
                        vec2 d2 = vec2(cos(dir+0.44), sin(dir+0.44));
                        vec3 g2 = gerstnerDisp(pos.xy, h*0.28, 0.028, t*0.95, d2, Q*0.92, tang, bino);

                        // Octave 3: Cross-swell -40° (~45m, ~5.4s)
                        vec2 d3 = vec2(cos(dir-0.7), sin(dir-0.7));
                        vec3 g3 = gerstnerDisp(pos.xy, h*0.18, 0.044, t*0.62, d3, Q*0.82, tang, bino);

                        // Octave 4: Medium-wavelength chop (~28m)
                        vec2 d4 = vec2(cos(dir+1.5), sin(dir+1.5));
                        vec3 g4 = gerstnerDisp(pos.xy, h*0.10, 0.072, t*1.35, d4, Q*0.7, tang, bino);

                        // Octave 5: Short chop -110° (~16m)
                        vec2 d5 = vec2(cos(dir-1.9), sin(dir-1.9));
                        vec3 g5 = gerstnerDisp(pos.xy, h*0.06, 0.12, t*1.8, d5, Q*0.55, tang, bino);

                        // Octave 6: Fine detail 60° (~9m)
                        vec2 d6 = vec2(cos(dir+1.05), sin(dir+1.05));
                        vec3 g6 = gerstnerDisp(pos.xy, h*0.035, 0.20, t*2.3, d6, Q*0.40, tang, bino);

                        // Octave 7: Micro ripple -170° (~5m)
                        vec2 d7 = vec2(cos(dir-2.96), sin(dir-2.96));
                        vec3 g7 = gerstnerDisp(pos.xy, h*0.018, 0.35, t*3.0, d7, Q*0.25, tang, bino);

                        // Octave 8: Capillary detail 130° (~3m)
                        vec2 d8 = vec2(cos(dir+2.27), sin(dir+2.27));
                        vec3 g8 = gerstnerDisp(pos.xy, h*0.008, 0.55, t*3.8, d8, Q*0.15, tang, bino);

                        // Sum all displacements
                        vec3 disp = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8;
                        pos.xy += disp.xy;
                        pos.z   = disp.z;

                        vElevation = pos.z;

                        // Foam: peaks above threshold
                        float foamStart = h * 0.30;
                        float foamFull  = h * 0.65;
                        vFoam = smoothstep(foamStart, foamFull, pos.z);

                        // ── Analytical normal from Gerstner partial derivatives ──
                        vWorldNormal = normalize(vec3(-tang.z, -bino.z, 1.0));
                    }

                    vec4 worldPos = modelMatrix * vec4(pos, 1.0);
                    vWorldPosition = worldPos.xyz;

                    vec4 mvPos = modelViewMatrix * vec4(pos, 1.0);
                    vViewPosition = -mvPos.xyz;
                    gl_Position = projectionMatrix * mvPos;

                    // Transform the analytically computed normal to world space
                    vWorldNormal = normalize(mat3(modelMatrix) * vWorldNormal);
                }
            `,

            fragmentShader: /* glsl */ `
                uniform sampler2D normalMap;
                uniform vec2 normalScale;

                uniform vec3  deepColor;
                uniform vec3  midColor;
                uniform vec3  shallowColor;
                uniform vec3  foamColor;
                uniform vec3  skyColor;
                uniform vec3  sunColor;
                uniform vec3  sssColor;
                uniform vec3  sunDirection;
                uniform float opacity;
                uniform float time;
                uniform float waveHeight;

                varying vec2  vUv;
                varying float vElevation;
                varying float vFoam;
                varying vec3  vViewPosition;
                varying vec3  vWorldPosition;
                varying vec3  vWorldNormal;

                // ── Noise utility ──
                float hash21(vec2 p) {
                    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
                }

                float noise2d(vec2 p) {
                    vec2 i = floor(p);
                    vec2 f = fract(p);
                    f = f * f * (3.0 - 2.0 * f);
                    float a = hash21(i);
                    float b = hash21(i + vec2(1.0, 0.0));
                    float c = hash21(i + vec2(0.0, 1.0));
                    float d = hash21(i + vec2(1.0, 1.0));
                    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
                }

                float fbm3(vec2 p) {
                    float v = 0.0;
                    v += 0.50 * noise2d(p); p *= 2.03;
                    v += 0.25 * noise2d(p); p *= 2.01;
                    v += 0.125 * noise2d(p);
                    return v;
                }

                void main() {
                    // ── Normal mapping: 3 layers at different scales/speeds ──
                    float t = time * 0.4;
                    vec2 uv1 = vUv        + vec2(t * 0.035, t * 0.018);
                    vec2 uv2 = vUv * 1.5  - vec2(t * 0.028, t * 0.04);
                    vec2 uv3 = vUv * 3.0  + vec2(-t * 0.012, t * 0.015);

                    vec3 n1 = texture2D(normalMap, uv1).rgb * 2.0 - 1.0;
                    vec3 n2 = texture2D(normalMap, uv2).rgb * 2.0 - 1.0;
                    vec3 n3 = texture2D(normalMap, uv3).rgb * 2.0 - 1.0;

                    // Blend normal maps with the analytical Gerstner normal
                    vec3 detailNormal = normalize(
                        n1 * normalScale.x +
                        n2 * normalScale.y * 0.6 +
                        n3 * 0.25
                    );

                    // Perturb the analytical normal with detail normal
                    vec3 N = normalize(vWorldNormal + detailNormal * 0.25);

                    // ── View direction ──
                    vec3 V = normalize(vViewPosition);

                    // ── Fresnel (Schlick, F0 for water IOR 1.33) ──
                    float F0 = 0.02;
                    float NdotV = max(0.0, dot(N, V));
                    float fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);

                    // ── Specular — tight sun disc (GGX-inspired) ──
                    vec3 H = normalize(sunDirection + V);
                    float NdotH = max(0.0, dot(N, H));
                    // Two-lobe specular: tight sun disc + soft spread
                    float specTight = pow(NdotH, 512.0) * 3.0;  // Sun disc
                    float specWide  = pow(NdotH, 64.0)  * 0.5;  // Broad glow
                    float spec = specTight + specWide;

                    // ── Subsurface scattering ──
                    float ssDot = max(0.0, dot(V, -sunDirection));
                    float sss = pow(ssDot, 5.0) * 0.30;

                    // ── Ocean color composition ──
                    // Normalized elevation: troughs ~-1, crests ~+1
                    float h = waveHeight > 0.01 ? waveHeight : 1.0;
                    float elevNorm = clamp(vElevation / h, -1.0, 1.5);

                    // Deep → mid → shallow based on elevation + Fresnel
                    float deepMix = smoothstep(-0.8, 0.3, elevNorm);
                    vec3 waterBase = mix(deepColor, midColor, deepMix);

                    float shallowMix = smoothstep(0.0, 1.0, elevNorm) * fresnel;
                    waterBase = mix(waterBase, shallowColor, shallowMix * 0.7);

                    // ── Sky/environment reflection ──
                    // Reflected sky gets blended by Fresnel at grazing angles
                    vec3 envRefl = mix(skyColor, vec3(0.35, 0.55, 0.65), fresnel * 0.4);
                    waterBase = mix(waterBase, envRefl, fresnel * 0.55);

                    // ── SSS: turquoise backlit glow through wave bodies ──
                    waterBase += sssColor * (sss + max(0.0, elevNorm) * 0.08);

                    // ── Sun specular ──
                    waterBase += sunColor * spec;

                    // ── Foam ──────────────────────────────────────────────
                    float totalFoam = 0.0;

                    // 1. Whitecap foam — only on the very highest crests
                    float whitecap = smoothstep(0.70, 1.4, elevNorm) * 0.75;

                    // 2. Data/sim-driven foam (only strong values)
                    float dataFoam = smoothstep(0.35, 0.8, vFoam) * 0.55;

                    totalFoam = max(whitecap, dataFoam);

                    // Organic foam texture — LOW frequency for natural streaky foam patches
                    float ft = time * 0.25;
                    float foamNoise = fbm3(vUv * 2.0 + vec2(ft * 0.06, -ft * 0.04));

                    // Foam dissolves with smooth, natural edges
                    float foamEdge = smoothstep(0.35, 0.60, foamNoise);

                    // Large-scale breakup (not pixel-level noise)
                    float foamBreak = noise2d(vUv * 5.0 + ft * 0.08);
                    foamEdge *= smoothstep(0.30, 0.55, foamBreak);

                    totalFoam *= foamEdge;
                    totalFoam = clamp(totalFoam, 0.0, 0.85);

                    // Foam is bright white, semi-translucent
                    waterBase = mix(waterBase, foamColor, totalFoam * 0.75);

                    // ── Atmospheric distance fog ──────────────────────────
                    float dist = length(vWorldPosition - cameraPosition);
                    float fogFactor = 1.0 - exp(-0.008 * dist);
                    vec3 fogCol = vec3(0.0, 0.10, 0.15);
                    waterBase = mix(waterBase, fogCol, fogFactor);

                    // ── Slight darkening at very far distances ──
                    float horizonDark = smoothstep(200.0, 450.0, dist) * 0.15;
                    waterBase *= (1.0 - horizonDark);

                    gl_FragColor = vec4(waterBase, opacity);
                }
            `,

            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: true
        });
    }

    // ══════════════════════════════════════════════════════════════
    //  LOAD XBeach SIMULATION DATA
    // ══════════════════════════════════════════════════════════════
    async loadSimulationData(url) {
        console.log(`🌊 Loading XBeach data from ${url} ...`);
        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const buffer = await response.arrayBuffer();
            console.log(`📦 Received ${(buffer.byteLength / 1024 / 1024).toFixed(1)} MB`);

            const headerBytes = new Uint8Array(buffer, 0, HEADER_BYTES);
            let headerEnd = headerBytes.indexOf(0);
            if (headerEnd < 0) headerEnd = HEADER_BYTES;
            const headerStr = new TextDecoder().decode(headerBytes.subarray(0, headerEnd));
            this.simMeta = JSON.parse(headerStr);

            const { nx, ny, frames } = this.simMeta;
            const frameSize = nx * ny;
            console.log(`📐 Grid: ${nx}×${ny}  Frames: ${frames}`);

            const zsOffset = HEADER_BYTES;
            const hOffset = zsOffset + frames * frameSize * 4;
            this.simZs = new Float32Array(buffer, zsOffset, frames * frameSize);
            this.simH = new Float32Array(buffer, hOffset, frames * frameSize);

            console.log(`📊 zs range: [${this.simZs.reduce((a, b) => Math.min(a, b)).toFixed(4)}, ${this.simZs.reduce((a, b) => Math.max(a, b)).toFixed(4)}]`);

            this.activateSimulationMode();
            this.simLoaded = true;
            this.simTime = 0;
            console.log('✅ XBeach simulation loaded!');
            return true;
        } catch (err) {
            console.error('❌ Failed to load XBeach data:', err);
            return false;
        }
    }

    // ══════════════════════════════════════════════════════════════
    //  ACTIVATE SIMULATION MODE
    // ══════════════════════════════════════════════════════════════
    activateSimulationMode() {
        const { nx, ny, domain_width_m, domain_height_m } = this.simMeta;

        if (this.mesh) {
            this.mesh.geometry.dispose();
            this.scene.remove(this.mesh);
        }

        const geom = this.createSimulationGeometry();
        const vertexCount = nx * ny;

        const simElev = new Float32Array(vertexCount);
        const simWaveH = new Float32Array(vertexCount);
        geom.setAttribute('simElevation', new THREE.BufferAttribute(simElev, 1));
        geom.setAttribute('simWaveH', new THREE.BufferAttribute(simWaveH, 1));

        this.material.uniforms.simMode.value = 1.0;

        const scaleFactor = this.config.size / Math.max(domain_width_m, domain_height_m);

        this.mesh = new THREE.Mesh(geom, this.material);
        this.mesh.position.y = 0;
        this.mesh.rotation.x = -Math.PI / 2;
        this.mesh.scale.set(scaleFactor, scaleFactor, scaleFactor);
        this.scene.add(this.mesh);

        this.writeFrame(0);
    }

    // ══════════════════════════════════════════════════════════════
    //  WRITE FRAME DATA TO GEOMETRY
    // ══════════════════════════════════════════════════════════════
    writeFrame(frameIndex) {
        if (!this.simMeta || !this.mesh) return;

        const { nx, ny, frames } = this.simMeta;
        const frameSize = nx * ny;
        const fi = Math.min(Math.max(0, Math.floor(frameIndex)), frames - 1);
        const offset = fi * frameSize;

        const elevAttr = this.mesh.geometry.getAttribute('simElevation');
        const waveHAttr = this.mesh.geometry.getAttribute('simWaveH');

        const fi2 = Math.min(fi + 1, frames - 1);
        const frac = frameIndex - fi;
        const offset2 = fi2 * frameSize;

        for (let i = 0; i < frameSize; i++) {
            elevAttr.array[i] = this.simZs[offset + i] + (this.simZs[offset2 + i] - this.simZs[offset + i]) * frac;
            waveHAttr.array[i] = this.simH[offset + i] + (this.simH[offset2 + i] - this.simH[offset + i]) * frac;
        }

        elevAttr.needsUpdate = true;
        waveHAttr.needsUpdate = true;
    }

    // ══════════════════════════════════════════════════════════════
    //  UPDATE (per-frame)
    // ══════════════════════════════════════════════════════════════
    update(elapsedTime, deltaTime) {
        if (this.material && this.material.uniforms) {
            this.material.uniforms.time.value = elapsedTime * this.waveParams.speed;
        }

        if (this.simLoaded && this.simPlaying && this.simMeta) {
            const { dt, t_start, t_end } = this.simMeta;
            const duration = t_end - t_start;

            this.simTime += deltaTime * this.simSpeed;
            if (this.simTime >= duration) this.simTime = this.simTime % duration;

            this.writeFrame(this.simTime / dt);
        }
    }

    // ══════════════════════════════════════════════════════════════
    //  PLAYBACK CONTROLS
    // ══════════════════════════════════════════════════════════════
    setSimPlaying(playing) { this.simPlaying = playing; }
    setSimSpeed(speed) { this.simSpeed = speed; }

    setSimTime(seconds) {
        if (!this.simMeta) return;
        this.simTime = Math.max(0, Math.min(seconds, this.simMeta.t_end - this.simMeta.t_start));
        this.writeFrame(this.simTime / this.simMeta.dt);
    }

    getSimTime() { return this.simTime; }
    getSimDuration() { return this.simMeta ? this.simMeta.t_end - this.simMeta.t_start : 0; }
    getSimProgress() { const d = this.getSimDuration(); return d > 0 ? this.simTime / d : 0; }

    // ══════════════════════════════════════════════════════════════
    //  SWITCH BACK TO PROCEDURAL
    // ══════════════════════════════════════════════════════════════
    deactivateSimulation() {
        if (!this.simLoaded) return;
        if (this.mesh) { this.mesh.geometry.dispose(); this.scene.remove(this.mesh); }

        const geom = this.createWaterGeometry();
        this.material.uniforms.simMode.value = 0.0;
        this.mesh = new THREE.Mesh(geom, this.material);
        this.mesh.position.y = 0;
        this.mesh.rotation.x = -Math.PI / 2;
        this.scene.add(this.mesh);
        this.simLoaded = false;
    }

    // ══════════════════════════════════════════════════════════════
    //  PARAMETER SETTERS
    // ══════════════════════════════════════════════════════════════
    setWaveHeight(height) {
        this.waveParams.height = height;
        if (this.material?.uniforms) {
            this.material.uniforms.waveHeight.value = height;
        }
    }

    setWaveDirection(direction) {
        this.waveParams.direction = direction;
        if (this.material?.uniforms) {
            this.material.uniforms.waveDirection.value = direction * Math.PI / 180;
        }
    }

    setWavePeriod(period) {
        if (period > 0) {
            this.waveParams.frequency = 1.0 / period;
            if (this.material?.uniforms) {
                this.material.uniforms.waveFrequency.value = this.waveParams.frequency;
            }
        }
    }

    setVisible(visible) { if (this.mesh) this.mesh.visible = visible; }

    dispose() {
        if (this.mesh) {
            this.mesh.geometry.dispose();
            this.material.dispose();
            this.scene.remove(this.mesh);
        }
        this.simZs = null;
        this.simH = null;
    }
}
