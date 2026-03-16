/**
 * Bathymetry Module - Realistic Mediterranean seafloor
 * Features: procedural sand ripples, posidonia patches, animated caustics
 *           with chromatic dispersion, depth-based color grading
 */

import * as THREE from 'three';

export class Bathymetry {
    constructor(scene) {
        this.scene = scene;
        this.mesh = null;

        // Terrain configuration
        this.config = {
            width: 400,          // 400m x 400m area
            depth: 400,
            segments: 128,       // Higher resolution
            maxDepth: 35,        // Maximum depth in meters
            minDepth: 15,        // Minimum depth (near coast)
            noiseScale: 0.02,    // Terrain roughness
            observatoryDepth: 20 // Observatory at 20m
        };
    }

    async init() {
        const geometry = this.createTerrainGeometry();
        const material = this.createTerrainMaterial();

        this.mesh = new THREE.Mesh(geometry, material);
        this.mesh.receiveShadow = true;
        this.mesh.rotation.x = -Math.PI / 2;

        this.scene.add(this.mesh);

        return Promise.resolve();
    }

    createTerrainGeometry() {
        const { width, depth, segments, noiseScale } = this.config;

        const geometry = new THREE.PlaneGeometry(width, depth, segments, segments);
        const positions = geometry.attributes.position.array;

        for (let i = 0; i < positions.length; i += 3) {
            const x = positions[i];
            const y = positions[i + 1];

            const baseDepth = this.config.observatoryDepth;

            // Gentle slope
            const slope = (x / width) * 2.0;

            // Natural terrain (sand ripples handled in shader)
            const noise = this.fbmNoise(x * noiseScale, y * noiseScale) * 0.5;

            const finalDepth = baseDepth + slope + noise;
            positions[i + 2] = -finalDepth;
        }

        geometry.computeVertexNormals();

        return geometry;
    }

    createTerrainMaterial() {
        const loader = new THREE.TextureLoader();
        const sandTexture = loader.load('assets/textures/sand_texture.png');
        const posidoniaTexture = loader.load('assets/textures/posidonia_texture.png');

        sandTexture.wrapS = sandTexture.wrapT = THREE.RepeatWrapping;
        posidoniaTexture.wrapS = posidoniaTexture.wrapT = THREE.RepeatWrapping;

        return new THREE.ShaderMaterial({
            uniforms: {
                // Textures
                sandTexture: { value: sandTexture },
                posidoniaTexture: { value: posidoniaTexture },

                // Texture tiling
                sandScale: { value: 25.0 },
                posidoniaScale: { value: 12.0 },

                // Sand color palette (warmer, Mediterranean)
                sandColorWarm: { value: new THREE.Color(0xD4B896) },  // Warm sand
                sandColorCool: { value: new THREE.Color(0xB8A07A) },  // Cool shadow
                deepColor: { value: new THREE.Color(0x0C2233) },  // Deep sea

                // Depth parameters
                minDepth: { value: this.config.minDepth },
                maxDepth: { value: this.config.maxDepth },

                // Fog (underwater)
                fogColor: { value: new THREE.Color(0x061a28) },
                fogDensity: { value: 0.007 },

                // Animation
                time: { value: 0 },
                currentDirection: { value: 0.0 },
                currentSpeed: { value: 0.1 },
                causticIntensity: { value: 0.6 },
                waveHeight: { value: 0.8 }
            },
            vertexShader: `
                varying float vDepth;
                varying vec3 vNormal;
                varying vec3 vPosition;
                varying vec2 vWorldXZ;
                varying vec2 vUv;

                void main() {
                    vDepth = -position.z;
                    vNormal = normalize(normalMatrix * normal);
                    vPosition = (modelMatrix * vec4(position, 1.0)).xyz;
                    vWorldXZ = position.xy;
                    vUv = uv;

                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D sandTexture;
                uniform sampler2D posidoniaTexture;
                uniform float sandScale;
                uniform float posidoniaScale;

                uniform vec3 sandColorWarm;
                uniform vec3 sandColorCool;
                uniform vec3 deepColor;
                uniform float minDepth;
                uniform float maxDepth;
                uniform vec3 fogColor;
                uniform float fogDensity;
                uniform float time;
                uniform float currentDirection;
                uniform float currentSpeed;
                uniform float causticIntensity;
                uniform float waveHeight;

                varying float vDepth;
                varying vec3 vNormal;
                varying vec3 vPosition;
                varying vec2 vWorldXZ;
                varying vec2 vUv;

                // ── Noise functions ──
                float hash(vec2 p) {
                    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
                }

                float noise(vec2 p) {
                    vec2 i = floor(p);
                    vec2 f = fract(p);
                    f = f * f * (3.0 - 2.0 * f);
                    float a = hash(i);
                    float b = hash(i + vec2(1.0, 0.0));
                    float c = hash(i + vec2(0.0, 1.0));
                    float d = hash(i + vec2(1.0, 1.0));
                    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
                }

                float fbm(vec2 p) {
                    float v = 0.0;
                    float amp = 0.5;
                    for (int i = 0; i < 4; i++) {
                        v += amp * noise(p);
                        p *= 2.01;
                        amp *= 0.5;
                    }
                    return v;
                }

                // ── Procedural sand ripples ──
                float sandRipples(vec2 uv, float curDir) {
                    // Ripples perpendicular to current
                    float cd = curDir + 1.5708; // +90°
                    vec2 rDir = vec2(cos(cd), sin(cd));
                    float proj = dot(uv, rDir);

                    // Multi-scale ripples
                    float r1 = sin(proj * 12.0 + noise(uv * 3.0) * 2.0) * 0.5 + 0.5;
                    float r2 = sin(proj * 25.0 + noise(uv * 6.0) * 1.5) * 0.5 + 0.5;
                    float r3 = sin(proj * 50.0 + noise(uv * 12.0) * 1.0) * 0.5 + 0.5;

                    return r1 * 0.5 + r2 * 0.3 + r3 * 0.2;
                }

                // ── Single-channel caustic pattern ──
                float causticSingle(vec2 uv, float t) {
                    uv *= 10.0;
                    float v = 0.0;
                    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));

                    for(int i = 0; i < 3; i++) {
                        v += sin(uv.x * 2.0 + t * 0.6) * sin(uv.y * 2.0 - t * 0.4);
                        v += sin(uv.x * 1.5 - t * 0.3 + uv.y * 0.7) * 0.5;
                        uv = rot * uv * 1.3;
                    }
                    return pow(max(v * 0.3 + 0.5, 0.0), 6.0);
                }

                // ── Voronoi caustics with chromatic dispersion ──
                vec3 causticsChromatic(vec2 uv, float t) {
                    float cR = causticSingle(uv + vec2(0.003, 0.0), t);
                    float cG = causticSingle(uv, t);
                    float cB = causticSingle(uv - vec2(0.003, 0.0), t);
                    return vec3(cR, cG, cB);
                }

                // ── Sand sparkles (specular glints from grains) ──
                float sparkles(vec2 uv, float t) {
                    float n = hash(uv * 600.0 + t * 0.005);
                    return pow(n, 25.0);
                }

                void main() {
                    // Texture coordinates
                    vec2 sandUV = vUv * sandScale;
                    vec2 posidoniaUV = vUv * posidoniaScale;

                    // ── Sample base textures ──
                    vec3 colSand = texture2D(sandTexture, sandUV).rgb;

                    // Warm up sand color with Mediterranean tones
                    colSand = mix(colSand, sandColorWarm, 0.45);

                    // ── Procedural sand ripples ──
                    float ripple = sandRipples(vWorldXZ * 0.15, currentDirection);
                    // Ripples create subtle light/shadow variation
                    colSand *= 0.85 + ripple * 0.3;
                    // Slightly cooler in troughs
                    colSand = mix(colSand, sandColorCool, (1.0 - ripple) * 0.15);

                    // ── Sand sparkles (bright grain reflections) ──
                    vec3 viewDir = normalize(cameraPosition - vPosition);
                    vec3 sunDir = normalize(vec3(0.5, 1.0, 0.5));
                    vec3 h = normalize(viewDir + sunDir);
                    float NdotH = max(0.0, dot(vNormal, h));

                    float sparkleVal = sparkles(vUv, time);
                    colSand += vec3(1.0, 0.95, 0.85) * sparkleVal * (NdotH * 2.5);

                    // ── Posidonia seagrass ──
                    vec3 colPosidonia = texture2D(posidoniaTexture, posidoniaUV).rgb;

                    // Vegetation distribution mask (organic patches)
                    float vegNoise = fbm(vWorldXZ * 0.025);
                    float vegMask = smoothstep(0.38, 0.58, vegNoise);

                    // Gentle sway animation on posidonia
                    float sway = sin(time * 0.8 + vWorldXZ.x * 0.1) * 0.03;
                    colPosidonia *= 1.0 + sway;

                    // Mix sand and posidonia
                    vec3 baseColor = mix(colSand, colPosidonia, vegMask);

                    // ── Depth-based color gradient ──
                    float depthFactor = clamp((vDepth - minDepth) / (maxDepth - minDepth), 0.0, 1.0);

                    // Smooth gradient: shallow warm sand → deep cold blue
                    baseColor = mix(baseColor, deepColor, depthFactor * 0.65);

                    // ── Lighting (diffuse + ambient) ──
                    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
                    float diffuse = max(dot(vNormal, lightDir), 0.35);
                    baseColor *= diffuse;

                    // ── Caustics — chromatic dispersion ──
                    vec3 causticColor = causticsChromatic(vWorldXZ * 0.05, time);

                    // Fade caustics with depth (strong shallow, absent deep)
                    float depthFade = 1.0 - smoothstep(5.0, 28.0, vDepth);
                    // Scale with wave conditions
                    float causticStrength = causticIntensity * (0.5 + waveHeight * 0.5) * depthFade;

                    baseColor += causticColor * causticStrength;

                    // ── Underwater fog ──
                    float dist = length(vPosition - cameraPosition);
                    float fogFactor = 1.0 - exp(-fogDensity * dist);
                    baseColor = mix(baseColor, fogColor, fogFactor);

                    gl_FragColor = vec4(baseColor, 1.0);
                }
            `,
            side: THREE.DoubleSide
        });
    }

    // Update caustics and current effects
    updateFromData(currentSpeed, currentDir, waveHeight) {
        if (this.mesh && this.mesh.material && this.mesh.material.uniforms) {
            this.mesh.material.uniforms.currentDirection.value = currentDir * Math.PI / 180;
            this.mesh.material.uniforms.currentSpeed.value = currentSpeed;
            if (waveHeight !== undefined) {
                this.mesh.material.uniforms.waveHeight.value = waveHeight;
            }
        }
    }

    // Animate caustics
    update(elapsedTime) {
        if (this.mesh && this.mesh.material && this.mesh.material.uniforms) {
            this.mesh.material.uniforms.time.value = elapsedTime;
        }
    }

    // Fractal Brownian Motion noise
    fbmNoise(x, y) {
        let value = 0;
        let amplitude = 1;
        let frequency = 1;

        for (let i = 0; i < 4; i++) {
            value += amplitude * this.noise2D(x * frequency, y * frequency);
            amplitude *= 0.5;
            frequency *= 2;
        }

        return value;
    }

    // Simple 2D noise function (pseudo-random based on position)
    noise2D(x, y) {
        const n = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
        return (n - Math.floor(n)) * 2 - 1;
    }

    lerp(a, b, t) {
        return a + (b - a) * t;
    }

    setVisible(visible) {
        if (this.mesh) this.mesh.visible = visible;
    }

    dispose() {
        if (this.mesh) {
            this.mesh.geometry.dispose();
            this.mesh.material.dispose();
            this.scene.remove(this.mesh);
        }
    }
}
