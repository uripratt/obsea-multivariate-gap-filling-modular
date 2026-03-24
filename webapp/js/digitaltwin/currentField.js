/**
 * Current Field Module - 3D visualization of ocean currents
 * Displays current vectors as animated arrows
 */

import * as THREE from 'three';

export class CurrentField {
    constructor(scene) {
        this.scene = scene;
        this.arrowsGroup = null;
        this.particlesGroup = null;
        this.arrows = [];
        this.particles = null;

        // Current parameters (from AWAC data)
        this.currentData = {
            speed: 0.15,      // m/s
            direction: 45,    // degrees from North
            uComponent: 0.1,  // East-West component
            vComponent: 0.1,  // North-South component
            wComponent: 0.0   // Vertical component
        };

        // Field configuration
        this.config = {
            gridSize: 5,           // 5x5 grid of arrows
            gridSpacing: 40,       // 40m between arrows
            arrowScale: 10,        // Visual scale factor
            arrowColor: 0x00ffaa,  // Cyan-green
            particleCount: 2000,   // Number of flow particles
            depthLayers: 3         // Depth layers for 3D field
        };
    }

    async init() {
        // Create arrow field
        this.createArrowField();

        // Create particle system for flow visualization
        this.createParticleSystem();

        return Promise.resolve();
    }

    createArrowField() {
        this.arrowsGroup = new THREE.Group();
        this.arrowsGroup.name = 'currentField';

        const { gridSize, gridSpacing, arrowScale, arrowColor, depthLayers } = this.config;
        const depths = [-5, -15, -25]; // Depth layers
        const halfGrid = (gridSize - 1) / 2;

        // Create arrows at each grid point
        for (let layer = 0; layer < depthLayers; layer++) {
            const depth = depths[layer];
            const layerOpacity = 1 - layer * 0.2;

            for (let i = 0; i < gridSize; i++) {
                for (let j = 0; j < gridSize; j++) {
                    const x = (i - halfGrid) * gridSpacing;
                    const z = (j - halfGrid) * gridSpacing;

                    const arrow = this.createArrow(arrowColor, layerOpacity);
                    arrow.position.set(x, depth, z);

                    // Store reference for animation
                    this.arrows.push({
                        mesh: arrow,
                        basePosition: new THREE.Vector3(x, depth, z),
                        phase: Math.random() * Math.PI * 2 // Random phase for variation
                    });

                    this.arrowsGroup.add(arrow);
                }
            }
        }

        this.scene.add(this.arrowsGroup);

        // Initial update
        this.updateArrows();
    }

    createArrow(color, opacity = 1.0) {
        const group = new THREE.Group();

        // Arrow shaft (cylinder)
        const shaftGeometry = new THREE.CylinderGeometry(0.15, 0.15, 1, 8);
        const shaftMaterial = new THREE.MeshPhongMaterial({
            color: color,
            transparent: true,
            opacity: opacity * 0.8,
            emissive: color,
            emissiveIntensity: 0.3
        });
        const shaft = new THREE.Mesh(shaftGeometry, shaftMaterial);
        shaft.rotation.z = Math.PI / 2; // Point horizontally
        shaft.position.x = 0.5; // Center offset
        group.add(shaft);

        // Arrow head (cone)
        const headGeometry = new THREE.ConeGeometry(0.4, 0.8, 8);
        const headMaterial = new THREE.MeshPhongMaterial({
            color: color,
            transparent: true,
            opacity: opacity,
            emissive: color,
            emissiveIntensity: 0.5
        });
        const head = new THREE.Mesh(headGeometry, headMaterial);
        head.rotation.z = -Math.PI / 2; // Point in positive X
        head.position.x = 1.4;
        group.add(head);

        // Glow effect
        const glowGeometry = new THREE.SphereGeometry(0.3, 8, 8);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: opacity * 0.3
        });
        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
        glow.position.x = 1.4;
        group.add(glow);

        return group;
    }

    createParticleSystem() {
        const { particleCount } = this.config;

        // Create particle geometry
        const positions = new Float32Array(particleCount * 3);
        const velocities = new Float32Array(particleCount * 3);
        const lifetimes = new Float32Array(particleCount);

        // Initialize particles
        for (let i = 0; i < particleCount; i++) {
            this.resetParticle(i, positions, velocities, lifetimes);
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        // Store velocities and lifetimes for animation
        this.particleVelocities = velocities;
        this.particleLifetimes = lifetimes;

        // Create material
        const material = new THREE.PointsMaterial({
            color: 0x66ffcc,
            size: 1.5,
            transparent: true,
            opacity: 0.6,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });

        this.particles = new THREE.Points(geometry, material);
        this.scene.add(this.particles);
    }

    resetParticle(index, positions, velocities, lifetimes) {
        const idx = index * 3;

        // Random position in the current field area
        positions[idx] = (Math.random() - 0.5) * 300;     // X
        positions[idx + 1] = -5 - Math.random() * 25;     // Y (depth)
        positions[idx + 2] = (Math.random() - 0.5) * 300; // Z

        // Velocity based on current data
        const speed = this.currentData.speed * 20; // Scale for visibility
        const dir = this.currentData.direction * Math.PI / 180;

        velocities[idx] = Math.sin(dir) * speed;      // X velocity
        velocities[idx + 1] = this.currentData.wComponent * 5; // Y velocity
        velocities[idx + 2] = Math.cos(dir) * speed;  // Z velocity

        // Random lifetime
        lifetimes[index] = Math.random();
    }

    updateArrows() {
        const { arrowScale } = this.config;
        const { speed, direction, uComponent, vComponent } = this.currentData;

        // Convert direction to radians (meteorological convention: from North, clockwise)
        const dirRad = (270 - direction) * Math.PI / 180;

        this.arrows.forEach(({ mesh, basePosition, phase }) => {
            // Scale arrow based on speed
            const scale = Math.max(0.5, speed * arrowScale);
            mesh.scale.set(scale, 1, 1);

            // Rotate to point in current direction
            mesh.rotation.y = dirRad;

            // Add some local variation based on position
            const localVariation = Math.sin(basePosition.x * 0.05 + phase) * 0.1;
            mesh.rotation.y += localVariation;
        });
    }

    updateFromData(speed, direction) {
        this.currentData.speed = speed;
        this.currentData.direction = direction;

        // Calculate UV components
        const dirRad = direction * Math.PI / 180;
        this.currentData.uComponent = speed * Math.sin(dirRad);
        this.currentData.vComponent = speed * Math.cos(dirRad);

        this.updateArrows();

        // ── Recalculate ALL particle velocities for immediate response ──
        if (this.particleVelocities) {
            const velocities = this.particleVelocities;
            const scaledSpeed = speed * 25; // Amplified for visibility
            const flowDir = direction * Math.PI / 180;

            for (let i = 0; i < velocities.length / 3; i++) {
                const idx = i * 3;
                // Add slight random variation per particle
                const jitter = 0.8 + Math.random() * 0.4;
                velocities[idx] = Math.sin(flowDir) * scaledSpeed * jitter;
                velocities[idx + 1] = this.currentData.wComponent * 5;
                velocities[idx + 2] = Math.cos(flowDir) * scaledSpeed * jitter;
            }
        }
    }

    update(elapsedTime, deltaTime) {
        // Animate arrow pulsation
        this.arrows.forEach(({ mesh, phase }, index) => {
            // Subtle pulsation
            const pulse = 1 + Math.sin(elapsedTime * 2 + phase) * 0.1;
            const baseScale = Math.max(0.5, this.currentData.speed * this.config.arrowScale);
            mesh.scale.x = baseScale * pulse;

            // Subtle oscillation
            mesh.rotation.y += Math.sin(elapsedTime * 0.5 + phase) * 0.001;
        });

        // Animate particles
        if (this.particles && this.particles.geometry) {
            const positions = this.particles.geometry.attributes.position.array;
            const velocities = this.particleVelocities;
            const lifetimes = this.particleLifetimes;

            for (let i = 0; i < positions.length / 3; i++) {
                const idx = i * 3;

                // Update lifetime
                lifetimes[i] -= deltaTime * 0.1;

                if (lifetimes[i] <= 0) {
                    // Reset particle
                    this.resetParticle(i, positions, velocities, lifetimes);
                } else {
                    // Move particle
                    positions[idx] += velocities[idx] * deltaTime;
                    positions[idx + 1] += velocities[idx + 1] * deltaTime;
                    positions[idx + 2] += velocities[idx + 2] * deltaTime;

                    // Add some turbulence
                    positions[idx] += (Math.random() - 0.5) * 0.2;
                    positions[idx + 2] += (Math.random() - 0.5) * 0.2;

                    // Wrap around boundaries
                    if (Math.abs(positions[idx]) > 150) positions[idx] *= -0.9;
                    if (Math.abs(positions[idx + 2]) > 150) positions[idx + 2] *= -0.9;
                }
            }

            this.particles.geometry.attributes.position.needsUpdate = true;
        }
    }

    setScale(scale) {
        this.config.arrowScale = scale;
        this.updateArrows();
    }

    setVisible(visible) {
        if (this.arrowsGroup) this.arrowsGroup.visible = visible;
        if (this.particles) this.particles.visible = visible;
    }

    dispose() {
        if (this.arrowsGroup) {
            this.arrowsGroup.traverse(obj => {
                if (obj.geometry) obj.geometry.dispose();
                if (obj.material) obj.material.dispose();
            });
            this.scene.remove(this.arrowsGroup);
        }

        if (this.particles) {
            this.particles.geometry.dispose();
            this.particles.material.dispose();
            this.scene.remove(this.particles);
        }
    }
}
