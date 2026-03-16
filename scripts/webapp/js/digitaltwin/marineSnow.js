/**
 * Marine Snow Effect
 * Simulates suspended organic particles in the water column.
 * Adds depth and sense of scale/current to the underwater scene.
 */

import * as THREE from 'three';

export class MarineSnow {
    constructor(scene, count = 15000) {
        this.scene = scene;
        this.count = count;
        this.geometry = null;
        this.material = null;
        this.points = null;
        this.velocities = [];
    }

    init() {
        // Create geometry
        this.geometry = new THREE.BufferGeometry();
        const positions = [];
        const sizes = [];

        // Distribute particles in a large volume around the center
        // Bounds should match scene bounds (approx 200m radius)
        const range = 200;
        const depthRange = 50; // 0 to -50

        for (let i = 0; i < this.count; i++) {
            // Random position
            const x = (Math.random() - 0.5) * range; // -100 to 100
            const y = (Math.random() * -depthRange); // 0 to -50
            const z = (Math.random() - 0.5) * range; // -100 to 100

            positions.push(x, y, z);

            // Random size variation
            sizes.push(Math.random() * 0.5 + 0.1);

            // Assign random drift velocity
            this.velocities.push({
                x: (Math.random() - 0.5) * 0.02,
                y: -(Math.random() * 0.05 + 0.01), // Slowly sinking
                z: (Math.random() - 0.5) * 0.02
            });
        }

        this.geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        this.geometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));

        // Create texture (simple soft circle)
        const loader = new THREE.TextureLoader();
        const texture = this.createParticleTexture();

        // Shader or PointsMaterial? PointsMaterial is cheaper.
        this.material = new THREE.PointsMaterial({
            color: 0xffffff,
            size: 0.2,
            map: texture,
            transparent: true,
            opacity: 0.6,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
            sizeAttenuation: true
        });

        this.points = new THREE.Points(this.geometry, this.material);
        this.scene.add(this.points);
    }

    // Procedurally create a soft circular texture to avoid loading external file
    createParticleTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 32;
        canvas.height = 32;
        const context = canvas.getContext('2d');
        const gradient = context.createRadialGradient(16, 16, 0, 16, 16, 16);
        gradient.addColorStop(0, 'rgba(255,255,255,1)');
        gradient.addColorStop(1, 'rgba(255,255,255,0)');
        context.fillStyle = gradient;
        context.fillRect(0, 0, 32, 32);
        const texture = new THREE.CanvasTexture(canvas);
        return texture;
    }

    update(time) {
        if (!this.points) return;

        const positions = this.geometry.attributes.position.array;
        const range = 200;
        const depthRange = 50;

        for (let i = 0; i < this.count; i++) {
            const i3 = i * 3;
            const vel = this.velocities[i];

            // Update position
            // Subtle swaying motion
            positions[i3] += vel.x + Math.sin(time + positions[i3 + 1]) * 0.002;
            positions[i3 + 1] += vel.y; // Sink
            positions[i3 + 2] += vel.z + Math.cos(time + positions[i3 + 1]) * 0.002;

            // Reset if out of bounds (wrap around loop)
            if (positions[i3 + 1] < -depthRange) {
                positions[i3 + 1] = 0; // Back to top
            }
            if (Math.abs(positions[i3]) > range / 2) positions[i3] *= -0.9;
            if (Math.abs(positions[i3 + 2]) > range / 2) positions[i3 + 2] *= -0.9;
        }

        this.geometry.attributes.position.needsUpdate = true;
    }

    setVisible(visible) {
        if (this.points) this.points.visible = visible;
    }
}
