/**
 * Observatory Module - 3D model of the OBSEA underwater observatory
 * Creates a simplified representation of the seafloor instruments
 */

import * as THREE from 'three';

export class Observatory {
    constructor(scene) {
        this.scene = scene;
        this.group = null;
        this.instruments = {};
        this.lights = [];

        // Observatory position
        this.position = new THREE.Vector3(0, -20, 0); // 20m depth

        // Configuration
        this.config = {
            baseSize: 8,       // Base platform size in meters
            instrumentScale: 1,
            showLabels: true
        };
    }

    async init() {
        this.group = new THREE.Group();
        this.group.position.copy(this.position);

        // Create main components
        this.createBasePlatform();
        this.createCTDSensor();
        this.createAWACProfiler();
        this.createCableJunction();
        this.createInstrumentLights();
        this.createMarkerBeacon();
        this.createBiotope();

        this.scene.add(this.group);

        return Promise.resolve();
    }

    createBasePlatform() {
        // Renamed to createCageStructure, but keeping method name for compatibility or updating call site
        this.createCageStructure();
    }

    createCageStructure() {
        const cageGroup = new THREE.Group();
        cageGroup.name = 'OBSEA_Cage';

        // Cage dimensions (approx based on photos)
        const width = 4;
        const height = 2.5;
        const depth = 4;

        // Create procedural grid texture
        const gridTexture = this.createGridTexture();
        gridTexture.wrapS = gridTexture.wrapT = THREE.RepeatWrapping;
        gridTexture.repeat.set(4, 2.5);

        // Material for the cage mesh
        const cageMaterial = new THREE.MeshStandardMaterial({
            color: 0xdddddd, // Slightly darker white/grey
            map: gridTexture,
            alphaMap: gridTexture,
            transparent: true,
            side: THREE.DoubleSide,
            metalness: 0.7, // More metallic
            roughness: 0.4, // Smoother for reflections
            bumpMap: gridTexture, // Add bumps
            bumpScale: 0.05
        });

        // The Cage Structure (Main box)
        const boxGeo = new THREE.BoxGeometry(width, height, depth);
        // We only want the sides and top, not the bottom? Or maybe bottom too.
        // Let's use individual planes for better control if needed, but box is fine.
        const cageMesh = new THREE.Mesh(boxGeo, cageMaterial);
        cageMesh.position.y = height / 2;
        cageMesh.castShadow = true;
        cageGroup.add(cageMesh);

        // Add a solid frame (edges) using meshes instead of lines for lighting
        const frameMaterial = new THREE.MeshStandardMaterial({
            color: 0x8899aa,
            metalness: 0.8,
            roughness: 0.2
        });

        // Create frame bars (simplified box frame)
        const thickness = 0.1;
        function createBar(w, h, d, x, y, z) {
            const bar = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), frameMaterial);
            bar.position.set(x, y, z);
            bar.castShadow = true;
            cageGroup.add(bar);
        }

        // Vertical pillars (corners)
        createBar(thickness, height, thickness, -width / 2, height / 2, -depth / 2);
        createBar(thickness, height, thickness, width / 2, height / 2, -depth / 2);
        createBar(thickness, height, thickness, -width / 2, height / 2, depth / 2);
        createBar(thickness, height, thickness, width / 2, height / 2, depth / 2);

        // Horizontal bars (top and bottom)
        // Top
        createBar(width, thickness, thickness, 0, height, -depth / 2);
        createBar(width, thickness, thickness, 0, height, depth / 2);
        createBar(thickness, thickness, depth, -width / 2, height, 0);
        createBar(thickness, thickness, depth, width / 2, height, 0);
        // Bottom
        createBar(width, thickness, thickness, 0, 0, -depth / 2);
        createBar(width, thickness, thickness, 0, 0, depth / 2);
        createBar(thickness, thickness, depth, -width / 2, 0, 0);
        createBar(thickness, thickness, depth, width / 2, 0, 0);

        // Concrete pads / feet
        const footGeo = new THREE.CylinderGeometry(0.4, 0.4, 0.2, 8);
        const footMat = new THREE.MeshStandardMaterial({ color: 0x999999 });

        const footPositions = [
            { x: width / 2, z: depth / 2 },
            { x: -width / 2, z: depth / 2 },
            { x: width / 2, z: -depth / 2 },
            { x: -width / 2, z: -depth / 2 }
        ];

        footPositions.forEach(pos => {
            const foot = new THREE.Mesh(footGeo, footMat);
            foot.position.set(pos.x, 0.1, pos.z);
            cageGroup.add(foot);
        });

        // Add the logo plate (from photo)
        const plateGeo = new THREE.PlaneGeometry(1.2, 0.8);
        const plateMat = new THREE.MeshBasicMaterial({ color: 0x003399, side: THREE.DoubleSide }); // Blue plate
        const plate = new THREE.Mesh(plateGeo, plateMat);
        plate.position.set(0, height * 0.6, depth / 2 + 0.05); // On front face
        cageGroup.add(plate);

        this.group.add(cageGroup);
        this.instruments.platform = cageGroup;

        // Position instruments INSIDE the cage now
        // We need to adjust their relative positions later
    }

    createGridTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');

        // Background transparent
        ctx.fillStyle = 'rgba(0,0,0,0)';
        ctx.fillRect(0, 0, 512, 512);

        // Draw grid lines
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 15; // Thick lines

        const step = 64;

        // Draw grid
        ctx.beginPath();
        for (let i = 0; i <= 512; i += step) {
            ctx.moveTo(i, 0);
            ctx.lineTo(i, 512);
            ctx.moveTo(0, i);
            ctx.lineTo(512, i);
        }
        ctx.stroke();

        // Border
        ctx.lineWidth = 30;
        ctx.strokeRect(0, 0, 512, 512);

        const texture = new THREE.CanvasTexture(canvas);
        return texture;
    }

    createCTDSensor() {
        // SBE37 CTD Sensor - Place inside cage
        const ctdGroup = new THREE.Group();
        ctdGroup.name = 'CTD_SBE37';

        // Main sensor body
        const bodyGeometry = new THREE.CylinderGeometry(0.1, 0.1, 0.6, 12);
        const bodyMaterial = new THREE.MeshStandardMaterial({ color: 0xddddcc });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        ctdGroup.add(body);

        ctdGroup.position.set(1.0, 1.0, 0.5); // Inside cage
        this.group.add(ctdGroup);
        this.instruments.ctd = ctdGroup;
    }

    createAWACProfiler() {
        // Nortek AWAC - Place inside cage
        const awacGroup = new THREE.Group();
        awacGroup.name = 'AWAC';

        const bodyGeometry = new THREE.CylinderGeometry(0.2, 0.2, 0.4, 16);
        const bodyMaterial = new THREE.MeshStandardMaterial({ color: 0xffdd00 });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        awacGroup.add(body);

        awacGroup.position.set(-1.0, 0.8, -0.5); // Inside cage
        this.group.add(awacGroup);
        this.instruments.awac = awacGroup;
    }

    createCableJunction() {
        // Underwater cable - connecting to cage
        const cableGroup = new THREE.Group();

        // Curve to the cage
        const curve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(-40, 0, -40),
            new THREE.Vector3(-20, 0.2, -20),
            new THREE.Vector3(-5, 0.5, -5), // Entrance to cage area
            new THREE.Vector3(0, 0.5, -2)   // Into cage
        ]);

        const geometry = new THREE.TubeGeometry(curve, 64, 0.08, 8, false);
        const material = new THREE.MeshStandardMaterial({ color: 0x222222 });
        const mesh = new THREE.Mesh(geometry, material);
        cableGroup.add(mesh);

        this.group.add(cableGroup);
    }

    createInstrumentLights() {
        // Blinking LEDs inside cage
        const lightPositions = [
            { pos: [1.0, 1.2, 0.5], color: 0x00ff00 },      // CTD
            { pos: [-1.0, 1.0, -0.5], color: 0x00aaff }   // AWAC
        ];

        lightPositions.forEach(config => {
            const light = new THREE.PointLight(config.color, 0.8, 4);
            light.position.set(...config.pos);
            this.group.add(light);
            this.lights.push(light);
        });
    }

    createMarkerBeacon() {
        // Buoy attached to cage corner
        const buoyGroup = new THREE.Group();

        // Buoy body
        const buoyGeometry = new THREE.SphereGeometry(0.6, 16, 16);
        const buoyMaterial = new THREE.MeshPhongMaterial({ color: 0xff6600 });
        const buoy = new THREE.Mesh(buoyGeometry, buoyMaterial);
        buoyGroup.add(buoy);

        // Line
        const lineGeo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(2, -22, 2) // Relative to buoy position (0,22,0) -> cage corner (2,0,2)
        ]);
        const lineMat = new THREE.LineBasicMaterial({ color: 0x333333 });
        const line = new THREE.Line(lineGeo, lineMat);
        buoyGroup.add(line);

        buoyGroup.position.set(0, 22, 0);
        this.group.add(buoyGroup);
        this.instruments.buoy = buoyGroup;
    }

    createBiotope() {
        // The Artificial Reef (Concrete Blocks)
        // Located near the cage (approx 6-8m away based on photos)
        const reefGroup = new THREE.Group();
        reefGroup.name = 'Biotope';

        // Texture for concrete with moss (simulated with noise/color for now if load fails)
        const concreteMat = new THREE.MeshStandardMaterial({
            color: 0x778877, // Greenish grey
            roughness: 0.9,
            bumpScale: 0.2
        });

        // Create a stack of blocks
        const blockSize = 1.2;
        const positions = [
            // Base layer
            { x: 0, y: 0, z: 0 }, { x: 1.3, y: 0, z: 0 }, { x: -1.3, y: 0, z: 0 },
            { x: 0, y: 0, z: 1.3 }, { x: 1.3, y: 0, z: 1.3 }, { x: -1.3, y: 0, z: 1.3 },

            // Second layer
            { x: 0.6, y: 1.2, z: 0.6 }, { x: -0.6, y: 1.2, z: 0.6 },
            { x: 0.6, y: 1.2, z: -0.6 }, { x: -0.6, y: 1.2, z: -0.6 },

            // Third layer
            { x: 0, y: 2.4, z: 0 }
        ];

        const blockGeo = new THREE.BoxGeometry(blockSize, blockSize, blockSize);

        positions.forEach(pos => {
            const block = new THREE.Mesh(blockGeo, concreteMat);
            block.position.set(pos.x, pos.y + blockSize / 2, pos.z);

            // Random rotation for natural look
            block.rotation.y = (Math.random() - 0.5) * 0.2;

            reefGroup.add(block);
        });

        // Add fish school around reef
        this.createFishSchool(reefGroup, 0, 3, 0, 15);

        // Position the entire reef relative to the cage
        reefGroup.position.set(8, 0, 5);
        this.group.add(reefGroup);
    }

    // Helper method for seaweeds (unchanged but redefined here if needed)
    createSeaweed(parent, x, z, height) {
        // Simplified seaweed
        const geometry = new THREE.ConeGeometry(0.2, height, 4);
        const material = new THREE.MeshBasicMaterial({ color: 0x228833 });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(x, height / 2, z);
        parent.add(mesh);
    }

    createRock(parent, x, z, size) {
        // Irregular rock shape using dodecahedron
        const rockGeometry = new THREE.DodecahedronGeometry(size, 1);

        // Deform vertices for natural look
        const positions = rockGeometry.attributes.position.array;
        for (let i = 0; i < positions.length; i += 3) {
            positions[i] *= 0.8 + Math.random() * 0.4;
            positions[i + 1] *= 0.6 + Math.random() * 0.4;
            positions[i + 2] *= 0.8 + Math.random() * 0.4;
        }
        rockGeometry.computeVertexNormals();

        const rockMaterial = new THREE.MeshPhongMaterial({
            color: 0x445544,
            roughness: 0.9,
            flatShading: true
        });

        const rock = new THREE.Mesh(rockGeometry, rockMaterial);
        rock.position.set(x, -size * 0.3, z);
        rock.rotation.set(
            Math.random() * 0.3,
            Math.random() * Math.PI,
            Math.random() * 0.3
        );
        rock.castShadow = true;
        rock.receiveShadow = true;

        parent.add(rock);
    }

    createFishSchool(parent, x, y, z, count) {
        const fishGroup = new THREE.Group();
        fishGroup.userData = { type: 'fishSchool', phase: Math.random() * Math.PI * 2 };

        const fishMaterial = new THREE.MeshPhongMaterial({
            color: 0x88aacc,
            metalness: 0.4,
            transparent: true,
            opacity: 0.9
        });

        for (let i = 0; i < count; i++) {
            // Simple fish shape using merged geometries
            const fishShape = new THREE.Group();

            // Body (elongated sphere)
            const bodyGeometry = new THREE.SphereGeometry(0.2, 8, 6);
            bodyGeometry.scale(2, 0.7, 0.5);
            const body = new THREE.Mesh(bodyGeometry, fishMaterial);
            fishShape.add(body);

            // Tail
            const tailGeometry = new THREE.ConeGeometry(0.15, 0.3, 4);
            tailGeometry.rotateZ(Math.PI / 2);
            const tail = new THREE.Mesh(tailGeometry, fishMaterial);
            tail.position.x = -0.45;
            fishShape.add(tail);

            // Position in school
            fishShape.position.set(
                (Math.random() - 0.5) * 4,
                (Math.random() - 0.5) * 2,
                (Math.random() - 0.5) * 3
            );
            fishShape.rotation.y = Math.random() * 0.3 - 0.15;
            fishShape.userData = { phase: Math.random() * Math.PI * 2 };

            fishGroup.add(fishShape);
        }

        fishGroup.position.set(x, y, z);
        parent.add(fishGroup);

        this.instruments.fishSchool = fishGroup;
    }

    update(elapsedTime, deltaTime) {
        // Animate LED lights (blinking)
        this.lights.forEach((light, index) => {
            const phase = index * Math.PI / 2;
            light.intensity = 0.3 + Math.sin(elapsedTime * 2 + phase) * 0.2;
        });

        // Animate seaweed swaying (if we stored references)
        // This would require storing seaweed meshes and animating their vertices

        // Animate fish school
        if (this.instruments.fishSchool) {
            const school = this.instruments.fishSchool;

            // School movement
            school.position.x = 8 + Math.sin(elapsedTime * 0.3) * 5;
            school.position.y = 3 + Math.sin(elapsedTime * 0.5) * 1;
            school.position.z = 5 + Math.cos(elapsedTime * 0.4) * 4;

            // Individual fish movement
            school.children.forEach((fish, i) => {
                if (fish.userData && fish.userData.phase !== undefined) {
                    fish.position.x += Math.sin(elapsedTime * 3 + fish.userData.phase) * 0.01;
                    fish.position.y += Math.cos(elapsedTime * 2 + fish.userData.phase) * 0.005;
                    fish.rotation.y = Math.sin(elapsedTime * 4 + fish.userData.phase) * 0.1;
                }
            });
        }

        // Buoy bobbing motion
        if (this.instruments.buoy) {
            this.instruments.buoy.position.y = 21 + Math.sin(elapsedTime * 1.5) * 0.3;
            this.instruments.buoy.rotation.x = Math.sin(elapsedTime * 1.2) * 0.1;
            this.instruments.buoy.rotation.z = Math.cos(elapsedTime * 0.8) * 0.08;
        }
    }

    setVisible(visible) {
        if (this.group) {
            this.group.visible = visible;
        }
    }

    dispose() {
        if (this.group) {
            this.group.traverse(obj => {
                if (obj.geometry) obj.geometry.dispose();
                if (obj.material) {
                    if (Array.isArray(obj.material)) {
                        obj.material.forEach(m => m.dispose());
                    } else {
                        obj.material.dispose();
                    }
                }
            });
            this.scene.remove(this.group);
        }
    }
}
