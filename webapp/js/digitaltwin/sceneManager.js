import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

export class SceneManager {
    constructor(container) {
        this.container = container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.composer = null;

        // OBSEA coordinates (converted to scene coordinates)
        this.observatoryPosition = new THREE.Vector3(0, -20, 0);

        // Scene boundaries
        this.sceneBounds = {
            width: 500,
            depth: 500,
            maxDepth: 40
        };
    }

    async init() {
        this.createScene();
        this.createCamera();
        this.createRenderer();
        this.createPostProcessing(); // New step
        this.createLights();
        this.createControls();
        this.createHelpers();

        window.addEventListener('resize', () => this.onWindowResize());

        return Promise.resolve();
    }

    createScene() {
        this.scene = new THREE.Scene();

        // Deep ocean fog - Teal/Blue tint for Mediterranean realism
        // Density 0.010 gives visibility of ~80-120m
        this.scene.fog = new THREE.FogExp2(0x001e2b, 0.010);

        // Background gradient
        this.scene.background = new THREE.Color(0x001e2b);
    }

    createCamera() {
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 2000);
        this.camera.position.set(50, 10, 50); // Closer startup view
        this.camera.lookAt(this.observatoryPosition);
    }

    createRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            antialias: true, // MSAA (Note: EffectComposer might disable this depending on setup)
            alpha: true,
            powerPreference: 'high-performance'
        });

        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        // Realism settings
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 0.9;

        this.container.appendChild(this.renderer.domElement);
    }

    createPostProcessing() {
        // Render Pass
        const renderScene = new RenderPass(this.scene, this.camera);

        // Bloom Pass (Glow)
        const bloomPass = new UnrealBloomPass(
            new THREE.Vector2(this.container.clientWidth, this.container.clientHeight),
            1.5, // Strength (Intensity)
            0.4, // Radius
            0.85 // Threshold (High to only glow bright lights)
        );
        bloomPass.strength = 1.0;
        bloomPass.radius = 0.6;
        bloomPass.threshold = 0.5; // Lower threshold for water specular glow

        this.composer = new EffectComposer(this.renderer);
        this.composer.addPass(renderScene);
        this.composer.addPass(bloomPass);
    }

    createLights() {
        // Ambient light (richer underwater fill)
        const ambient = new THREE.AmbientLight(0x1a4466, 0.8);
        this.scene.add(ambient);

        // Main Sun (Caustic source direction)
        const sun = new THREE.DirectionalLight(0xf0f8ff, 1.8);
        sun.position.set(80, 150, 40);
        sun.castShadow = true;

        // Soften shadows for underwater diffusion
        sun.shadow.mapSize.width = 2048;
        sun.shadow.mapSize.height = 2048;
        sun.shadow.bias = -0.0001;

        this.scene.add(sun);

        // Spotlights for the Observatory (artificial lights)
        // Simulate a marker beacon or instrument LED
        const beaconLight = new THREE.PointLight(0xffaa00, 2, 20);
        beaconLight.position.set(0, -18, 0); // On top of cage
        this.scene.add(beaconLight);
    }

    createControls() {
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.target.copy(this.observatoryPosition);
        this.controls.minDistance = 5;
        this.controls.maxDistance = 150;
        this.controls.maxPolarAngle = Math.PI * 0.9;
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.update();
    }

    createHelpers() {
        // Minimal helpers for realism
        // Remove grid if we have terrain
    }

    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
        if (this.composer) {
            this.composer.setSize(width, height);
        }
    }

    render() {
        if (this.controls) this.controls.update();

        // Use Composer if available, else fallback
        if (this.composer) {
            this.composer.render();
        } else {
            this.renderer.render(this.scene, this.camera);
        }
    }

    dispose() {
        this.controls?.dispose();
        this.renderer?.dispose();
        this.composer?.dispose();
        if (this.renderer?.domElement) {
            this.container.removeChild(this.renderer.domElement);
        }
    }
}
