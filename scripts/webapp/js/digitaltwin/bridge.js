import { DigitalTwinApp } from './main.js';

// Expose to global scope so app.js (classic script) can access it
window.DigitalTwinAppClass = DigitalTwinApp;
console.log('🧊 Digital Twin Bridge Loaded');
