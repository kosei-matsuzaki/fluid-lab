#!/usr/bin/env node
// Assembles the single-file deliverable (fluid-lab.html) from src/.
// The Artifact platform requires one self-contained HTML (strict CSP,
// no external files), so the readable sources live in src/ and this
// script splices them back together. Run: node build.js
'use strict';
const fs = require('fs');
const path = require('path');
const read = f => fs.readFileSync(path.join(__dirname, 'src', f), 'utf8');
const JS_PARTS = [
  '00-shared.js',        // IIFE open + GL/color helpers
  '10-ink2d.js',         // ink: 2D stable fluids (GPU)
  '20-water2d.js',       // water 2D: FLIP/PIC + metaball surface
  '30-water3d-earth.js', // water 3D + earth: FLIP3D, SWE ocean, terrain, raymarcher
  '90-app.js',           // mode registry, sidebar UI, input (+ IIFE close)
];
const html = read('template.html')
  .replace('/*__STYLE__*/\n', read('style.css'))
  .replace('/*__SCRIPT__*/\n', JS_PARTS.map(read).join(''));
fs.writeFileSync(path.join(__dirname, 'fluid-lab.html'), html);
console.log('built fluid-lab.html (' + html.length + ' bytes)');
