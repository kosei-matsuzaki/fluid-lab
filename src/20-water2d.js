function createWater2D () {
  const canvas = makeCanvas();
  canvas.style.cursor = 'crosshair';
  const gl = canvas.getContext('webgl2', { alpha: false, depth: false, antialias: true })
          || canvas.getContext('webgl', { alpha: false, depth: false, antialias: true });
  if (!gl) { canvas.remove(); return { error: true }; }
  const gl2 = typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext;
  const floatOK = gl2 && !!gl.getExtension('EXT_color_buffer_float');

  const FLUID = 0, AIR = 1, SOLID = 2;

  class FlipFluid {
    constructor (density, width, height, spacing, particleRadius, maxParticles) {
      this.density = density;
      this.fNumX = Math.floor(width / spacing) + 1;
      this.fNumY = Math.floor(height / spacing) + 1;
      this.h = Math.max(width / this.fNumX, height / this.fNumY);
      this.fInvSpacing = 1.0 / this.h;
      const nc = this.fNumCells = this.fNumX * this.fNumY;
      this.u = new Float32Array(nc);
      this.v = new Float32Array(nc);
      this.du = new Float32Array(nc);
      this.dv = new Float32Array(nc);
      this.prevU = new Float32Array(nc);
      this.prevV = new Float32Array(nc);
      this.s = new Float32Array(nc);
      this.cellType = new Int32Array(nc);
      this.particleDensity = new Float32Array(nc);
      this.particleRestDensity = 0.0;
      this.maxParticles = maxParticles;
      this.particlePos = new Float32Array(2 * maxParticles);
      this.particleVel = new Float32Array(2 * maxParticles);
      this.particleColor = new Float32Array(3 * maxParticles);
      this.particleRadius = particleRadius;
      this.pInvSpacing = 1.0 / (2.2 * particleRadius);
      this.pNumX = Math.floor(width * this.pInvSpacing) + 1;
      this.pNumY = Math.floor(height * this.pInvSpacing) + 1;
      this.pNumCells = this.pNumX * this.pNumY;
      this.numCellParticles = new Int32Array(this.pNumCells);
      this.firstCellParticle = new Int32Array(this.pNumCells + 1);
      this.cellParticleIds = new Int32Array(maxParticles);
      this.numParticles = 0;
    }
    integrateParticles (dt, gravity) {
      const pos = this.particlePos, vel = this.particleVel;
      const maxV = 25, maxV2 = maxV * maxV;   // safety cap against runaway particles
      for (let i = 0; i < this.numParticles; i++) {
        // slight viscous damping so residual currents settle to rest
        let vx = vel[2 * i] * 0.997, vy = (vel[2 * i + 1] - dt * gravity) * 0.997;
        const v2 = vx * vx + vy * vy;
        if (v2 > maxV2) { const k = maxV / Math.sqrt(v2); vx *= k; vy *= k; }
        vel[2 * i] = vx; vel[2 * i + 1] = vy;
        pos[2 * i] += vx * dt;
        pos[2 * i + 1] += vy * dt;
      }
    }
    pushParticlesApart (numIters) {
      this.numCellParticles.fill(0);
      for (let i = 0; i < this.numParticles; i++) {
        const xi = Math.max(0, Math.min(this.pNumX - 1, Math.floor(this.particlePos[2 * i] * this.pInvSpacing)));
        const yi = Math.max(0, Math.min(this.pNumY - 1, Math.floor(this.particlePos[2 * i + 1] * this.pInvSpacing)));
        this.numCellParticles[xi * this.pNumY + yi]++;
      }
      let first = 0;
      for (let i = 0; i < this.pNumCells; i++) {
        first += this.numCellParticles[i];
        this.firstCellParticle[i] = first;
      }
      this.firstCellParticle[this.pNumCells] = first;
      for (let i = 0; i < this.numParticles; i++) {
        const xi = Math.max(0, Math.min(this.pNumX - 1, Math.floor(this.particlePos[2 * i] * this.pInvSpacing)));
        const yi = Math.max(0, Math.min(this.pNumY - 1, Math.floor(this.particlePos[2 * i + 1] * this.pInvSpacing)));
        this.cellParticleIds[--this.firstCellParticle[xi * this.pNumY + yi]] = i;
      }
      const minDist = 2.0 * this.particleRadius;
      const minDist2 = minDist * minDist;
      const colorDiffusion = 0.001;
      for (let iter = 0; iter < numIters; iter++) {
        for (let i = 0; i < this.numParticles; i++) {
          const px = this.particlePos[2 * i], py = this.particlePos[2 * i + 1];
          const pxi = Math.floor(px * this.pInvSpacing), pyi = Math.floor(py * this.pInvSpacing);
          const x0 = Math.max(pxi - 1, 0), y0 = Math.max(pyi - 1, 0);
          const x1 = Math.min(pxi + 1, this.pNumX - 1), y1 = Math.min(pyi + 1, this.pNumY - 1);
          for (let xi = x0; xi <= x1; xi++) {
            for (let yi = y0; yi <= y1; yi++) {
              const cell = xi * this.pNumY + yi;
              const start = this.firstCellParticle[cell], end = this.firstCellParticle[cell + 1];
              for (let k = start; k < end; k++) {
                const id = this.cellParticleIds[k];
                if (id <= i) continue;   // visit each pair once
                let dx = this.particlePos[2 * id] - px;
                let dy = this.particlePos[2 * id + 1] - py;
                const d2 = dx * dx + dy * dy;
                if (d2 > minDist2 || d2 === 0) continue;
                const d = Math.sqrt(d2);
                const sc = 0.5 * (minDist - d) / d;
                dx *= sc; dy *= sc;
                this.particlePos[2 * i] -= dx; this.particlePos[2 * i + 1] -= dy;
                this.particlePos[2 * id] += dx; this.particlePos[2 * id + 1] += dy;
                for (let ch = 0; ch < 3; ch++) {
                  const c0 = this.particleColor[3 * i + ch];
                  const c1 = this.particleColor[3 * id + ch];
                  const mid = (c0 + c1) * 0.5;
                  this.particleColor[3 * i + ch] = c0 + (mid - c0) * colorDiffusion;
                  this.particleColor[3 * id + ch] = c1 + (mid - c1) * colorDiffusion;
                }
              }
            }
          }
        }
      }
    }
    handleParticleCollisions (obstacle) {
      const h = this.h, r = this.particleRadius;
      const minX = h + r, maxX = (this.fNumX - 1) * h - r;
      const minY = h + r, maxY = (this.fNumY - 1) * h - r;
      const useObs = obstacle.active;
      const minObsDist = obstacle.radius + r;
      const minObsDist2 = minObsDist * minObsDist;
      for (let i = 0; i < this.numParticles; i++) {
        let x = this.particlePos[2 * i], y = this.particlePos[2 * i + 1];
        if (useObs) {
          const dx = x - obstacle.x, dy = y - obstacle.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < minObsDist2) {
            const d = Math.sqrt(d2) || 1e-6;
            x = obstacle.x + dx / d * minObsDist;
            y = obstacle.y + dy / d * minObsDist;
            this.particleVel[2 * i] = obstacle.vx;
            this.particleVel[2 * i + 1] = obstacle.vy;
          }
        }
        if (x < minX) { x = minX; this.particleVel[2 * i] = 0; }
        if (x > maxX) { x = maxX; this.particleVel[2 * i] = 0; }
        if (y < minY) { y = minY; this.particleVel[2 * i + 1] = 0; }
        if (y > maxY) { y = maxY; this.particleVel[2 * i + 1] = 0; }
        this.particlePos[2 * i] = x;
        this.particlePos[2 * i + 1] = y;
      }
    }
    updateParticleDensity () {
      const n = this.fNumY, h1 = this.fInvSpacing, h2 = 0.5 * this.h;
      const d = this.particleDensity;
      d.fill(0);
      for (let i = 0; i < this.numParticles; i++) {
        const x = Math.max(this.h, Math.min((this.fNumX - 1) * this.h, this.particlePos[2 * i]));
        const y = Math.max(this.h, Math.min((this.fNumY - 1) * this.h, this.particlePos[2 * i + 1]));
        const x0 = ((x - h2) * h1) | 0, tx = (x - h2 - x0 * this.h) * h1;
        const x1 = Math.min(x0 + 1, this.fNumX - 2);
        const y0 = ((y - h2) * h1) | 0, ty = (y - h2 - y0 * this.h) * h1;
        const y1 = Math.min(y0 + 1, this.fNumY - 2);
        d[x0 * n + y0] += (1 - tx) * (1 - ty);
        d[x1 * n + y0] += tx * (1 - ty);
        d[x1 * n + y1] += tx * ty;
        d[x0 * n + y1] += (1 - tx) * ty;
      }
      if (this.particleRestDensity === 0.0) {
        let sum = 0, count = 0;
        for (let i = 0; i < this.fNumCells; i++) {
          if (this.cellType[i] === FLUID) { sum += d[i]; count++; }
        }
        if (count > 0) this.particleRestDensity = sum / count;
      }
    }
    transferVelocities (toGrid, flipRatio) {
      const n = this.fNumY, h = this.h, h1 = this.fInvSpacing, h2 = 0.5 * h;
      if (toGrid) {
        this.prevU.set(this.u);
        this.prevV.set(this.v);
        this.du.fill(0); this.dv.fill(0);
        this.u.fill(0); this.v.fill(0);
        for (let i = 0; i < this.fNumCells; i++) {
          this.cellType[i] = this.s[i] === 0 ? SOLID : AIR;
        }
        for (let i = 0; i < this.numParticles; i++) {
          const xi = Math.max(0, Math.min(this.fNumX - 1, Math.floor(this.particlePos[2 * i] * h1)));
          const yi = Math.max(0, Math.min(this.fNumY - 1, Math.floor(this.particlePos[2 * i + 1] * h1)));
          const cell = xi * n + yi;
          if (this.cellType[cell] === AIR) this.cellType[cell] = FLUID;
        }
      }
      for (let component = 0; component < 2; component++) {
        const dx = component === 0 ? 0 : h2;
        const dy = component === 0 ? h2 : 0;
        const f = component === 0 ? this.u : this.v;
        const prevF = component === 0 ? this.prevU : this.prevV;
        const dArr = component === 0 ? this.du : this.dv;
        const offset = component === 0 ? n : 1;
        for (let i = 0; i < this.numParticles; i++) {
          const x = Math.max(h, Math.min((this.fNumX - 1) * h, this.particlePos[2 * i]));
          const y = Math.max(h, Math.min((this.fNumY - 1) * h, this.particlePos[2 * i + 1]));
          const x0 = Math.min(((x - dx) * h1) | 0, this.fNumX - 2);
          const tx = (x - dx - x0 * h) * h1;
          const x1 = Math.min(x0 + 1, this.fNumX - 2);
          const y0 = Math.min(((y - dy) * h1) | 0, this.fNumY - 2);
          const ty = (y - dy - y0 * h) * h1;
          const y1 = Math.min(y0 + 1, this.fNumY - 2);
          const w0 = (1 - tx) * (1 - ty), w1 = tx * (1 - ty), w2 = tx * ty, w3 = (1 - tx) * ty;
          const nr0 = x0 * n + y0, nr1 = x1 * n + y0, nr2 = x1 * n + y1, nr3 = x0 * n + y1;
          if (toGrid) {
            const pv = this.particleVel[2 * i + component];
            f[nr0] += pv * w0; dArr[nr0] += w0;
            f[nr1] += pv * w1; dArr[nr1] += w1;
            f[nr2] += pv * w2; dArr[nr2] += w2;
            f[nr3] += pv * w3; dArr[nr3] += w3;
          } else {
            const v0 = (this.cellType[nr0] !== AIR || this.cellType[nr0 - offset] !== AIR) ? 1 : 0;
            const v1 = (this.cellType[nr1] !== AIR || this.cellType[nr1 - offset] !== AIR) ? 1 : 0;
            const v2 = (this.cellType[nr2] !== AIR || this.cellType[nr2 - offset] !== AIR) ? 1 : 0;
            const v3 = (this.cellType[nr3] !== AIR || this.cellType[nr3 - offset] !== AIR) ? 1 : 0;
            const d = v0 * w0 + v1 * w1 + v2 * w2 + v3 * w3;
            const pv = this.particleVel[2 * i + component];
            if (d > 0) {
              const picV = (v0 * w0 * f[nr0] + v1 * w1 * f[nr1] + v2 * w2 * f[nr2] + v3 * w3 * f[nr3]) / d;
              const corr = (v0 * w0 * (f[nr0] - prevF[nr0]) + v1 * w1 * (f[nr1] - prevF[nr1])
                          + v2 * w2 * (f[nr2] - prevF[nr2]) + v3 * w3 * (f[nr3] - prevF[nr3])) / d;
              this.particleVel[2 * i + component] = (1 - flipRatio) * picV + flipRatio * (pv + corr);
            }
          }
        }
        if (toGrid) {
          for (let i = 0; i < f.length; i++) {
            if (dArr[i] > 0) f[i] /= dArr[i];
          }
          for (let i = 0; i < this.fNumX; i++) {
            for (let j = 0; j < this.fNumY; j++) {
              const solid = this.cellType[i * n + j] === SOLID;
              if (solid || (i > 0 && this.cellType[(i - 1) * n + j] === SOLID)) this.u[i * n + j] = this.prevU[i * n + j];
              if (solid || (j > 0 && this.cellType[i * n + j - 1] === SOLID)) this.v[i * n + j] = this.prevV[i * n + j];
            }
          }
          // free-slip walls: tangential velocity in the wall layer mirrors the fluid
          for (let j = 0; j < this.fNumY; j++) {
            this.v[j] = this.v[n + j];
            this.v[(this.fNumX - 1) * n + j] = this.v[(this.fNumX - 2) * n + j];
          }
          for (let i = 0; i < this.fNumX; i++) {
            this.u[i * n] = this.u[i * n + 1];
            this.u[i * n + this.fNumY - 1] = this.u[i * n + this.fNumY - 2];
          }
        }
      }
    }
    solveIncompressibility (numIters, overRelaxation) {
      // FLIP delta baseline: snapshot the grid AFTER particle transfer, BEFORE projection.
      // (Snapshotting last frame's grid instead makes moving particles self-amplify.)
      this.prevU.set(this.u);
      this.prevV.set(this.v);
      const n = this.fNumY;
      const s = this.s, u = this.u, v = this.v, t = this.cellType, pd = this.particleDensity;
      const rest = this.particleRestDensity;
      // collect fluid cells once per frame, then sweep only those each iteration
      if (!this.fluidCells) this.fluidCells = new Int32Array(this.fNumCells);
      const list = this.fluidCells;
      let m = 0;
      for (let i = 1; i < this.fNumX - 1; i++) {
        const base = i * n;
        for (let j = 1; j < this.fNumY - 1; j++) {
          if (t[base + j] === FLUID) list[m++] = base + j;
        }
      }
      for (let iter = 0; iter < numIters; iter++) {
        for (let k = 0; k < m; k++) {
          const c = list[k];
          const sx0 = s[c - n], sx1 = s[c + n];
          const sy0 = s[c - 1], sy1 = s[c + 1];
          const sSum = sx0 + sx1 + sy0 + sy1;
          if (sSum === 0) continue;
          let div = u[c + n] - u[c] + v[c + 1] - v[c];
          if (rest > 0) {
            // deadband + soft gain: fix real clumping without simmering at rest
            const compression = pd[c] - rest * 1.05;
            if (compression > 0) div -= 0.7 * compression;
          }
          const pv = (-div / sSum) * overRelaxation;
          u[c] -= sx0 * pv;
          u[c + n] += sx1 * pv;
          v[c] -= sy0 * pv;
          v[c + 1] += sy1 * pv;
        }
      }
    }
    updateParticleColors () {
      const h1 = this.fInvSpacing, n = this.fNumY;
      const rest = this.particleRestDensity;
      for (let i = 0; i < this.numParticles; i++) {
        for (let ch = 0; ch < 3; ch++) {
          const c = this.particleColor[3 * i + ch];
          this.particleColor[3 * i + ch] = c + (BASE[ch] - c) * 0.03;
        }
        if (rest > 0) {
          // foam = sparse AND fast: a calm free surface is sparse too, but must stay blue
          const vx = this.particleVel[2 * i], vy = this.particleVel[2 * i + 1];
          if (vx * vx + vy * vy > 4.0) {
            const xi = Math.max(1, Math.min(this.fNumX - 1, Math.floor(this.particlePos[2 * i] * h1)));
            const yi = Math.max(1, Math.min(this.fNumY - 1, Math.floor(this.particlePos[2 * i + 1] * h1)));
            if (this.particleDensity[xi * n + yi] / rest < 0.6) {
              this.particleColor[3 * i] = FOAM[0];
              this.particleColor[3 * i + 1] = FOAM[1];
              this.particleColor[3 * i + 2] = FOAM[2];
            }
          }
        }
      }
    }
    setObstacle (obstacle) {
      const n = this.fNumY, h = this.h;
      if (!this.staticS) {
        // walls are static: build them once, then only patch cells the paddle touches
        this.staticS = true;
        for (let i = 0; i < this.fNumX; i++) {
          for (let j = 0; j < this.fNumY; j++) {
            const border = i === 0 || i === this.fNumX - 1 || j === 0 || j === this.fNumY - 1;
            this.s[i * n + j] = border ? 0 : 1;
          }
        }
        this.obsCells = [];
      }
      for (let k = 0; k < this.obsCells.length; k++) this.s[this.obsCells[k]] = 1;
      this.obsCells.length = 0;
      if (!obstacle.active) return;
      const r = obstacle.radius, r2 = r * r;
      const i0 = Math.max(1, ((obstacle.x - r) / h | 0));
      const i1 = Math.min(this.fNumX - 2, ((obstacle.x + r) / h | 0) + 1);
      const j0 = Math.max(1, ((obstacle.y - r) / h | 0));
      const j1 = Math.min(this.fNumY - 2, ((obstacle.y + r) / h | 0) + 1);
      for (let i = i0; i <= i1; i++) {
        for (let j = j0; j <= j1; j++) {
          const dx = (i + 0.5) * h - obstacle.x, dy = (j + 0.5) * h - obstacle.y;
          if (dx * dx + dy * dy < r2) {
            const idx = i * n + j;
            this.s[idx] = 0;
            this.obsCells.push(idx);
            // never write paddle velocity into faces that touch a wall cell
            if (i > 1) this.u[idx] = obstacle.vx;
            if (i < this.fNumX - 2) this.u[idx + n] = obstacle.vx;
            if (j > 1) this.v[idx] = obstacle.vy;
            if (j < this.fNumY - 2) this.v[idx + 1] = obstacle.vy;
          }
        }
      }
    }
    simulate (dt, gravity, flipRatio, obstacle) {
      this.integrateParticles(dt, gravity);
      this.pushParticlesApart(2);
      this.handleParticleCollisions(obstacle);
      this.setObstacle(obstacle);
      this.transferVelocities(true, flipRatio);
      this.updateParticleDensity();
      this.solveIncompressibility(40, 1.9);
      this.transferVelocities(false, flipRatio);
      this.tick = (this.tick || 0) + 1;
      if ((this.tick & 1) === 0) this.updateParticleColors();   // cosmetic pass, half rate
    }
  }

  const params = { gravity: 9.8, flipRatio: 0.10, waterAmount: 0.5, res: 100, paused: false, showParticles: false };
  const obstacle = { active: false, x: 0, y: 0, vx: 0, vy: 0, radius: 0.35 };
  let fluid = null;
  let simWidth = 0, simHeight = 3.0;

  function setupScene () {
    simHeight = 3.0;
    simWidth = simHeight * canvas.width / canvas.height;
    const res = Math.round(params.res);
    const spacing = simHeight / res;
    const particleRadius = 0.3 * spacing;
    const relW = 0.55, relH = params.waterAmount / 0.55 * 0.75;
    const dx = 2.0 * particleRadius;
    const dy = Math.sqrt(3) / 2 * dx;
    const numX = Math.floor((relW * simWidth - 2 * spacing - 2 * particleRadius) / dx);
    const numY = Math.floor((Math.min(0.92, relH) * simHeight - 2 * spacing - 2 * particleRadius) / dy);
    const maxParticles = numX * numY;
    fluid = new FlipFluid(1000, simWidth, simHeight, spacing, particleRadius, maxParticles);
    fluid.numParticles = maxParticles;
    let p = 0;
    for (let i = 0; i < numX; i++) {
      for (let j = 0; j < numY; j++) {
        fluid.particlePos[p++] = spacing + particleRadius + dx * i + (j % 2 === 0 ? 0 : particleRadius);
        fluid.particlePos[p++] = spacing + particleRadius + dy * j;
      }
    }
    for (let i = 0; i < maxParticles; i++) {
      fluid.particleColor[3 * i] = BASE[0];
      fluid.particleColor[3 * i + 1] = BASE[1];
      fluid.particleColor[3 * i + 2] = BASE[2];
    }
    obstacle.radius = 0.12 * simHeight;
  }

  const pointProg = glProgram(gl, `
attribute vec2 aPos;
attribute vec3 aColor;
uniform vec2 uDomain;
uniform float uPointSize;
varying vec3 vColor;
void main () {
  vColor = aColor;
  gl_Position = vec4(aPos / uDomain * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = uPointSize;
}`, `
precision mediump float;
varying vec3 vColor;
void main () {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(d, d);
  if (r2 > 1.0) discard;
  float a = smoothstep(1.0, 0.55, r2);
  gl_FragColor = vec4(vColor, a);
}`);
  const discProg = glProgram(gl, `
attribute vec2 aPos;
varying vec2 vUv;
void main () { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`, `
precision mediump float;
varying vec2 vUv;
uniform vec2 uDomain;
uniform vec2 uCenter;
uniform float uRadius;
void main () {
  vec2 p = vUv * uDomain;
  float d = distance(p, uCenter);
  float body = smoothstep(uRadius, uRadius - 0.015, d);
  float ring = smoothstep(0.012, 0.0, abs(d - uRadius * 0.72));
  vec3 col = vec3(0.16, 0.20, 0.27) + ring * vec3(0.12);
  gl_FragColor = vec4(col, body * 0.95);
}`);

  // ---- metaball surface rendering: accumulate a density+color field, then shade it ----
  const fieldProg = glProgram(gl, `
attribute vec2 aPos;
attribute vec3 aColor;
uniform vec2 uDomain;
uniform float uPointSize;
varying vec3 vColor;
void main () {
  vColor = aColor;
  gl_Position = vec4(aPos / uDomain * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = uPointSize;
}`, `
precision mediump float;
varying vec3 vColor;
uniform float uW;
void main () {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(d, d);
  if (r2 > 1.0) discard;
  float w = uW * exp(-3.0 * r2);
  gl_FragColor = vec4(vColor * w, w);
}`);
  const surfProg = glProgram(gl, `
attribute vec2 aPos;
varying vec2 vUv;
void main () { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`, `
precision mediump float;
varying vec2 vUv;
uniform sampler2D uField;
uniform vec2 uTexel;
uniform float uT0;
uniform float uT1;
vec3 wallColor (vec2 uv) {
  // tank back wall: gradient with faint tile lines, so refraction is visible
  vec3 c = mix(vec3(0.020, 0.028, 0.045), vec3(0.058, 0.075, 0.105), uv.y);
  float tileY = smoothstep(0.94, 1.0, abs(fract(uv.y * 12.0) - 0.5) * 2.0);
  float tileX = smoothstep(0.96, 1.0, abs(fract(uv.x * 18.0) - 0.5) * 2.0);
  c += (tileY + tileX) * 0.014;
  float vg = distance(uv, vec2(0.5, 0.55));
  c *= 1.0 - 0.35 * vg * vg;
  return c;
}
void main () {
  vec4 f = texture2D(uField, vUv);
  float d = f.a;
  float mask = smoothstep(uT0, uT1, d);
  if (mask < 0.004) discard;
  vec3 tint = f.rgb / max(d, 1e-4);

  float dl = texture2D(uField, vUv - vec2(uTexel.x, 0.0)).a;
  float dr = texture2D(uField, vUv + vec2(uTexel.x, 0.0)).a;
  float db = texture2D(uField, vUv - vec2(0.0, uTexel.y)).a;
  float du = texture2D(uField, vUv + vec2(0.0, uTexel.y)).a;
  vec2 grad = vec2(dl - dr, db - du);

  // only the interface band curves; the interior stays flat so the body reads
  // as clear water instead of a lumpy jelly of metaballs
  float edge = 1.0 - smoothstep(uT1, uT1 * 2.6, d);
  vec3 n = normalize(vec3(grad * (0.25 + 0.75 * edge), uT1 * 1.2));

  // march upward to the free surface: real depth below the waterline
  float stp = uTexel.y * 6.0;
  float surfSteps = 12.0;
  for (int i = 1; i <= 12; i++) {
    if (texture2D(uField, vUv + vec2(0.0, float(i) * stp)).a < uT1) { surfSteps = float(i); break; }
  }
  float depthT = clamp(surfSteps / 12.0, 0.0, 1.0);

  // transmission: the wall seen through the water, absorbed by depth (red dies first)
  vec2 ruv = vUv + n.xy * (0.045 + 0.030 * mask);
  vec3 absorbK = vec3(2.6, 1.1, 0.55);
  vec3 dyeFilter = mix(vec3(1.0), clamp(tint * 2.2, 0.0, 1.0), 0.75);
  vec3 trans = wallColor(ruv) * exp(-absorbK * (0.25 + 2.4 * depthT)) * dyeFilter * 2.4;
  // in-scattered light, brightest just under the surface
  trans += tint * 0.16 * exp(-3.5 * depthT);

  // fresnel-weighted environment reflection (bright ceiling light, dark floor)
  float F = 0.02 + 0.98 * pow(1.0 - max(n.z, 0.0), 5.0);
  vec3 env = mix(vec3(0.05, 0.065, 0.09), vec3(0.48, 0.58, 0.72),
                 clamp(0.5 + 1.8 * n.y, 0.0, 1.0));
  vec3 light = normalize(vec3(-0.35, 0.55, 0.75));
  float spec = pow(max(dot(n, normalize(light + vec3(0.0, 0.0, 1.0))), 0.0), 180.0);

  vec3 col = mix(trans, env, clamp(F * 1.5, 0.0, 0.85)) + spec * 1.6 * (0.3 + 0.7 * edge);

  // foam: white (low-density splash) particles brighten instead of tinting
  float lum = (tint.r + tint.g + tint.b) * 0.3333;
  float foam = smoothstep(0.55, 0.8, lum);
  col += foam * vec3(0.5, 0.55, 0.6) * (0.35 + 0.65 * edge);

  // thin bright line right at the waterline
  float crest = smoothstep(uT1 * 0.95, uT1 * 1.2, d) * (1.0 - smoothstep(uT1 * 1.2, uT1 * 1.7, d));
  col += crest * vec3(0.10, 0.15, 0.22);

  gl_FragColor = vec4(col, mask);
}`);
  const blur2Prog = glProgram(gl, `
attribute vec2 aPos;
varying vec2 vUv;
void main () { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`, `
precision mediump float;
varying vec2 vUv;
uniform sampler2D uTex;
uniform vec2 uDir;
void main () {
  vec4 c = texture2D(uTex, vUv) * 0.312;
  c += (texture2D(uTex, vUv + uDir) + texture2D(uTex, vUv - uDir)) * 0.235;
  c += (texture2D(uTex, vUv + uDir * 2.0) + texture2D(uTex, vUv - uDir * 2.0)) * 0.109;
  gl_FragColor = c;
}`);
  const bgProg = glProgram(gl, `
attribute vec2 aPos;
varying vec2 vUv;
void main () { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`, `
precision mediump float;
varying vec2 vUv;
void main () {
  vec3 c = mix(vec3(0.020, 0.028, 0.045), vec3(0.058, 0.075, 0.105), vUv.y);
  float tileY = smoothstep(0.94, 1.0, abs(fract(vUv.y * 12.0) - 0.5) * 2.0);
  float tileX = smoothstep(0.96, 1.0, abs(fract(vUv.x * 18.0) - 0.5) * 2.0);
  c += (tileY + tileX) * 0.014;
  float vg = distance(vUv, vec2(0.5, 0.55));
  c *= 1.0 - 0.35 * vg * vg;
  gl_FragColor = vec4(c, 1.0);
}`);
  let field = null, fieldS = null, fieldW = 0, fieldH = 0;
  function makeFieldFBO (w, h) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (floatOK) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return { tex, fbo };
  }
  function ensureField () {
    const w = Math.max(1, canvas.width >> 1), h = Math.max(1, canvas.height >> 1);
    if (field && fieldW === w && fieldH === h) return;
    fieldW = w; fieldH = h;
    for (const f of [field, fieldS]) {
      if (f) { gl.deleteTexture(f.tex); gl.deleteFramebuffer(f.fbo); }
    }
    field = makeFieldFBO(w, h);
    fieldS = makeFieldFBO(w, h);
  }

  const quadBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1]), gl.STATIC_DRAW);
  const posBuf = gl.createBuffer();
  const colBuf = gl.createBuffer();
  const aPosPoint = gl.getAttribLocation(pointProg, 'aPos');
  const aColPoint = gl.getAttribLocation(pointProg, 'aColor');
  const aPosDisc = gl.getAttribLocation(discProg, 'aPos');
  function drawQuad2 (prog) {
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    const a = gl.getAttribLocation(prog, 'aPos');
    gl.enableVertexAttribArray(a);
    gl.vertexAttribPointer(a, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(a);
  }

  function uploadParticles () {
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, fluid.particlePos.subarray(0, 2 * fluid.numParticles), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
    gl.bufferData(gl.ARRAY_BUFFER, fluid.particleColor.subarray(0, 3 * fluid.numParticles), gl.DYNAMIC_DRAW);
  }
  function drawParticles (prog, pointSize) {
    gl.useProgram(prog);
    gl.uniform2f(gl.getUniformLocation(prog, 'uDomain'), simWidth, simHeight);
    gl.uniform1f(gl.getUniformLocation(prog, 'uPointSize'), pointSize);
    const aP = gl.getAttribLocation(prog, 'aPos');
    const aC = gl.getAttribLocation(prog, 'aColor');
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.enableVertexAttribArray(aP);
    gl.vertexAttribPointer(aP, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
    gl.enableVertexAttribArray(aC);
    gl.vertexAttribPointer(aC, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, fluid.numParticles);
    gl.disableVertexAttribArray(aC);
    gl.disableVertexAttribArray(aP);
  }

  function render () {
    uploadParticles();
    if (!params.showParticles) {
      // density field pass (half resolution, additive)
      ensureField();
      gl.bindFramebuffer(gl.FRAMEBUFFER, field.fbo);
      gl.viewport(0, 0, fieldW, fieldH);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.useProgram(fieldProg);
      gl.uniform1f(gl.getUniformLocation(fieldProg, 'uW'), floatOK ? 1.0 : 0.3);
      drawParticles(fieldProg, 5.0 * fluid.particleRadius / simWidth * fieldW);
      gl.disable(gl.BLEND);
      // smooth the field so the surface reads as one liquid (separable blur)
      gl.useProgram(blur2Prog);
      gl.uniform1i(gl.getUniformLocation(blur2Prog, 'uTex'), 0);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fieldS.fbo);
      gl.bindTexture(gl.TEXTURE_2D, field.tex);
      gl.uniform2f(gl.getUniformLocation(blur2Prog, 'uDir'), 1 / fieldW, 0);
      drawQuad2(blur2Prog);
      gl.bindFramebuffer(gl.FRAMEBUFFER, field.fbo);
      gl.bindTexture(gl.TEXTURE_2D, fieldS.tex);
      gl.uniform2f(gl.getUniformLocation(blur2Prog, 'uDir'), 0, 1 / fieldH);
      drawQuad2(blur2Prog);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.020, 0.027, 0.047, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(bgProg);
    drawQuad2(bgProg);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    if (params.showParticles) {
      drawParticles(pointProg, 2.4 * fluid.particleRadius / simWidth * canvas.width);
    } else {
      gl.useProgram(surfProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, field.tex);
      gl.uniform1i(gl.getUniformLocation(surfProg, 'uField'), 0);
      gl.uniform2f(gl.getUniformLocation(surfProg, 'uTexel'), 1 / fieldW, 1 / fieldH);
      gl.uniform1f(gl.getUniformLocation(surfProg, 'uT0'), floatOK ? 0.28 : 0.084);
      gl.uniform1f(gl.getUniformLocation(surfProg, 'uT1'), floatOK ? 0.62 : 0.186);
      drawQuad2(surfProg);
    }
    if (obstacle.active) {
      gl.useProgram(discProg);
      gl.uniform2f(gl.getUniformLocation(discProg, 'uDomain'), simWidth, simHeight);
      gl.uniform2f(gl.getUniformLocation(discProg, 'uCenter'), obstacle.x, obstacle.y);
      gl.uniform1f(gl.getUniformLocation(discProg, 'uRadius'), obstacle.radius);
      gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
      gl.enableVertexAttribArray(aPosDisc);
      gl.vertexAttribPointer(aPosDisc, 2, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.disableVertexAttribArray(aPosDisc);
    }
    gl.disable(gl.BLEND);
  }

  function resizeCanvas () {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const w = Math.floor(canvas.clientWidth * dpr);
    const h = Math.floor(canvas.clientHeight * dpr);
    if (w && h && (canvas.width !== w || canvas.height !== h)) {
      canvas.width = w; canvas.height = h;
      setupScene();
    }
  }
  resizeCanvas();
  if (!fluid) setupScene();

  let lastPointer = null;
  let simAcc = 1, simLast = performance.now();
  function toSim (e) {
    const rect = canvas.getBoundingClientRect();
    // keep the whole paddle disc clear of the walls (see 3D version)
    const m = obstacle.radius + (fluid ? fluid.h : 0.05);
    return [
      Math.min(simWidth - m, Math.max(m, (e.clientX - rect.left) / rect.width * simWidth)),
      Math.min(simHeight - m, Math.max(m, (1 - (e.clientY - rect.top) / rect.height) * simHeight)),
    ];
  }
  canvas.addEventListener('pointerdown', e => {
    canvas.setPointerCapture(e.pointerId);
    const [x, y] = toSim(e);
    obstacle.active = true;
    obstacle.x = x; obstacle.y = y;
    obstacle.vx = 0; obstacle.vy = 0;
    lastPointer = { x, y, t: performance.now() };
  });
  canvas.addEventListener('pointermove', e => {
    if (!obstacle.active) return;
    const [x, y] = toSim(e);
    const now = performance.now();
    const dt = Math.max((now - lastPointer.t) / 1000, 1e-3);
    obstacle.vx = (x - lastPointer.x) / dt;
    obstacle.vy = (y - lastPointer.y) / dt;
    const sp = Math.hypot(obstacle.vx, obstacle.vy);
    if (sp > 12) { const k = 12 / sp; obstacle.vx *= k; obstacle.vy *= k; }
    obstacle.x = x; obstacle.y = y;
    lastPointer = { x, y, t: now };
  });
  const endPointer = () => { obstacle.active = false; obstacle.vx = 0; obstacle.vy = 0; };
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);

  return {
    canvas, params,
    hint: 'ドラッグで水をかき混ぜる',
    sub: '水 2D — 重力つき液体(FLIP法)',
    controls: [
      { label: '重力', min: 0, max: 20, step: 0.1,
        get: () => params.gravity, set: v => { params.gravity = v; }, fmt: v => v.toFixed(1) },
      { label: 'しぶき (FLIP率)', min: 0, max: 0.2, step: 0.01,
        get: () => params.flipRatio, set: v => { params.flipRatio = v; }, fmt: v => v.toFixed(2) },
      { label: '水の量', min: 0.15, max: 0.85, step: 0.05,
        get: () => params.waterAmount, set: v => { params.waterAmount = v; setupScene(); }, fmt: v => v.toFixed(2) },
      { label: '解像度', min: 60, max: 140, step: 10,
        get: () => params.res, set: v => { params.res = v; setupScene(); }, fmt: v => String(Math.round(v)) },
    ],
    actions: [
      { label: 'リセット (R)', onClick: setupScene },
      { label: '粒子表示 (V)', pressed: () => params.showParticles,
        onClick: () => { params.showParticles = !params.showParticles; } },
    ],
    onKey (e) {
      if (e.key === 'r' || e.key === 'R') { setupScene(); return true; }
      if (e.key === 'v' || e.key === 'V') { params.showParticles = !params.showParticles; return true; }
      return false;
    },
    frame (now) {
      resizeCanvas();
      const elapsed = Math.min((now - simLast) / 1000, 0.1);
      simLast = now;
      if (!params.paused) {
        // decouple sim rate from display rate: at most ~60 steps/s on any monitor
        simAcc += elapsed;
        if (simAcc >= 1 / 65) {
          fluid.simulate(1 / 60, params.gravity, params.flipRatio, obstacle);
          if (!obstacle.active) { obstacle.vx = 0; obstacle.vy = 0; }
          simAcc = Math.min(simAcc - 1 / 60, 1 / 60);
        }
      }
      render();
    },
  };
}

// =====================================================================
// MODULE 4: Water 3D — 3D FLIP in a cubic tank, full feature set
// =====================================================================
