function createWater3D (planetMode) {
  const canvas = makeCanvas('grab');
  const gl = canvas.getContext('webgl2', { alpha: false, depth: true, antialias: true })
          || canvas.getContext('webgl', { alpha: false, depth: true, antialias: true });
  if (!gl) { canvas.remove(); return { error: true }; }
  const gl2 = typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext;
  const surfaceOK = gl2 && !!gl.getExtension('EXT_color_buffer_float');
  const halfLinear = surfaceOK && !!gl.getExtension('OES_texture_half_float_linear');

  const FLUID = 0, AIR = 1, SOLID = 2;

  class FlipFluid3 {
    constructor (width, height, depth, spacing, particleRadius, maxParticles, shapeFn) {
      this.shapeFn = shapeFn || null;   // static container shape: (x,y,z) => is solid
      this.cx = width * 0.5;            // domain center (cube), used for radial gravity
      this.nX = Math.floor(width / spacing) + 1;
      this.nY = Math.floor(height / spacing) + 1;
      this.nZ = Math.floor(depth / spacing) + 1;
      this.h = Math.max(width / this.nX, height / this.nY, depth / this.nZ);
      this.inv = 1.0 / this.h;
      this.sx = this.nY * this.nZ;
      this.sy = this.nZ;
      const nc = this.numCells = this.nX * this.nY * this.nZ;
      this.u = new Float32Array(nc);
      this.v = new Float32Array(nc);
      this.w = new Float32Array(nc);
      this.du = new Float32Array(nc);
      this.dv = new Float32Array(nc);
      this.dw = new Float32Array(nc);
      this.prevU = new Float32Array(nc);
      this.prevV = new Float32Array(nc);
      this.prevW = new Float32Array(nc);
      this.s = new Float32Array(nc);
      this.cellType = new Int32Array(nc);
      this.particleDensity = new Float32Array(nc);
      this.particleRestDensity = 0.0;
      this.maxParticles = maxParticles;
      this.particlePos = new Float32Array(3 * maxParticles);
      this.particleVel = new Float32Array(3 * maxParticles);
      this.particleColor = new Float32Array(3 * maxParticles);
      this.particleRadius = particleRadius;
      this.pInv = 1.0 / (2.2 * particleRadius);
      this.pNumX = Math.floor(width * this.pInv) + 1;
      this.pNumY = Math.floor(height * this.pInv) + 1;
      this.pNumZ = Math.floor(depth * this.pInv) + 1;
      this.pNumCells = this.pNumX * this.pNumY * this.pNumZ;
      this.numCellParticles = new Int32Array(this.pNumCells);
      this.firstCellParticle = new Int32Array(this.pNumCells + 1);
      this.cellParticleIds = new Int32Array(maxParticles);
      this.numParticles = 0;
    }
    integrateParticles (dt, gx, gy, gz, radial) {
      const pos = this.particlePos, vel = this.particleVel;
      const maxV = 8, maxV2 = maxV * maxV;   // safety cap against runaway particles
      const gmag = Math.sqrt(gx * gx + gy * gy + gz * gz);
      // globe mode: leveling currents in a shallow ocean are slow, so damp far
      // less or the sea freezes into a lumpy shape before it can flatten
      const damp = radial ? 0.9995 : 0.997;
      for (let i = 0; i < this.numParticles; i++) {
        let ax = gx, ay = gy, az = gz;
        if (radial) {
          // planet mode: gravity points at the center
          const ddx = this.cx - pos[3 * i], ddy = this.cx - pos[3 * i + 1], ddz = this.cx - pos[3 * i + 2];
          const dd = Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz) || 1e-6;
          ax = ddx / dd * gmag; ay = ddy / dd * gmag; az = ddz / dd * gmag;
          if (this.tideK > 0) {
            // tidal (differential) gravity of the moon and the sun:
            // a = T * (3 (r·m)m - r) — bulges on the near AND far sides
            const rx = -ddx, ry = -ddy, rz = -ddz;
            const m = this.tideDir, s2 = this.tideDir2;
            const dm = rx * m[0] + ry * m[1] + rz * m[2];
            ax += this.tideK * (3 * dm * m[0] - rx);
            ay += this.tideK * (3 * dm * m[1] - ry);
            az += this.tideK * (3 * dm * m[2] - rz);
            const ds = rx * s2[0] + ry * s2[1] + rz * s2[2];
            ax += this.tideK2 * (3 * ds * s2[0] - rx);
            ay += this.tideK2 * (3 * ds * s2[1] - ry);
            az += this.tideK2 * (3 * ds * s2[2] - rz);
          }
          const w = this.spin || 0;
          if (w > 0) {
            // rotating frame (spin axis +y): centrifugal + Coriolis
            ax += w * w * (pos[3 * i] - this.cx) - 2 * w * vel[3 * i + 2];
            az += w * w * (pos[3 * i + 2] - this.cx) + 2 * w * vel[3 * i];
          }
        }
        let vx = (vel[3 * i] + dt * ax) * damp;
        let vy = (vel[3 * i + 1] + dt * ay) * damp;
        let vz = (vel[3 * i + 2] + dt * az) * damp;
        const v2 = vx * vx + vy * vy + vz * vz;
        if (v2 > maxV2) { const k = maxV / Math.sqrt(v2); vx *= k; vy *= k; vz *= k; }
        vel[3 * i] = vx; vel[3 * i + 1] = vy; vel[3 * i + 2] = vz;
        pos[3 * i] += vx * dt;
        pos[3 * i + 1] += vy * dt;
        pos[3 * i + 2] += vz * dt;
      }
    }
    pushParticlesApart (numIters) {
      const pos = this.particlePos;
      const pNumY = this.pNumY, pNumZ = this.pNumZ;
      const psx = pNumY * pNumZ, psy = pNumZ;
      this.numCellParticles.fill(0);
      for (let i = 0; i < this.numParticles; i++) {
        const xi = Math.max(0, Math.min(this.pNumX - 1, Math.floor(pos[3 * i] * this.pInv)));
        const yi = Math.max(0, Math.min(pNumY - 1, Math.floor(pos[3 * i + 1] * this.pInv)));
        const zi = Math.max(0, Math.min(pNumZ - 1, Math.floor(pos[3 * i + 2] * this.pInv)));
        this.numCellParticles[xi * psx + yi * psy + zi]++;
      }
      let first = 0;
      for (let i = 0; i < this.pNumCells; i++) {
        first += this.numCellParticles[i];
        this.firstCellParticle[i] = first;
      }
      this.firstCellParticle[this.pNumCells] = first;
      for (let i = 0; i < this.numParticles; i++) {
        const xi = Math.max(0, Math.min(this.pNumX - 1, Math.floor(pos[3 * i] * this.pInv)));
        const yi = Math.max(0, Math.min(pNumY - 1, Math.floor(pos[3 * i + 1] * this.pInv)));
        const zi = Math.max(0, Math.min(pNumZ - 1, Math.floor(pos[3 * i + 2] * this.pInv)));
        this.cellParticleIds[--this.firstCellParticle[xi * psx + yi * psy + zi]] = i;
      }
      const minDist = 2.0 * this.particleRadius;
      const minDist2 = minDist * minDist;
      for (let iter = 0; iter < numIters; iter++) {
        for (let i = 0; i < this.numParticles; i++) {
          const px = pos[3 * i], py = pos[3 * i + 1], pz = pos[3 * i + 2];
          const pxi = Math.floor(px * this.pInv), pyi = Math.floor(py * this.pInv), pzi = Math.floor(pz * this.pInv);
          const x0 = Math.max(pxi - 1, 0), y0 = Math.max(pyi - 1, 0), z0 = Math.max(pzi - 1, 0);
          const x1 = Math.min(pxi + 1, this.pNumX - 1), y1 = Math.min(pyi + 1, pNumY - 1), z1 = Math.min(pzi + 1, pNumZ - 1);
          for (let xi = x0; xi <= x1; xi++) {
            for (let yi = y0; yi <= y1; yi++) {
              for (let zi = z0; zi <= z1; zi++) {
                const cell = xi * psx + yi * psy + zi;
                const start = this.firstCellParticle[cell], end = this.firstCellParticle[cell + 1];
                for (let k = start; k < end; k++) {
                  const id = this.cellParticleIds[k];
                  if (id <= i) continue;   // visit each pair once
                  let dx = pos[3 * id] - px;
                  let dy = pos[3 * id + 1] - py;
                  let dz = pos[3 * id + 2] - pz;
                  const d2 = dx * dx + dy * dy + dz * dz;
                  if (d2 > minDist2 || d2 === 0) continue;
                  const d = Math.sqrt(d2);
                  const sc = 0.5 * (minDist - d) / d;
                  dx *= sc; dy *= sc; dz *= sc;
                  pos[3 * i] -= dx; pos[3 * i + 1] -= dy; pos[3 * i + 2] -= dz;
                  pos[3 * id] += dx; pos[3 * id + 1] += dy; pos[3 * id + 2] += dz;
                  const col = this.particleColor;
                  for (let ch = 0; ch < 3; ch++) {
                    const c0 = col[3 * i + ch], c1 = col[3 * id + ch];
                    const mid = (c0 + c1) * 0.5;
                    col[3 * i + ch] = c0 + (mid - c0) * 0.003;
                    col[3 * id + ch] = c1 + (mid - c1) * 0.003;
                  }
                }
              }
            }
          }
        }
      }
    }
    handleParticleCollisions (obstacle, balls) {
      const h = this.h, r = this.particleRadius;
      const minX = h + r, maxX = (this.nX - 1) * h - r;
      const minY = h + r, maxY = (this.nY - 1) * h - r;
      const minZ = h + r, maxZ = (this.nZ - 1) * h - r;
      const useObs = obstacle.active;
      const minObsDist = obstacle.radius + r;
      const minObsDist2 = minObsDist * minObsDist;
      const pos = this.particlePos, vel = this.particleVel, col = this.particleColor;
      const nBalls = balls.length;
      for (let i = 0; i < this.numParticles; i++) {
        let x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
        if (useObs) {
          const dx = x - obstacle.x, dy = y - obstacle.y, dz = z - obstacle.z;
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < minObsDist2) {
            const d = Math.sqrt(d2) || 1e-6;
            x = obstacle.x + dx / d * minObsDist;
            y = obstacle.y + dy / d * minObsDist;
            z = obstacle.z + dz / d * minObsDist;
            vel[3 * i] = obstacle.vx;
            vel[3 * i + 1] = obstacle.vy;
            vel[3 * i + 2] = obstacle.vz;
            if (obstacle.dye) {
              col[3 * i] = obstacle.dye[0];
              col[3 * i + 1] = obstacle.dye[1];
              col[3 * i + 2] = obstacle.dye[2];
            }
          }
        }
        for (let b = 0; b < nBalls; b++) {
          const ball = balls[b];
          const md = ball.r + r;
          const dx = x - ball.x, dy = y - ball.y, dz = z - ball.z;
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < md * md) {
            const d = Math.sqrt(d2) || 1e-6;
            x = ball.x + dx / d * md;
            y = ball.y + dy / d * md;
            z = ball.z + dz / d * md;
            vel[3 * i] = 0.5 * (vel[3 * i] + ball.vx);
            vel[3 * i + 1] = 0.5 * (vel[3 * i + 1] + ball.vy);
            vel[3 * i + 2] = 0.5 * (vel[3 * i + 2] + ball.vz);
          }
        }
        if (x < minX) { x = minX; vel[3 * i] = 0; }
        if (x > maxX) { x = maxX; vel[3 * i] = 0; }
        if (y < minY) { y = minY; vel[3 * i + 1] = 0; }
        if (y > maxY) { y = maxY; vel[3 * i + 1] = 0; }
        if (z < minZ) { z = minZ; vel[3 * i + 2] = 0; }
        if (z > maxZ) { z = maxZ; vel[3 * i + 2] = 0; }
        pos[3 * i] = x; pos[3 * i + 1] = y; pos[3 * i + 2] = z;
      }
    }
    updateParticleDensity () {
      const h = this.h, inv = this.inv, h2 = 0.5 * h;
      const sx = this.sx, sy = this.sy;
      const d = this.particleDensity;
      d.fill(0);
      for (let i = 0; i < this.numParticles; i++) {
        const x = Math.max(h, Math.min((this.nX - 1) * h, this.particlePos[3 * i]));
        const y = Math.max(h, Math.min((this.nY - 1) * h, this.particlePos[3 * i + 1]));
        const z = Math.max(h, Math.min((this.nZ - 1) * h, this.particlePos[3 * i + 2]));
        const x0 = Math.min(((x - h2) * inv) | 0, this.nX - 2), tx = (x - h2 - x0 * h) * inv;
        const y0 = Math.min(((y - h2) * inv) | 0, this.nY - 2), ty = (y - h2 - y0 * h) * inv;
        const z0 = Math.min(((z - h2) * inv) | 0, this.nZ - 2), tz = (z - h2 - z0 * h) * inv;
        const base = x0 * sx + y0 * sy + z0;
        const rx = 1 - tx, ry = 1 - ty, rz = 1 - tz;
        d[base] += rx * ry * rz;
        d[base + sx] += tx * ry * rz;
        d[base + sy] += rx * ty * rz;
        d[base + sx + sy] += tx * ty * rz;
        d[base + 1] += rx * ry * tz;
        d[base + sx + 1] += tx * ry * tz;
        d[base + sy + 1] += rx * ty * tz;
        d[base + sx + sy + 1] += tx * ty * tz;
      }
      if (this.particleRestDensity === 0.0) {
        let sum = 0, count = 0;
        for (let i = 0; i < this.numCells; i++) {
          if (this.cellType[i] === FLUID) { sum += d[i]; count++; }
        }
        if (count > 0) this.particleRestDensity = sum / count;
      }
    }
    transferVelocities (toGrid, flipRatio) {
      const h = this.h, inv = this.inv, h2 = 0.5 * h;
      const sx = this.sx, sy = this.sy;
      const pos = this.particlePos, vel = this.particleVel;
      if (toGrid) {
        this.prevU.set(this.u); this.prevV.set(this.v); this.prevW.set(this.w);
        this.du.fill(0); this.dv.fill(0); this.dw.fill(0);
        this.u.fill(0); this.v.fill(0); this.w.fill(0);
        for (let i = 0; i < this.numCells; i++) {
          this.cellType[i] = this.s[i] === 0 ? SOLID : AIR;
        }
        for (let i = 0; i < this.numParticles; i++) {
          const xi = Math.max(0, Math.min(this.nX - 1, Math.floor(pos[3 * i] * inv)));
          const yi = Math.max(0, Math.min(this.nY - 1, Math.floor(pos[3 * i + 1] * inv)));
          const zi = Math.max(0, Math.min(this.nZ - 1, Math.floor(pos[3 * i + 2] * inv)));
          const cell = xi * sx + yi * sy + zi;
          if (this.cellType[cell] === AIR) this.cellType[cell] = FLUID;
        }
      }
      for (let component = 0; component < 3; component++) {
        const dx = component === 0 ? 0 : h2;
        const dy = component === 1 ? 0 : h2;
        const dz = component === 2 ? 0 : h2;
        const f = component === 0 ? this.u : component === 1 ? this.v : this.w;
        const prevF = component === 0 ? this.prevU : component === 1 ? this.prevV : this.prevW;
        const dArr = component === 0 ? this.du : component === 1 ? this.dv : this.dw;
        const offset = component === 0 ? sx : component === 1 ? sy : 1;
        for (let i = 0; i < this.numParticles; i++) {
          const x = Math.max(h, Math.min((this.nX - 1) * h, pos[3 * i]));
          const y = Math.max(h, Math.min((this.nY - 1) * h, pos[3 * i + 1]));
          const z = Math.max(h, Math.min((this.nZ - 1) * h, pos[3 * i + 2]));
          const x0 = Math.min(((x - dx) * inv) | 0, this.nX - 2), tx = (x - dx - x0 * h) * inv;
          const y0 = Math.min(((y - dy) * inv) | 0, this.nY - 2), ty = (y - dy - y0 * h) * inv;
          const z0 = Math.min(((z - dz) * inv) | 0, this.nZ - 2), tz = (z - dz - z0 * h) * inv;
          const rx = 1 - tx, ry = 1 - ty, rz = 1 - tz;
          const base = x0 * sx + y0 * sy + z0;
          const n0 = base, w0 = rx * ry * rz;
          const n1 = base + sx, w1 = tx * ry * rz;
          const n2 = base + sy, w2 = rx * ty * rz;
          const n3 = base + sx + sy, w3 = tx * ty * rz;
          const n4 = base + 1, w4 = rx * ry * tz;
          const n5 = base + sx + 1, w5 = tx * ry * tz;
          const n6 = base + sy + 1, w6 = rx * ty * tz;
          const n7 = base + sx + sy + 1, w7 = tx * ty * tz;
          if (toGrid) {
            const pv = vel[3 * i + component];
            f[n0] += pv * w0; dArr[n0] += w0;
            f[n1] += pv * w1; dArr[n1] += w1;
            f[n2] += pv * w2; dArr[n2] += w2;
            f[n3] += pv * w3; dArr[n3] += w3;
            f[n4] += pv * w4; dArr[n4] += w4;
            f[n5] += pv * w5; dArr[n5] += w5;
            f[n6] += pv * w6; dArr[n6] += w6;
            f[n7] += pv * w7; dArr[n7] += w7;
          } else {
            const t = this.cellType;
            const v0 = (t[n0] !== AIR || t[n0 - offset] !== AIR) ? 1 : 0;
            const v1 = (t[n1] !== AIR || t[n1 - offset] !== AIR) ? 1 : 0;
            const v2 = (t[n2] !== AIR || t[n2 - offset] !== AIR) ? 1 : 0;
            const v3 = (t[n3] !== AIR || t[n3 - offset] !== AIR) ? 1 : 0;
            const v4 = (t[n4] !== AIR || t[n4 - offset] !== AIR) ? 1 : 0;
            const v5 = (t[n5] !== AIR || t[n5 - offset] !== AIR) ? 1 : 0;
            const v6 = (t[n6] !== AIR || t[n6 - offset] !== AIR) ? 1 : 0;
            const v7 = (t[n7] !== AIR || t[n7 - offset] !== AIR) ? 1 : 0;
            const d = v0 * w0 + v1 * w1 + v2 * w2 + v3 * w3 + v4 * w4 + v5 * w5 + v6 * w6 + v7 * w7;
            if (d > 0) {
              const pv = vel[3 * i + component];
              const picV = (v0 * w0 * f[n0] + v1 * w1 * f[n1] + v2 * w2 * f[n2] + v3 * w3 * f[n3]
                          + v4 * w4 * f[n4] + v5 * w5 * f[n5] + v6 * w6 * f[n6] + v7 * w7 * f[n7]) / d;
              const corr = (v0 * w0 * (f[n0] - prevF[n0]) + v1 * w1 * (f[n1] - prevF[n1])
                          + v2 * w2 * (f[n2] - prevF[n2]) + v3 * w3 * (f[n3] - prevF[n3])
                          + v4 * w4 * (f[n4] - prevF[n4]) + v5 * w5 * (f[n5] - prevF[n5])
                          + v6 * w6 * (f[n6] - prevF[n6]) + v7 * w7 * (f[n7] - prevF[n7])) / d;
              vel[3 * i + component] = (1 - flipRatio) * picV + flipRatio * (pv + corr);
            }
          }
        }
        if (toGrid) {
          for (let i = 0; i < f.length; i++) {
            if (dArr[i] > 0) f[i] /= dArr[i];
          }
        }
      }
      if (toGrid) {
        // solid boundary condition: block only the INTO-solid component of each
        // face and keep the along/outward flow. Full pinning acted as a no-slip
        // glue on the voxelized terrain, so thin sheets of water never drained.
        const t = this.cellType;
        const u2 = this.u, v2 = this.v, w2 = this.w;
        for (let i = 0; i < this.nX; i++) {
          for (let j = 0; j < this.nY; j++) {
            for (let k = 0; k < this.nZ; k++) {
              const idx = i * sx + j * sy + k;
              const solid = t[idx] === SOLID;
              const lsX = i > 0 && t[idx - sx] === SOLID;
              if (solid && lsX) u2[idx] = this.prevU[idx];
              else if (lsX) { if (u2[idx] < 0) u2[idx] = 0; }
              else if (solid) { if (u2[idx] > 0) u2[idx] = 0; }
              const lsY = j > 0 && t[idx - sy] === SOLID;
              if (solid && lsY) v2[idx] = this.prevV[idx];
              else if (lsY) { if (v2[idx] < 0) v2[idx] = 0; }
              else if (solid) { if (v2[idx] > 0) v2[idx] = 0; }
              const lsZ = k > 0 && t[idx - 1] === SOLID;
              if (solid && lsZ) w2[idx] = this.prevW[idx];
              else if (lsZ) { if (w2[idx] < 0) w2[idx] = 0; }
              else if (solid) { if (w2[idx] > 0) w2[idx] = 0; }
            }
          }
        }
        // moving spheres (paddle, balls) re-impose their stamped face velocities
        const oc = this.obsCells || [];
        for (let q = 0; q < oc.length; q++) {
          const idx = oc[q];
          u2[idx] = this.prevU[idx]; u2[idx + sx] = this.prevU[idx + sx];
          v2[idx] = this.prevV[idx]; v2[idx + sy] = this.prevV[idx + sy];
          w2[idx] = this.prevW[idx]; w2[idx + 1] = this.prevW[idx + 1];
        }
        // free-slip walls: tangential velocity inside the wall layer mirrors the
        // adjacent fluid, so water slides along walls instead of being sheared to rest
        const nX = this.nX, nY = this.nY, nZ = this.nZ;
        const u = this.u, v = this.v, w = this.w;
        for (let j = 0; j < nY; j++) {
          for (let k = 0; k < nZ; k++) {
            const a = j * sy + k, b = sx + j * sy + k;
            const a2 = (nX - 1) * sx + j * sy + k, b2 = (nX - 2) * sx + j * sy + k;
            v[a] = v[b]; w[a] = w[b];
            v[a2] = v[b2]; w[a2] = w[b2];
          }
        }
        for (let i = 0; i < nX; i++) {
          for (let k = 0; k < nZ; k++) {
            const a = i * sx + k, b = i * sx + sy + k;
            const a2 = i * sx + (nY - 1) * sy + k, b2 = i * sx + (nY - 2) * sy + k;
            u[a] = u[b]; w[a] = w[b];
            u[a2] = u[b2]; w[a2] = w[b2];
          }
        }
        for (let i = 0; i < nX; i++) {
          for (let j = 0; j < nY; j++) {
            const a = i * sx + j * sy, b = a + 1;
            const a2 = i * sx + j * sy + nZ - 1, b2 = a2 - 1;
            u[a] = u[b]; v[a] = v[b];
            u[a2] = u[b2]; v[a2] = v[b2];
          }
        }
      }
    }
    solveIncompressibility (numIters, overRelaxation) {
      // FLIP delta baseline: snapshot the grid AFTER particle transfer, BEFORE projection.
      // (Snapshotting last frame's grid instead makes moving particles self-amplify.)
      this.prevU.set(this.u);
      this.prevV.set(this.v);
      this.prevW.set(this.w);
      const sx = this.sx, sy = this.sy;
      const t = this.cellType, s = this.s;
      const u = this.u, v = this.v, w = this.w, pd = this.particleDensity;
      const rest = this.particleRestDensity;
      // collect fluid cells once per frame, then sweep only those each iteration
      if (!this.fluidCells) this.fluidCells = new Int32Array(this.numCells);
      const list = this.fluidCells;
      let m = 0;
      for (let i = 1; i < this.nX - 1; i++) {
        for (let j = 1; j < this.nY - 1; j++) {
          const rowBase = i * sx + j * sy;
          for (let k = 1; k < this.nZ - 1; k++) {
            if (t[rowBase + k] === FLUID) list[m++] = rowBase + k;
          }
        }
      }
      for (let iter = 0; iter < numIters; iter++) {
        for (let q = 0; q < m; q++) {
          const c = list[q];
          const sx0 = s[c - sx], sx1 = s[c + sx];
          const sy0 = s[c - sy], sy1 = s[c + sy];
          const sz0 = s[c - 1], sz1 = s[c + 1];
          const sSum = sx0 + sx1 + sy0 + sy1 + sz0 + sz1;
          if (sSum === 0) continue;
          let div = u[c + sx] - u[c] + v[c + sy] - v[c] + w[c + 1] - w[c];
          if (rest > 0) {
            // deadband + soft gain: fix real clumping without simmering at rest
            const compression = pd[c] - rest * 1.05;
            if (compression > 0) div -= 0.7 * compression;
          }
          const p = (-div / sSum) * overRelaxation;
          u[c] -= sx0 * p; u[c + sx] += sx1 * p;
          v[c] -= sy0 * p; v[c + sy] += sy1 * p;
          w[c] -= sz0 * p; w[c + 1] += sz1 * p;
        }
      }
    }
    updateParticleColors () {
      const inv = this.inv, sx = this.sx, sy = this.sy;
      const rest = this.particleRestDensity;
      const col = this.particleColor, pos = this.particlePos;
      for (let i = 0; i < this.numParticles; i++) {
        for (let ch = 0; ch < 3; ch++) {
          const c = col[3 * i + ch];
          col[3 * i + ch] = c + (BASE[ch] - c) * 0.012;
        }
        if (rest > 0) {
          // foam = sparse AND fast: a calm free surface is sparse too, but must stay blue
          const vel = this.particleVel;
          const vx = vel[3 * i], vy = vel[3 * i + 1], vz = vel[3 * i + 2];
          if (vx * vx + vy * vy + vz * vz > 4.0) {
            const xi = Math.max(1, Math.min(this.nX - 2, Math.floor(pos[3 * i] * inv)));
            const yi = Math.max(1, Math.min(this.nY - 2, Math.floor(pos[3 * i + 1] * inv)));
            const zi = Math.max(1, Math.min(this.nZ - 2, Math.floor(pos[3 * i + 2] * inv)));
            if (this.particleDensity[xi * sx + yi * sy + zi] / rest < 0.5) {
              col[3 * i] = FOAM[0]; col[3 * i + 1] = FOAM[1]; col[3 * i + 2] = FOAM[2];
            }
          }
        }
      }
    }
    sampleVelocity (x, y, z) {
      const h = this.h, inv = this.inv, h2 = 0.5 * h;
      const sx = this.sx, sy = this.sy;
      const out = [0, 0, 0];
      x = Math.max(h, Math.min((this.nX - 1) * h, x));
      y = Math.max(h, Math.min((this.nY - 1) * h, y));
      z = Math.max(h, Math.min((this.nZ - 1) * h, z));
      for (let component = 0; component < 3; component++) {
        const dx = component === 0 ? 0 : h2;
        const dy = component === 1 ? 0 : h2;
        const dz = component === 2 ? 0 : h2;
        const f = component === 0 ? this.u : component === 1 ? this.v : this.w;
        const x0 = Math.min(((x - dx) * inv) | 0, this.nX - 2), tx = (x - dx - x0 * h) * inv;
        const y0 = Math.min(((y - dy) * inv) | 0, this.nY - 2), ty = (y - dy - y0 * h) * inv;
        const z0 = Math.min(((z - dz) * inv) | 0, this.nZ - 2), tz = (z - dz - z0 * h) * inv;
        const rx = 1 - tx, ry = 1 - ty, rz = 1 - tz;
        const base = x0 * sx + y0 * sy + z0;
        out[component] =
          rx * ry * rz * f[base] + tx * ry * rz * f[base + sx]
          + rx * ty * rz * f[base + sy] + tx * ty * rz * f[base + sx + sy]
          + rx * ry * tz * f[base + 1] + tx * ry * tz * f[base + sx + 1]
          + rx * ty * tz * f[base + sy + 1] + tx * ty * tz * f[base + sx + sy + 1];
      }
      return out;
    }
    submergedFraction (x, y, z) {
      if (this.particleRestDensity === 0) return 0;
      const xi = Math.max(1, Math.min(this.nX - 2, Math.floor(x * this.inv)));
      const yi = Math.max(1, Math.min(this.nY - 2, Math.floor(y * this.inv)));
      const zi = Math.max(1, Math.min(this.nZ - 2, Math.floor(z * this.inv)));
      return Math.min(1, this.particleDensity[xi * this.sx + yi * this.sy + zi] / this.particleRestDensity);
    }
    addParticle (x, y, z, vx, vy, vz, cr, cg, cb) {
      if (this.numParticles >= this.maxParticles) return false;
      const i = this.numParticles++;
      this.particlePos[3 * i] = x; this.particlePos[3 * i + 1] = y; this.particlePos[3 * i + 2] = z;
      this.particleVel[3 * i] = vx; this.particleVel[3 * i + 1] = vy; this.particleVel[3 * i + 2] = vz;
      this.particleColor[3 * i] = cr; this.particleColor[3 * i + 1] = cg; this.particleColor[3 * i + 2] = cb;
      return true;
    }
    drainAt (cx, cz, radius, maxY) {
      const pos = this.particlePos, vel = this.particleVel, col = this.particleColor;
      const r2 = radius * radius;
      for (let i = 0; i < this.numParticles; i++) {
        const dx = pos[3 * i] - cx, dz = pos[3 * i + 2] - cz;
        if (pos[3 * i + 1] < maxY && dx * dx + dz * dz < r2) {
          const last = --this.numParticles;
          for (let ch = 0; ch < 3; ch++) {
            pos[3 * i + ch] = pos[3 * last + ch];
            vel[3 * i + ch] = vel[3 * last + ch];
            col[3 * i + ch] = col[3 * last + ch];
          }
          i--;
        }
      }
    }
    setObstacle (obstacle, balls) {
      const sx = this.sx, sy = this.sy, h = this.h;
      if (!this.staticS) {
        // walls are static: build them once, then only patch cells the paddle touches
        this.staticS = true;
        for (let i = 0; i < this.nX; i++) {
          for (let j = 0; j < this.nY; j++) {
            for (let k = 0; k < this.nZ; k++) {
              const solid = i === 0 || i === this.nX - 1 || j === 0 || j === this.nY - 1 || k === 0 || k === this.nZ - 1
                || (this.shapeFn && this.shapeFn((i + 0.5) * h, (j + 0.5) * h, (k + 0.5) * h));
              this.s[i * sx + j * sy + k] = solid ? 0 : 1;
            }
          }
        }
        this.obsCells = [];
      }
      for (let q = 0; q < this.obsCells.length; q++) this.s[this.obsCells[q]] = 1;
      this.obsCells.length = 0;
      if (obstacle.active) {
        this.stampSphere(obstacle.x, obstacle.y, obstacle.z, obstacle.radius,
          obstacle.vx, obstacle.vy, obstacle.vz);
      }
      // balls are solid too, so the pressure solve makes water flow around them
      for (let q = 0; q < balls.length; q++) {
        const b = balls[q];
        this.stampSphere(b.x, b.y, b.z, b.r, b.vx, b.vy, b.vz);
      }
    }

    // mark a moving sphere as solid cells carrying its velocity (paddle, balls)
    stampSphere (cx, cy, cz, r, vx, vy, vz) {
      const sx = this.sx, sy = this.sy, h = this.h;
      const r2 = r * r;
      const i0 = Math.max(1, ((cx - r) / h | 0));
      const i1 = Math.min(this.nX - 2, ((cx + r) / h | 0) + 1);
      const j0 = Math.max(1, ((cy - r) / h | 0));
      const j1 = Math.min(this.nY - 2, ((cy + r) / h | 0) + 1);
      const k0 = Math.max(1, ((cz - r) / h | 0));
      const k1 = Math.min(this.nZ - 2, ((cz + r) / h | 0) + 1);
      for (let i = i0; i <= i1; i++) {
        for (let j = j0; j <= j1; j++) {
          for (let k = k0; k <= k1; k++) {
            const dx = (i + 0.5) * h - cx;
            const dy = (j + 0.5) * h - cy;
            const dz = (k + 0.5) * h - cz;
            if (dx * dx + dy * dy + dz * dz < r2) {
              const idx = i * sx + j * sy + k;
              if (this.s[idx] !== 1) continue;   // don't stamp into static walls or other spheres
              this.s[idx] = 0;
              this.obsCells.push(idx);
              // never write sphere velocity into faces that touch a wall cell
              if (i > 1) this.u[idx] = vx;
              if (i < this.nX - 2) this.u[idx + sx] = vx;
              if (j > 1) this.v[idx] = vy;
              if (j < this.nY - 2) this.v[idx + sy] = vy;
              if (k > 1) this.w[idx] = vz;
              if (k < this.nZ - 2) this.w[idx + 1] = vz;
            }
          }
        }
      }
    }
    simulate (dt, gx, gy, gz, flipRatio, obstacle, balls, radial) {
      this.integrateParticles(dt, gx, gy, gz, radial);
      this.pushParticlesApart(2);
      this.handleParticleCollisions(obstacle, balls);
      if (this.postCollide) this.postCollide();   // analytic container-shape collisions
      this.setObstacle(obstacle, balls);
      this.transferVelocities(true, flipRatio);
      this.updateParticleDensity();
      this.solveIncompressibility(24, 1.9);
      this.transferVelocities(false, flipRatio);
      this.tick = (this.tick || 0) + 1;
      if ((this.tick & 1) === 0) this.updateParticleColors();   // cosmetic pass, half rate
    }
  }

  const params = {
    gravity: 9.8,
    flipRatio: planetMode ? 0.10 : 0.60,
    waterAmount: planetMode ? 0.40 : 0.45,
    timeScale: 1.0,
    tiltX: 0, tiltZ: 0,
    res: planetMode ? 50 : 38,
    spin: 0.06, tide: 0.3,
    view: 'space',               // camera: 'space' (stars fixed) | 'sun' (sun fixed)
    shape: planetMode ? 'planet' : 'box',   // 'box' | 'cyl' | 'bowl' | 'planet'
    paused: false, stir: false, pour: false, drain: false,
    showParticles: !surfaceOK,   // surface rendering needs float color buffers
  };
  const SHAPE_LABELS = { box: '立方体', cyl: '円筒', bowl: 'ボウル', planet: '地球' };
  const SHAPE_ORDER = ['box', 'cyl', 'bowl'];   // the globe is its own mode

  // ---- whole-globe terrain: radius table on a lat/long grid ----
  // A full planet filling the box (base radius 0.345) with Earth-like gentle
  // relief (a few % of the radius) and a thin global ocean shell.
  // WebGL2 + float render targets unlocks the hi-res pipeline: 1024x512
  // terrain and ocean grids, GPU-solved, with sub-grid detail added in the
  // shader so even extreme close-ups never resolve individual cells
  const hiRes = gl2 && surfaceOK;
  const TER_W = hiRes ? 2048 : 512, TER_H = hiRes ? 1024 : 256;
  const TER_RMIN = 0.28, TER_RRANGE = 0.14;
  let terrain = null;      // radius per (longitude, latitude)
  let terrainTex = null;
  let seaR = 0.358;        // sea level radius
  let lightAngle = Math.atan2(0.35, 0.4);   // sun azimuth: advances with the spin
  function getLightDir () {
    // axial tilt: the sun's elevation swings once per year (the seasons);
    // the year phase is the sun's lag behind the planet's rotation
    const season = Math.sin(lightAngle - spinAngle);
    const y = 0.45 + 0.25 * season;
    const h = Math.sqrt(Math.max(0.2, 1 - y * y));
    return normalize([Math.cos(lightAngle) * h, y, Math.sin(lightAngle) * h]);
  }
  // the moon orbits in a tilted plane; its differential gravity raises the tides
  let moonAngle = Math.random() * Math.PI * 2;
  function getMoonDir () {
    const c = Math.cos(moonAngle), s = Math.sin(moonAngle);
    return normalize([c, s * 0.342, s * 0.940]);
  }
  // rain falls from a few drifting storm cells, not uniformly over the globe
  const storms = [];
  function randomUnitVec () {
    const a = Math.random() * Math.PI * 2, cb = 2 * Math.random() - 1;
    const sb = Math.sqrt(1 - cb * cb);
    return [sb * Math.cos(a), cb, sb * Math.sin(a)];
  }
  function initStorms () {
    storms.length = 0;
    for (let s = 0; s < 4; s++) {
      storms.push({ dir: randomUnitVec(), axis: randomUnitVec(), speed: 0.10 + Math.random() * 0.15 });
    }
  }
  function rotateVec (v, k, th) {
    const c = Math.cos(th), s = Math.sin(th);
    const cx = k[1] * v[2] - k[2] * v[1];
    const cy = k[2] * v[0] - k[0] * v[2];
    const cz = k[0] * v[1] - k[1] * v[0];
    const dot = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
    return [
      v[0] * c + cx * s + k[0] * dot * (1 - c),
      v[1] * c + cy * s + k[1] * dot * (1 - c),
      v[2] * c + cz * s + k[2] * dot * (1 - c),
    ];
  }
  function updateStorms (dt) {
    for (const st of storms) {
      st.dir = rotateVec(st.dir, st.axis, st.speed * dt);
      st.axis[0] += (Math.random() - 0.5) * 0.02;
      st.axis[1] += (Math.random() - 0.5) * 0.02;
      st.axis[2] += (Math.random() - 0.5) * 0.02;
      const l = Math.hypot(st.axis[0], st.axis[1], st.axis[2]) || 1;
      st.axis[0] /= l; st.axis[1] /= l; st.axis[2] /= l;
    }
  }

  // ---- spherical shallow-water ocean (globe mode) ----
  // The global ocean is a thin film, which is exactly the shallow-water regime:
  // height h and depth-averaged velocity (u east, v south) on a lat/long grid.
  // Tides enter as the tangential tidal acceleration, rotation as Coriolis.
  // On WebGL2 with float render targets the solver runs on the GPU at the
  // full terrain resolution — one cell per terrain-table texel, fine enough
  // for rivers. The CPU solver stays as a 256x128 fallback.
  const gpuSW = hiRes;
  const SW_W = gpuSW ? TER_W : 256, SW_H = gpuSW ? TER_H : 128;
  let swH = null, swU = null, swV = null, swBed = null;
  let swH2 = null, swU2 = null, swV2 = null;
  let swTex = null, swData = null;
  const swSin = new Float32Array(SW_H);      // clamped sin(theta) per row
  const swSinF = new Float32Array(SW_H + 1); // sin(theta) at row faces
  const swCosT = new Float32Array(SW_H);
  const swCosL = new Float32Array(SW_W), swSinL = new Float32Array(SW_W);
  // ---- GPU shallow water: each substep is two fragment passes over the
  // float grid (momentum, then continuity), ping-ponging texA <-> texB.
  // Rendering samples the state via a tiny encode pass; the CPU never
  // touches the ocean at all.
  let swGPU = null, swRainDt = 0;
  function initSWEGPU () {
    if (!swGPU) {
      const V = `#version 300 es
in vec2 aPos; void main () { gl_Position = vec4(aPos, 0.0, 1.0); }`;
      const HEAD = `#version 300 es
precision highp float;
precision highp sampler2D;
out vec4 o;
const float PI = 3.14159265;
const vec2 DIM = vec2(${SW_W}.0, ${SW_H}.0);
`;
      const mom = glProgram(gl, V, HEAD + `
uniform sampler2D uState;
uniform sampler2D uBed;
uniform float uDt, uG, uR, uSpin, uDrag, uTkM, uTkS, uRain;
uniform vec3 uMoon, uSun;
uniform vec4 uStorms[4];
float eta (ivec2 c) { return texelFetch(uBed, c, 0).r + texelFetch(uState, c, 0).r; }
void main () {
  ivec2 c = ivec2(gl_FragCoord.xy);
  int W = int(DIM.x), H = int(DIM.y);
  vec4 s = texelFetch(uState, c, 0);
  float th = (float(c.y) + 0.5) / DIM.y * PI;
  float st = max(0.18, sin(th)), ct = cos(th);
  float la = (float(c.x) + 0.5) / DIM.x * 2.0 * PI - PI;
  float cl = cos(la), sl = sin(la);
  vec3 dd = vec3(st * cl, ct, st * sl);
  float h = s.r;
  if (uRain > 0.0) {
    for (int k = 0; k < 4; k++) {
      if (uStorms[k].w < 0.5) continue;
      float ang = acos(clamp(dot(dd, uStorms[k].xyz), -1.0, 1.0));
      float rr = ang * ang / (0.085 * 0.085);
      if (rr < 1.0) h += 0.09 * uRain * (1.0 - rr);
    }
  }
  if (c.y == 0 || c.y == H - 1) { o = vec4(h, 0.0, 0.0, 0.0); return; }
  float dx = uR * st * (2.0 * PI / DIM.x), dy = uR * (PI / DIM.y);
  float au0 = -uG * (eta(ivec2((c.x + 1) % W, c.y)) - eta(ivec2((c.x + W - 1) % W, c.y))) / (2.0 * dx);
  float av0 = -uG * (eta(ivec2(c.x, c.y + 1)) - eta(ivec2(c.x, c.y - 1))) / (2.0 * dy);
  vec3 a = uTkM * (3.0 * dot(dd, uMoon) * uMoon - dd) * uR
         + uTkS * (3.0 * dot(dd, uSun) * uSun - dd) * uR;
  float f = 2.0 * uSpin * ct;
  float au = au0 + a.x * (-sl) + a.z * cl + f * s.b;
  float av = av0 + a.x * ct * cl + a.y * (-st) + a.z * ct * sl - f * s.g;
  float dryFac = h < 0.0008 ? 0.55 : 1.0;
  o = vec4(h, (s.g + uDt * au) * uDrag * dryFac, (s.b + uDt * av) * uDrag * dryFac, 0.0);
}`);
      const cont = glProgram(gl, V, HEAD + `
uniform sampler2D uState;
uniform float uDt, uR;
void main () {
  ivec2 c = ivec2(gl_FragCoord.xy);
  int W = int(DIM.x), H = int(DIM.y);
  if (c.y == 0) { o = vec4(texelFetch(uState, ivec2(c.x, 1), 0).r, 0.0, 0.0, 0.0); return; }
  if (c.y == H - 1) { o = vec4(texelFetch(uState, ivec2(c.x, H - 2), 0).r, 0.0, 0.0, 0.0); return; }
  vec4 s = texelFetch(uState, c, 0);
  vec4 sE = texelFetch(uState, ivec2((c.x + 1) % W, c.y), 0);
  vec4 sW = texelFetch(uState, ivec2((c.x + W - 1) % W, c.y), 0);
  vec4 sN = texelFetch(uState, ivec2(c.x, c.y - 1), 0);
  vec4 sS = texelFetch(uState, ivec2(c.x, c.y + 1), 0);
  float th = (float(c.y) + 0.5) / DIM.y * PI;
  float st = max(0.18, sin(th));
  float dx = uR * st * (2.0 * PI / DIM.x);
  float dyM = uR * (PI / DIM.y) * st;
  float fsN = max(0.18, sin(float(c.y) / DIM.y * PI));
  float fsS = max(0.18, sin(float(c.y + 1) / DIM.y * PI));
  float ue = 0.5 * (s.g + sE.g), uw = 0.5 * (sW.g + s.g);
  float vn = 0.5 * (sN.b + s.b), vs = 0.5 * (s.b + sS.b);
  float FE = ue * (ue > 0.0 ? s.r : sE.r);
  float FW = uw * (uw > 0.0 ? sW.r : s.r);
  float FN = vn * (vn > 0.0 ? sN.r : s.r);
  float FS = vs * (vs > 0.0 ? s.r : sS.r);
  float div = (FE - FW) / dx + (FS * fsS - FN * fsN) / dyM;
  o = vec4(max(0.0, s.r - uDt * div), s.g, s.b, 0.0);
}`);
      const init = glProgram(gl, V, HEAD + `
uniform sampler2D uBed;
uniform float uSeaR;
void main () {
  float bed = texelFetch(uBed, ivec2(gl_FragCoord.xy), 0).r;
  o = vec4(max(0.0, uSeaR - bed), 0.0, 0.0, 0.0);
}`);
      const enc = glProgram(gl, V, HEAD + `
uniform sampler2D uState;
uniform sampler2D uBed;
uniform float uPB;
void main () {
  ivec2 c = ivec2(gl_FragCoord.xy);
  float bed = texelFetch(uBed, c, 0).r;
  vec4 sv = texelFetch(uState, c, 0);
  float h = sv.r;
  float e2 = h > 0.0008 ? (bed + h - (uPB - 0.10)) / 0.24 : 0.0;
  o = vec4(e2, h / 0.12, min(1.0, length(sv.gb) * 1.5), 1.0);
}`);
      const mkTex = (internal, format, type, filter, wrapS) => {
        const t = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrapS);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D, 0, internal, SW_W, SW_H, 0, format, type, null);
        return t;
      };
      const texA = mkTex(gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST, gl.CLAMP_TO_EDGE);
      const texB = mkTex(gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST, gl.CLAMP_TO_EDGE);
      const bed = mkTex(gl.R32F, gl.RED, gl.FLOAT, gl.NEAREST, gl.CLAMP_TO_EDGE);
      const disp = mkTex(gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR, gl.REPEAT);
      const mkFbo = t => {
        const f = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, f);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
        return f;
      };
      const U = (p, n2) => gl.getUniformLocation(p, n2);
      swGPU = {
        mom, cont, init, enc, texA, texB, bed, disp,
        fboA: mkFbo(texA), fboB: mkFbo(texB), fboD: mkFbo(disp),
        momU: {
          uDt: U(mom, 'uDt'), uG: U(mom, 'uG'), uR: U(mom, 'uR'), uSpin: U(mom, 'uSpin'),
          uDrag: U(mom, 'uDrag'), uTkM: U(mom, 'uTkM'), uTkS: U(mom, 'uTkS'),
          uRain: U(mom, 'uRain'), uMoon: U(mom, 'uMoon'), uSun: U(mom, 'uSun'),
          uStorms: [0, 1, 2, 3].map(k => U(mom, 'uStorms[' + k + ']'))
        },
        contU: { uDt: U(cont, 'uDt'), uR: U(cont, 'uR') },
        initU: { uSeaR: U(init, 'uSeaR') },
        encU: { uPB: U(enc, 'uPB') }
      };
      gl.useProgram(mom);
      gl.uniform1i(U(mom, 'uState'), 0); gl.uniform1i(U(mom, 'uBed'), 1);
      gl.useProgram(cont); gl.uniform1i(U(cont, 'uState'), 0);
      gl.useProgram(init); gl.uniform1i(U(init, 'uBed'), 0);
      gl.useProgram(enc);
      gl.uniform1i(U(enc, 'uState'), 0); gl.uniform1i(U(enc, 'uBed'), 1);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swTex = disp;   // drawScene samples this as uWater
    }
    const r = swGPU;
    gl.bindTexture(gl.TEXTURE_2D, r.bed);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, SW_W, SW_H, 0, gl.RED, gl.FLOAT, terrain);
    gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST);
    gl.viewport(0, 0, SW_W, SW_H);
    gl.useProgram(r.init);
    gl.uniform1f(r.initU.uSeaR, seaR);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, r.bed);
    gl.bindFramebuffer(gl.FRAMEBUFFER, r.fboA); drawQuad(r.init);
    gl.bindFramebuffer(gl.FRAMEBUFFER, r.fboB); drawQuad(r.init);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    swRainDt = 0;
  }
  function sweStepGPU (dt) {
    const r = swGPU;
    if (!r) return;
    gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST);
    gl.viewport(0, 0, SW_W, SW_H);
    const m = getMoonDir(), s2 = getLightDir();
    gl.bindFramebuffer(gl.FRAMEBUFFER, r.fboB);
    gl.useProgram(r.mom);
    gl.uniform1f(r.momU.uDt, dt);
    gl.uniform1f(r.momU.uG, params.gravity);
    gl.uniform1f(r.momU.uR, PBASE);
    gl.uniform1f(r.momU.uSpin, params.spin);
    gl.uniform1f(r.momU.uDrag, 1 - Math.min(0.5, 0.35 * dt));
    gl.uniform1f(r.momU.uTkM, params.tide);
    gl.uniform1f(r.momU.uTkS, params.tide * 0.45);
    gl.uniform3f(r.momU.uMoon, m[0], m[1], m[2]);
    gl.uniform3f(r.momU.uSun, s2[0], s2[1], s2[2]);
    gl.uniform1f(r.momU.uRain, swRainDt);
    for (let k = 0; k < 4; k++) {
      const st2 = storms[k];
      if (st2 && swRainDt > 0) gl.uniform4f(r.momU.uStorms[k], st2.dir[0], st2.dir[1], st2.dir[2], 1);
      else gl.uniform4f(r.momU.uStorms[k], 0, 1, 0, 0);
    }
    swRainDt = 0;
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, r.bed);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, r.texA);
    drawQuad(r.mom);
    gl.bindFramebuffer(gl.FRAMEBUFFER, r.fboA);
    gl.useProgram(r.cont);
    gl.uniform1f(r.contU.uDt, dt);
    gl.uniform1f(r.contU.uR, PBASE);
    gl.bindTexture(gl.TEXTURE_2D, r.texB);
    drawQuad(r.cont);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
  function encodeWaterGPU () {
    const r = swGPU;
    if (!r) return;
    gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST);
    gl.viewport(0, 0, SW_W, SW_H);
    gl.bindFramebuffer(gl.FRAMEBUFFER, r.fboD);
    gl.useProgram(r.enc);
    gl.uniform1f(r.encU.uPB, PBASE);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, r.bed);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, r.texA);
    drawQuad(r.enc);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
  function initSWE () {
    if (gpuSW) { initSWEGPU(); return; }
    const n = SW_W * SW_H;
    swH = new Float32Array(n); swU = new Float32Array(n); swV = new Float32Array(n);
    swH2 = new Float32Array(n); swU2 = new Float32Array(n); swV2 = new Float32Array(n);
    swBed = new Float32Array(n);
    swData = new Uint8Array(n * 4);
    for (let j = 0; j < SW_H; j++) {
      const th = (j + 0.5) / SW_H * Math.PI;
      swSin[j] = Math.max(0.18, Math.sin(th));
      swCosT[j] = Math.cos(th);
    }
    for (let j = 0; j <= SW_H; j++) swSinF[j] = Math.max(0.18, Math.sin(j / SW_H * Math.PI));
    for (let i = 0; i < SW_W; i++) {
      const la = (i + 0.5) / SW_W * Math.PI * 2 - Math.PI;
      swCosL[i] = Math.cos(la); swSinL[i] = Math.sin(la);
    }
    for (let j = 0; j < SW_H; j++) {
      const st = Math.sin((j + 0.5) / SW_H * Math.PI), ct = swCosT[j];
      for (let i = 0; i < SW_W; i++) {
        const idx = j * SW_W + i;
        swBed[idx] = terrainAt(st * swCosL[i], ct, st * swSinL[i]);
        swH[idx] = Math.max(0, seaR - swBed[idx]);   // calm ocean at sea level
      }
    }
    if (!swTex) {
      swTex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, swTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, SW_W, SW_H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }
  }
  function sweStep (dt) {
    if (gpuSW) { sweStepGPU(dt); return; }
    const W = SW_W, H = SW_H;
    const g = params.gravity, R = PBASE;
    const dLam = 2 * Math.PI / W, dThe = Math.PI / H;
    const drag = 1 - Math.min(0.5, 0.35 * dt) * 1;
    const m = getMoonDir(), s2 = getLightDir();
    const Tk = params.tide, Tk2 = params.tide * 0.45;
    const w = params.spin;
    // momentum: pressure gradient of the free surface + tides + Coriolis
    for (let j = 1; j < H - 1; j++) {
      const st = swSin[j], ct = swCosT[j];
      const dx = R * st * dLam, dy = R * dThe;
      const f = 2 * w * ct;
      for (let i = 0; i < W; i++) {
        const idx = j * W + i;
        const iE = j * W + ((i + 1) % W), iW = j * W + ((i - 1 + W) % W);
        const iN = idx - W, iS = idx + W;
        const au0 = -g * ((swBed[iE] + swH[iE]) - (swBed[iW] + swH[iW])) / (2 * dx);
        const av0 = -g * ((swBed[iS] + swH[iS]) - (swBed[iN] + swH[iN])) / (2 * dy);
        const cl = swCosL[i], sl = swSinL[i];
        const ddx = st * cl, ddy = ct, ddz = st * sl;
        const dm = ddx * m[0] + ddy * m[1] + ddz * m[2];
        let ax = Tk * (3 * dm * m[0] - ddx) * R;
        let ay = Tk * (3 * dm * m[1] - ddy) * R;
        let az = Tk * (3 * dm * m[2] - ddz) * R;
        const ds = ddx * s2[0] + ddy * s2[1] + ddz * s2[2];
        ax += Tk2 * (3 * ds * s2[0] - ddx) * R;
        ay += Tk2 * (3 * ds * s2[1] - ddy) * R;
        az += Tk2 * (3 * ds * s2[2] - ddz) * R;
        const uu = swU[idx], vv = swV[idx];
        const au = au0 + ax * (-sl) + az * cl + f * vv;
        const av = av0 + ax * ct * cl + ay * (-st) + az * ct * sl - f * uu;
        const dryFac = swH[idx] < 0.0008 ? 0.55 : 1.0;
        swU2[idx] = (uu + dt * au) * drag * dryFac;
        swV2[idx] = (vv + dt * av) * drag * dryFac;
      }
    }
    // continuity: upwind fluxes with the spherical metric (conserves volume)
    for (let j = 1; j < H - 1; j++) {
      const st = swSin[j];
      const dx = R * st * dLam, dyM = R * dThe * st;
      const sN = swSinF[j], sS = swSinF[j + 1];
      for (let i = 0; i < W; i++) {
        const idx = j * W + i;
        const iE = j * W + ((i + 1) % W), iW = j * W + ((i - 1 + W) % W);
        const iN = idx - W, iS = idx + W;
        const ue = 0.5 * (swU2[idx] + swU2[iE]);
        const uw = 0.5 * (swU2[iW] + swU2[idx]);
        const vn = 0.5 * (swV2[iN] + swV2[idx]);
        const vs = 0.5 * (swV2[idx] + swV2[iS]);
        const FE = ue * (ue > 0 ? swH[idx] : swH[iE]);
        const FW = uw * (uw > 0 ? swH[iW] : swH[idx]);
        const FN = vn * (vn > 0 ? swH[iN] : swH[idx]);
        const FS = vs * (vs > 0 ? swH[idx] : swH[iS]);
        const div = (FE - FW) / dx + (FS * sS - FN * sN) / dyM;
        swH2[idx] = Math.max(0, swH[idx] - dt * div);
      }
    }
    // polar caps: mirror the first interior row, no flow
    for (let i = 0; i < W; i++) {
      swH2[i] = swH2[W + i]; swU2[i] = 0; swV2[i] = 0;
      swH2[(H - 1) * W + i] = swH2[(H - 2) * W + i];
      swU2[(H - 1) * W + i] = 0; swV2[(H - 1) * W + i] = 0;
    }
    let t3 = swH; swH = swH2; swH2 = t3;
    t3 = swU; swU = swU2; swU2 = t3;
    t3 = swV; swV = swV2; swV2 = t3;
  }
  function rainSWE (dt) {
    if (gpuSW) { swRainDt += dt; return; }   // deposited inside the next momentum pass
    const rate = 0.09 * dt;
    for (const st2 of storms) {
      const d = st2.dir;
      const la = Math.atan2(d[2], d[0]);
      const th = Math.acos(Math.max(-1, Math.min(1, d[1])));
      const ci = Math.round((la + Math.PI) / (2 * Math.PI) * SW_W - 0.5);
      const cj = Math.round(th / Math.PI * SW_H - 0.5);
      for (let dj = -3; dj <= 3; dj++) {
        for (let di = -3; di <= 3; di++) {
          const rr = (di * di + dj * dj) / 9.0;
          if (rr > 1) continue;
          const ii = ((ci + di) % SW_W + SW_W) % SW_W;
          const jj = Math.max(1, Math.min(SW_H - 2, cj + dj));
          swH[jj * SW_W + ii] += rate * (1 - rr);
        }
      }
    }
  }
  function uploadWaterTex () {
    if (gpuSW) { encodeWaterGPU(); return; }
    if (!swTex || !swH) return;
    const lo = PBASE - 0.10;
    for (let p2 = 0; p2 < SW_W * SW_H; p2++) {
      const hh = swH[p2];
      let enc = 0;
      if (hh > 0.0012) {
        enc = Math.max(1, Math.min(255, Math.round((swBed[p2] + hh - lo) / 0.24 * 255)));
      }
      swData[4 * p2] = enc;
      swData[4 * p2 + 1] = Math.max(0, Math.min(255, Math.round(hh / 0.12 * 255)));
      swData[4 * p2 + 2] = 0;
      swData[4 * p2 + 3] = 255;
    }
    gl.bindTexture(gl.TEXTURE_2D, swTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, SW_W, SW_H, gl.RGBA, gl.UNSIGNED_BYTE, swData);
  }
  // hi-res terrain synthesis on the GPU: the noise passes over 2M cells would
  // take seconds in JS. The result is read back so river carving (inherently
  // sequential) and the ocean-bed upload keep working on the CPU array.
  let tgRes = null;
  function buildTerrainGPU (cont, CW, CH, thr, off2, off3) {
    if (!tgRes) {
      const V = `#version 300 es
in vec2 aPos; void main () { gl_Position = vec4(aPos, 0.0, 1.0); }`;
      const prog = glProgram(gl, V, `#version 300 es
precision highp float;
out vec4 o;
const float PI = 3.14159265;
const vec2 DIM = vec2(${TER_W}.0, ${TER_H}.0);
uniform sampler2D uCont;
uniform float uThr, uPB;
uniform vec3 uOff2, uOff3;
float hashT (vec3 q3) { q3 = mod(q3, 289.0); return fract(sin(dot(q3, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
float vno (vec3 x) {
  vec3 ip = floor(x), fp = fract(x);
  fp = fp * fp * (3.0 - 2.0 * fp);
  float n000 = hashT(ip),                       n100 = hashT(ip + vec3(1.0, 0.0, 0.0));
  float n010 = hashT(ip + vec3(0.0, 1.0, 0.0)), n110 = hashT(ip + vec3(1.0, 1.0, 0.0));
  float n001 = hashT(ip + vec3(0.0, 0.0, 1.0)), n101 = hashT(ip + vec3(1.0, 0.0, 1.0));
  float n011 = hashT(ip + vec3(0.0, 1.0, 1.0)), n111 = hashT(ip + vec3(1.0, 1.0, 1.0));
  return mix(mix(mix(n000, n100, fp.x), mix(n010, n110, fp.x), fp.y),
             mix(mix(n001, n101, fp.x), mix(n011, n111, fp.x), fp.y), fp.z);
}
void main () {
  vec2 uv = gl_FragCoord.xy / DIM;
  float phi = uv.y * PI;
  float th = uv.x * 2.0 * PI;
  vec3 d = vec3(sin(phi) * cos(th), cos(phi), sin(phi) * sin(th));
  float s = clamp((texture(uCont, uv).r - uThr) / 0.07 + 0.5, 0.0, 1.0);
  float landness = s * s * (3.0 - 2.0 * s);
  float rid = 0.0, ra = 1.0, rf = 5.0, rt = 0.0;
  for (int k = 0; k < 3; k++) {
    float v = 1.0 - abs(2.0 * vno(d * rf + uOff2) - 1.0);
    rid += ra * v * v; rt += ra; ra *= 0.5; rf *= 2.1;
  }
  rid /= rt;
  float mount = pow(rid, 2.5) * 0.11 * landness;
  float det = (vno(d * 12.0 + uOff3) - 0.5) * 0.012
            + (vno(d * 34.0 + uOff3.yzx + 11.0) - 0.5) * 0.005;
  float r = uPB - 0.075 + landness * 0.085 + mount + det;
  o = vec4(clamp(r, uPB - 0.085, uPB + 0.11), 0.0, 0.0, 1.0);
}`);
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, TER_W, TER_H, 0, gl.RGBA, gl.FLOAT, null);
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      const contTex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, contTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      tgRes = { prog, fbo, contTex, buf: new Float32Array(TER_W * TER_H * 4) };
    }
    const r = tgRes;
    gl.bindTexture(gl.TEXTURE_2D, r.contTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, CW, CH, 0, gl.RED, gl.FLOAT, cont);
    gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, r.fbo);
    gl.viewport(0, 0, TER_W, TER_H);
    gl.useProgram(r.prog);
    gl.uniform1i(gl.getUniformLocation(r.prog, 'uCont'), 0);
    gl.uniform1f(gl.getUniformLocation(r.prog, 'uThr'), thr);
    gl.uniform1f(gl.getUniformLocation(r.prog, 'uPB'), PBASE);
    gl.uniform3f(gl.getUniformLocation(r.prog, 'uOff2'), off2[0], off2[1], off2[2]);
    gl.uniform3f(gl.getUniformLocation(r.prog, 'uOff3'), off3[0], off3[1], off3[2]);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, r.contTex);
    drawQuad(r.prog);
    gl.readPixels(0, 0, TER_W, TER_H, gl.RGBA, gl.FLOAT, r.buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    for (let p2 = 0; p2 < TER_W * TER_H; p2++) terrain[p2] = r.buf[4 * p2];
  }
  function buildPlanet () {
    terrain = new Float32Array(TER_W * TER_H);
    const off = [Math.random() * 100, Math.random() * 100, Math.random() * 100];
    const off2 = [Math.random() * 100, Math.random() * 100, Math.random() * 100];
    const off3 = [Math.random() * 100, Math.random() * 100, Math.random() * 100];
    const fract = x => x - Math.floor(x);
    const hash = (i, j, k) => fract(Math.sin(i * 127.1 + j * 311.7 + k * 74.7) * 43758.5453);
    const sm = t => t * t * (3 - 2 * t);
    function noise (x, y, z) {
      const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
      const tx = sm(x - xi), ty = sm(y - yi), tz = sm(z - zi);
      let v = 0;
      for (let c = 0; c < 8; c++) {
        const cx = c & 1, cy = (c >> 1) & 1, cz = (c >> 2) & 1;
        const w = (cx ? tx : 1 - tx) * (cy ? ty : 1 - ty) * (cz ? tz : 1 - tz);
        v += w * hash(xi + cx, yi + cy, zi + cz);
      }
      return v;
    }
    // pass 1: low-frequency continental field (a few large landmasses) —
    // computed on a fixed coarse grid (it only holds continent-scale shapes)
    // and bilinearly upsampled in pass 2, so hi-res generation stays fast
    const CW = 512, CH = 256;
    const cont = new Float32Array(CW * CH);
    for (let j = 0; j < CH; j++) {
      const phi = (j + 0.5) / CH * Math.PI;
      for (let i = 0; i < CW; i++) {
        const th = (i + 0.5) / CW * Math.PI * 2;
        const dx = Math.sin(phi) * Math.cos(th);
        const dy = Math.cos(phi);
        const dz = Math.sin(phi) * Math.sin(th);
        let c = 0, ca = 1, cf = 1.15, ct = 0;
        for (let o = 0; o < 3; o++) {
          c += ca * noise(dx * cf + off[0], dy * cf + off[1], dz * cf + off[2]);
          ct += ca; ca *= 0.4; cf *= 2.2;
        }
        cont[j * CW + i] = c / ct;
      }
    }
    const cAt = (ii, jj) => cont[Math.max(0, Math.min(CH - 1, jj)) * CW + (((ii % CW) + CW) % CW)];
    // set the coastline from this generation's actual distribution, so the
    // land fraction is stable (~48%) no matter how the noise fell
    const sorted = Float32Array.from(cont).sort();
    const thr = sorted[Math.floor(sorted.length * 0.52)];
    // pass 2: hypsometry + ridged mountains + detail
    if (hiRes) {
      buildTerrainGPU(cont, CW, CH, thr, off2, off3);
    } else for (let j = 0; j < TER_H; j++) {
      const phi = (j + 0.5) / TER_H * Math.PI;
      for (let i = 0; i < TER_W; i++) {
        const th = (i + 0.5) / TER_W * Math.PI * 2;
        const dx = Math.sin(phi) * Math.cos(th);
        const dy = Math.cos(phi);
        const dz = Math.sin(phi) * Math.sin(th);
        const cxf = (i + 0.5) / TER_W * CW - 0.5, cyf = (j + 0.5) / TER_H * CH - 0.5;
        const cx0 = Math.floor(cxf), cy0 = Math.floor(cyf);
        const cfx = cxf - cx0, cfy = cyf - cy0;
        const cval = cAt(cx0, cy0) * (1 - cfx) * (1 - cfy) + cAt(cx0 + 1, cy0) * cfx * (1 - cfy)
                   + cAt(cx0, cy0 + 1) * (1 - cfx) * cfy + cAt(cx0 + 1, cy0 + 1) * cfx * cfy;
        const s = Math.max(0, Math.min(1, (cval - thr) / 0.07 + 0.5));
        const landness = s * s * (3 - 2 * s);
        let rid = 0, ra = 1, rf = 5.0, rt = 0;
        for (let o = 0; o < 3; o++) {
          const v = 1 - Math.abs(2 * noise(dx * rf + off2[0], dy * rf + off2[1], dz * rf + off2[2]) - 1);
          rid += ra * v * v;
          rt += ra; ra *= 0.5; rf *= 2.1;
        }
        rid /= rt;
        const mount = Math.pow(rid, 2.5) * 0.11 * landness;
        const det = (noise(dx * 12 + off3[0], dy * 12 + off3[1], dz * 12 + off3[2]) - 0.5) * 0.012
                  + (noise(dx * 34 + off3[1], dy * 34 + off3[2], dz * 34 + off3[0]) - 0.5) * 0.005;
        const r = PBASE - 0.075 + landness * 0.085 + mount + det;
        terrain[j * TER_W + i] = Math.min(PBASE + 0.11, Math.max(PBASE - 0.085, r));
      }
    }
    // rivers: trace downhill from mountain springs, carving a valley as we go,
    // until the channel reaches the sea — real drainage instead of arcs
    const W = TER_W, H = TER_H;
    const RSC = TER_W / 512;   // carving widths/slopes are defined in 512-grid units
    const idxOf = (i2, j2) => j2 * W + i2;
    for (let sIdx = 0; sIdx < 16; sIdx++) {
      let ci = 0, cj = 0, tries = 0;
      do {
        ci = (Math.random() * W) | 0;
        cj = (H * (0.15 + 0.7 * Math.random())) | 0;
        tries++;
      } while (terrain[idxOf(ci, cj)] < seaR + 0.03 && tries < 80);
      if (terrain[idxOf(ci, cj)] < seaR + 0.03) continue;
      let floorH = terrain[idxOf(ci, cj)];
      for (let stp2 = 0; stp2 < 900 * RSC; stp2++) {
        if (terrain[idxOf(ci, cj)] < seaR - 0.008) break;   // reached the sea
        floorH = Math.max(PBASE - 0.085, Math.min(floorH - 0.00025 / RSC, terrain[idxOf(ci, cj)]));
        for (let dj = -2 * RSC; dj <= 2 * RSC; dj++) {
          for (let di = -2 * RSC; di <= 2 * RSC; di++) {
            const rr = Math.hypot(di, dj);
            if (rr > 2.2 * RSC) continue;
            const ii = ((ci + di) % W + W) % W;
            const jj = Math.max(0, Math.min(H - 1, cj + dj));
            const target = floorH + rr * 0.004 / RSC;
            if (terrain[idxOf(ii, jj)] > target) terrain[idxOf(ii, jj)] = target;
          }
        }
        let best = 1e9, bi = ci, bj = cj;
        for (let dj = -1; dj <= 1; dj++) {
          for (let di = -1; di <= 1; di++) {
            if (di === 0 && dj === 0) continue;
            const ii = ((ci + di) % W + W) % W;
            const jj = Math.max(0, Math.min(H - 1, cj + dj));
            const v = terrain[idxOf(ii, jj)] + Math.random() * 0.0004;
            if (v < best) { best = v; bi = ii; bj = jj; }
          }
        }
        ci = bi; cj = bj;
      }
    }
    if (!terrainTex) terrainTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, terrainTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (gl2) {
      // full float precision: 8-bit heights terrace the coastline and rivers
      const data = new Float32Array(TER_W * TER_H * 4);
      for (let p = 0; p < TER_W * TER_H; p++) {
        const v = (terrain[p] - (PBASE - 0.10)) / 0.24;
        data[4 * p] = v; data[4 * p + 1] = v; data[4 * p + 2] = v; data[4 * p + 3] = 1;
      }
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, TER_W, TER_H, 0, gl.RGBA, gl.FLOAT, data);
    } else {
      const data = new Uint8Array(TER_W * TER_H * 4);
      for (let p = 0; p < TER_W * TER_H; p++) {
        const v = Math.round((terrain[p] - (PBASE - 0.10)) / 0.24 * 255);
        data[4 * p] = v; data[4 * p + 1] = v; data[4 * p + 2] = v; data[4 * p + 3] = 255;
      }
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, TER_W, TER_H, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    }
  }
  // true surface normal of the heightfield (radial direction tilted by the
  // terrain gradient) — collisions must use THIS, not the radial direction,
  // or gravity's downhill component is cancelled and nothing ever flows
  function terrainNormalAt (dx, dy, dz, d) {
    const ndx = dx / d, ndy = dy / d, ndz = dz / d;
    let t1x = -ndz, t1z = ndx;
    let l = Math.hypot(t1x, t1z);
    if (l < 1e-4) { t1x = 1; t1z = 0; l = 1; }
    t1x /= l; t1z /= l;
    const t2x = ndy * t1z, t2y = ndz * t1x - ndx * t1z, t2z = -ndy * t1x;
    const e = 0.012;
    const sp = (ox, oy, oz) => {
      const vx = ndx + ox, vy = ndy + oy, vz = ndz + oz;
      const vl = Math.hypot(vx, vy, vz);
      const rr = terrainAt(vx, vy, vz);
      return [vx / vl * rr, vy / vl * rr, vz / vl * rr];
    };
    const pa = sp(t1x * e, 0, t1z * e), pb = sp(-t1x * e, 0, -t1z * e);
    const pc = sp(t2x * e, t2y * e, t2z * e), pd = sp(-t2x * e, -t2y * e, -t2z * e);
    const ax = pa[0] - pb[0], ay = pa[1] - pb[1], az = pa[2] - pb[2];
    const bx = pc[0] - pd[0], by = pc[1] - pd[1], bz = pc[2] - pd[2];
    let nx = by * az - bz * ay, ny = bz * ax - bx * az, nz = bx * ay - by * ax;
    const nl = Math.hypot(nx, ny, nz) || 1e-6;
    nx /= nl; ny /= nl; nz /= nl;
    if (nx * ndx + ny * ndy + nz * ndz < 0) { nx = -nx; ny = -ny; nz = -nz; }
    return [nx, ny, nz];
  }
  function terrainAt (dx, dy, dz) {
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;
    const u = Math.atan2(dz / d, dx / d) / (Math.PI * 2) + 0.5;
    const v = Math.acos(Math.max(-1, Math.min(1, dy / d))) / Math.PI;
    const x = u * TER_W - 0.5, y = v * TER_H - 0.5;
    const xi = Math.floor(x), yi = Math.floor(y);
    const tx = x - xi, ty = y - yi;
    const x0 = ((xi % TER_W) + TER_W) % TER_W, x1 = (x0 + 1) % TER_W;
    const y0 = Math.max(0, Math.min(TER_H - 1, yi)), y1 = Math.max(0, Math.min(TER_H - 1, yi + 1));
    return (terrain[y0 * TER_W + x0] * (1 - tx) + terrain[y0 * TER_W + x1] * tx) * (1 - ty)
         + (terrain[y1 * TER_W + x0] * (1 - tx) + terrain[y1 * TER_W + x1] * tx) * ty;
  }
  // cyl: vertical cylinder r=0.48; bowl: sphere centered (0.5, 0.9, 0.5) r=0.75
  function shapeSolidCell (x, y, z) {
    if (params.shape === 'cyl') {
      const dx = x - 0.5, dz = z - 0.5;
      return dx * dx + dz * dz > 0.2304;
    }
    if (params.shape === 'bowl') {
      const dx = x - 0.5, dy = y - 0.9, dz = z - 0.5;
      return dx * dx + dy * dy + dz * dz > 0.5625;
    }
    if (params.shape === 'planet') {
      const c = BOX / 2;
      const dx = x - c, dy = y - c, dz = z - c;
      return Math.sqrt(dx * dx + dy * dy + dz * dz) < terrainAt(dx, dy, dz);
    }
    return false;
  }
  function shapeBlocked (x, y, z, rad) {
    if (params.shape === 'cyl') {
      const dx = x - 0.5, dz = z - 0.5;
      const m = 0.48 - rad;
      return dx * dx + dz * dz > m * m;
    }
    if (params.shape === 'bowl') {
      const dx = x - 0.5, dy = y - 0.9, dz = z - 0.5;
      const m = 0.75 - rad;
      return dx * dx + dy * dy + dz * dz > m * m;
    }
    if (params.shape === 'planet') {
      // seed the water as a shell floating above the surface, so on reset it
      // rains down over the whole globe and gathers into the basins
      // rain genesis: the water starts as a shell in the sky and rains down;
      // with surface-normal sliding it drains into the basins on its own
      const c = BOX / 2;
      const dx = x - c, dy = y - c, dz = z - c;
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      return r < PBASE + 0.10 || r > PBASE + 0.104 + params.waterAmount * 0.038;
    }
    return false;
  }
  function shapeClampBall (b) {
    if (params.shape === 'cyl') {
      const dx = b.x - 0.5, dz = b.z - 0.5;
      const maxR = 0.48 - b.r;
      const d2 = dx * dx + dz * dz;
      if (d2 > maxR * maxR) {
        const d = Math.sqrt(d2) || 1e-6;
        b.x = 0.5 + dx / d * maxR;
        b.z = 0.5 + dz / d * maxR;
        const vn = (b.vx * dx + b.vz * dz) / d;
        if (vn > 0) { b.vx -= 1.4 * vn * dx / d; b.vz -= 1.4 * vn * dz / d; }
      }
    } else if (params.shape === 'bowl') {
      const dx = b.x - 0.5, dy = b.y - 0.9, dz = b.z - 0.5;
      const maxR = 0.75 - b.r;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > maxR * maxR) {
        const d = Math.sqrt(d2) || 1e-6;
        b.x = 0.5 + dx / d * maxR;
        b.y = 0.9 + dy / d * maxR;
        b.z = 0.5 + dz / d * maxR;
        const vn = (b.vx * dx + b.vy * dy + b.vz * dz) / d;
        if (vn > 0) {
          b.vx -= 1.4 * vn * dx / d;
          b.vy -= 1.4 * vn * dy / d;
          b.vz -= 1.4 * vn * dz / d;
        }
      }
    } else if (params.shape === 'planet') {
      const c = BOX / 2;
      const dx = b.x - c, dy = b.y - c, dz = b.z - c;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;
      const tr = terrainAt(dx, dy, dz) + b.r;
      if (d < tr) {
        b.x = c + dx / d * tr;
        b.y = c + dy / d * tr;
        b.z = c + dz / d * tr;
        const n = terrainNormalAt(dx, dy, dz, d);
        const vn = b.vx * n[0] + b.vy * n[1] + b.vz * n[2];
        if (vn < 0) {
          b.vx -= 1.3 * vn * n[0];
          b.vy -= 1.3 * vn * n[1];
          b.vz -= 1.3 * vn * n[2];
        }
      }
    }
  }
  function shapeParticleCollide () {
    if (params.shape === 'box') return;
    const pos = fluid.particlePos, vel = fluid.particleVel, r = fluid.particleRadius;
    if (params.shape === 'planet') {
      const c = BOX / 2, outer = PBASE + 0.2;
      for (let i = 0; i < fluid.numParticles; i++) {
        const dx = pos[3 * i] - c, dy = pos[3 * i + 1] - c, dz = pos[3 * i + 2] - c;
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;
        const tr = terrainAt(dx, dy, dz) + r;
        if (d < tr) {
          pos[3 * i] = c + dx / d * tr;
          pos[3 * i + 1] = c + dy / d * tr;
          pos[3 * i + 2] = c + dz / d * tr;
          // slide along the slope: only the surface-normal velocity is removed
          const n = terrainNormalAt(dx, dy, dz, d);
          const vn = vel[3 * i] * n[0] + vel[3 * i + 1] * n[1] + vel[3 * i + 2] * n[2];
          if (vn < 0) {
            vel[3 * i] -= vn * n[0];
            vel[3 * i + 1] -= vn * n[1];
            vel[3 * i + 2] -= vn * n[2];
          }
        } else if (d > outer) {
          pos[3 * i] = c + dx / d * outer;
          pos[3 * i + 1] = c + dy / d * outer;
          pos[3 * i + 2] = c + dz / d * outer;
        }
      }
      return;
    }
    for (let i = 0; i < fluid.numParticles; i++) {
      if (params.shape === 'cyl') {
        const dx = pos[3 * i] - 0.5, dz = pos[3 * i + 2] - 0.5;
        const maxR = 0.48 - r;
        const d2 = dx * dx + dz * dz;
        if (d2 > maxR * maxR) {
          const d = Math.sqrt(d2) || 1e-6;
          pos[3 * i] = 0.5 + dx / d * maxR;
          pos[3 * i + 2] = 0.5 + dz / d * maxR;
          const vn = (vel[3 * i] * dx + vel[3 * i + 2] * dz) / d;
          if (vn > 0) { vel[3 * i] -= vn * dx / d; vel[3 * i + 2] -= vn * dz / d; }
        }
      } else {
        const dx = pos[3 * i] - 0.5, dy = pos[3 * i + 1] - 0.9, dz = pos[3 * i + 2] - 0.5;
        const maxR = 0.75 - r;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > maxR * maxR) {
          const d = Math.sqrt(d2) || 1e-6;
          pos[3 * i] = 0.5 + dx / d * maxR;
          pos[3 * i + 1] = 0.9 + dy / d * maxR;
          pos[3 * i + 2] = 0.5 + dz / d * maxR;
          const vn = (vel[3 * i] * dx + vel[3 * i + 1] * dy + vel[3 * i + 2] * dz) / d;
          if (vn > 0) {
            vel[3 * i] -= vn * dx / d;
            vel[3 * i + 1] -= vn * dy / d;
            vel[3 * i + 2] -= vn * dz / d;
          }
        }
      }
    }
  }
  const obstacle = { active: false, x: 0.5, y: 0.5, z: 0.5, vx: 0, vy: 0, vz: 0, radius: 0.13, dye: null };
  let BOX = 1.0;         // domain side length: 2.0 in globe mode, 1.0 otherwise
  let PBASE = 0.78;      // planet base radius (globe mode)
  let fluid = null;

  const BALL_COLORS = [
    [1.00, 0.45, 0.42], [1.00, 0.76, 0.32], [0.42, 0.90, 0.66],
    [0.72, 0.62, 1.00], [0.95, 0.95, 0.98],
  ];
  const MAX_BALLS = 5;
  const balls = [];
  let nextBallColor = 0;
  function spawnBall () {
    const ball = {
      x: BOX / 2 - 0.15 + Math.random() * 0.3,
      y: BOX - 0.18,
      z: BOX / 2 - 0.15 + Math.random() * 0.3,
      vx: 0, vy: 0, vz: 0,
      r: 0.09,
      color: BALL_COLORS[nextBallColor++ % BALL_COLORS.length],
    };
    if (balls.length >= MAX_BALLS) balls.shift();
    balls.push(ball);
  }
  // a fast inbound ball from space, aimed at the planet
  function spawnMeteor () {
    const a = Math.random() * Math.PI * 2, cb = 2 * Math.random() - 1;
    const sb = Math.sqrt(1 - cb * cb);
    const dxn = sb * Math.cos(a), dyn = cb, dzn = sb * Math.sin(a);
    const R0 = BOX / 2 - 0.06, c0 = BOX / 2;
    const ball = {
      x: c0 + dxn * R0, y: c0 + dyn * R0, z: c0 + dzn * R0,
      vx: -dxn * 5, vy: -dyn * 5, vz: -dzn * 5,
      r: 0.09, color: [0.90, 0.52, 0.30],
    };
    if (balls.length >= MAX_BALLS) balls.shift();
    balls.push(ball);
  }
  function updateBalls (dt, gx, gy, gz) {
    const RHO = 2.4;
    for (const b of balls) {
      // on the globe, each ball's gravity points at the planet center
      let bgx = gx, bgy = gy, bgz = gz;
      if (params.shape === 'planet') {
        const ddx = BOX / 2 - b.x, ddy = BOX / 2 - b.y, ddz = BOX / 2 - b.z;
        const dd = Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz) || 1e-6;
        const gm = Math.sqrt(gx * gx + gy * gy + gz * gz);
        bgx = ddx / dd * gm; bgy = ddy / dd * gm; bgz = ddz / dd * gm;
        const w = params.spin;
        bgx += w * w * (b.x - BOX / 2) - 2 * w * b.vz;
        bgz += w * w * (b.z - BOX / 2) + 2 * w * b.vx;
      }
      // sample around the ball, not inside it: its own cells are solid-stamped
      const off = b.r * 1.5;
      const f = (
        fluid.submergedFraction(b.x + off, b.y, b.z) +
        fluid.submergedFraction(b.x - off, b.y, b.z) +
        fluid.submergedFraction(b.x, b.y, b.z + off) +
        fluid.submergedFraction(b.x, b.y, b.z - off) +
        fluid.submergedFraction(b.x, b.y - off, b.z)
      ) / 5;
      const lift = 1 - f * RHO;
      let fu = 0, fv = 0, fw = 0;
      for (let q = 0; q < 6; q++) {
        const ox = q === 0 ? off : q === 1 ? -off : 0;
        const oy = q === 2 ? off : q === 3 ? -off : 0;
        const oz = q === 4 ? off : q === 5 ? -off : 0;
        const s = fluid.sampleVelocity(b.x + ox, b.y + oy, b.z + oz);
        fu += s[0]; fv += s[1]; fw += s[2];
      }
      fu /= 6; fv /= 6; fw /= 6;
      const drag = 6 * f;
      b.vx += (bgx * lift + (fu - b.vx) * drag) * dt;
      b.vy += (bgy * lift + (fv - b.vy) * drag) * dt;
      b.vz += (bgz * lift + (fw - b.vz) * drag) * dt;
      b.x += b.vx * dt; b.y += b.vy * dt; b.z += b.vz * dt;
      const lo = fluid.h + b.r, hi = BOX - fluid.h - b.r;
      if (b.x < lo) { b.x = lo; b.vx = -b.vx * 0.4; }
      if (b.x > hi) { b.x = hi; b.vx = -b.vx * 0.4; }
      if (b.y < lo) { b.y = lo; b.vy = -b.vy * 0.4; }
      if (b.y > hi) { b.y = hi; b.vy = -b.vy * 0.4; }
      if (b.z < lo) { b.z = lo; b.vz = -b.vz * 0.4; }
      if (b.z > hi) { b.z = hi; b.vz = -b.vz * 0.4; }
      shapeClampBall(b);
    }
    for (let i = 0; i < balls.length; i++) {
      for (let j = i + 1; j < balls.length; j++) {
        const a = balls[i], c = balls[j];
        const md = a.r + c.r;
        const dx = c.x - a.x, dy = c.y - a.y, dz = c.z - a.z;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < md * md && d2 > 0) {
          const d = Math.sqrt(d2);
          const push = 0.5 * (md - d) / d;
          a.x -= dx * push; a.y -= dy * push; a.z -= dz * push;
          c.x += dx * push; c.y += dy * push; c.z += dz * push;
        }
      }
    }
  }
  function shake () {
    const vel = fluid.particleVel;
    for (let i = 0; i < fluid.numParticles; i++) {
      vel[3 * i] += (Math.random() - 0.5) * 4;
      vel[3 * i + 1] += Math.random() * 3;
      vel[3 * i + 2] += (Math.random() - 0.5) * 4;
    }
    for (const b of balls) {
      b.vx += (Math.random() - 0.5) * 2;
      b.vy += Math.random() * 1.5;
      b.vz += (Math.random() - 0.5) * 2;
    }
  }

  function setupScene () {
    const planet = params.shape === 'planet';
    BOX = planet ? 3.0 : 1.0;   // globe mode triples the domain, keeping cell size
    CENTER[0] = CENTER[1] = CENTER[2] = BOX / 2;
    camTarget[0] = CENTER[0]; camTarget[1] = CENTER[1]; camTarget[2] = CENTER[2];
    const res = Math.round(params.res * (planet ? 0.8 : 1));
    const spacing = 1.0 / res;
    const r = 0.37 * spacing;
    const dx = 2.0 * r;
    const relW = 0.42;
    if (planet) {
      PBASE = 1.25;
      seaR = PBASE - 0.03 + params.waterAmount * 0.04;   // sea level
      buildPlanet();
      initStorms();
      initSWE();          // the globe ocean is shallow-water, not particles
      fluid = null;
      balls.length = 0;
      boxEdgeCount = 0;
      return;
    }
    const x0 = spacing + r, y0 = spacing + r, z0 = spacing + r;
    const full = Math.floor((BOX - 2 * spacing - 2 * r) / dx);
    const numX = planet ? full : Math.floor((relW * BOX - 2 * spacing) / dx);
    const numY = planet ? full : Math.floor((Math.min(0.9, params.waterAmount * 2.0) * BOX - 2 * spacing) / dx);
    const numZ = full;
    const capacity = planet ? 140000 : Math.floor(numX * numY * numZ * 1.8) + 8000;
    fluid = new FlipFluid3(BOX, BOX, BOX, spacing, r, capacity,
      params.shape === 'box' ? null : shapeSolidCell);
    fluid.postCollide = shapeParticleCollide;
    balls.length = 0;
    params.tiltX = 0; params.tiltZ = 0;
    let count = 0;
    for (let i = 0; i < numX; i++) {
      for (let j = 0; j < numY; j++) {
        for (let k = 0; k < numZ; k++) {
          const px = x0 + dx * i + (Math.random() - 0.5) * 0.2 * r;
          const py = y0 + dx * j + (Math.random() - 0.5) * 0.2 * r;
          const pz = z0 + dx * k + (Math.random() - 0.5) * 0.2 * r;
          if (shapeBlocked(px, py, pz, r)) continue;   // seed only inside the container
          if (count >= fluid.maxParticles) continue;
          fluid.particlePos[3 * count] = px;
          fluid.particlePos[3 * count + 1] = py;
          fluid.particlePos[3 * count + 2] = pz;
          count++;
        }
      }
    }
    fluid.numParticles = count;
    for (let i = 0; i < count; i++) {
      fluid.particleColor[3 * i] = BASE[0];
      fluid.particleColor[3 * i + 1] = BASE[1];
      fluid.particleColor[3 * i + 2] = BASE[2];
    }
    boxEdgeCount = buildBoxEdges();
  }

  const cam = { theta: 0.7, phi: 0.32, dist: planetMode ? 5.6 : 2.4 };
  let camSunOffset = cam.theta - lightAngle;   // framing offset for the sun-fixed view
  // the planet's accumulated rotation, integrated in SIM time — the star-fixed
  // camera must follow this clock, not the wall clock, or frame drops desync
  // the visible spin from the sun, the terminator and the tides
  let spinAngle = 0;
  let waveTime = 0;   // sim-time clock for the animated water micro-waves
  let camSpaceOffset = cam.theta;
  const CENTER = [0.5, 0.5, 0.5];
  const camTarget = [0.5, 0.5, 0.5];           // free look-at point (pan with Shift+drag)
  const CAM_PRESETS = [
    { name: '標準', view: 'space', dist: 5.6, phi: 0.32, offset: 0 },
    { name: '昼と夜', view: 'sun', dist: 6.5, phi: 0.15, offset: Math.PI / 2 },
    { name: '地球と月', view: 'moon', dist: 7.5, phi: 0.12 },
    { name: '太陽と地球と月', view: 'sun', dist: 18, phi: -0.12, offset: 0.42 },
  ];
  let camPreset = 0;
  function applyCamPreset () {
    const p = CAM_PRESETS[camPreset];
    params.view = p.view;
    cam.dist = p.dist;
    cam.phi = p.phi;
    if (p.offset !== undefined) camSunOffset = p.offset;
    camTarget[0] = CENTER[0]; camTarget[1] = CENTER[1]; camTarget[2] = CENTER[2];
  }
  const FOV_TAN = Math.tan((40 * Math.PI / 180) / 2);
  function cameraBasis () {
    const cp = Math.cos(cam.phi), sp = Math.sin(cam.phi);
    const ct = Math.cos(cam.theta), st = Math.sin(cam.theta);
    const eye = [
      camTarget[0] + cam.dist * cp * st,
      camTarget[1] + cam.dist * sp,
      camTarget[2] + cam.dist * cp * ct,
    ];
    const fwd = normalize([camTarget[0] - eye[0], camTarget[1] - eye[1], camTarget[2] - eye[2]]);
    const right = normalize(cross(fwd, [0, 1, 0]));
    const up = cross(right, fwd);
    return { eye, fwd, right, up };
  }

  const PROJ = `
uniform vec3 uEye;
uniform vec3 uRight;
uniform vec3 uUp;
uniform vec3 uFwd;
uniform float uTanA;
uniform float uAspect;
vec4 project (vec3 p) {
  vec3 rel = p - uEye;
  vec3 vp = vec3(dot(rel, uRight), dot(rel, uUp), dot(rel, uFwd));
  float n = 0.05, f = 80.0;
  return vec4(vp.x / (uTanA * uAspect), vp.y / uTanA, ((f + n) * vp.z - 2.0 * f * n) / (f - n), vp.z);
}`;
  const pointProg = glProgram(gl, `
attribute vec3 aPos;
attribute vec3 aColor;
uniform float uPointScale;
varying vec3 vColor;
${PROJ}
void main () {
  vColor = aColor;
  vec4 cp = project(aPos);
  gl_Position = cp;
  gl_PointSize = uPointScale / max(cp.w, 0.05);
}`, `
precision mediump float;
varying vec3 vColor;
void main () {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  d.y = -d.y;
  float r2 = dot(d, d);
  if (r2 > 1.0) discard;
  vec3 nrm = vec3(d.x, d.y, sqrt(1.0 - r2));
  float diff = 0.45 + 0.55 * max(dot(nrm, normalize(vec3(0.4, 0.7, 0.6))), 0.0);
  gl_FragColor = vec4(vColor * diff, 1.0);
}`);
  const lineProg = glProgram(gl, `
attribute vec3 aPos;
${PROJ}
void main () { gl_Position = project(aPos); }`, `
precision mediump float;
uniform vec4 uColor;
void main () { gl_FragColor = uColor; }`);
  const discProg = glProgram(gl, `
attribute vec3 aPos;
uniform float uPointScale;
${PROJ}
void main () {
  vec4 cp = project(aPos);
  gl_Position = cp;
  gl_PointSize = uPointScale / max(cp.w, 0.05);
}`, `
precision mediump float;
uniform vec3 uColor;
void main () {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  d.y = -d.y;
  float r2 = dot(d, d);
  if (r2 > 1.0) discard;
  vec3 nrm = vec3(d.x, d.y, sqrt(1.0 - r2));
  float diff = 0.4 + 0.6 * max(dot(nrm, normalize(vec3(0.4, 0.7, 0.6))), 0.0);
  gl_FragColor = vec4(uColor * diff, 1.0);
}`);

  // ---- screen-space fluid rendering (liquid surface mode) ----
  const QUAD_VS = `
attribute vec2 aPos;
varying vec2 vUv;
void main () { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;
  const waterDepthProg = surfaceOK ? glProgram(gl, `
attribute vec3 aPos;
uniform float uPointScale;
varying float vDepth;
${PROJ}
void main () {
  vec4 cp = project(aPos);
  vDepth = cp.w;
  gl_Position = cp;
  gl_PointSize = uPointScale / max(cp.w, 0.05);
}`, `
precision mediump float;
varying float vDepth;
uniform float uRadiusW;
void main () {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(d, d);
  if (r2 > 1.0) discard;
  // spherical depth: the impostor bulges toward the camera for smoother normals
  gl_FragColor = vec4(vDepth - sqrt(1.0 - r2) * uRadiusW, 0.0, 0.0, 1.0);
}`) : null;
  const thickProg = surfaceOK ? glProgram(gl, `
attribute vec3 aPos;
attribute vec3 aColor;
uniform float uPointScale;
varying vec3 vColor;
${PROJ}
void main () {
  vColor = aColor;
  vec4 cp = project(aPos);
  gl_Position = cp;
  gl_PointSize = uPointScale / max(cp.w, 0.05);
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
}`) : null;
  const blurProg = surfaceOK ? glProgram(gl, QUAD_VS, `
precision highp float;
varying vec2 vUv;
uniform sampler2D uTex;
uniform vec2 uDir;
void main () {
  float c = texture2D(uTex, vUv).r;
  if (c <= 0.001) { gl_FragColor = vec4(0.0); return; }
  float sum = c;
  float wsum = 1.0;
  for (int i = 1; i <= 8; i++) {
    float fi = float(i);
    float g = exp(-fi * fi / 32.0);
    float s1 = texture2D(uTex, vUv + uDir * fi).r;
    float s2 = texture2D(uTex, vUv - uDir * fi).r;
    if (s1 > 0.001 && abs(s1 - c) < 0.35) { sum += s1 * g; wsum += g; }
    if (s2 > 0.001 && abs(s2 - c) < 0.35) { sum += s2 * g; wsum += g; }
  }
  gl_FragColor = vec4(sum / wsum, 0.0, 0.0, 1.0);
}`) : null;
  const compProg = surfaceOK ? glProgram(gl, QUAD_VS, `
precision highp float;
varying vec2 vUv;
uniform sampler2D uDepthTex;
uniform sampler2D uThickTex;
uniform sampler2D uScene;
uniform sampler2D uSceneDepth;
uniform vec2 uTexel;
uniform float uTanA;
uniform float uAspect;
uniform vec3 uLightV;
uniform vec3 uLightW;
uniform vec3 uRightW;
uniform vec3 uUpW;
uniform vec3 uFwdW;
uniform float uPlanetOn;
uniform vec3 uCW;
uniform vec3 uEyeW;
vec3 vpos (vec2 uv, float d) {
  vec2 ndc = uv * 2.0 - 1.0;
  ndc.x *= uAspect;
  return vec3(ndc * uTanA * d, d);
}
void main () {
  float d = texture2D(uDepthTex, vUv).r;
  if (d <= 0.001) discard;
  // scene depth test: geometry in front of the water surface (e.g. the exposed
  // top of a floating ball) must occlude the water
  float zn = texture2D(uSceneDepth, vUv).r * 2.0 - 1.0;
  float sceneVz = 8.0 / (80.05 - zn * 79.95);
  if (d >= sceneVz - 0.01) discard;
  float dl = texture2D(uDepthTex, vUv - vec2(uTexel.x, 0.0)).r;
  float dr = texture2D(uDepthTex, vUv + vec2(uTexel.x, 0.0)).r;
  float db = texture2D(uDepthTex, vUv - vec2(0.0, uTexel.y)).r;
  float du = texture2D(uDepthTex, vUv + vec2(0.0, uTexel.y)).r;
  if (dl <= 0.001) dl = d;
  if (dr <= 0.001) dr = d;
  if (db <= 0.001) db = d;
  if (du <= 0.001) du = d;
  // silhouette guard: keep neighbour depths near the centre so the edge of the
  // water body does not produce wild normals and noisy rims
  float rng = 0.06 + 0.04 * d;
  dl = clamp(dl, d - rng, d + rng);
  dr = clamp(dr, d - rng, d + rng);
  db = clamp(db, d - rng, d + rng);
  du = clamp(du, d - rng, d + rng);
  vec3 pc = vpos(vUv, d);
  vec3 px = vpos(vUv + vec2(uTexel.x, 0.0), dr) - vpos(vUv - vec2(uTexel.x, 0.0), dl);
  vec3 py = vpos(vUv + vec2(0.0, uTexel.y), du) - vpos(vUv - vec2(0.0, uTexel.y), db);
  vec3 n = normalize(cross(px, py));
  if (n.z > 0.0) n = -n;
  vec4 th = texture2D(uThickTex, vUv);
  float thickness = th.a;
  vec3 tint = th.rgb / max(th.a, 1e-3);
  vec3 V = -normalize(pc);
  vec3 pWorld = uEyeW + pc.x * uRightW + pc.y * uUpW + pc.z * uFwdW;
  float diff = 0.35 + 0.65 * max(dot(n, uLightV), 0.0);
  // two-lobe specular with per-cell sun glitter, so the surface sparkles the
  // way real water does instead of showing one plastic highlight
  vec3 Hv = normalize(uLightV + V);
  float ndh = max(dot(n, Hv), 0.0);
  float spec = pow(ndh, 140.0) * 1.1 + pow(ndh, 24.0) * 0.10;
  float glit = fract(sin(dot(floor(pWorld.xz * 220.0) + floor(pWorld.y * 220.0),
                             vec2(12.9898, 78.233))) * 43758.5453);
  spec *= 0.65 + 0.7 * glit;
  // globe: no light reaches the night side of the planet
  float day = 1.0;
  if (uPlanetOn > 0.5) {
    vec3 ndW = normalize(pWorld - uCW);
    day = 0.18 + 0.82 * smoothstep(-0.15, 0.35, dot(ndW, normalize(uLightW)));
  }

  // world-space environment reflection, weighted by real fresnel
  vec3 nW = normalize(n.x * uRightW + n.y * uUpW + n.z * uFwdW);
  vec3 dirW = normalize(pc.x * uRightW + pc.y * uUpW + pc.z * uFwdW);
  vec3 rW = reflect(dirW, nW);
  vec3 env = mix(vec3(0.045, 0.060, 0.085), vec3(0.42, 0.52, 0.66),
                 smoothstep(-0.15, 0.65, rW.y));
  env += vec3(0.10, 0.12, 0.15) * exp(-abs(rW.y) * 5.0);   // bright horizon band
  env += vec3(0.9) * pow(max(dot(rW, normalize(uLightW)), 0.0), 48.0);
  env *= 0.25 + 0.75 * day;
  float F = 0.02 + 0.98 * pow(1.0 - max(dot(n, V), 0.0), 5.0);

  // transmission: the scene behind, refracted by the surface normal and
  // absorbed by thickness (red dies first), tinted by the dye
  vec2 ruv = vUv + n.xy * (0.06 * clamp(thickness, 0.25, 1.4));
  vec3 sceneCol = texture2D(uScene, ruv).rgb;
  vec3 absorbK = vec3(1.6, 0.65, 0.30);
  vec3 dyeFilter = mix(vec3(1.0), clamp(tint * 2.2, 0.0, 1.0), 0.5);
  vec3 transmitted = sceneCol * exp(-absorbK * min(thickness, 2.5) * 0.16) * dyeFilter * 1.25;
  transmitted += tint * 0.09 * (0.4 + 0.6 * diff) * clamp(thickness, 0.0, 1.0);
  if (uPlanetOn > 0.5) {
    // an ocean seen from space is opaque: the seabed shows only through
    // very shallow water, and the body carries the day/night lighting
    float clearT = exp(-thickness * 2.6);
    vec3 oceanBody = vec3(0.016, 0.075, 0.16) * (0.35 + 0.65 * diff) * day;
    transmitted = mix(oceanBody, transmitted, clamp(clearT, 0.0, 1.0));
  }

  vec3 col = mix(transmitted, env, clamp(F * 1.5, 0.0, 0.85)) + spec * 0.7 * day;
  // foam: white splash particles brighten instead of tinting
  float lum = (tint.r + tint.g + tint.b) * 0.3333;
  col += smoothstep(0.6, 0.85, lum) * 0.30 * clamp(thickness, 0.0, 1.0);
  // opaque where there is water (transmission already carries the scene);
  // fade out only the very thin splash edges
  float alpha = smoothstep(0.01, 0.10, thickness);
  gl_FragColor = vec4(col, alpha);
}`) : null;

  // raycast environment: gradient sky and a tiled floor under the tank
  const BG_VS = gl2 ? `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main () { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }` : QUAD_VS;
  const bg3Prog = glProgram(gl, BG_VS, (gl2 ? `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragColor;
#define texture2D texture
#define OUTCOL fragColor
#define WRITE_DEPTH(z) gl_FragDepth = (z)
` : `
precision highp float;
varying vec2 vUv;
#define OUTCOL gl_FragColor
#define WRITE_DEPTH(z)
`) + `
uniform vec3 uEye;
uniform vec3 uRight;
uniform vec3 uUp;
uniform vec3 uFwd;
uniform float uTanA;
uniform float uAspect;
uniform float uPlanet;
uniform float uSeaR;
uniform vec3 uC;
uniform float uPBase;
uniform float uPRad2;
uniform vec3 uLightDir;
uniform vec3 uMoonDir;
uniform float uMoonDist;
uniform float uMoonR;
uniform sampler2D uTerrain;
uniform sampler2D uWater;
uniform float uWTime;
uniform vec2 uTerDim;
uniform vec2 uWatDim;
vec2 sphUV (vec3 d) {
  return vec2(atan(d.z, d.x) / 6.28318 + 0.5, acos(clamp(d.y, -1.0, 1.0)) / 3.14159);
}
float terrainR (vec3 d) {
  return (uPBase - 0.10) + texture2D(uTerrain, sphUV(d)).r * 0.24;
}
float waterR (vec3 d) {
  return (uPBase - 0.10) + texture2D(uWater, sphUV(d)).r * 0.24;
}
float waterH (vec3 d) { return texture2D(uWater, sphUV(d)).g * 0.12; }
float surfR (vec3 d) { return max(terrainR(d), waterR(d)); }
float hash3 (vec3 q) { return fract(sin(dot(q, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
float vnoise (vec3 x) {
  vec3 ip = floor(x), fp = fract(x);
  fp = fp * fp * (3.0 - 2.0 * fp);
  float n000 = hash3(ip),                       n100 = hash3(ip + vec3(1.0, 0.0, 0.0));
  float n010 = hash3(ip + vec3(0.0, 1.0, 0.0)), n110 = hash3(ip + vec3(1.0, 1.0, 0.0));
  float n001 = hash3(ip + vec3(0.0, 0.0, 1.0)), n101 = hash3(ip + vec3(1.0, 0.0, 1.0));
  float n011 = hash3(ip + vec3(0.0, 1.0, 1.0)), n111 = hash3(ip + vec3(1.0, 1.0, 1.0));
  return mix(mix(mix(n000, n100, fp.x), mix(n010, n110, fp.x), fp.y),
             mix(mix(n001, n101, fp.x), mix(n011, n111, fp.x), fp.y), fp.z);
}
// smooth (B-spline bicubic, 4 bilinear taps) height sampling used for the
// shading normals: plain bilinear leaves cell-diamond facets that read as a
// visible grid even when the silhouette is smooth
float bicR (sampler2D tex, vec2 uv, vec2 res) {
  vec2 st = uv * res - 0.5;
  vec2 iuv = floor(st), f = fract(st);
  vec2 f2 = f * f, f3 = f2 * f;
  vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
  vec2 w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
  vec2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
  vec2 w3 = f3 / 6.0;
  vec2 g0 = w0 + w1, g1 = w2 + w3;
  vec2 h0 = (iuv - 0.5 + w1 / g0) / res;
  vec2 h1 = (iuv + 1.5 + w3 / g1) / res;
  return (texture2D(tex, vec2(h0.x, h0.y)).r * g0.x + texture2D(tex, vec2(h1.x, h0.y)).r * g1.x) * g0.y
       + (texture2D(tex, vec2(h0.x, h1.y)).r * g0.x + texture2D(tex, vec2(h1.x, h1.y)).r * g1.x) * g1.y;
}
float terrainRS (vec3 d) { return (uPBase - 0.10) + bicR(uTerrain, sphUV(d), uTerDim) * 0.24; }
float waterRS (vec3 d) { return (uPBase - 0.10) + bicR(uWater, sphUV(d), uWatDim) * 0.24; }
// one travelling ocean wave with peaked crests and wide flat troughs
// (exp of sine, the cheap stand-in for a trochoid); returns the crest value
// and accumulates the surface slope into grad
float waveG (vec2 q, vec2 wd, float kf, float sp, float wt, out vec2 grad) {
  float ph = dot(q, wd) * kf + wt * sp;
  float e = exp(1.8 * (sin(ph) - 1.0));
  grad = wd * (1.8 * cos(ph) * e);
  return e;
}
void main () {
  vec2 ndc = vUv * 2.0 - 1.0;
  vec3 dir = normalize(uFwd + uTanA * (ndc.x * uAspect * uRight + ndc.y * uUp));
  vec3 sky = mix(vec3(0.024, 0.032, 0.050), vec3(0.085, 0.105, 0.145), smoothstep(-0.2, 0.7, dir.y));
  vec3 col = sky;
  bool hit = false;
  float fragZ = 1.0;
  float tP = 1e9;
  if (uPlanet > 0.5) {
    // the whole globe in space: raymarch the spherical heightfield
    col = vec3(0.008, 0.012, 0.022);   // deep space
    vec3 ro = uEye - uC;
    float b = dot(ro, dir);
    float cc = dot(ro, ro) - uPRad2;   // bounding sphere just above the peaks
    float disc = b * b - cc;
    if (disc > 0.0) {
      float t = max(-b - sqrt(disc), 0.0);
      float tExit = -b + sqrt(disc);
      float stp = (tExit - t) / 64.0;
      t += stp * 0.5;
      vec3 p = ro;
      for (int i = 0; i < 64; i++) {
        p = ro + dir * t;
        if (length(p) < surfR(normalize(p))) { hit = true; break; }
        t += stp;
      }
      if (hit) {
        float lo = t - stp, hi2 = t;
        for (int i = 0; i < 5; i++) {
          float mid = (lo + hi2) * 0.5;
          vec3 pm = ro + dir * mid;
          if (length(pm) < surfR(normalize(pm))) hi2 = mid; else lo = mid;
        }
        t = (lo + hi2) * 0.5;
        p = ro + dir * t;
        vec3 nd = normalize(p);
        bool isWater = waterR(nd) > terrainR(nd) + 0.0006;
        vec3 t1 = normalize(cross(nd, vec3(0.0, 1.0, 0.001)));
        vec3 t2 = cross(nd, t1);
        float e = isWater ? 0.007 : 0.005;
        vec3 pa, pb, pc2, pd;
        if (isWater) {
          pa = normalize(nd + t1 * e) * waterRS(normalize(nd + t1 * e));
          pb = normalize(nd - t1 * e) * waterRS(normalize(nd - t1 * e));
          pc2 = normalize(nd + t2 * e) * waterRS(normalize(nd + t2 * e));
          pd = normalize(nd - t2 * e) * waterRS(normalize(nd - t2 * e));
        } else {
          pa = normalize(nd + t1 * e) * terrainRS(normalize(nd + t1 * e));
          pb = normalize(nd - t1 * e) * terrainRS(normalize(nd - t1 * e));
          pc2 = normalize(nd + t2 * e) * terrainRS(normalize(nd + t2 * e));
          pd = normalize(nd - t2 * e) * terrainRS(normalize(nd - t2 * e));
        }
        vec3 n = normalize(cross(pc2 - pd, pa - pb));
        if (dot(n, nd) < 0.0) n = -n;
        vec3 L = normalize(uLightDir);
        float day = 0.18 + 0.82 * smoothstep(-0.15, 0.35, dot(nd, L));
        float rim = pow(1.0 - abs(dot(dir, nd)), 2.5);
        if (isWater) {
          // shallow-water ocean: the sim gives the broad shape (tides, basin
          // waves); a directional wave field with peaked trochoid-like crests,
          // wind patches and whitecaps supplies the sea state below grid scale
          float hW = waterH(nd);
          float flow = texture2D(uWater, sphUV(nd)).b;   // |current| from the sim
          vec3 V2 = -dir;
          vec2 q = vec2(dot(p, t1), dot(p, t2));
          float wt = uWTime;
          // the sea is not uniformly rough: slow-drifting gust patches
          // (named "gust" — "patch" is a reserved word in GLSL ES)
          float gust = 0.35 + 0.65 * vnoise(p * 7.0 + vec3(wt * 0.10, 0.0, -wt * 0.07));
          float depthF = clamp(hW * 40.0 + 0.10, 0.15, 1.0);
          float rough = gust * depthF;
          vec2 g = vec2(0.0), gw;
          float cr = 0.0;
          cr += 0.35 * waveG(q, vec2(0.98, 0.20), 130.0, 1.9, wt, gw); g += 0.35 * gw;
          cr += 0.30 * waveG(q, vec2(0.60, -0.80), 205.0, 2.4, wt, gw); g += 0.30 * gw;
          cr += 0.26 * waveG(q, vec2(-0.30, 0.95), 340.0, 3.1, wt, gw); g += 0.26 * gw;
          cr += 0.22 * waveG(q, vec2(0.86, -0.51), 560.0, 3.9, wt, gw); g += 0.22 * gw;
          cr += 0.18 * waveG(q, vec2(-0.71, -0.71), 920.0, 5.0, wt, gw); g += 0.18 * gw;
          // sub-grid chop fading in as the camera closes on the surface
          float closeF = smoothstep(2.5, 0.6, t);
          if (closeF > 0.01) {
            cr += closeF * 0.16 * waveG(q, vec2(0.26, 0.97), 1600.0, 6.6, wt, gw);
            g += closeF * 0.16 * gw;
            g += closeF * 0.5 * vec2(vnoise(p * 900.0 + wt * 0.6) - 0.5,
                                     vnoise(p * 900.0 + 13.1 - wt * 0.5) - 0.5);
          }
          vec3 nW = normalize(n + (t1 * g.x + t2 * g.y) * 0.30 * rough);
          // body colour: turquoise shallows into deep blue, seabed sand only
          // showing through the very first metres
          vec3 shallowC = vec3(0.06, 0.34, 0.36);
          vec3 deepC = vec3(0.010, 0.055, 0.13);
          vec3 body = mix(shallowC, deepC, clamp(hW * 22.0, 0.0, 1.0));
          vec3 bedCol = mix(vec3(0.40, 0.35, 0.24), body, clamp(hW * 55.0, 0.15, 1.0));
          // crests catch a touch more light so the waves shade themselves
          float diffW = (0.35 + 0.65 * max(dot(n, L), 0.0)) * (0.86 + 0.20 * cr);
          float F = 0.02 + 0.98 * pow(1.0 - max(dot(nW, V2), 0.0), 5.0);
          vec3 rWv = reflect(dir, nW);
          float sunR = max(dot(rWv, L), 0.0);
          vec3 env = mix(vec3(0.05, 0.075, 0.11), vec3(0.30, 0.38, 0.50),
                         smoothstep(-0.2, 0.9, sunR));
          env += vec3(1.0, 0.85, 0.55) * pow(sunR, 60.0) * 1.6;   // mirrored sun glow
          vec3 Hn = normalize(L + V2);
          float ndh = max(dot(nW, Hn), 0.0);
          float glit = fract(sin(dot(floor(p.xz * 300.0) + floor(p.y * 300.0),
                                     vec2(12.9898, 78.233))) * 43758.5453);
          float spec = (pow(ndh, 420.0) * 2.2 + pow(ndh, 48.0) * 0.14) * (0.55 + 0.9 * glit);
          // foam: aligned steep crests break into whitecaps, fast tide/runoff
          // currents over shallows churn into surf, and the last metres of a
          // beach always carry a foam line
          float capN = vnoise(p * 480.0 + vec3(0.0, wt * 0.35, 0.0));
          float cap = smoothstep(0.82, 1.10, cr * (0.75 + 0.5 * capN) * rough);
          float surfF = smoothstep(0.25, 0.7, flow) * smoothstep(0.030, 0.006, hW);
          float fn = 0.5 + 0.5 * sin(q.x * 640.0 + wt * 2.1) * sin(q.y * 610.0 - wt * 1.8);
          float foamBand = (1.0 - smoothstep(0.0022, 0.0045, hW)) * smoothstep(0.0006, 0.0012, hW);
          float foam = clamp(cap * 0.85 + surfF * 0.7 + foamBand * (0.35 + 0.45 * fn), 0.0, 1.0);
          vec3 colW = mix(bedCol * diffW, env, clamp(F * 1.6, 0.0, 0.8)) + spec * (1.0 - foam);
          col = colW * day * 1.35 + vec3(0.20, 0.35, 0.60) * rim * 0.35 * day;
          col = mix(col, vec3(0.92, 0.96, 1.0) * (0.30 + 0.70 * day), foam * 0.85);
        } else {
          float h = length(p) - uSeaR;   // altitude relative to sea level
          float slope = 1.0 - clamp(dot(n, nd), 0.0, 1.0);
          vec3 sand = vec3(0.42, 0.36, 0.24);
          vec3 grassC = vec3(0.11, 0.27, 0.10);
          vec3 dryC = vec3(0.30, 0.28, 0.16);
          vec3 rock = vec3(0.27, 0.24, 0.20);
          vec3 snow = vec3(0.80, 0.82, 0.86);
          vec3 land;
          if (h < 0.0) {
            land = mix(sand, vec3(0.10, 0.09, 0.075), clamp(-h * 35.0, 0.0, 1.0));
          } else {
            vec3 veg = mix(grassC, dryC, clamp(h * 22.0 - 0.15, 0.0, 1.0));
            land = mix(sand, veg, clamp(h * 120.0, 0.0, 1.0));            // beach into vegetation
            land = mix(land, rock, clamp((h - 0.035) * 28.0, 0.0, 1.0));  // uplands turn rocky
            land = mix(land, snow, clamp((h - 0.072) * 45.0, 0.0, 1.0));  // snow caps
          }
          land = mix(land, rock, clamp(slope * 3.4 - 0.35, 0.0, 0.75));   // steep faces are bare rock
          // sub-grid ground detail: albedo mottling always, plus bump normals
          // fading in close-up so the height grid never reads as flat facets
          land *= 0.88 + 0.24 * vnoise(p * 90.0);
          float fadeG = smoothstep(3.0, 0.7, t);
          if (fadeG > 0.01) {
            float d1 = (vnoise(p * 260.0) - 0.5) + 0.5 * (vnoise(p * 760.0) - 0.5);
            float d2 = (vnoise(p * 260.0 + 41.3) - 0.5) + 0.5 * (vnoise(p * 760.0 + 17.9) - 0.5);
            n = normalize(n + (t1 * d1 + t2 * d2) * 0.6 * fadeG);
            land *= 0.86 + 0.28 * vnoise(p * 900.0);
          }
          float diff = 0.38 + 0.62 * max(dot(n, L), 0.0);
          col = land * diff * day * 1.35 + vec3(0.20, 0.35, 0.60) * rim * 0.35 * day;
        }
        float vz = max(t * dot(dir, uFwd), 0.051);
        fragZ = clamp(((80.05 * vz - 8.0) / (79.95 * vz)) * 0.5 + 0.5, 0.0, 1.0);
        tP = t;
      }
    }
    // the moon as a real sphere orbiting close by (can cross in front or hide behind)
    {
      vec3 M = uC + normalize(uMoonDir) * uMoonDist;
      vec3 oc = uEye - M;
      float bb = dot(oc, dir);
      float c2 = dot(oc, oc) - uMoonR * uMoonR;
      float ddm = bb * bb - c2;
      if (ddm > 0.0) {
        float tM = -bb - sqrt(ddm);
        if (tM > 0.0 && (!hit || tM < tP)) {
          vec3 nM = normalize(uEye + dir * tM - M);
          float lit = 0.04 + 0.96 * max(dot(nM, normalize(uLightDir)), 0.0);
          col = vec3(0.72, 0.73, 0.76) * lit;
          float vzm = max(tM * dot(dir, uFwd), 0.051);
          fragZ = clamp(((80.05 * vzm - 8.0) / (79.95 * vzm)) * 0.5 + 0.5, 0.0, 1.0);
          hit = true;
        }
      }
    }
    if (!hit) {
      vec3 Ls = normalize(uLightDir);
      // thin blue atmosphere hugging the sunlit limb
      float tCA = -dot(ro, dir);
      if (tCA > 0.0) {
        vec3 cp2 = ro + dir * tCA;
        float q = length(cp2);
        float dayA = 0.2 + 0.8 * smoothstep(-0.2, 0.4, dot(normalize(cp2), Ls));
        col += vec3(0.25, 0.45, 0.80) * exp(-max(q - uPBase, 0.0) * 22.0) * 0.55 * dayA;
      }
      // the sun: a big bright disc with a strong glare
      float sunD = max(dot(dir, Ls), 0.0);
      if (sunD > 0.99939) col = vec3(2.0, 1.85, 1.55);
      col += vec3(1.0, 0.93, 0.80) * (pow(sunD, 1200.0) * 1.2 + pow(sunD, 60.0) * 0.2);
    }
  }
  if (!hit && uPlanet < 0.5 && dir.y < -0.002) {
    float t = (-0.02 - uEye.y) / dir.y;
    if (t > 0.0) {
      vec3 p = uEye + dir * t;
      vec2 g = abs(fract(p.xz * 4.0) - 0.5) * 2.0;
      float line = smoothstep(0.92, 1.0, max(g.x, g.y));
      float dist = length(p.xz - uEye.xz);
      float fade = clamp(exp(-dist * 0.55) * 1.2, 0.0, 1.0);
      col = mix(sky, vec3(0.035, 0.045, 0.065) + line * 0.022, fade);
    }
  }
  float d0 = distance(vUv, vec2(0.5));
  WRITE_DEPTH(fragZ);
  OUTCOL = vec4(col * (1.0 - 0.3 * d0 * d0), 1.0);
}`);
  const blitProg = glProgram(gl, QUAD_VS, `
precision mediump float;
varying vec2 vUv;
uniform sampler2D uTex;
void main () { gl_FragColor = texture2D(uTex, vUv); }`);

  let scene = null;
  function ensureScene () {
    const w = canvas.width, h = canvas.height;
    if (scene && scene.w === w && scene.h === h) return;
    if (scene) {
      gl.deleteTexture(scene.tex);
      gl.deleteTexture(scene.dtex);
      gl.deleteFramebuffer(scene.fbo);
    }
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    // depth as a texture so the water composite can test against scene geometry
    const dtex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, dtex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, w, h, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, dtex, 0);
    scene = { tex, dtex, fbo, w, h };
  }

  const quadBuf3 = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf3);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1]), gl.STATIC_DRAW);
  function drawQuad (prog) {
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf3);
    const a = gl.getAttribLocation(prog, 'aPos');
    gl.enableVertexAttribArray(a);
    gl.vertexAttribPointer(a, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(a);
  }

  let ssf = null;
  function ssfTex (internal, format, filter) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, ssf.w, ssf.h, 0, format, gl.HALF_FLOAT, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return { tex, fbo };
  }
  function ensureSSF () {
    const w = Math.max(1, Math.round(canvas.width * 0.67));
    const h = Math.max(1, Math.round(canvas.height * 0.67));
    if (ssf && ssf.w === w && ssf.h === h) return;
    if (ssf) {
      for (const k of ['depth', 'blurA', 'blurB', 'thick']) {
        gl.deleteTexture(ssf[k].tex);
        gl.deleteFramebuffer(ssf[k].fbo);
      }
      gl.deleteRenderbuffer(ssf.rb);
    }
    const filt = halfLinear ? gl.LINEAR : gl.NEAREST;
    ssf = { w, h };
    ssf.depth = ssfTex(gl.R16F, gl.RED, filt);
    ssf.rb = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, ssf.rb);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, ssf.depth.fbo);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, ssf.rb);
    ssf.blurA = ssfTex(gl.R16F, gl.RED, filt);
    ssf.blurB = ssfTex(gl.R16F, gl.RED, filt);
    ssf.thick = ssfTex(gl.RGBA16F, gl.RGBA, filt);
  }

  const posBuf = gl.createBuffer();
  const colBuf = gl.createBuffer();
  const lineBuf = gl.createBuffer();
  const obsBuf = gl.createBuffer();
  let boxEdgeCount = 0;
  function buildBoxEdges () {
    const lo = fluid.h, hx = (fluid.nX - 1) * fluid.h, hy = (fluid.nY - 1) * fluid.h, hz = (fluid.nZ - 1) * fluid.h;
    const verts = [];
    const seg = (ax, ay, az, bx, by, bz) => { verts.push(ax, ay, az, bx, by, bz); };
    const circle = (cy, radius) => {
      const N = 48;
      for (let s = 0; s < N; s++) {
        const a0 = (s / N) * Math.PI * 2, a1 = ((s + 1) / N) * Math.PI * 2;
        seg(0.5 + Math.cos(a0) * radius, cy, 0.5 + Math.sin(a0) * radius,
            0.5 + Math.cos(a1) * radius, cy, 0.5 + Math.sin(a1) * radius);
      }
    };
    // outer box frame for spatial context (a planet floats free in space)
    if (params.shape !== 'planet') {
      const c = [
        [lo, lo, lo], [hx, lo, lo], [hx, lo, hz], [lo, lo, hz],
        [lo, hy, lo], [hx, hy, lo], [hx, hy, hz], [lo, hy, hz],
      ];
      const e = [0,1, 1,2, 2,3, 3,0, 4,5, 5,6, 6,7, 7,4, 0,4, 1,5, 2,6, 3,7];
      for (let i = 0; i < e.length; i += 2) {
        seg(c[e[i]][0], c[e[i]][1], c[e[i]][2], c[e[i + 1]][0], c[e[i + 1]][1], c[e[i + 1]][2]);
      }
    }
    if (params.shape === 'cyl') {
      circle(lo, 0.48);
      circle(hy, 0.48);
      for (let q = 0; q < 4; q++) {
        const a = q * Math.PI / 2 + Math.PI / 4;
        const x = 0.5 + Math.cos(a) * 0.48, z = 0.5 + Math.sin(a) * 0.48;
        seg(x, lo, z, x, hy, z);
      }
    } else if (params.shape === 'bowl') {
      for (const cy of [0.2, 0.32, 0.45]) {
        const rr = Math.min(0.49, Math.sqrt(Math.max(0, 0.5625 - (cy - 0.9) * (cy - 0.9))));
        circle(cy, rr);
      }
    }
    const arr = new Float32Array(verts);
    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
    gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
    return verts.length / 3;
  }

  function setProjUniforms (prog) {
    const { eye, fwd, right, up } = cameraBasis();
    gl.uniform3f(gl.getUniformLocation(prog, 'uEye'), eye[0], eye[1], eye[2]);
    gl.uniform3f(gl.getUniformLocation(prog, 'uRight'), right[0], right[1], right[2]);
    gl.uniform3f(gl.getUniformLocation(prog, 'uUp'), up[0], up[1], up[2]);
    gl.uniform3f(gl.getUniformLocation(prog, 'uFwd'), fwd[0], fwd[1], fwd[2]);
    gl.uniform1f(gl.getUniformLocation(prog, 'uTanA'), FOV_TAN);
    gl.uniform1f(gl.getUniformLocation(prog, 'uAspect'), canvas.width / canvas.height);
  }

  function uploadParticles3 () {
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, fluid.particlePos.subarray(0, 3 * fluid.numParticles), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
    gl.bufferData(gl.ARRAY_BUFFER, fluid.particleColor.subarray(0, 3 * fluid.numParticles), gl.DYNAMIC_DRAW);
  }
  function drawParticles3 (prog, withColor) {
    const aP = gl.getAttribLocation(prog, 'aPos');
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.enableVertexAttribArray(aP);
    gl.vertexAttribPointer(aP, 3, gl.FLOAT, false, 0, 0);
    let aC = -1;
    if (withColor) {
      aC = gl.getAttribLocation(prog, 'aColor');
      gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
      gl.enableVertexAttribArray(aC);
      gl.vertexAttribPointer(aC, 3, gl.FLOAT, false, 0, 0);
    }
    gl.drawArrays(gl.POINTS, 0, fluid.numParticles);
    gl.disableVertexAttribArray(aP);
    if (aC >= 0) gl.disableVertexAttribArray(aC);
  }

  function renderWaterSurfaceTargets () {
    ensureSSF();
    // view-space depth of the front-most particles
    gl.bindFramebuffer(gl.FRAMEBUFFER, ssf.depth.fbo);
    gl.viewport(0, 0, ssf.w, ssf.h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.useProgram(waterDepthProg);
    setProjUniforms(waterDepthProg);
    gl.uniform1f(gl.getUniformLocation(waterDepthProg, 'uPointScale'),
      2.8 * fluid.particleRadius * (ssf.h * 0.5) / FOV_TAN);
    gl.uniform1f(gl.getUniformLocation(waterDepthProg, 'uRadiusW'), 1.4 * fluid.particleRadius);
    drawParticles3(waterDepthProg, false);
    gl.disable(gl.DEPTH_TEST);
    // accumulated thickness + dye color
    gl.bindFramebuffer(gl.FRAMEBUFFER, ssf.thick.fbo);
    gl.viewport(0, 0, ssf.w, ssf.h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(thickProg);
    setProjUniforms(thickProg);
    gl.uniform1f(gl.getUniformLocation(thickProg, 'uPointScale'),
      5.0 * fluid.particleRadius * (ssf.h * 0.5) / FOV_TAN);
    gl.uniform1f(gl.getUniformLocation(thickProg, 'uW'), 0.30);
    drawParticles3(thickProg, true);
    gl.disable(gl.BLEND);
    // depth-aware separable blur; the kernel widens as the camera closes in,
    // so particles always merge into one continuous surface at any zoom
    const pixPer = 2.8 * fluid.particleRadius * (ssf.h * 0.5) / FOV_TAN / cam.dist;
    const blurW = Math.min(5.0, Math.max(1.2, pixPer * 0.35));
    gl.useProgram(blurProg);
    gl.uniform1i(gl.getUniformLocation(blurProg, 'uTex'), 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, ssf.blurA.fbo);
    gl.bindTexture(gl.TEXTURE_2D, ssf.depth.tex);
    gl.uniform2f(gl.getUniformLocation(blurProg, 'uDir'), blurW / ssf.w, 0);
    drawQuad(blurProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, ssf.blurB.fbo);
    gl.bindTexture(gl.TEXTURE_2D, ssf.blurA.tex);
    gl.uniform2f(gl.getUniformLocation(blurProg, 'uDir'), 0, blurW / ssf.h);
    drawQuad(blurProg);
    // second smoothing round flattens the remaining per-particle bumps
    gl.bindFramebuffer(gl.FRAMEBUFFER, ssf.blurA.fbo);
    gl.bindTexture(gl.TEXTURE_2D, ssf.blurB.tex);
    gl.uniform2f(gl.getUniformLocation(blurProg, 'uDir'), blurW / ssf.w, 0);
    drawQuad(blurProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, ssf.blurB.fbo);
    gl.bindTexture(gl.TEXTURE_2D, ssf.blurA.tex);
    gl.uniform2f(gl.getUniformLocation(blurProg, 'uDir'), 0, blurW / ssf.h);
    drawQuad(blurProg);
  }

  function compositeWater () {
    gl.useProgram(compProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, ssf.blurB.tex);
    gl.uniform1i(gl.getUniformLocation(compProg, 'uDepthTex'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, ssf.thick.tex);
    gl.uniform1i(gl.getUniformLocation(compProg, 'uThickTex'), 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, scene.tex);
    gl.uniform1i(gl.getUniformLocation(compProg, 'uScene'), 2);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, scene.dtex);
    gl.uniform1i(gl.getUniformLocation(compProg, 'uSceneDepth'), 3);
    gl.uniform2f(gl.getUniformLocation(compProg, 'uTexel'), 1.5 / ssf.w, 1.5 / ssf.h);
    gl.uniform1f(gl.getUniformLocation(compProg, 'uTanA'), FOV_TAN);
    gl.uniform1f(gl.getUniformLocation(compProg, 'uAspect'), canvas.width / canvas.height);
    const { eye, right, up, fwd } = cameraBasis();
    const L = getLightDir();
    gl.uniform3f(gl.getUniformLocation(compProg, 'uLightW'), L[0], L[1], L[2]);
    gl.uniform1f(gl.getUniformLocation(compProg, 'uPlanetOn'), params.shape === 'planet' ? 1 : 0);
    gl.uniform3f(gl.getUniformLocation(compProg, 'uCW'), CENTER[0], CENTER[1], CENTER[2]);
    gl.uniform3f(gl.getUniformLocation(compProg, 'uEyeW'), eye[0], eye[1], eye[2]);
    gl.uniform3f(gl.getUniformLocation(compProg, 'uLightV'),
      L[0] * right[0] + L[1] * right[1] + L[2] * right[2],
      L[0] * up[0] + L[1] * up[1] + L[2] * up[2],
      L[0] * fwd[0] + L[1] * fwd[1] + L[2] * fwd[2]);
    gl.uniform3f(gl.getUniformLocation(compProg, 'uRightW'), right[0], right[1], right[2]);
    gl.uniform3f(gl.getUniformLocation(compProg, 'uUpW'), up[0], up[1], up[2]);
    gl.uniform3f(gl.getUniformLocation(compProg, 'uFwdW'), fwd[0], fwd[1], fwd[2]);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    drawQuad(compProg);
    gl.disable(gl.BLEND);
    gl.activeTexture(gl.TEXTURE0);
  }

  function drawScene (targetFbo, surface) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFbo);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.020, 0.027, 0.047, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // raycast environment (sky gradient + tiled floor, or the planet terrain);
    // on WebGL2 the shader also writes terrain depth for correct occlusion
    if (gl2) { gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.ALWAYS); }
    else gl.disable(gl.DEPTH_TEST);
    gl.useProgram(bg3Prog);
    setProjUniforms(bg3Prog);
    gl.uniform1f(gl.getUniformLocation(bg3Prog, 'uPlanet'), params.shape === 'planet' ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(bg3Prog, 'uSeaR'), seaR);
    gl.uniform3f(gl.getUniformLocation(bg3Prog, 'uC'), CENTER[0], CENTER[1], CENTER[2]);
    gl.uniform1f(gl.getUniformLocation(bg3Prog, 'uPBase'), PBASE);
    gl.uniform1f(gl.getUniformLocation(bg3Prog, 'uPRad2'), (PBASE + 0.06) * (PBASE + 0.06));
    const LD = getLightDir();
    gl.uniform3f(gl.getUniformLocation(bg3Prog, 'uLightDir'), LD[0], LD[1], LD[2]);
    const MD = getMoonDir();
    gl.uniform3f(gl.getUniformLocation(bg3Prog, 'uMoonDir'), MD[0], MD[1], MD[2]);
    gl.uniform1f(gl.getUniformLocation(bg3Prog, 'uMoonDist'), 4.2);
    gl.uniform1f(gl.getUniformLocation(bg3Prog, 'uMoonR'), 0.34);
    gl.uniform1i(gl.getUniformLocation(bg3Prog, 'uTerrain'), 0);
    gl.uniform1i(gl.getUniformLocation(bg3Prog, 'uWater'), 1);
    gl.uniform1f(gl.getUniformLocation(bg3Prog, 'uWTime'), waveTime);
    gl.uniform2f(gl.getUniformLocation(bg3Prog, 'uTerDim'), TER_W, TER_H);
    gl.uniform2f(gl.getUniformLocation(bg3Prog, 'uWatDim'), SW_W, SW_H);
    gl.activeTexture(gl.TEXTURE1);
    if (swTex) gl.bindTexture(gl.TEXTURE_2D, swTex);
    gl.activeTexture(gl.TEXTURE0);
    if (terrainTex) gl.bindTexture(gl.TEXTURE_2D, terrainTex);
    drawQuad(bg3Prog);
    if (gl2) gl.depthFunc(gl.LESS);
    else gl.enable(gl.DEPTH_TEST);

    gl.useProgram(lineProg);
    setProjUniforms(lineProg);
    gl.uniform4f(gl.getUniformLocation(lineProg, 'uColor'), 0.72, 0.78, 0.85, 0.22);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
    const aLine = gl.getAttribLocation(lineProg, 'aPos');
    gl.enableVertexAttribArray(aLine);
    gl.vertexAttribPointer(aLine, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.LINES, 0, boxEdgeCount);
    gl.disableVertexAttribArray(aLine);
    gl.disable(gl.BLEND);

    if (!surface) {
      gl.useProgram(pointProg);
      setProjUniforms(pointProg);
      gl.uniform1f(gl.getUniformLocation(pointProg, 'uPointScale'),
        2.4 * fluid.particleRadius * (canvas.height * 0.5) / FOV_TAN);
      drawParticles3(pointProg, true);
    }

    const spheres = [];
    if (obstacle.active) {
      spheres.push({ x: obstacle.x, y: obstacle.y, z: obstacle.z, r: obstacle.radius,
                     color: obstacle.dye || [0.20, 0.24, 0.31] });
    }
    for (const b of balls) spheres.push(b);
    if (params.drain && params.shape !== 'planet') {
      spheres.push({ x: 0.5, y: fluid.h, z: 0.5, r: 0.1, color: [0.03, 0.04, 0.06] });
    }
    if (spheres.length > 0) {
      gl.useProgram(discProg);
      setProjUniforms(discProg);
      const aObs = gl.getAttribLocation(discProg, 'aPos');
      const uScale = gl.getUniformLocation(discProg, 'uPointScale');
      const uColor = gl.getUniformLocation(discProg, 'uColor');
      gl.bindBuffer(gl.ARRAY_BUFFER, obsBuf);
      gl.enableVertexAttribArray(aObs);
      for (const s of spheres) {
        gl.uniform1f(uScale, 2.0 * s.r * (canvas.height * 0.5) / FOV_TAN);
        gl.uniform3f(uColor, s.color[0], s.color[1], s.color[2]);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([s.x, s.y, s.z]), gl.DYNAMIC_DRAW);
        gl.vertexAttribPointer(aObs, 3, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.POINTS, 0, 1);
      }
      gl.disableVertexAttribArray(aObs);
    }
    gl.disable(gl.DEPTH_TEST);
  }

  function render () {
    if (planetMode) {
      // shallow-water earth: terrain and ocean are one raymarched scene
      uploadWaterTex();
      drawScene(null, true);
      return;
    }
    const surface = surfaceOK && !params.showParticles;
    uploadParticles3();
    if (surface) {
      renderWaterSurfaceTargets();
      ensureScene();
      drawScene(scene.fbo, true);
      // show the scene, then composite the refracting water surface on top
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.disable(gl.DEPTH_TEST);
      gl.useProgram(blitProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, scene.tex);
      gl.uniform1i(gl.getUniformLocation(blitProg, 'uTex'), 0);
      drawQuad(blitProg);
      compositeWater();
    } else {
      drawScene(null, false);
    }
  }

  function resizeCanvas () {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor(canvas.clientWidth * dpr);
    const h = Math.floor(canvas.clientHeight * dpr);
    if (w && h && (canvas.width !== w || canvas.height !== h)) { canvas.width = w; canvas.height = h; }
  }
  resizeCanvas();

  let lastInteraction = performance.now();
  const pointers = new Map();
  let pinchDist = 0;
  let stirHue = Math.random();
  let stirPhase = 0;
  let simAcc = 1, simLast = performance.now();
  function clampDist (d) { return Math.min(planetMode ? 40.0 : 9.0, Math.max(1.3, d)); }
  function screenToSim (clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    let nx = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ny = (1 - (clientY - rect.top) / rect.height) * 2 - 1;
    nx *= rect.width / rect.height;
    const { eye, fwd, right, up } = cameraBasis();
    const dir = normalize([
      fwd[0] + FOV_TAN * (nx * right[0] + ny * up[0]),
      fwd[1] + FOV_TAN * (nx * right[1] + ny * up[1]),
      fwd[2] + FOV_TAN * (nx * right[2] + ny * up[2]),
    ]);
    const rel = [CENTER[0] - eye[0], CENTER[1] - eye[1], CENTER[2] - eye[2]];
    const t = (rel[0] * fwd[0] + rel[1] * fwd[1] + rel[2] * fwd[2])
            / (dir[0] * fwd[0] + dir[1] * fwd[1] + dir[2] * fwd[2]);
    // keep the whole paddle sphere clear of the walls, so it can never
    // stamp solid cells or velocities into the boundary layer
    const m = obstacle.radius + fluid.h;
    return [
      Math.min(BOX - m, Math.max(m, eye[0] + dir[0] * t)),
      Math.min(BOX - m, Math.max(m, eye[1] + dir[1] * t)),
      Math.min(BOX - m, Math.max(m, eye[2] + dir[2] * t)),
    ];
  }

  canvas.addEventListener('pointerdown', e => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
    }
    lastInteraction = performance.now();
  });
  canvas.addEventListener('pointermove', e => {
    const prev = pointers.get(e.pointerId);
    if (!prev) return;
    const dx = e.clientX - prev.x, dy = e.clientY - prev.y;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinchDist > 0) cam.dist = clampDist(cam.dist * (pinchDist / d));
      pinchDist = d;
    } else if (e.shiftKey && params.shape !== 'planet') {
      // shift+drag tilts the tank (steers the gravity direction)
      params.tiltX = Math.max(-0.5, Math.min(0.5, params.tiltX + dx * 0.004));
      params.tiltZ = Math.max(-0.5, Math.min(0.5, params.tiltZ + dy * 0.004));
    } else {
      cam.theta -= dx * 0.008;
      cam.phi = Math.min(1.35, Math.max(-1.2, cam.phi + dy * 0.008));
    }
    lastInteraction = performance.now();
  });
  const endPointer = e => {
    pointers.delete(e.pointerId);
    pinchDist = 0;
  };
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    cam.dist = clampDist(cam.dist * Math.exp(e.deltaY * 0.001));
    lastInteraction = performance.now();
  }, { passive: false });
  canvas.addEventListener('dblclick', () => {
    camTarget[0] = CENTER[0]; camTarget[1] = CENTER[1]; camTarget[2] = CENTER[2];
  });

  setupScene();
  let pourHue = Math.random();

  const stirAction = { label: 'かき混ぜ (X)', pressed: () => params.stir,
    onClick: () => {
      params.stir = !params.stir;
      if (!params.stir) obstacle.active = false;
    } };
  const particleToggle = { label: '粒子表示 (V)', pressed: () => params.showParticles,
    onClick: () => { if (surfaceOK) params.showParticles = !params.showParticles; } };
  const commonControls = [
    { label: '重力', min: 0, max: 20, step: 0.1,
      get: () => params.gravity, set: v => { params.gravity = v; }, fmt: v => v.toFixed(1) },
    { label: 'しぶき (FLIP率)', min: planetMode ? 0 : 0.2, max: planetMode ? 0.2 : 1.0, step: 0.01,
      get: () => params.flipRatio, set: v => { params.flipRatio = v; }, fmt: v => v.toFixed(2) },
    { label: '水の量', min: planetMode ? 0.1 : 0.2, max: planetMode ? 0.5 : 0.7, step: 0.05,
      get: () => params.waterAmount, set: v => { params.waterAmount = v; setupScene(); }, fmt: v => v.toFixed(2) },
    { label: '解像度', min: planetMode ? 34 : 30, max: planetMode ? 50 : 46, step: 4,
      get: () => params.res, set: v => { params.res = v; setupScene(); }, fmt: v => String(Math.round(v)) },
  ];
  const speedControl = { label: '再生速度', min: 0.5, max: 1.5, step: 0.1,
    get: () => params.timeScale, set: v => { params.timeScale = v; }, fmt: v => v.toFixed(1) };

  return {
    canvas, params,
    hint: planetMode
      ? 'ドラッグで回転 ・ ホイールで拡大\n「カメラ」(C) で 標準 / 昼と夜 / 地球と月 / ワイド を切替\n「雨」(R) で降雨'
      : 'ドラッグで回転 ・ ホイール / ピンチで拡大\nShift+ドラッグで水槽を傾ける\n「かき混ぜ」(X) で自動かき混ぜ',
    sub: planetMode ? '地球 — 自転する全球の水循環' : '水 3D — 重力つき液体(3D FLIP法)',
    controls: planetMode
      ? [commonControls[0], commonControls[2],
         { label: '自転', min: 0, max: 0.12, step: 0.01,
           get: () => params.spin, set: v => { params.spin = v; }, fmt: v => v.toFixed(2) },
         { label: '潮汐 (誇張)', min: 0, max: 0.6, step: 0.05,
           get: () => params.tide, set: v => { params.tide = v; }, fmt: v => v.toFixed(2) },
         speedControl]
      : [...commonControls, speedControl],
    actions: planetMode
      ? [
          { label: '雨 (R)', pressed: () => params.pour, onClick: () => { params.pour = !params.pour; } },
          { label: () => 'カメラ (C): ' + CAM_PRESETS[camPreset].name,
            onClick: () => {
              camPreset = (camPreset + 1) % CAM_PRESETS.length;
              applyCamPreset();
            } },
          { label: '新しい地形', onClick: setupScene },
        ]
      : [
          stirAction,
          { label: () => '形状: ' + SHAPE_LABELS[params.shape],
            onClick: () => {
              params.shape = SHAPE_ORDER[(SHAPE_ORDER.indexOf(params.shape) + 1) % SHAPE_ORDER.length];
              setupScene();
            } },
          { label: 'リセット', onClick: setupScene },
          particleToggle,
        ],
    onKey (e) {
      if (planetMode) {
        if (e.key === 'r' || e.key === 'R') { params.pour = !params.pour; return true; }
        if (e.key === 'c' || e.key === 'C') {
          camPreset = (camPreset + 1) % CAM_PRESETS.length;
          applyCamPreset();
          return true;
        }
        return false;
      }
      if (e.key === 'x' || e.key === 'X') {
        params.stir = !params.stir;
        if (!params.stir) obstacle.active = false;
        return true;
      }
      if (e.key === 'v' || e.key === 'V') { if (surfaceOK) params.showParticles = !params.showParticles; return true; }
      return false;
    },
    frame (now) {
      resizeCanvas();
      const elapsed = Math.min((now - simLast) / 1000, 0.1);
      simLast = now;
      simAcc += elapsed;
      // decouple sim rate from display rate: at most ~60 steps/s on any monitor
      const doStep = !params.paused && simAcc >= 1 / 65;
      if (doStep) {
        simAcc = Math.min(simAcc - 1 / 60, 1 / 60);
        const dt = params.timeScale / 60;
        if (planetMode) {
          // true ratios: solar day = sidereal day x (1 + 1/365.25),
          // and the moon laps the sky once per 1/(1 - 1/27.32) days
          spinAngle += params.spin * dt;
          lightAngle += params.spin * (1.0 - 1.0 / 365.25) * dt;
          moonAngle += params.spin * (1.0 - 1.0 / 27.32) * dt;
          if (params.pour) {
            updateStorms(dt);
            rainSWE(dt);
          }
          // shallow-water substeps sized to the gravity-wave CFL limit,
          // adapting to the grid spacing and the current gravity setting
          const dxMin = PBASE * 0.18 * 2 * Math.PI / SW_W;
          const cMax = Math.sqrt(Math.max(4, params.gravity) * 0.09);
          const SUB = gpuSW ? Math.min(120, Math.max(10, Math.ceil(dt / (0.35 * dxMin / cMax))))
                            : Math.min(10, Math.max(3, Math.ceil(dt / 0.0022)));
          for (let s = 0; s < SUB; s++) sweStep(dt / SUB);
          waveTime += dt;
        } else {
          const g = params.gravity;
          const tilt = Math.hypot(params.tiltX, params.tiltZ);
          const gx = g * Math.sin(params.tiltX);
          const gz = g * Math.sin(params.tiltZ);
          const gy = -g * Math.cos(Math.min(tilt, 1.2));
          fluid.spin = 0;
          fluid.tideK = 0;
          if (params.pour) {
            pourHue = (pourHue + 0.0012) % 1;
            const c = HSVtoRGB(pourHue, 0.7, 1.0);
            for (let i = 0; i < 6; i++) {
              const a = Math.random() * Math.PI * 2, rr = Math.random() * 0.045;
              if (!fluid.addParticle(0.5 + Math.cos(a) * rr, BOX - 2.5 * fluid.h, 0.5 + Math.sin(a) * rr,
                0, -1.6, 0, c[0], c[1], c[2])) break;
            }
          }
          if (params.stir) {
            // auto-stir: while enabled, the paddle orbits the tank by itself
            stirPhase += dt * 2.4;
            stirHue = (stirHue + 0.03 * dt) % 1;
            const orbitR = 0.27;
            obstacle.active = true;
            obstacle.dye = HSVtoRGB(stirHue, 0.8, 1.0);
            obstacle.x = 0.5 + Math.cos(stirPhase) * orbitR;
            obstacle.z = 0.5 + Math.sin(stirPhase) * orbitR;
            obstacle.y = 0.30 + 0.08 * Math.sin(stirPhase * 0.7);
            obstacle.vx = -Math.sin(stirPhase) * orbitR * 2.4;
            obstacle.vz = Math.cos(stirPhase) * orbitR * 2.4;
            obstacle.vy = 0.056 * Math.cos(stirPhase * 0.7) * 2.4;
          } else {
            obstacle.active = false;
          }
          if (params.drain) fluid.drainAt(0.5, 0.5, 0.11, 0.18);
          updateBalls(dt, gx, gy, gz);
          fluid.simulate(dt, gx, gy, gz, params.flipRatio, obstacle, balls, false);
        }
      }
      if (pointers.size > 0) {
        camSunOffset = cam.theta - lightAngle;     // remember the user's framing
        camSpaceOffset = cam.theta - spinAngle;
      } else if (!reducedMotion) {
        if (planetMode && !params.paused) {
          if (params.view === 'sun') {
            // sun-fixed view: the sun is pinned on screen; the planet spins
            // through day and night and the moon orbits past with its phases
            cam.theta = lightAngle + camSunOffset;
          } else if (params.view === 'moon') {
            // moon-tracking view: keep the earth-moon pair framed side-on
            const md = getMoonDir();
            cam.theta = Math.atan2(md[0], md[2]) + Math.PI / 2;
            camTarget[0] = CENTER[0] + md[0] * 2.1;
            camTarget[1] = CENTER[1] + md[1] * 2.1;
            camTarget[2] = CENTER[2] + md[2] * 2.1;
          } else {
            // star-fixed view, locked to SIM time so it can never outrun the sun
            cam.theta = spinAngle + camSpaceOffset;
          }
        } else if (now - lastInteraction > 4000) {
          cam.theta += 0.06 / 60;
        }
      }
      render();
    },
  };
}

// ================= mode management & UI =================
