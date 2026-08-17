function createInk2D () {
  const canvas = makeCanvas();
  canvas.style.cursor = 'crosshair';
  const gl = canvas.getContext('webgl2', { alpha: false, depth: false, stencil: false, antialias: false });
  if (!gl || !gl.getExtension('EXT_color_buffer_float')) { canvas.remove(); return { error: true }; }
  const supportLinear = !!gl.getExtension('OES_texture_half_float_linear');

  const params = {
    simRes: 160,
    dyeRes: Math.min(1024, Math.floor(window.devicePixelRatio * 720)),
    pressureIters: 24,
    curl: 10,
    velDissipation: 0.99,
    dyeDissipation: 1 - (1 - 0.5) * 0.045,
    splatRadius: 0.40 / 100,
    splatForce: 6000,
    paused: false,
  };

  const VERT = `
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv, vL, vR, vT, vB;
uniform vec2 texelSize;
void main () {
  vUv = aPosition * 0.5 + 0.5;
  vL = vUv - vec2(texelSize.x, 0.0);
  vR = vUv + vec2(texelSize.x, 0.0);
  vT = vUv + vec2(0.0, texelSize.y);
  vB = vUv - vec2(0.0, texelSize.y);
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;
  const FRAG = {
    clear: `
precision highp float;
varying vec2 vUv;
uniform sampler2D uTexture;
uniform float value;
void main () { gl_FragColor = value * texture2D(uTexture, vUv); }`,
    splat: `
precision highp float;
varying vec2 vUv;
uniform sampler2D uTarget;
uniform float aspectRatio;
uniform vec3 color;
uniform vec2 point;
uniform float radius;
void main () {
  vec2 p = vUv - point;
  p.x *= aspectRatio;
  vec3 splat = exp(-dot(p, p) / radius) * color;
  gl_FragColor = vec4(texture2D(uTarget, vUv).xyz + splat, 1.0);
}`,
    advection: `
precision highp float;
varying vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 texelSize;
uniform vec2 dyeTexelSize;
uniform float dt;
uniform float dissipation;
#ifdef MANUAL_FILTERING
vec4 bilerp (sampler2D sam, vec2 uv, vec2 tsize) {
  vec2 st = uv / tsize - 0.5;
  vec2 iuv = floor(st);
  vec2 fuv = fract(st);
  vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * tsize);
  vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * tsize);
  vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * tsize);
  vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * tsize);
  return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
}
#endif
void main () {
#ifdef MANUAL_FILTERING
  vec2 coord = vUv - dt * bilerp(uVelocity, vUv, texelSize).xy * texelSize;
  vec4 result = bilerp(uSource, coord, dyeTexelSize);
#else
  vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
  vec4 result = texture2D(uSource, coord);
#endif
  gl_FragColor = dissipation * result;
}`,
    divergence: `
precision highp float;
varying vec2 vUv, vL, vR, vT, vB;
uniform sampler2D uVelocity;
void main () {
  float L = texture2D(uVelocity, vL).x;
  float R = texture2D(uVelocity, vR).x;
  float T = texture2D(uVelocity, vT).y;
  float B = texture2D(uVelocity, vB).y;
  vec2 C = texture2D(uVelocity, vUv).xy;
  if (vL.x < 0.0) L = -C.x;
  if (vR.x > 1.0) R = -C.x;
  if (vT.y > 1.0) T = -C.y;
  if (vB.y < 0.0) B = -C.y;
  gl_FragColor = vec4(0.5 * (R - L + T - B), 0.0, 0.0, 1.0);
}`,
    curl: `
precision highp float;
varying vec2 vUv, vL, vR, vT, vB;
uniform sampler2D uVelocity;
void main () {
  float L = texture2D(uVelocity, vL).y;
  float R = texture2D(uVelocity, vR).y;
  float T = texture2D(uVelocity, vT).x;
  float B = texture2D(uVelocity, vB).x;
  gl_FragColor = vec4(R - L - T + B, 0.0, 0.0, 1.0);
}`,
    vorticity: `
precision highp float;
varying vec2 vUv, vL, vR, vT, vB;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform float curl;
uniform float dt;
void main () {
  float L = texture2D(uCurl, vL).x;
  float R = texture2D(uCurl, vR).x;
  float T = texture2D(uCurl, vT).x;
  float B = texture2D(uCurl, vB).x;
  float C = texture2D(uCurl, vUv).x;
  vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
  force /= length(force) + 0.0001;
  force *= curl * C;
  force.y *= -1.0;
  vec2 velocity = texture2D(uVelocity, vUv).xy + force * dt;
  velocity = clamp(velocity, -1000.0, 1000.0);
  gl_FragColor = vec4(velocity, 0.0, 1.0);
}`,
    pressure: `
precision highp float;
varying vec2 vUv, vL, vR, vT, vB;
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
void main () {
  float L = texture2D(uPressure, vL).x;
  float R = texture2D(uPressure, vR).x;
  float T = texture2D(uPressure, vT).x;
  float B = texture2D(uPressure, vB).x;
  float divergence = texture2D(uDivergence, vUv).x;
  gl_FragColor = vec4((L + R + B + T - divergence) * 0.25, 0.0, 0.0, 1.0);
}`,
    gradientSubtract: `
precision highp float;
varying vec2 vUv, vL, vR, vT, vB;
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
void main () {
  float L = texture2D(uPressure, vL).x;
  float R = texture2D(uPressure, vR).x;
  float T = texture2D(uPressure, vT).x;
  float B = texture2D(uPressure, vB).x;
  vec2 velocity = texture2D(uVelocity, vUv).xy - vec2(R - L, T - B);
  gl_FragColor = vec4(velocity, 0.0, 1.0);
}`,
    display: `
precision highp float;
varying vec2 vUv;
uniform sampler2D uTexture;
void main () {
  vec3 c = texture2D(uTexture, vUv).rgb;
  c = c / (1.0 + c * 0.35);
  float d = distance(vUv, vec2(0.5));
  c *= 1.0 - 0.35 * d * d;
  gl_FragColor = vec4(c, 1.0);
}`,
  };

  const defines = supportLinear ? '' : '#define MANUAL_FILTERING\n';
  let progs;
  try {
    progs = {
      clear: glProgramU(gl, VERT, FRAG.clear),
      splat: glProgramU(gl, VERT, FRAG.splat),
      advection: glProgramU(gl, VERT, defines + FRAG.advection),
      divergence: glProgramU(gl, VERT, FRAG.divergence),
      curl: glProgramU(gl, VERT, FRAG.curl),
      vorticity: glProgramU(gl, VERT, FRAG.vorticity),
      pressure: glProgramU(gl, VERT, FRAG.pressure),
      gradientSubtract: glProgramU(gl, VERT, FRAG.gradientSubtract),
      display: glProgramU(gl, VERT, FRAG.display),
    };
  } catch (e) { canvas.remove(); return { error: true }; }

  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(0);

  function blit (target) {
    if (target == null) {
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    } else {
      gl.viewport(0, 0, target.width, target.height);
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    }
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  }

  const filtering = supportLinear ? gl.LINEAR : gl.NEAREST;
  function createFBO (w, h, internalFormat, format, type, filter) {
    const texture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    return {
      texture, fbo, width: w, height: h,
      texelSizeX: 1 / w, texelSizeY: 1 / h,
      attach (id) { gl.activeTexture(gl.TEXTURE0 + id); gl.bindTexture(gl.TEXTURE_2D, texture); return id; },
    };
  }
  function createDoubleFBO (w, h, internalFormat, format, type, filter) {
    let a = createFBO(w, h, internalFormat, format, type, filter);
    let b = createFBO(w, h, internalFormat, format, type, filter);
    return {
      width: w, height: h, texelSizeX: 1 / w, texelSizeY: 1 / h,
      get read () { return a; }, get write () { return b; },
      swap () { const t = a; a = b; b = t; },
    };
  }
  function getResolution (base) {
    let aspect = gl.drawingBufferWidth / gl.drawingBufferHeight;
    if (aspect < 1) aspect = 1 / aspect;
    const min = Math.round(base), max = Math.round(base * aspect);
    return gl.drawingBufferWidth > gl.drawingBufferHeight
      ? { width: max, height: min } : { width: min, height: max };
  }

  let velocity, dye, divergence, curlFBO, pressure;
  function initFBOs () {
    const sim = getResolution(params.simRes);
    const dyeR = getResolution(params.dyeRes);
    velocity = createDoubleFBO(sim.width, sim.height, gl.RG16F, gl.RG, gl.HALF_FLOAT, filtering);
    dye = createDoubleFBO(dyeR.width, dyeR.height, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, filtering);
    divergence = createFBO(sim.width, sim.height, gl.R16F, gl.RED, gl.HALF_FLOAT, gl.NEAREST);
    curlFBO = createFBO(sim.width, sim.height, gl.R16F, gl.RED, gl.HALF_FLOAT, gl.NEAREST);
    pressure = createDoubleFBO(sim.width, sim.height, gl.R16F, gl.RED, gl.HALF_FLOAT, gl.NEAREST);
  }
  function resizeCanvas () {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor(canvas.clientWidth * dpr);
    const h = Math.floor(canvas.clientHeight * dpr);
    if (w && h && (canvas.width !== w || canvas.height !== h)) {
      canvas.width = w; canvas.height = h;
      initFBOs();
    }
  }
  resizeCanvas();

  function step (dt) {
    gl.disable(gl.BLEND);
    let u = progs.curl;
    gl.useProgram(u.program);
    gl.uniform2f(u.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(u.uniforms.uVelocity, velocity.read.attach(0));
    blit(curlFBO);

    u = progs.vorticity;
    gl.useProgram(u.program);
    gl.uniform2f(u.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(u.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(u.uniforms.uCurl, curlFBO.attach(1));
    gl.uniform1f(u.uniforms.curl, params.curl);
    gl.uniform1f(u.uniforms.dt, dt);
    blit(velocity.write);
    velocity.swap();

    u = progs.divergence;
    gl.useProgram(u.program);
    gl.uniform2f(u.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(u.uniforms.uVelocity, velocity.read.attach(0));
    blit(divergence);

    u = progs.clear;
    gl.useProgram(u.program);
    gl.uniform1i(u.uniforms.uTexture, pressure.read.attach(0));
    gl.uniform1f(u.uniforms.value, 0.8);
    blit(pressure.write);
    pressure.swap();

    u = progs.pressure;
    gl.useProgram(u.program);
    gl.uniform2f(u.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(u.uniforms.uDivergence, divergence.attach(0));
    for (let i = 0; i < params.pressureIters; i++) {
      gl.uniform1i(u.uniforms.uPressure, pressure.read.attach(1));
      blit(pressure.write);
      pressure.swap();
    }

    u = progs.gradientSubtract;
    gl.useProgram(u.program);
    gl.uniform2f(u.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(u.uniforms.uPressure, pressure.read.attach(0));
    gl.uniform1i(u.uniforms.uVelocity, velocity.read.attach(1));
    blit(velocity.write);
    velocity.swap();

    u = progs.advection;
    gl.useProgram(u.program);
    gl.uniform2f(u.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    if (!supportLinear) gl.uniform2f(u.uniforms.dyeTexelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(u.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(u.uniforms.uSource, velocity.read.attach(0));
    gl.uniform1f(u.uniforms.dt, dt);
    gl.uniform1f(u.uniforms.dissipation, params.velDissipation);
    blit(velocity.write);
    velocity.swap();

    if (!supportLinear) gl.uniform2f(u.uniforms.dyeTexelSize, dye.texelSizeX, dye.texelSizeY);
    gl.uniform1i(u.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(u.uniforms.uSource, dye.read.attach(1));
    gl.uniform1f(u.uniforms.dissipation, params.dyeDissipation);
    blit(dye.write);
    dye.swap();
  }

  function render () {
    const u = progs.display;
    gl.useProgram(u.program);
    gl.uniform1i(u.uniforms.uTexture, dye.read.attach(0));
    blit(null);
  }

  let hue = Math.random();
  function nextColor () {
    hue = (hue + 0.13) % 1;
    const c = HSVtoRGB(hue, 0.85, 1.0);
    return [c[0] * 0.2, c[1] * 0.2, c[2] * 0.2];
  }
  function splat (x, y, dx, dy, color) {
    const aspect = canvas.width / canvas.height;
    const u = progs.splat;
    gl.useProgram(u.program);
    gl.uniform1i(u.uniforms.uTarget, velocity.read.attach(0));
    gl.uniform1f(u.uniforms.aspectRatio, aspect);
    gl.uniform2f(u.uniforms.point, x, y);
    gl.uniform3f(u.uniforms.color, dx, dy, 0);
    gl.uniform1f(u.uniforms.radius, params.splatRadius * (aspect >= 1 ? aspect : 1));
    blit(velocity.write);
    velocity.swap();
    gl.uniform1i(u.uniforms.uTarget, dye.read.attach(0));
    gl.uniform3f(u.uniforms.color, color[0], color[1], color[2]);
    blit(dye.write);
    dye.swap();
  }
  function randomBurst (count) {
    for (let i = 0; i < count; i++) {
      const color = nextColor().map(v => v * 10);
      const angle = Math.random() * Math.PI * 2;
      const mag = 800 + Math.random() * 800;
      splat(0.2 + Math.random() * 0.6, 0.2 + Math.random() * 0.6,
            Math.cos(angle) * mag, Math.sin(angle) * mag, color);
    }
  }
  function clearDye () {
    const u = progs.clear;
    gl.useProgram(u.program);
    gl.uniform1i(u.uniforms.uTexture, dye.read.attach(0));
    gl.uniform1f(u.uniforms.value, 0);
    blit(dye.write);
    dye.swap();
    gl.uniform1i(u.uniforms.uTexture, velocity.read.attach(0));
    blit(velocity.write);
    velocity.swap();
  }

  const pointer = { down: false, moved: false, x: 0, y: 0, dx: 0, dy: 0, color: nextColor() };
  let lastInteraction = performance.now();
  function pointerPos (e) {
    const rect = canvas.getBoundingClientRect();
    return [(e.clientX - rect.left) / rect.width, 1 - (e.clientY - rect.top) / rect.height];
  }
  canvas.addEventListener('pointerdown', e => {
    canvas.setPointerCapture(e.pointerId);
    const [x, y] = pointerPos(e);
    pointer.down = true; pointer.x = x; pointer.y = y;
    pointer.color = nextColor();
  });
  canvas.addEventListener('pointermove', e => {
    if (!pointer.down) return;
    const [x, y] = pointerPos(e);
    pointer.dx = (x - pointer.x) * params.splatForce;
    pointer.dy = (y - pointer.y) * params.splatForce;
    pointer.x = x; pointer.y = y;
    pointer.moved = Math.abs(pointer.dx) > 0.1 || Math.abs(pointer.dy) > 0.1;
    lastInteraction = performance.now();
  });
  const endPointer = () => { pointer.down = false; pointer.moved = false; };
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);

  let lastTime = performance.now();
  randomBurst(reducedMotion ? 3 : 8);

  return {
    canvas, params,
    hint: 'ドラッグしてインクを流す',
    sub: 'インク 2D — 発光インクの流れ',
    controls: [
      { label: '渦の強さ', min: 0, max: 20, step: 1,
        get: () => params.curl, set: v => { params.curl = v; }, fmt: v => String(Math.round(v)) },
      { label: 'インクの持続', min: 0, max: 1, step: 0.05,
        get: () => 1 - (1 - params.dyeDissipation) / 0.045,
        set: v => { params.dyeDissipation = 1 - (1 - v) * 0.045; }, fmt: v => v.toFixed(2) },
      { label: '筆の太さ', min: 0.05, max: 0.75, step: 0.05,
        get: () => params.splatRadius * 100, set: v => { params.splatRadius = v / 100; }, fmt: v => v.toFixed(2) },
    ],
    actions: [
      { label: 'ひと吹き (B)', onClick: () => randomBurst(6 + Math.floor(Math.random() * 5)) },
      { label: 'クリア (C)', onClick: clearDye },
    ],
    onKey (e) {
      if (e.key === 'b' || e.key === 'B') { randomBurst(6 + Math.floor(Math.random() * 5)); return true; }
      if (e.key === 'c' || e.key === 'C') { clearDye(); return true; }
      return false;
    },
    frame (now) {
      const dt = Math.min((now - lastTime) / 1000, 1 / 30);
      lastTime = now;
      resizeCanvas();
      if (!params.paused) {
        if (pointer.moved) {
          splat(pointer.x, pointer.y, pointer.dx, pointer.dy, pointer.color.map(v => v * 10));
          pointer.moved = false;
        }
        if (!reducedMotion && now - lastInteraction > 6000) {
          randomBurst(1);
          lastInteraction = now - 3000;
        }
        step(dt);
      }
      render();
    },
  };
}

// =====================================================================
// MODULE 3: Water 2D — FLIP particles in a tank
// =====================================================================
