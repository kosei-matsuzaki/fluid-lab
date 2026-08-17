const MODES = [
  { key: 'ink2', factory: createInk2D, name: 'インク', desc: '発光インクの流れ(2D)' },
  { key: 'water2', factory: createWater2D, name: '水 2D', desc: '重力つき液体(FLIP法)' },
  { key: 'water3', factory: () => createWater3D(false), name: '水 3D', desc: '立体水槽(3D FLIP法)' },
  { key: 'earth', factory: () => createWater3D(true), name: '地球', desc: '自転する全球の水循環' },
];
const modules = {};
let modeKey = 'ink2';
let active = null;

const hintEl = document.getElementById('hint');
const ctlGrid = document.getElementById('ctlGrid');
const actionSlot = document.getElementById('actionSlot');
const pauseBtn = document.getElementById('pauseBtn');
const errBox = document.getElementById('errBox');
const sidebar = document.getElementById('sidebar');
const collapseBtn = document.getElementById('collapseBtn');
const modeList = document.getElementById('modeList');
const modeNameEl = document.getElementById('modeName');
const modeDescEl = document.getElementById('modeDesc');
let hintTimer = 0;

const ICONS = {
  ink2: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M12 3.5c3.2 4.6 6 7.6 6 10.7a6 6 0 1 1-12 0c0-3.1 2.8-6.1 6-10.7z"/></svg>',
  water2: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M3 9.5c2.2-2.4 4.3-2.4 6.5 0s4.3 2.4 6.5 0c1.7-1.9 3.3-2.2 5-0.6"/><path d="M3 15.5c2.2-2.4 4.3-2.4 6.5 0s4.3 2.4 6.5 0c1.7-1.9 3.3-2.2 5-0.6"/></svg>',
  water3: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><path d="M12 3l8 4.6v8.8L12 21l-8-4.6V7.6L12 3z"/><path d="M12 12l8-4.4M12 12L4 7.6M12 12v9"/></svg>',
  earth: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="12" cy="12" r="8.2"/><ellipse cx="12" cy="12" rx="3.6" ry="8.2"/><path d="M4.2 9.5h15.6M4.2 14.5h15.6"/></svg>',
};
const SVG_PAUSE = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="7" y="5.5" width="3.4" height="13" rx="1"/><rect x="13.6" y="5.5" width="3.4" height="13" rx="1"/></svg>';
const SVG_PLAY = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5.6v12.8l10-6.4z"/></svg>';
function updatePauseBtn () {
  const paused = active && !active.error && active.params.paused;
  pauseBtn.innerHTML = paused ? SVG_PLAY : SVG_PAUSE;
  pauseBtn.setAttribute('aria-pressed', String(!!paused));
}
pauseBtn.title = '一時停止 / 再開 (Space)';
collapseBtn.title = 'パネル開閉 (S)';


function sliderFill (input) {
  const pct = (input.value - input.min) / (input.max - input.min) * 100;
  input.style.background = `linear-gradient(90deg, var(--fill) ${pct}%, var(--track) ${pct}%)`;
}

function buildControls (mod) {
  ctlGrid.textContent = '';
  for (const c of mod.controls) {
    const wrap = document.createElement('div');
    wrap.className = 'ctl';
    const head = document.createElement('div');
    head.className = 'ctl-head';
    const label = document.createElement('label');
    label.textContent = c.label;
    const out = document.createElement('output');
    out.textContent = c.fmt(c.get());
    head.appendChild(label);
    head.appendChild(out);
    const input = document.createElement('input');
    input.type = 'range';
    input.min = c.min; input.max = c.max; input.step = c.step;
    input.value = c.get();
    input.setAttribute('aria-label', c.label);
    input.addEventListener('input', () => {
      const v = +input.value;
      c.set(v);
      out.textContent = c.fmt(v);
      sliderFill(input);
    });
    wrap.appendChild(head);
    wrap.appendChild(input);
    ctlGrid.appendChild(wrap);
    sliderFill(input);
  }
}
function buildActions (mod) {
  actionSlot.textContent = '';
  for (const a of mod.actions) {
    const btn = document.createElement('button');
    const setLabel = () => { btn.textContent = typeof a.label === 'function' ? a.label() : a.label; };
    setLabel();
    if (a.pressed) {
      btn.classList.add('tgl');
      btn.setAttribute('aria-pressed', String(a.pressed()));
    }
    btn.addEventListener('click', () => {
      a.onClick();
      if (a.pressed) btn.setAttribute('aria-pressed', String(a.pressed()));
      setLabel();
    });
    actionSlot.appendChild(btn);
  }
}
function showHint (text) {
  hintEl.textContent = '';
  const lines = text.split('\n');
  lines.forEach((l, i) => {
    hintEl.appendChild(document.createTextNode(l));
    if (i < lines.length - 1) hintEl.appendChild(document.createElement('br'));
  });
  hintEl.classList.remove('gone');
  clearTimeout(hintTimer);
  hintTimer = setTimeout(() => hintEl.classList.add('gone'), 6000);
}

function buildModeList () {
  modeList.textContent = '';
  MODES.forEach((m, i) => {
    const btn = document.createElement('button');
    btn.className = 'rail-btn';
    btn.innerHTML = ICONS[m.key] || '';
    btn.title = m.name + ' (' + (i + 1) + ')';
    btn.setAttribute('aria-label', m.name + ' (' + (i + 1) + 'キー)');
    btn.setAttribute('aria-pressed', String(m.key === modeKey));
    btn.addEventListener('click', () => {
      if (modeKey === m.key) return;
      modeKey = m.key;
      switchMode();
    });
    modeList.appendChild(btn);
  });
}
function syncModeList () {
  [...modeList.children].forEach((b, i) =>
    b.setAttribute('aria-pressed', String(MODES[i].key === modeKey)));
}

function switchMode () {
  if (active && active.canvas) active.canvas.classList.remove('active');
  const m = MODES.find(m => m.key === modeKey);
  if (!(modeKey in modules)) modules[modeKey] = m.factory();
  const mod = modules[modeKey];
  active = mod;
  syncModeList();
  modeNameEl.textContent = m.name;
  modeDescEl.textContent = m.desc;
  if (mod.error) {
    errBox.classList.add('show');
    ctlGrid.textContent = '';
    actionSlot.textContent = '';
    hintEl.classList.add('gone');
    updatePauseBtn();
    return;
  }
  errBox.classList.remove('show');
  mod.canvas.classList.add('active');
  buildControls(mod);
  buildActions(mod);
  updatePauseBtn();
  showHint(mod.hint);
}

function setPanel (open) {
  sidebar.classList.toggle('collapsed', !open);
  collapseBtn.setAttribute('aria-expanded', String(open));
}
collapseBtn.addEventListener('click', () => setPanel(sidebar.classList.contains('collapsed')));
if (window.innerWidth < 640) setPanel(false);
pauseBtn.addEventListener('click', () => {
  if (!active || active.error) return;
  active.params.paused = !active.params.paused;
  updatePauseBtn();
});
window.addEventListener('keydown', e => {
  if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON')) return;
  const idx = ['1', '2', '3', '4'].indexOf(e.key);
  if (idx >= 0 && idx < MODES.length) {
    if (MODES[idx].key !== modeKey) { modeKey = MODES[idx].key; switchMode(); }
    e.preventDefault();
    return;
  }
  if (e.key === ' ') {
    if (active && !active.error) {
      active.params.paused = !active.params.paused;
      updatePauseBtn();
    }
    e.preventDefault();
    return;
  }
  if (e.key === 's' || e.key === 'S') {
    setPanel(sidebar.classList.contains('collapsed'));
    e.preventDefault();
    return;
  }
  if (active && !active.error && active.onKey && active.onKey(e)) {
    e.preventDefault();
    buildActions(active);   // resync toggle-button states changed via keyboard
  }
});
stage.addEventListener('pointerdown', () => {
  hintEl.classList.add('gone');
  clearTimeout(hintTimer);
});

buildModeList();
switchMode();

function loop (now) {
  if (active && !active.error) active.frame(now);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

})();
