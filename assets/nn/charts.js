/* charts.js — small canvas plotting helpers, Tufte-flavoured: thin marks,
 * recessive axes, lots of paper. Colours are read from CSS custom properties
 * so every plot tracks light/dark automatically. Classic script, no modules.
 */

function themeColors() {
  const cs = getComputedStyle(document.documentElement);
  const g = n => cs.getPropertyValue(n).trim();
  return {
    c0: g('--c0'), c1: g('--c1'), c2: g('--c2'), cthe: g('--muted'),
    ink: g('--ink'), muted: g('--muted'), grid: g('--grid'), paper: g('--paper'),
  };
}

// Size a canvas to its container width at a given aspect, hi-DPI aware.
// Returns { ctx, w, h } in CSS pixels.
function fitCanvas(canvas, aspect) {
  const cssW = canvas.parentElement.clientWidth;
  const cssH = Math.round(cssW * aspect);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.width = cssW + 'px';
  canvas.style.height = cssH + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  return { ctx, w: cssW, h: cssH };
}

// Linear map builder from data range to pixel range.
function scale(d0, d1, p0, p1) {
  const m = (p1 - p0) / (d1 - d0);
  return v => p0 + (v - d0) * m;
}

function hexToRgb(hex) {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}
function mix(a, b, t) {
  const A = hexToRgb(a), B = hexToRgb(b);
  return `rgb(${Math.round(A[0] + (B[0] - A[0]) * t)},${Math.round(A[1] + (B[1] - A[1]) * t)},${Math.round(A[2] + (B[2] - A[2]) * t)})`;
}

/* Decision-boundary plot for a binary classifier.
 * Fills a coarse probability field (muted), inks a crisp 0.5 contour, then
 * draws the data points on top with a paper-coloured separation ring. */
function drawDecision(canvas, model, data, opts) {
  opts = opts || {};
  const { ctx, w, h } = fitCanvas(canvas, 0.85);
  const t = themeColors();
  const pad = 10;
  const xr = data.xr, yr = data.yr;
  const sx = scale(xr[0], xr[1], pad, w - pad);
  const sy = scale(yr[0], yr[1], h - pad, pad);
  const ix = scale(pad, w - pad, xr[0], xr[1]);      // pixel -> data
  const iy = scale(h - pad, pad, yr[0], yr[1]);

  // Probability field on a block grid.
  const step = 5;
  const cols = Math.ceil((w - 2 * pad) / step), rows = Math.ceil((h - 2 * pad) / step);
  const grid = [];
  const batch = [];
  for (let r = 0; r <= rows; r++) for (let c = 0; c <= cols; c++) {
    const px = pad + c * step, py = pad + r * step;
    batch.push([ix(px), iy(py)]);
  }
  const probs = model.predictProba(batch);
  let k = 0;
  for (let r = 0; r <= rows; r++) { grid.push([]); for (let c = 0; c <= cols; c++) grid[r].push(probs[k++]); }

  // fill — sharpen around 0.5 so even a near-chance model's two half-planes
  // are distinguishable (otherwise a linear model paints one flat muddy tint)
  k = 0;
  for (let r = 0; r <= rows; r++) for (let c = 0; c <= cols; c++) {
    const p = probs[k++];
    const fp = Math.max(0, Math.min(1, 0.5 + (p - 0.5) * 2.2));
    ctx.fillStyle = mix(t.c0, t.c1, fp);
    ctx.globalAlpha = 0.22;
    ctx.fillRect(pad + c * step - step / 2, pad + r * step - step / 2, step, step);
  }
  ctx.globalAlpha = 1;

  // crisp 0.5 contour: ink a cell whenever it straddles the boundary
  ctx.fillStyle = t.ink;
  for (let r = 0; r <= rows; r++) for (let c = 0; c <= cols; c++) {
    const p = grid[r][c];
    const right = c < cols ? grid[r][c + 1] : p;
    const down = r < rows ? grid[r + 1][c] : p;
    if ((p - 0.5) * (right - 0.5) < 0 || (p - 0.5) * (down - 0.5) < 0) {
      ctx.globalAlpha = 0.9;
      ctx.fillRect(pad + c * step - 1, pad + r * step - 1, 2, 2);
    }
  }
  ctx.globalAlpha = 1;

  // points
  for (let i = 0; i < data.X.length; i++) {
    const [x, y] = data.X[i];
    const px = sx(x), py = sy(y);
    ctx.beginPath();
    ctx.arc(px, py, 3.1, 0, Math.PI * 2);
    ctx.fillStyle = data.Y[i] === 1 ? t.c1 : t.c0;
    ctx.fill();
    ctx.lineWidth = 1; ctx.strokeStyle = t.paper; ctx.stroke();
  }
}

/* Small line chart with a real y-axis: three tick gridlines (min / mid / max),
 * an x-axis label, thin series, final value labelled directly.
 * series: [{ ys, color, label }]. Kept compact — the panel caps its width. */
function drawLines(canvas, series, opts) {
  opts = opts || {};
  const { ctx, w, h } = fitCanvas(canvas, opts.aspect || 0.42);
  const t = themeColors();
  const padL = 42, padR = 60, padT = 12, padB = 24;
  let n = 0, ymin = opts.ymin != null ? opts.ymin : Infinity, ymax = opts.ymax != null ? opts.ymax : -Infinity;
  for (const s of series) {
    n = Math.max(n, s.ys.length);
    if (opts.ymin == null) for (const v of s.ys) if (v < ymin) ymin = v;
    if (opts.ymax == null) for (const v of s.ys) if (v > ymax) ymax = v;
  }
  if (!isFinite(ymin)) ymin = 0;
  if (!isFinite(ymax)) ymax = 1;
  if (ymax <= ymin) ymax = ymin + 1;
  ymax += (ymax - ymin) * 0.06;                 // headroom above the peak
  const sx = scale(0, Math.max(1, n - 1), padL, w - padR);
  const sy = scale(ymin, ymax, h - padB, padT);

  // y gridlines + labels at min, mid, max
  ctx.font = '11px var(--mono, monospace)'; ctx.textBaseline = 'middle'; ctx.textAlign = 'right';
  for (const tv of [ymin, (ymin + ymax) / 2, ymax]) {
    const py = sy(tv);
    ctx.strokeStyle = t.grid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, py + 0.5); ctx.lineTo(w - padR, py + 0.5); ctx.stroke();
    ctx.fillStyle = t.muted; ctx.fillText(fmt(tv, opts.dp), padL - 6, py);
  }
  if (opts.xlabel) {
    ctx.fillStyle = t.muted; ctx.textAlign = 'right'; ctx.textBaseline = 'alphabetic';
    ctx.fillText(opts.xlabel + ' →', w - padR, h - 6);
  }

  for (const s of series) {
    if (!s.ys.length) continue;
    ctx.strokeStyle = s.color; ctx.lineWidth = 1.75; ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i < s.ys.length; i++) { const px = sx(i), py = sy(s.ys[i]); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
    ctx.stroke();
    const last = s.ys[s.ys.length - 1];
    ctx.fillStyle = s.color; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
    ctx.font = '12px var(--mono, monospace)';
    ctx.fillText(' ' + (s.label ? s.label + ' ' : '') + fmt(last, opts.dp), w - padR + 2, sy(last));
  }
}

function fmt(v, dp) { return dp === 0 ? Math.round(v).toString() : v.toFixed(dp == null ? 2 : dp); }

/* Labelled scatter for the embedding projection. pts: [{x,y,label,color}]. */
function drawScatter(canvas, pts, opts) {
  opts = opts || {};
  const { ctx, w, h } = fitCanvas(canvas, opts.aspect || 0.72);
  const t = themeColors();
  const pad = 34;
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const p of pts) { xmin = Math.min(xmin, p.x); xmax = Math.max(xmax, p.x); ymin = Math.min(ymin, p.y); ymax = Math.max(ymax, p.y); }
  const mx = (xmax - xmin) * 0.12 || 1, my = (ymax - ymin) * 0.12 || 1;
  const sx = scale(xmin - mx, xmax + mx, pad, w - pad);
  const sy = scale(ymin - my, ymax + my, h - pad, pad);

  // faint zero cross
  ctx.strokeStyle = t.grid; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, sy(0)); ctx.lineTo(w - pad, sy(0)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(sx(0), pad); ctx.lineTo(sx(0), h - pad); ctx.stroke();

  ctx.textBaseline = 'middle'; ctx.font = '13px var(--serif, serif)';
  for (const p of pts) {
    const px = sx(p.x), py = sy(p.y);
    ctx.beginPath(); ctx.arc(px, py, 3.4, 0, Math.PI * 2);
    ctx.fillStyle = p.color; ctx.fill();
    ctx.lineWidth = 1; ctx.strokeStyle = t.paper; ctx.stroke();
    ctx.fillStyle = t.ink; ctx.textAlign = 'left';
    ctx.fillText(' ' + p.label, px + 3, py);
  }
}

/* Generalization-gap chart for S1-4: three dataset sizes on the x, train and
 * test accuracy as two series, the gap between them shaded. */
function drawGap(canvas, rows) {
  // rows: [{ n, train, test }]
  const { ctx, w, h } = fitCanvas(canvas, 0.55);
  const t = themeColors();
  const padL = 40, padR = 58, padT = 16, padB = 30;
  const xs = rows.map((_, i) => padL + i * ((w - padL - padR) / Math.max(1, rows.length - 1)));
  const sy = scale(0.4, 1.0, h - padB, padT);

  // y gridlines at .5 .75 1.0
  ctx.strokeStyle = t.grid; ctx.lineWidth = 1; ctx.fillStyle = t.muted;
  ctx.font = '11px var(--mono, monospace)'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  [0.5, 0.75, 1.0].forEach(v => { ctx.beginPath(); ctx.moveTo(padL, sy(v)); ctx.lineTo(w - padR, sy(v)); ctx.stroke(); ctx.fillText((v * 100).toFixed(0) + '%', padL - 6, sy(v)); });

  // gap band
  ctx.beginPath();
  rows.forEach((r, i) => { const p = ctx[i === 0 ? 'moveTo' : 'lineTo'](xs[i], sy(r.train)); });
  for (let i = rows.length - 1; i >= 0; i--) ctx.lineTo(xs[i], sy(rows[i].test));
  ctx.closePath(); ctx.fillStyle = t.muted; ctx.globalAlpha = 0.13; ctx.fill(); ctx.globalAlpha = 1;

  const drawSeries = (key, color, label) => {
    ctx.strokeStyle = color; ctx.lineWidth = 1.75; ctx.beginPath();
    rows.forEach((r, i) => ctx[i === 0 ? 'moveTo' : 'lineTo'](xs[i], sy(r[key])));
    ctx.stroke();
    rows.forEach((r, i) => { ctx.beginPath(); ctx.arc(xs[i], sy(r[key]), 3.2, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill(); ctx.lineWidth = 1; ctx.strokeStyle = t.paper; ctx.stroke(); });
    const last = rows[rows.length - 1];
    ctx.fillStyle = color; ctx.textAlign = 'left'; ctx.font = '12px var(--mono, monospace)';
    ctx.fillText(' ' + label, xs[xs.length - 1] + 6, sy(last[key]));
  };
  drawSeries('train', t.c1, 'train');
  drawSeries('test', t.c0, 'test');

  // x labels
  ctx.fillStyle = t.muted; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.font = '11px var(--mono, monospace)';
  rows.forEach((r, i) => ctx.fillText('n=' + r.n, xs[i], h - padB + 6));
}
