/* experiments.js — the four proofs. Each run() reads its parameters from the
 * on-page controls, generates data, trains real nets live in the browser, and
 * animates the result. Nothing is precomputed. Changing a control supersedes
 * any in-flight training for that section (see RUNTOK). Depends on nn.js and
 * charts.js. Classic script.
 */

const REDRAW = {};     // section id -> repaint fn, called on resize
const RUNTOK = {};     // section id -> current run token; stale loops self-cancel

// Read free-text controls, validating and clamping to a safe range. The
// sanitised value is written back into the field so you always see what
// actually ran (e.g. typing 500 for a capped-at-64 width shows 64).
function pvNum(id, def, min, max) {
  const el = document.getElementById(id);
  if (!el) return def;
  let v = parseFloat(el.value);
  if (isNaN(v)) v = def;
  v = Math.max(min, Math.min(max, v));
  el.value = String(v);
  return v;
}
function pvInt(id, def, min, max) {
  const el = document.getElementById(id);
  if (!el) return def;
  let v = parseInt(el.value, 10);
  if (isNaN(v)) v = def;
  v = Math.max(min, Math.min(max, Math.round(v)));
  el.value = String(v);
  return v;
}
function pvAct(id, def, allowed) {
  const el = document.getElementById(id);
  if (!el) return def;
  let s = (el.value || '').trim().toLowerCase();
  if (!allowed.includes(s)) s = def;
  el.value = s;
  return s;
}
const ACTS = ['relu', 'gelu', 'tanh', 'sigmoid', 'leaky'];

function loopTrain(sec, totalSteps, perFrame, stepFn, frameFn, doneFn) {
  const tok = (RUNTOK[sec] = (RUNTOK[sec] || 0) + 1);   // supersede prior run
  let s = 0;
  function frame() {
    if (RUNTOK[sec] !== tok) return;                    // a newer run took over
    for (let k = 0; k < perFrame && s < totalSteps; k++) { stepFn(s); s++; }
    frameFn(s);
    if (s < totalSteps) requestAnimationFrame(frame); else if (doneFn) doneFn();
  }
  requestAnimationFrame(frame);
}

function sampleBatch(X, Y, bs) {
  const N = X.length;
  if (bs >= N) return { X, Y };
  const bx = [], by = [];
  for (let i = 0; i < bs; i++) { const j = (Math.random() * N) | 0; bx.push(X[j]); by.push(Y[j]); }
  return { X: bx, Y: by };
}

const pct = v => (v * 100).toFixed(1) + '%';
function setText(id, s) { const el = document.getElementById(id); if (el) el.textContent = s; }

/* ---------- data ---------- */

function makeRings(n, noise) {
  const X = [], Y = [], half = Math.floor(n / 2);
  for (let i = 0; i < n; i++) {
    const outer = i >= half;
    const r = (outer ? 2.0 : 0.8) + randn() * noise;
    const th = Math.random() * Math.PI * 2;
    X.push([r * Math.cos(th), r * Math.sin(th)]);
    Y.push(outer ? 1 : 0);
  }
  return { X, Y, xr: [-3, 3], yr: [-3, 3] };
}

function makeBlobs(n, flip) {
  const X = [], Y = [], Ytrue = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2, cx = c === 0 ? -1.1 : 1.1;
    let y = c;
    if (Math.random() < flip) y = 1 - y;   // label noise
    X.push([cx + randn() * 1.15, randn() * 1.15]);
    Y.push(y);         // noisy label — what the net trains on, and what the loss curve uses
    Ytrue.push(c);     // true label — used to score accuracy, so the gap is a real gen. gap
  }
  return { X, Y, Ytrue, xr: [-4.5, 4.5], yr: [-3.5, 3.5] };
}
function makeBlobsClean(n) {
  const X = [], Y = [];
  for (let i = 0; i < n; i++) { const c = i % 2, cx = c === 0 ? -1.1 : 1.1; X.push([cx + randn() * 1.15, randn() * 1.15]); Y.push(c); }
  return { X, Y };
}

const GRAMMAR = { animals: ['cat', 'dog', 'cow'], verbs: ['eat', 'chase', 'see'], fruits: ['apple', 'mango', 'banana'] };
function catOf(tok) {
  if (GRAMMAR.animals.includes(tok)) return 'animals';
  if (GRAMMAR.verbs.includes(tok)) return 'verbs';
  if (GRAMMAR.fruits.includes(tok)) return 'fruits';
  return 'the';
}
function makeGrammar(nSent) {
  const vocab = ['the', ...GRAMMAR.animals, ...GRAMMAR.verbs, ...GRAMMAR.fruits];
  const idx = {}; vocab.forEach((t, i) => idx[t] = i);
  const pick = a => a[(Math.random() * a.length) | 0];
  const pairs = [];
  for (let s = 0; s < nSent; s++) {
    const sent = ['the', pick(GRAMMAR.animals), pick(GRAMMAR.verbs), 'the', pick(GRAMMAR.fruits)];
    for (let i = 0; i < sent.length - 1; i++) pairs.push([idx[sent[i]], idx[sent[i + 1]]]);
  }
  return { vocab, idx, pairs };
}
function oneHot(i, n) { const v = new Array(n).fill(0); v[i] = 1; return v; }

/* ---------- linear algebra for the embedding projection ---------- */
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function powerIter(cov, d, deflate) {
  let v = Array.from({ length: d }, () => randn());
  for (let it = 0; it < 300; it++) {
    const nv = new Array(d).fill(0);
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) nv[i] += cov[i][j] * v[j];
    if (deflate) { const p = dot(nv, deflate); for (let i = 0; i < d; i++) nv[i] -= p * deflate[i]; }
    const norm = Math.sqrt(dot(nv, nv)) || 1; v = nv.map(x => x / norm);
  }
  return v;
}
function pca2(rows) {
  const n = rows.length, d = rows[0].length, mean = new Array(d).fill(0);
  for (const r of rows) for (let i = 0; i < d; i++) mean[i] += r[i] / n;
  const C = rows.map(r => r.map((v, i) => v - mean[i]));
  const cov = Array.from({ length: d }, () => new Array(d).fill(0));
  for (const r of C) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cov[i][j] += r[i] * r[j] / n;
  const v1 = powerIter(cov, d, null), v2 = powerIter(cov, d, v1);
  return C.map(r => ({ x: dot(r, v1), y: dot(r, v2) }));
}

/* ===================== 01 : activations ===================== */
function runS1() {
  const act = pvAct('s1-act', 'relu', ACTS);
  const layers = pvInt('s1-layers', 1, 1, 4);
  const units = pvInt('s1-units', 16, 2, 64);
  const noise = pvNum('s1-noise', 0.17, 0.01, 0.6);
  const steps = pvInt('s1-steps', 500, 50, 3000);
  const lr = pvNum('s1-lr', 0.05, 0.0005, 0.5);
  const data = makeRings(300, noise);
  const hidden = new Array(layers).fill(units);
  // Both nets are the exact same size. The only difference: the left one has
  // no activation on its hidden layer (all-linear, so it collapses to a line),
  // the right one uses the chosen activation.
  const noact = new MLP([2, ...hidden, 1], new Array(layers + 1).fill('linear'), 'bce');
  const withact = new MLP([2, ...hidden, 1], new Array(layers).fill(act).concat(['linear']), 'bce');
  const lossL = [], lossR = [];
  const draw = () => {
    drawDecision(document.getElementById('s1-linear'), noact, data);
    drawDecision(document.getElementById('s1-relu'), withact, data);
    const c = themeColors();
    drawLines(document.getElementById('s1-loss'), [
      { ys: lossL, color: c.c2, label: 'none' },
      { ys: lossR, color: c.c0, label: act },
    ], { ymin: 0, dp: 2, xlabel: 'epochs', aspect: 0.85 });
  };
  REDRAW.s1 = draw;
  loopTrain('s1', steps, 8,
    () => { lossL.push(noact.trainStep(data.X, data.Y, lr)); lossR.push(withact.trainStep(data.X, data.Y, lr)); },
    () => { setText('s1-linear-acc', pct(noact.accuracy(data.X, data.Y))); setText('s1-relu-acc', pct(withact.accuracy(data.X, data.Y))); draw(); }
  );
}

/* ===================== 02 : depth without nonlinearity ===================== */
function runS2() {
  const L = pvInt('s2-depth', 5, 2, 10), W = pvInt('s2-width', 16, 2, 64), act = pvAct('s2-act', 'relu', ACTS);
  const steps = pvInt('s2-steps', 900, 50, 3000);
  const lr = pvNum('s2-lr', 0.02, 0.0005, 0.5);
  const data = makeRings(300, 0.17);
  const hidden = new Array(L - 1).fill(W);
  const sizes = [2, ...hidden, 1];
  const one = new MLP([2, 1], ['linear'], 'bce');
  const deepLin = new MLP(sizes, new Array(L).fill('linear'), 'bce');
  const deepNL = new MLP(sizes, new Array(L - 1).fill(act).concat(['linear']), 'bce');
  // product of the deep-linear stack's weight matrices -> one 1×2 matrix.
  // A chain of linear maps is one linear map, so the whole stack collapses to this.
  const collapsed = () => {
    let M = deepLin.layers[0].W.map(r => r.slice());
    for (let l = 1; l < deepLin.layers.length; l++) {
      const A = deepLin.layers[l].W, nOut = A.length, nMid = A[0].length, nIn = M[0].length;
      const C = Array.from({ length: nOut }, () => new Array(nIn).fill(0));
      for (let o = 0; o < nOut; o++) for (let k = 0; k < nMid; k++) { const av = A[o][k]; for (let i = 0; i < nIn; i++) C[o][i] += av * M[k][i]; }
      M = C;
    }
    return M[0];
  };
  const draw = () => {
    drawDecision(document.getElementById('s2-one'), one, data);
    drawDecision(document.getElementById('s2-five'), deepLin, data);
    drawDecision(document.getElementById('s2-relu'), deepNL, data);
    const w = one.layers[0].W[0], m = collapsed();
    setText('s2-one-w', `[ ${w[0].toFixed(3)}   ${w[1].toFixed(3)} ]`);
    setText('s2-matrix', `[ ${m[0].toFixed(3)}   ${m[1].toFixed(3)} ]`);
  };
  REDRAW.s2 = draw;
  loopTrain('s2', steps, 12,
    () => { one.trainStep(data.X, data.Y, lr); deepLin.trainStep(data.X, data.Y, lr); deepNL.trainStep(data.X, data.Y, lr); },
    () => {
      setText('s2-one-acc', pct(one.accuracy(data.X, data.Y)));
      setText('s2-five-acc', pct(deepLin.accuracy(data.X, data.Y)));
      setText('s2-relu-acc', pct(deepNL.accuracy(data.X, data.Y)));
      draw();
    }
  );
}

/* ===================== S1-3 : embeddings from next-token ===================== */
function runS3() {
  const d = pvInt('s3-dim', 8, 1, 32), nSent = pvInt('s3-sent', 600, 20, 3000), steps = pvInt('s3-steps', 2600, 100, 8000);
  const lr = pvNum('s3-lr', 0.06, 0.0005, 0.5);
  const g = makeGrammar(nSent);
  const V = g.vocab.length;
  const model = new MLP([V, d, V], ['linear', 'linear'], 'softmax');
  const X = g.pairs.map(p => oneHot(p[0], V));
  const Y = g.pairs.map(p => p[1]);
  const loss = [];
  const catColor = () => { const c = themeColors(); return { animals: c.c0, verbs: c.c2, fruits: c.c1, the: c.muted }; };
  const embVec = tok => { const L = model.layers[0]; const v = []; for (let o = 0; o < L.nOut; o++) v.push(L.W[o][tok]); return v; };
  const draw = () => {
    const rows = g.vocab.map((_, i) => embVec(i));
    const proj = pca2(rows);
    const cc = catColor();
    const pts = g.vocab.map((tok, i) => ({ x: proj[i].x, y: proj[i].y, label: tok, color: cc[catOf(tok)] }));
    drawScatter(document.getElementById('s3-embed'), pts);
    drawLines(document.getElementById('s3-loss'), [{ ys: loss, color: themeColors().c0, label: 'loss' }], { ymin: 0, dp: 2, xlabel: 'steps' });
    // nearest-neighbour purity, computed live from the current embeddings
    const content = g.vocab.map((t, i) => ({ t, i })).filter(o => o.t !== 'the');
    let same = 0;
    for (const o of content) {
      let best = -1, bd = Infinity;
      for (const p of content) if (p.i !== o.i) {
        let dd = 0; for (let k = 0; k < d; k++) { const diff = rows[o.i][k] - rows[p.i][k]; dd += diff * diff; }
        if (dd < bd) { bd = dd; best = p.i; }
      }
      if (catOf(g.vocab[best]) === catOf(o.t)) same++;
    }
    setText('s3-neighbors', `${same} of ${content.length} content tokens have a same-category token as their nearest neighbour.`);
  };
  REDRAW.s3 = draw;
  loopTrain('s3', steps, 40,
    () => { const b = sampleBatch(X, Y, 64); loss.push(model.trainStep(b.X, b.Y, lr)); },
    () => { if (loss.length % 4 === 0) draw(); },
    () => draw()
  );
}

/* ===================== S1-4 : memorization vs generalization ===================== */
function runS4() {
  const W = pvInt('s4-width', 40, 2, 128), flip = pvNum('s4-noise', 0.08, 0, 0.5);
  const steps = pvInt('s4-steps', 1600, 100, 5000);
  const lr = pvNum('s4-lr', 0.02, 0.0005, 0.5);
  const sizes = [20, 200, 2000];
  const test = makeBlobsClean(1000);
  const rows = sizes.map(n => ({ n, train: 0.5, test: 0.5 }));
  const LOSS_COLORS = ['#6c71c4', '#2aa198', '#d33682'];   // n = 20, 200, 2000
  const runs = sizes.map((n, i) => {
    const train = makeBlobs(n, flip);
    const model = new MLP([2, W, W, 1], ['relu', 'relu', 'linear'], 'bce');
    return { n, i, train, model, loss: [], color: LOSS_COLORS[i], canvas: ['s4-b20', 's4-b200', 's4-b2000'][i] };
  });
  const draw = () => {
    for (const r of runs) drawDecision(document.getElementById(r.canvas), r.model, { X: r.train.X, Y: r.train.Y, xr: r.train.xr, yr: r.train.yr });
    drawGap(document.getElementById('s4-gap'), rows);
    setText('s4-table', rows.map(r => `n=${r.n}: train ${pct(r.train)} / test ${pct(r.test)} / gap ${pct(r.train - r.test)}`).join('  ·  '));
    drawLines(document.getElementById('s4-loss'), runs.map(r => ({ ys: r.loss, color: r.color, label: 'n=' + r.n })), { ymin: 0, dp: 2, xlabel: 'steps' });
  };
  REDRAW.s4 = draw;
  loopTrain('s4', steps, 25,
    () => { for (const r of runs) { const b = sampleBatch(r.train.X, r.train.Y, Math.min(64, r.n)); r.model.trainStep(b.X, b.Y, lr); } },
    (s) => {
      for (const r of runs) {
        rows[r.i].train = r.model.accuracy(r.train.X, r.train.Ytrue);   // true labels
        rows[r.i].test = r.model.accuracy(test.X, test.Y);              // true labels
        r.loss.push(r.model.loss(r.train.X, r.train.Y));               // noisy labels it fits
      }
      if (s % 4 === 0 || s >= steps) draw();
    }
  );
}

function runAll() { runS1(); runS2(); runS3(); runS4(); }
