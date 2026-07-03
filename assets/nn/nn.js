/* nn.js — tiny neural nets in the browser, zero dependencies.
 *
 * Everything here is small enough that plain nested loops are fast, so the
 * code stays literal: dense layers, ReLU/sigmoid/tanh/linear, Adam, and two
 * output heads (sigmoid + BCE for binary, softmax + cross-entropy for
 * multiclass). Loaded as a classic script; classes below are visible to the
 * other scripts on the page by load order. No modules, so it also runs from
 * a double-clicked file:// index.html.
 */

function randn() {
  // Box-Muller. Standard normal, used for weight init.
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

// GELU (tanh approximation) and its derivative — needed for backprop.
const GELU_K = 0.7978845608028654;   // sqrt(2/pi)
function gelu(x) { return 0.5 * x * (1 + Math.tanh(GELU_K * (x + 0.044715 * x * x * x))); }
function geluPrime(x) {
  const u = GELU_K * (x + 0.044715 * x * x * x);
  const t = Math.tanh(u);
  const du = GELU_K * (1 + 3 * 0.044715 * x * x);
  return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du;
}

function softmax(row) {
  let m = -Infinity;
  for (const z of row) if (z > m) m = z;
  const out = new Array(row.length);
  let s = 0;
  for (let i = 0; i < row.length; i++) { out[i] = Math.exp(row[i] - m); s += out[i]; }
  for (let i = 0; i < row.length; i++) out[i] /= s;
  return out;
}

/* One fully-connected layer. Weights W are [nOut][nIn], bias b is [nOut].
 * Optimised with Adam; each layer keeps its own moment estimates. */
class Dense {
  constructor(nIn, nOut, activation) {
    this.nIn = nIn; this.nOut = nOut; this.activation = activation;
    // He init for relu-family, Xavier otherwise. Either is fine at this scale.
    const heFamily = activation === 'relu' || activation === 'gelu' || activation === 'leaky';
    const scale = heFamily ? Math.sqrt(2 / nIn) : Math.sqrt(1 / nIn);
    this.W = []; this.b = new Array(nOut).fill(0);
    this.mW = []; this.vW = [];
    for (let o = 0; o < nOut; o++) {
      const w = new Array(nIn), mw = new Array(nIn).fill(0), vw = new Array(nIn).fill(0);
      for (let i = 0; i < nIn; i++) w[i] = randn() * scale;
      this.W.push(w); this.mW.push(mw); this.vW.push(vw);
    }
    this.mb = new Array(nOut).fill(0); this.vb = new Array(nOut).fill(0);
  }

  actArray(z) {
    const a = new Array(z.length);
    switch (this.activation) {
      case 'relu':    for (let o = 0; o < z.length; o++) a[o] = z[o] > 0 ? z[o] : 0; break;
      case 'leaky':   for (let o = 0; o < z.length; o++) a[o] = z[o] > 0 ? z[o] : 0.01 * z[o]; break;
      case 'gelu':    for (let o = 0; o < z.length; o++) a[o] = gelu(z[o]); break;
      case 'sigmoid': for (let o = 0; o < z.length; o++) a[o] = sigmoid(z[o]); break;
      case 'tanh':    for (let o = 0; o < z.length; o++) a[o] = Math.tanh(z[o]); break;
      default:        for (let o = 0; o < z.length; o++) a[o] = z[o];            // linear
    }
    return a;
  }

  // X: [N][nIn] -> A: [N][nOut]. Caches inputs and pre-activations for backward.
  forward(X) {
    const N = X.length, A = new Array(N), Z = new Array(N);
    for (let n = 0; n < N; n++) {
      const x = X[n], z = new Array(this.nOut);
      for (let o = 0; o < this.nOut; o++) {
        let s = this.b[o]; const w = this.W[o];
        for (let i = 0; i < this.nIn; i++) s += w[i] * x[i];
        z[o] = s;
      }
      Z[n] = z; A[n] = this.actArray(z);
    }
    this.xCache = X; this.zCache = Z; this.aCache = A;
    return A;
  }

  // dA: [N][nOut] = d loss / d (this layer's output). Returns dX: [N][nIn].
  // The 1/N averaging is baked into the top-level output gradient, so nothing
  // here re-divides; grads just sum over the batch.
  backward(dA, lr, t) {
    const N = dA.length;
    const gW = this.W.map(r => new Array(r.length).fill(0));
    const gb = new Array(this.nOut).fill(0);
    const dX = new Array(N);
    for (let n = 0; n < N; n++) {
      const z = this.zCache[n], a = this.aCache[n], x = this.xCache[n];
      const dz = new Array(this.nOut);
      for (let o = 0; o < this.nOut; o++) {
        let d = dA[n][o];
        switch (this.activation) {
          case 'relu':    d *= z[o] > 0 ? 1 : 0; break;
          case 'leaky':   d *= z[o] > 0 ? 1 : 0.01; break;
          case 'gelu':    d *= geluPrime(z[o]); break;
          case 'sigmoid': d *= a[o] * (1 - a[o]); break;
          case 'tanh':    d *= 1 - a[o] * a[o]; break;
          // linear: derivative 1
        }
        dz[o] = d;
      }
      const dx = new Array(this.nIn).fill(0);
      for (let o = 0; o < this.nOut; o++) {
        const w = this.W[o], gw = gW[o], dzo = dz[o];
        gb[o] += dzo;
        for (let i = 0; i < this.nIn; i++) { gw[i] += dzo * x[i]; dx[i] += dzo * w[i]; }
      }
      dX[n] = dx;
    }
    this.adam(gW, gb, lr, t);
    return dX;
  }

  adam(gW, gb, lr, t) {
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
    for (let o = 0; o < this.nOut; o++) {
      const w = this.W[o], mw = this.mW[o], vw = this.vW[o], gw = gW[o];
      for (let i = 0; i < this.nIn; i++) {
        mw[i] = b1 * mw[i] + (1 - b1) * gw[i];
        vw[i] = b2 * vw[i] + (1 - b2) * gw[i] * gw[i];
        w[i] -= lr * (mw[i] / c1) / (Math.sqrt(vw[i] / c2) + eps);
      }
      this.mb[o] = b1 * this.mb[o] + (1 - b1) * gb[o];
      this.vb[o] = b2 * this.vb[o] + (1 - b2) * gb[o] * gb[o];
      this.b[o] -= lr * (this.mb[o] / c1) / (Math.sqrt(this.vb[o] / c2) + eps);
    }
  }
}

/* A stack of Dense layers with one of two loss heads.
 *   outputType 'bce'     -> last layer emits 1 logit, sigmoid + binary cross-entropy
 *   outputType 'softmax' -> last layer emits K logits, softmax + cross-entropy
 * The final layer is always constructed 'linear'; the head applies the
 * nonlinearity, so the "no activations" experiments are literally this class
 * with every hidden activation set to 'linear'. */
class MLP {
  constructor(sizes, activations, outputType) {
    this.layers = [];
    for (let i = 0; i < sizes.length - 1; i++) {
      this.layers.push(new Dense(sizes[i], sizes[i + 1], activations[i]));
    }
    this.outputType = outputType;
    this.t = 0;
  }

  logits(X) { let A = X; for (const L of this.layers) A = L.forward(A); return A; }

  predictProba(X) {
    const Z = this.logits(X);
    if (this.outputType === 'bce') return Z.map(r => sigmoid(r[0]));
    return Z.map(r => softmax(r));
  }

  // Y is [N] of 0/1 for bce, or [N] of class indices for softmax.
  trainStep(X, Y, lr) {
    this.t++;
    const Z = this.logits(X), N = X.length;
    const dLogits = new Array(N);
    let loss = 0;
    if (this.outputType === 'bce') {
      for (let n = 0; n < N; n++) {
        const p = sigmoid(Z[n][0]), y = Y[n];
        loss += -(y * Math.log(p + 1e-9) + (1 - y) * Math.log(1 - p + 1e-9));
        dLogits[n] = [(p - y) / N];
      }
    } else {
      for (let n = 0; n < N; n++) {
        const p = softmax(Z[n]), y = Y[n];
        loss += -Math.log(p[y] + 1e-9);
        const g = p.slice(); g[y] -= 1;
        for (let k = 0; k < g.length; k++) g[k] /= N;
        dLogits[n] = g;
      }
    }
    let dA = dLogits;
    for (let i = this.layers.length - 1; i >= 0; i--) dA = this.layers[i].backward(dA, lr, this.t);
    return loss / N;
  }

  accuracy(X, Y) {
    const P = this.predictProba(X);
    let ok = 0;
    for (let n = 0; n < X.length; n++) {
      if (this.outputType === 'bce') { if ((P[n] >= 0.5 ? 1 : 0) === Y[n]) ok++; }
      else { let arg = 0; for (let k = 1; k < P[n].length; k++) if (P[n][k] > P[n][arg]) arg = k; if (arg === Y[n]) ok++; }
    }
    return ok / X.length;
  }

  // Forward-only mean loss (no backprop, no optimizer update) — safe to call
  // for monitoring between real training steps.
  loss(X, Y) {
    const P = this.predictProba(X);
    let s = 0;
    if (this.outputType === 'bce') {
      for (let n = 0; n < X.length; n++) { const p = P[n], y = Y[n]; s += -(y * Math.log(p + 1e-9) + (1 - y) * Math.log(1 - p + 1e-9)); }
    } else {
      for (let n = 0; n < X.length; n++) s += -Math.log(P[n][Y[n]] + 1e-9);
    }
    return s / X.length;
  }
}
