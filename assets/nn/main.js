/* main.js — wiring for the embedded post. No standalone theme toggle: the
 * charts read CSS custom properties and repaint when the blog's switcher
 * flips html.theme-dark. */
const RERUN = { s1: runS1, s2: runS2, s3: runS3, s4: runS4 };
function bind(id, fn) { const el = document.getElementById(id); if (el) el.addEventListener('click', fn); }
function redrawAll() { for (const k in REDRAW) try { REDRAW[k](); } catch (e) {} }
function initNN() {
  runAll();
  bind('s1-run', runS1); bind('s2-run', runS2); bind('s3-run', runS3); bind('s4-run', runS4);
  bind('run-all', runAll);
  document.querySelectorAll('input[data-sec]').forEach(inp => {
    inp.addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); const fn = RERUN[inp.dataset.sec]; if (fn) fn(); }
    });
  });
  // blog switcher toggles html.theme-dark; repaint the charts on that change
  new MutationObserver(redrawAll).observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
  let t;
  window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(redrawAll, 150); });
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initNN); else initNN();
