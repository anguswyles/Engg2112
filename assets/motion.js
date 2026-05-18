/* ─────────────────────────────────────────────────────────────────────
 * Motion layer — count-up, scroll reveal, magnetic hover, cursor spotlight.
 * Dash auto-includes everything in /assets, so no script tag needed.
 * Uses IntersectionObserver + rAF — no JS framework, no jQuery.
 * Respects prefers-reduced-motion.
 * ──────────────────────────────────────────────────────────────────── */

(function () {
  'use strict';

  const PRM = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const EASE_OUT = (t) => 1 - Math.pow(1 - t, 3);   // cubic ease-out

  // Mark the root as JS-ready so scroll-reveal CSS engages.
  // Without this flag, .reveal-on-scroll elements stay fully visible.
  document.documentElement.classList.add('js-ready');

  // ── Count-up numerals ────────────────────────────────────────────────
  // Parses any [data-count] or .stat-value text, animates from 0 → target
  // when the element enters the viewport. Runs once.
  function animateNumber(el) {
    if (PRM) return;
    if (el.dataset.counted === '1') return;
    el.dataset.counted = '1';

    const raw = (el.dataset.count ?? el.textContent ?? '').trim();
    // Extract prefix, numeric body, suffix (e.g. "$1.2M", "98.7%", "297/301")
    const m = raw.match(/^([^\d\-+.]*)(-?\d[\d,]*(?:\.\d+)?(?:\/\d[\d,]*(?:\.\d+)?)?)([^]*)$/);
    if (!m) return;
    const [, prefix, body, suffix] = m;

    // Compound numbers like "297/301" — animate just the first half
    const slash = body.indexOf('/');
    if (slash !== -1) {
      const head = body.slice(0, slash);
      const tail = body.slice(slash);
      runCount(el, prefix, head, tail + suffix);
      return;
    }
    runCount(el, prefix, body, suffix);
  }

  function runCount(el, prefix, numStr, suffix) {
    const target = parseFloat(numStr.replace(/,/g, ''));
    if (isNaN(target)) return;
    const decimals = (numStr.split('.')[1] || '').length;
    const hasComma = numStr.includes(',');
    const dur = Math.min(1400, 700 + Math.abs(target) * 1.2);
    const start = performance.now();

    function fmt(v) {
      let s = v.toFixed(decimals);
      if (hasComma) {
        const [whole, dec] = s.split('.');
        s = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ',') + (dec ? '.' + dec : '');
      }
      return prefix + s + suffix;
    }

    function tick(now) {
      const t = Math.min(1, (now - start) / dur);
      const v = target * EASE_OUT(t);
      el.textContent = fmt(v);
      if (t < 1) requestAnimationFrame(tick);
      else el.textContent = prefix + numStr + suffix;
    }
    el.textContent = fmt(0);
    requestAnimationFrame(tick);
  }

  // ── Scroll reveal ────────────────────────────────────────────────────
  // Anything with .reveal-on-scroll fades in when it enters the viewport.
  // Sets --reveal-delay from data-delay (in ms) for elegant cascades.
  function setupRevealObserver() {
    if (PRM) {
      document.querySelectorAll('.reveal-on-scroll').forEach((el) => {
        el.classList.add('revealed');
      });
      return;
    }
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('revealed');
            obs.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: '0px 0px -60px 0px' },
    );
    document.querySelectorAll('.reveal-on-scroll:not(.revealed)').forEach((el) => {
      obs.observe(el);
    });
    return obs;
  }

  // ── Count-up trigger ─────────────────────────────────────────────────
  function setupCountObserver() {
    if (PRM) return;
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            animateNumber(entry.target);
            obs.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.4 },
    );
    document.querySelectorAll('.stat-value:not([data-counted])').forEach((el) => {
      obs.observe(el);
    });
    return obs;
  }

  // ── Cursor spotlight ─────────────────────────────────────────────────
  // Adds a subtle radial highlight that follows the cursor over .lift cards.
  // Pointer-events: none on the overlay so it doesn't block clicks.
  function setupSpotlight() {
    if (PRM) return;
    if (!window.matchMedia('(hover: hover) and (pointer: fine)').matches) return;

    document.addEventListener('mousemove', (e) => {
      const card = e.target.closest('.lift, .stat-value');
      if (!card) return;
      // throttle via rAF
      if (card._spotPending) return;
      card._spotPending = true;
      requestAnimationFrame(() => {
        const rect = card.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        card.style.setProperty('--spot-x', x + '%');
        card.style.setProperty('--spot-y', y + '%');
        card._spotPending = false;
      });
    }, { passive: true });
  }

  // ── Magnetic hover on nav links ──────────────────────────────────────
  // Subtle "pull toward cursor" effect, only on fine pointer devices.
  function setupMagnetic() {
    if (PRM) return;
    if (!window.matchMedia('(hover: hover) and (pointer: fine)').matches) return;

    let activeEl = null;
    document.addEventListener('mousemove', (e) => {
      const el = e.target.closest('.nav-link-item');
      if (!el) {
        if (activeEl) {
          activeEl.style.transform = '';
          activeEl = null;
        }
        return;
      }
      activeEl = el;
      const r = el.getBoundingClientRect();
      const dx = (e.clientX - (r.left + r.width / 2)) * 0.06;
      const dy = (e.clientY - (r.top + r.height / 2)) * 0.12;
      el.style.transform = `translate(${dx}px, ${dy}px)`;
    }, { passive: true });

    document.addEventListener('mouseleave', () => {
      if (activeEl) { activeEl.style.transform = ''; activeEl = null; }
    });
  }

  // ── Hero chart line draw-on ──────────────────────────────────────────
  // Plotly SVG paths get a stroke-dasharray reveal animation, once per element.
  function animatePlotPaths() {
    if (PRM) return;
    const plots = document.querySelectorAll('.js-plotly-plot:not([data-drawn])');
    plots.forEach((plot) => {
      const paths = plot.querySelectorAll('.scatterlayer .trace .js-line');
      if (!paths.length) return;
      plot.setAttribute('data-drawn', '1');
      paths.forEach((p, i) => {
        try {
          const len = p.getTotalLength();
          if (!len) return;
          p.style.strokeDasharray = len + ' ' + len;
          p.style.strokeDashoffset = len;
          p.style.transition = `stroke-dashoffset 1100ms cubic-bezier(0.77, 0, 0.175, 1) ${i * 120}ms`;
          // Force layout flush, then drop the offset to 0.
          // eslint-disable-next-line no-unused-expressions
          p.getBoundingClientRect();
          p.style.strokeDashoffset = '0';
        } catch (_) { /* ignore */ }
      });
    });
  }

  // ── Re-bind on Dash route changes ────────────────────────────────────
  // Dash swaps #page-content children when the user navigates between pages.
  // A MutationObserver re-runs our setup so newly mounted elements pick up
  // the count-up and reveal observers.
  function bindAll() {
    setupRevealObserver();
    setupCountObserver();
    // Plotly renders async; poll briefly for new charts.
    setTimeout(animatePlotPaths, 300);
    setTimeout(animatePlotPaths, 900);
    setTimeout(animatePlotPaths, 1800);
  }

  function init() {
    setupSpotlight();
    setupMagnetic();
    bindAll();

    const page = document.getElementById('page-content');
    if (page) {
      const mo = new MutationObserver(() => {
        bindAll();
      });
      mo.observe(page, { childList: true, subtree: true });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
