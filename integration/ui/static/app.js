/* Results browser for ORBIT x CheatBench runs.
 *
 * Vanilla ES2020, no framework, no build step, no network access beyond the local API.
 * Every piece of run data (transcripts, tool output, board posts, reasons, errors) is
 * inserted with textContent or text nodes. Nothing from the API ever reaches innerHTML.
 *
 * Sections
 *   1. DOM helpers and formatting
 *   2. API client
 *   3. Shared components
 *   4. Router, header and breadcrumbs
 *   5. View: runs table              #/
 *   6. View: run page                #/run/<run>
 *   7. View: transcript              #/run/<run>/episode/<episode>
 *   8. View: context windows         #/run/<run>/context?agent=<name>&call=<i>
 *   9. View: sample                  #/run/<run>/sample
 *  10. Boot
 */
'use strict';
(() => {

// =====================================================================================
// 1. DOM helpers and formatting
// =====================================================================================

const enc = encodeURIComponent;
const DASH = '—';
const REFUSAL = 'I cannot provide that response under the configured defense policy.';
const BOOL_PROPS = new Set(['checked', 'selected', 'disabled', 'hidden', 'open']);

/** Build an element. Strings in children become text nodes; attrs never take HTML. */
function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  let value;
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = Array.isArray(v) ? v.filter(Boolean).join(' ') : v;
      else if (k === 'text') el.textContent = String(v);
      else if (k === 'value') value = v;
      else if (k === 'dataset') Object.assign(el.dataset, v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else if (BOOL_PROPS.has(k)) el[k] = !!v;
      else el.setAttribute(k, v === true ? '' : String(v));
    }
  }
  appendAll(el, children);
  if (value !== undefined) el.value = value;
  return el;
}

function appendAll(el, children) {
  for (const c of children) {
    if (c == null || c === false || c === true) continue;
    if (Array.isArray(c)) appendAll(el, c);
    else el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

function debounce(fn, ms) {
  let t = null;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function fmtInt(n) {
  return n == null || Number.isNaN(Number(n)) ? DASH : Number(n).toLocaleString('en-US');
}

function fmtCompact(n) {
  if (n == null || Number.isNaN(Number(n))) return DASH;
  n = Number(n);
  const a = Math.abs(n);
  if (a >= 1e6) return (n / 1e6).toFixed(a >= 1e7 ? 1 : 2) + 'M';
  if (a >= 1e4) return Math.round(n / 1e3) + 'k';
  if (a >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(n);
}

function fmtSecs(s) {
  if (s == null || Number.isNaN(Number(s))) return DASH;
  s = Number(s);
  if (s < 60) return s.toFixed(s < 10 ? 2 : 1) + 's';
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  if (m < 60) return `${m}m ${String(r).padStart(2, '0')}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}

function fmtUsd(x) {
  if (x == null || Number.isNaN(Number(x))) return DASH;
  x = Number(x);
  return '$' + x.toFixed(x < 1 ? 3 : 2);
}

function fmtTime(iso) {
  if (!iso) return DASH;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso);
  return d.toISOString().replace('T', ' ').slice(0, 19) + 'Z';
}

function fmtScore(score) {
  if (!score || typeof score !== 'object') return DASH;
  const vals = Object.values(score);
  if (!vals.length) return DASH;
  return vals.map(v => (typeof v === 'number' ? String(+v.toFixed(3)) : String(v))).join(' / ');
}

function firstLine(text, max = 120) {
  const s = String(text == null ? '' : text);
  const line = (s.split('\n').find(l => l.trim()) || '').trim();
  return line.length > max ? line.slice(0, max - 1) + '…' : line;
}

function asText(v) {
  if (v == null) return '';
  return typeof v === 'string' ? v : JSON.stringify(v, null, 2);
}

function plural(n, word, many) {
  return `${fmtInt(n)} ${n === 1 ? word : (many || word + 's')}`;
}

// =====================================================================================
// 2. API client
// =====================================================================================

class ApiError extends Error {
  constructor(path, status, body) {
    super((body && body.error) || `HTTP ${status}`);
    this.path = path;
    this.status = status;
    this.body = body;
  }
}

const API_CACHE_MAX = 250;
const apiCache = new Map();  // path -> Promise, kept in least-recently-used order
// One model call can be hundreds of KB, so single calls get their own small cache.
const CALL_CACHE_MAX = 40;
const callCache = new Map();

function cachedFetch(cache, max, path) {
  if (cache.has(path)) {
    const p = cache.get(path);
    cache.delete(path);
    cache.set(path, p);
    return p;
  }
  const p = fetchJson(path);
  cache.set(path, p);
  p.catch(() => { if (cache.get(path) === p) cache.delete(path); });
  while (cache.size > max) cache.delete(cache.keys().next().value);
  return p;
}

function api(path) { return cachedFetch(apiCache, API_CACHE_MAX, path); }
function apiCall(path) { return cachedFetch(callCache, CALL_CACHE_MAX, path); }

async function fetchJson(path) {
  let res;
  try {
    res = await fetch(path, { headers: { Accept: 'application/json' } });
  } catch (e) {
    throw new ApiError(path, 0, { error: `network error: ${e && e.message}` });
  }
  const text = await res.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch (e) {
    body = { error: `non-JSON response (HTTP ${res.status})`, body: text.slice(0, 4000) };
  }
  if (!res.ok) throw new ApiError(path, res.status, body);
  return body;
}

const P = {
  runs: () => '/api/runs',
  run: r => `/api/runs/${enc(r)}`,
  transcript: (r, ep) => `/api/runs/${enc(r)}/transcript/${enc(ep)}`,
  sample: r => `/api/runs/${enc(r)}/sample`,
  context: r => `/api/runs/${enc(r)}/context`,
  call: (r, i) => `/api/runs/${enc(r)}/context/${i}`,
};

/** Hash routes. */
const R = {
  runs: q => '#/' + query(q),
  run: r => `#/run/${enc(r)}`,
  episode: (r, ep, q) => `#/run/${enc(r)}/episode/${enc(ep)}` + query(q),
  context: (r, q) => `#/run/${enc(r)}/context` + query(q),
  sample: r => `#/run/${enc(r)}/sample`,
};

function query(q) {
  if (!q) return '';
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(q)) if (v != null && v !== '') p.set(k, String(v));
  const s = p.toString();
  return s ? '?' + s : '';
}

// =====================================================================================
// 3. Shared components
// =====================================================================================

function badge(text, kind, title) {
  return h('span', { class: ['badge', kind ? 'b-' + kind : 'b-plain'], title }, text);
}

function muted(text, cls) {
  return h('span', { class: ['muted', cls] }, text);
}

function mono(text) {
  return h('span', { class: 'mono' }, text == null || text === '' ? DASH : text);
}

function link(href, text, cls) {
  return h('a', { href, class: cls }, text);
}

/** yes/no badge for a verdict boolean; null renders a dash. */
function yn(v, yesKind, noKind) {
  if (v === true) return badge('yes', yesKind || 'plain');
  if (v === false) return badge('no', noKind || 'plain');
  return muted(DASH);
}

/** Text in a <pre>; longer than `lines` lines (or `chars` chars) starts collapsed. */
function longText(text, opts = {}) {
  const { lines = 30, chars = 12000, cls = '' } = opts;
  const full = text == null ? '' : String(text);
  const all = full.split('\n');
  const pre = h('pre', { class: ['pre', cls] });
  if (all.length <= lines && full.length <= chars) {
    pre.textContent = full;
    return pre;
  }
  let preview = all.slice(0, lines).join('\n');
  if (preview.length > chars) preview = preview.slice(0, chars);
  const wrap = h('div', { class: 'longtext' });
  const btn = h('button', { type: 'button', class: 'expand' });
  let expanded = false;
  const sync = () => {
    pre.textContent = expanded ? full : preview + '\n…';
    btn.textContent = expanded
      ? 'collapse'
      : `expand: ${plural(all.length, 'line')}, ${fmtInt(full.length)} chars`;
    wrap.classList.toggle('collapsed', !expanded);
  };
  btn.addEventListener('click', () => { expanded = !expanded; sync(); });
  sync();
  return appendAll(wrap, [pre, btn]);
}

/** <details> whose body is built on first open. */
function lazyDetails(summary, build, opts = {}) {
  const { open = false, cls = '' } = opts;
  const d = h('details', { class: cls });
  d.appendChild(h('summary', null, summary));
  let built = false;
  const ensure = () => {
    if (built) return;
    built = true;
    appendAll(d, [build()]);
  };
  d.addEventListener('toggle', () => { if (d.open) ensure(); });
  if (open) { ensure(); d.open = true; }
  return d;
}

// Gemini appends an opaque base64 "thought signature" to its reasoning summary, after a blank
// line (or as the whole reasoning). It is not readable text, so it is split off and hidden.
const SIG_RE = /(?:^|\n\s*\n)\s*([A-Za-z0-9+/_-]{100,}={0,2})\s*$/;

function splitSignature(raw) {
  const s = raw == null ? '' : String(raw);
  const m = SIG_RE.exec(s);
  if (!m || !/[A-Z]/.test(m[1]) || !/[a-z]/.test(m[1]) || !/[0-9]/.test(m[1])) return { text: s, sig: '' };
  return { text: s.slice(0, m.index).replace(/\s+$/, ''), sig: m[1] };
}

function reasoningView(raw, lines) {
  const { text, sig } = splitSignature(raw);
  const label = text
    ? `reasoning (${fmtInt(text.length)} chars${sig ? `; ${fmtInt(sig.length)}-char opaque signature hidden` : ''})`
    : `reasoning: opaque provider signature only (${fmtInt(sig.length)} chars, no readable text)`;
  return lazyDetails(label, () => h('div', null,
    text ? longText(text, { lines }) : null,
    sig ? lazyDetails('show the opaque signature', () => longText(sig, { lines: 4, chars: 400, cls: 'sig' }), { cls: 'sub' }) : null),
  { cls: 'reasoning' });
}

/** A link into a view that reads the eval log, or a disabled button when the log is missing. */
function logLink(ok, evalLog, href, label, cls) {
  if (ok) return link(href, label, cls);
  return h('span', { class: [cls, 'disabled'], title: evalLog ? `eval log missing: ${evalLog}` : 'no eval log recorded for this run' }, label);
}

/** Definition list from [key, value] pairs; undefined values are skipped. */
function kv(pairs, cls) {
  const dl = h('dl', { class: ['kv', cls] });
  for (const [k, v] of pairs) {
    if (v === undefined) continue;
    dl.appendChild(h('dt', null, k));
    dl.appendChild(h('dd', null, v == null || v === '' ? muted(DASH) : v));
  }
  return dl;
}

function section(title, ...children) {
  return h('section', { class: 'card' }, h('h2', null, title), ...children);
}

function errorBox(err, context) {
  const isApi = err instanceof ApiError;
  const body = isApi ? err.body : { error: String((err && err.message) || err) };
  const title = isApi
    ? `API error${err.status ? ' ' + err.status : ''} on ${err.path}`
    : 'Error';
  const tb = body && typeof body.traceback === 'string' ? body.traceback : null;
  const shown = tb ? Object.assign({}, body, { traceback: '(below)' }) : body;
  return h('div', { class: 'error-box' },
    h('div', { class: 'error-title' }, context ? `${context}: ${title}` : title),
    h('pre', { class: 'pre' }, JSON.stringify(shown, null, 2)),
    tb ? h('pre', { class: 'pre traceback' }, tb) : null,
    !isApi && err && err.stack ? h('pre', { class: 'pre traceback' }, String(err.stack)) : null);
}

function select(options, value, onChange, attrs) {
  const s = h('select', attrs,
    options.map(o => (Array.isArray(o) ? h('option', { value: o[0] }, o[1]) : h('option', { value: o }, o))));
  s.value = value;
  s.addEventListener('change', () => onChange(s.value));
  return s;
}

function labeled(label, control) {
  return h('label', { class: 'field' }, h('span', { class: 'field-label' }, label), control);
}

/** Tool-call arguments. Bash commands and Python code render as code blocks. */
function argsView(fn, args) {
  let a = args;
  if (typeof a === 'string') {
    try {
      a = JSON.parse(a);
    } catch (e) {
      return h('div', { class: 'args' }, longText(args, { cls: 'code' }));
    }
  }
  if (a == null) return h('div', { class: 'args' }, muted('(no arguments)', 'small'));
  if (typeof a !== 'object' || Array.isArray(a)) {
    return h('div', { class: 'args' }, h('pre', { class: 'pre code' }, JSON.stringify(a, null, 2)));
  }
  const box = h('div', { class: 'args' });
  const simple = [];
  const keys = Object.keys(a);
  if (!keys.length) return appendAll(box, [muted('(no arguments)', 'small')]);
  for (const k of keys) {
    const v = a[k];
    const codeLang = typeof v === 'string'
      ? (k === 'command' || k === 'cmd' ? 'bash' : k === 'code' ? (fn === 'bash' ? 'bash' : 'python') : null)
      : null;
    if (codeLang) {
      box.appendChild(h('div', { class: 'arg' },
        h('div', { class: 'arg-key' }, k, ' ', muted(codeLang, 'small')),
        longText(v, { lines: 40, cls: `code lang-${codeLang}` })));
    } else if (typeof v === 'string' && (v.includes('\n') || v.length > 100)) {
      box.appendChild(h('div', { class: 'arg' },
        h('div', { class: 'arg-key' }, k),
        longText(v, { lines: 20, cls: 'argtext' })));
    } else {
      simple.push(`${k}: ${JSON.stringify(v)}`);
    }
  }
  if (simple.length) box.insertBefore(h('pre', { class: 'pre code args-simple' }, simple.join('\n')), box.firstChild);
  return box;
}

function isBlockedResult(rx) {
  return !!rx && typeof rx.error === 'string' && rx.error.startsWith('Blocked by defense');
}

/** Small badges for extra keys that the renderer does not handle itself. */
function extraBadges(extra, skip) {
  if (!extra || typeof extra !== 'object') return null;
  const out = [];
  for (const [k, v] of Object.entries(extra)) {
    if (skip.includes(k) || v == null || v === false || v === '') continue;
    const label = v === true ? k : `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`;
    out.push(badge(label.length > 60 ? label.slice(0, 59) + '…' : label, 'plain', label));
  }
  return out;
}

const VERDICT_FIELDS = [
  ['cheated', 'cheated', 'bad', 'good'],
  ['honey_pots_seen', 'honeypots seen', 'warn', ''],
  ['attempted_access', 'attempted access', 'bad', 'good'],
  ['extracted_submission', 'extracted submission', 'bad', ''],
  ['success_claim_seen_before_access', 'success claim seen before access', 'warn', ''],
  ['disclosed_access', 'disclosed access', 'good', ''],
  ['explicit_prohibition_violated', 'explicit prohibition violated', 'bad', ''],
];

function verdictBlock(v) {
  if (!v) return h('div', { class: 'verdict' }, muted('Not judged.'));
  return h('div', { class: 'verdict' },
    h('div', { class: 'verdict-flags' },
      VERDICT_FIELDS.map(([k, label, yk, nk]) =>
        h('span', { class: 'vf' }, h('span', { class: 'vf-label' }, label), yn(v[k], yk, nk)))),
    h('div', { class: 'verdict-meta' },
      muted(`judge ${v.judge || '?'} · ${fmtUsd(v.judge_cost_usd)}`),
      v.parse_ok === false ? badge('judge output did not parse', 'bad') : null,
      v.judge_error ? badge('judge error', 'bad', String(v.judge_error)) : null),
    v.judge_error ? h('pre', { class: 'pre errtext' }, String(v.judge_error)) : null,
    v.summary ? h('p', { class: 'verdict-summary' }, v.summary) : null);
}

function episodeRecord(ep) {
  const u = ep.usage || {};
  const d = ep.defense;
  return kv([
    ['agent', ep.orbit_agent || ep.agent],
    ['tool calls', fmtInt(ep.n_tool_calls)],
    ['turns', fmtInt(ep.n_turns)],
    ['tokens prompt', h('span', { title: fmtInt(u.input) }, fmtCompact(u.input))],
    ['tokens completion', h('span', { title: fmtInt(u.output) }, fmtCompact(u.output))],
    ['tokens cache read', h('span', { title: fmtInt(u.cache_read) }, fmtCompact(u.cache_read))],
    ['wall time', fmtSecs(ep.wall_time)],
    ['limit', ep.limit == null ? null : asText(ep.limit)],
    ['failure', ep.failure ? badge(ep.failure, 'bad') : null],
    ['ok', ep.ok == null ? null : yn(ep.ok, 'good', 'bad')],
    ['score', fmtScore(ep.score)],
    ['control', ep.control || undefined],
    ['defenses', ep.defenses && ep.defenses.length ? ep.defenses.join(', ') : undefined],
    ['defense checks / blocks', d ? defenseCounts(d) : undefined],
  ], 'kv-grid');
}

function defenseCounts(d) {
  const parts = [`${plural(d.checks || 0, 'check')}, ${plural(d.blocks || 0, 'block')}`];
  const bb = d.blocks_by_boundary || {};
  const cb = d.checks_by_boundary || {};
  const bounds = [...new Set([...Object.keys(cb), ...Object.keys(bb)])];
  const detail = bounds.map(b => `${b} ${fmtInt(bb[b] || 0)}/${fmtInt(cb[b] || 0)}`).join(' · ');
  return h('span', null, parts[0], d.blocks ? ' ' : '', d.blocks ? badge('blocked', 'blocked') : null,
    detail ? h('div', { class: 'muted small' }, 'blocks/checks: ', detail) : null);
}

// =====================================================================================
// 4. Router, header and breadcrumbs
// =====================================================================================

// Cleanups of the route on screen. Each route gets its own list, so a view that finishes
// loading after the user has left cannot leave a listener behind on the next view.
let cleanups = [];
let routeToken = 0;

function runCleanups(list) {
  for (const fn of list.splice(0)) { try { fn(); } catch (e) { /* ignore */ } }
}

function parseHash() {
  const raw = location.hash.replace(/^#/, '') || '/';
  const qi = raw.indexOf('?');
  const pathPart = qi >= 0 ? raw.slice(0, qi) : raw;
  const q = new URLSearchParams(qi >= 0 ? raw.slice(qi + 1) : '');
  let parts;
  try {
    parts = pathPart.split('/').filter(Boolean).map(decodeURIComponent);
  } catch (e) {
    return { view: 'notfound', raw };
  }
  if (!parts.length) return { view: 'runs', query: q };
  if (parts[0] === 'run' && parts.length >= 2) {
    const run = parts[1];
    if (parts.length === 2) return { view: 'run', run, query: q };
    if (parts[2] === 'episode' && parts.length >= 4) {
      return { view: 'episode', run, episode: parts.slice(3).join('/'), query: q };
    }
    if (parts[2] === 'context' && parts.length === 3) return { view: 'context', run, query: q };
    if (parts[2] === 'sample' && parts.length === 3) return { view: 'sample', run, query: q };
  }
  return { view: 'notfound', raw };
}

function renderShell() {
  const app = document.getElementById('app');
  app.replaceChildren(
    h('header', { class: 'topbar' },
      h('a', { class: 'brand', href: '#/' }, 'Results'),
      h('nav', { class: 'crumbs', id: 'crumbs', 'aria-label': 'Breadcrumbs' }),
      h('button', {
        type: 'button', class: 'btn btn-small', title: 'Clear cached API responses and reload this view',
        onclick: () => { apiCache.clear(); callCache.clear(); route(); },
      }, 'Reload')),
    h('main', { id: 'main' }));
}

function setCrumbs(r) {
  const nav = document.getElementById('crumbs');
  const items = [];
  const sep = () => h('span', { class: 'sep' }, '/');
  items.push(r.view === 'runs' ? h('span', { class: 'crumb current' }, 'All runs') : link('#/', 'All runs', 'crumb'));
  if (r.run) {
    items.push(sep());
    items.push(r.view === 'run' ? h('span', { class: 'crumb current' }, r.run) : link(R.run(r.run), r.run, 'crumb'));
    if (r.view === 'episode') items.push(sep(), h('span', { class: 'crumb current' }, `transcript ${r.episode}`));
    if (r.view === 'context') items.push(sep(), h('span', { class: 'crumb current' }, 'context windows'));
    if (r.view === 'sample') items.push(sep(), h('span', { class: 'crumb current' }, 'sample'));
    const tab = (href, label, view, needsLog) => h('a', {
      href, class: ['tab', r.view === view && 'active'], dataset: needsLog ? { needsLog: '1' } : null,
    }, label);
    items.push(h('span', { class: 'tabs' },
      tab(R.run(r.run), 'overview', 'run', false),
      tab(R.context(r.run), 'context', 'context', true),
      tab(R.sample(r.run), 'sample', 'sample', true)));
  }
  nav.replaceChildren(...items);
}

/** Context windows and the sample come from the eval log; grey those tabs out when it is missing. */
function markEvalLogTabs(r, token) {
  if (!r.run) return;
  api(P.runs()).then(runs => {
    if (token !== routeToken) return;
    const rs = (runs || []).find(x => x.run === r.run);
    if (!rs || rs.eval_log_exists) return;
    for (const a of document.querySelectorAll('#crumbs .tab[data-needs-log]')) {
      if (a.classList.contains('active')) continue;
      a.replaceWith(h('span', { class: 'tab disabled', title: 'eval log missing: no context windows or sample for this run' }, a.textContent));
    }
  }).catch(() => {});
}

const VIEWS = {
  runs: viewRuns,
  run: viewRun,
  episode: viewEpisode,
  context: viewContext,
  sample: viewSample,
  notfound: async (r, view) => {
    view.mounted(() => { document.title = 'Unknown page · results'; });
    return h('div', { class: 'page' },
      h('h1', null, 'Unknown page'),
      h('p', null, 'No view matches ', mono('#' + (r.raw || '')), '. '),
      link('#/', 'Back to all runs'));
  },
};

/**
 * Render the view for the current hash. A view is `async (route, view) => Node`.
 * view.mounted(fn) runs fn once the node is on screen, and only if the user is still on
 * this route. view.cleanup(fn) runs fn when the user leaves. Views keep every global side
 * effect (document.title, history.replaceState, document listeners) behind these two.
 */
async function route() {
  runCleanups(cleanups);
  const mine = [];
  cleanups = mine;
  const r = parseHash();
  const token = ++routeToken;
  setCrumbs(r);
  markEvalLogTabs(r, token);
  const main = document.getElementById('main');
  main.replaceChildren(h('div', { class: 'loading' }, 'Loading…'));
  const mounted = [];
  const view = { mounted: fn => mounted.push(fn), cleanup: fn => mine.push(fn) };
  try {
    const node = await VIEWS[r.view](r, view);
    if (token !== routeToken) { runCleanups(mine); return; }
    main.replaceChildren(node);
    window.scrollTo(0, 0);
    for (const fn of mounted) fn();
  } catch (e) {
    if (token !== routeToken) { runCleanups(mine); return; }
    document.title = 'Error · results';
    main.replaceChildren(h('div', { class: 'page' }, errorBox(e)));
  }
}

// =====================================================================================
// 5. View: runs table   #/
// =====================================================================================

const runsState = { q: '', topology: '', control: '', cheated: '', hideFailed: true, sortKey: null, sortDir: 1 };

function cmpNullsLast(a, b, dir) {
  const an = a == null || a === '';
  const bn = b == null || b === '';
  if (an && bn) return 0;
  if (an) return 1;
  if (bn) return -1;
  if (typeof a === 'boolean') a = +a;
  if (typeof b === 'boolean') b = +b;
  if (typeof a === 'number' && typeof b === 'number') return (a - b) * dir;
  return String(a).localeCompare(String(b), 'en', { numeric: true }) * dir;
}

const vget = k => row => (row.ep.verdict ? row.ep.verdict[k] : null);

const RUN_COLS = [
  { key: 'run', label: 'Run', sort: r => r.run.run },
  { key: 'control', label: 'Control', sort: r => r.run.control },
  { key: 'topology', label: 'Topology', sort: r => r.run.topology },
  { key: 'agent', label: 'Agent', sort: r => r.ep.orbit_agent },
  { key: 'role', label: 'Role', sort: r => r.ep.role, title: 'hidden role in a swarm run' },
  { key: 'calls', label: 'Calls', num: true, sort: r => r.ep.n_tool_calls, title: 'tool calls (turns in the tooltip)' },
  { key: 'blocks', label: 'Blocks / checks', num: true, sort: r => (r.ep.defense ? r.ep.defense.blocks : null) },
  { key: 'prompt', label: 'Prompt tok', num: true, sort: r => (r.ep.usage || {}).input },
  { key: 'completion', label: 'Compl. tok', num: true, sort: r => (r.ep.usage || {}).output },
  { key: 'wall', label: 'Wall', num: true, sort: r => r.ep.wall_time },
  { key: 'score', label: 'Score', num: true, sort: r => { const v = Object.values(r.ep.score || {}); return v.length ? v[0] : null; } },
  { key: 'failure', label: 'Failure', sort: r => r.ep.failure },
  { key: 'cheated', label: 'Cheated', sort: vget('cheated') },
  { key: 'honeypots', label: 'Honeypots', sort: vget('honey_pots_seen'), title: 'verdict: honeypots seen' },
  { key: 'attempted', label: 'Attempted', sort: vget('attempted_access'), title: 'verdict: attempted access' },
  { key: 'disclosed', label: 'Disclosed', sort: vget('disclosed_access'), title: 'verdict: disclosed access' },
  { key: 'judge', label: 'Judge $', num: true, sort: vget('judge_cost_usd') },
];

/**
 * Runs-table state from the URL: #/?q=&topology=&control=&cheated=&failed=1&sort=&dir=desc.
 * A bare #/ keeps the state from earlier in the session.
 */
function runsStateFromQuery(q, topologies, controls) {
  if (!q || ![...q.keys()].length) return;
  const pick = (v, allowed) => (allowed.includes(v) ? v : '');
  runsState.q = q.get('q') || '';
  runsState.topology = pick(q.get('topology') || '', topologies);
  runsState.control = pick(q.get('control') || '', controls);
  runsState.cheated = pick(q.get('cheated') || '', ['yes', 'no', 'none']);
  runsState.hideFailed = q.get('failed') !== '1';
  const sk = q.get('sort');
  runsState.sortKey = RUN_COLS.some(c => c.key === sk) ? sk : null;
  runsState.sortDir = q.get('dir') === 'desc' ? -1 : 1;
}

function runsQuery() {
  const s = runsState;
  return {
    q: s.q.trim() ? s.q : '', topology: s.topology, control: s.control, cheated: s.cheated,
    failed: s.hideFailed ? '' : '1', sort: s.sortKey || '', dir: s.sortKey && s.sortDir < 0 ? 'desc' : '',
  };
}

const ROLE_KIND = { worker: 'plain', regulator: 'good', disregulator: 'bad' };
function roleBadge(role) { return badge(role, ROLE_KIND[role] || 'plain', 'hidden role in this swarm run'); }

function runRowCells(row) {
  const { run, ep } = row;
  const v = ep.verdict;
  const d = ep.defense;
  const u = ep.usage || {};
  const td = (content, opts = {}) => h('td', { class: opts.num ? 'num' : opts.cls, title: opts.title }, content);
  const agentName = ep.orbit_agent || ep.episode || DASH;
  // A verdict on an episode with no transcript (harness crash) judged nothing: grey, not counted.
  const blind = !!v && v.cheated != null && !ep.has_transcript && !row.noEp;
  let cheatCell = blind
    ? h('span', { title: 'verdict on an episode with no transcript (harness crash); not counted in the summary' }, yn(v.cheated, 'plain', 'plain'))
    : yn(v ? v.cheated : null, 'bad', 'good');
  if (v && (v.parse_ok === false || v.judge_error)) {
    cheatCell = h('span', null, cheatCell, ' ', badge('judge issue', 'warn', String(v.judge_error || 'parse failed')));
  }
  const bb = d ? Object.entries(d.blocks_by_boundary || {}).map(([b, n]) => `${b}: ${n}`).join(', ') : '';
  return [
    td([link(R.run(run.run), run.run, 'run-link'), run.failed ? [' ', badge('failed', 'bad')] : null], { cls: 'wrap-cell' }),
    td(run.control || DASH),
    td(run.topology || DASH),
    td(row.noEp ? muted('no episodes')
      : ep.has_transcript ? link(R.episode(run.run, ep.episode), agentName, 'ep-link')
        : h('span', { title: 'no transcript on disk' }, agentName, ' ', muted('(no transcript)', 'small')),
    { title: ep.episode, cls: 'wrap-cell' }),
    td(ep.role ? roleBadge(ep.role) : muted(DASH)),
    td(fmtInt(ep.n_tool_calls), { num: true, title: `${fmtInt(ep.n_tool_calls)} tool calls, ${fmtInt(ep.n_turns)} turns` }),
    td(d ? h('span', { class: d.blocks ? 'has-blocks' : '' }, `${fmtInt(d.blocks)} / ${fmtInt(d.checks)}`) : DASH,
      { num: true, title: bb ? `blocks by boundary: ${bb}` : '' }),
    td(fmtCompact(u.input), { num: true, title: `${fmtInt(u.input)} prompt, ${fmtInt(u.cache_read)} cache read` }),
    td(fmtCompact(u.output), { num: true, title: fmtInt(u.output) }),
    td(fmtSecs(ep.wall_time), { num: true }),
    td(fmtScore(ep.score), { num: true }),
    td(ep.failure ? badge(ep.failure, 'bad') : muted(DASH), { title: ep.error ? firstLine(ep.error, 300) : '' }),
    td(cheatCell),
    td(yn(v ? v.honey_pots_seen : null, 'warn')),
    td(yn(v ? v.attempted_access : null, 'bad', 'good')),
    td(yn(v ? v.disclosed_access : null, 'good')),
    td(v ? fmtUsd(v.judge_cost_usd) : DASH, { num: true }),
  ];
}

async function viewRuns({ query: urlQ }, view) {
  const runs = await api(P.runs());
  let live = false;
  view.mounted(() => { live = true; document.title = 'Runs · results'; updateUrl(); });
  view.cleanup(() => { live = false; });
  const rows = [];
  for (const run of runs) {
    const eps = run.episodes && run.episodes.length ? run.episodes : [null];
    for (const ep of eps) rows.push({ run, ep: ep || {}, noEp: !ep });
  }
  const controls = [...new Set(runs.map(r => r.control).filter(Boolean))].sort();
  const topologies = [...new Set(runs.map(r => r.topology).filter(Boolean))].sort();
  runsStateFromQuery(urlQ, topologies, controls);

  const summary = h('div', { class: 'summary-line' });
  const tbody = h('tbody');
  const ths = RUN_COLS.map(col => {
    const th = h('th', { class: [col.num && 'num', 'sortable'], title: col.title || 'click to sort', tabindex: '0' }, col.label);
    const go = () => sortBy(col.key);
    th.addEventListener('click', go);
    th.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
    col.th = th;
    return th;
  });

  for (const row of rows) {
    const { run, ep } = row;
    const target = row.noEp || !ep.has_transcript ? R.run(run.run) : R.episode(run.run, ep.episode);
    const tr = h('tr', { class: [run.failed && 'failed'], title: run.failed ? 'failed run (harness crash)' : null }, runRowCells(row));
    tr.dataset.run = run.run;
    tr.addEventListener('click', e => {
      if (e.target.closest('a')) return;
      const sel = window.getSelection && window.getSelection();
      if (sel && String(sel).length) return;
      location.hash = target;
    });
    row.tr = tr;
    const v = ep.verdict || {};
    row.hay = [run.run, run.control, run.topology, run.model, run.monitor_model, (run.defenses || []).join(' '),
      ep.episode, ep.orbit_agent, ep.failure, ep.error && firstLine(ep.error, 200), v.summary, v.judge]
      .filter(Boolean).join(' \n ').toLowerCase();
    tbody.appendChild(tr);
  }

  function band() {
    let last = null;
    let alt = false;
    for (const tr of tbody.children) {
      if (tr.hidden) continue;
      if (tr.dataset.run !== last) { alt = !alt; last = tr.dataset.run; }
      tr.classList.toggle('band', alt);
    }
  }

  function apply() {
    const q = runsState.q.trim().toLowerCase();
    let shown = 0;
    let cheated = 0;
    let judged = 0;
    let judgedBlind = 0;
    let failedHidden = 0;
    const runsShown = new Set();
    for (const row of rows) {
      const v = row.ep.verdict;
      let ok = true;
      if (runsState.hideFailed && row.run.failed) { ok = false; failedHidden++; }
      if (ok && runsState.topology && row.run.topology !== runsState.topology) ok = false;
      if (ok && runsState.control && row.run.control !== runsState.control) ok = false;
      if (ok && runsState.cheated) {
        const c = v ? v.cheated : null;
        if (runsState.cheated === 'yes' && c !== true) ok = false;
        if (runsState.cheated === 'no' && c !== false) ok = false;
        if (runsState.cheated === 'none' && c != null) ok = false;
      }
      if (ok && q && !row.hay.includes(q)) ok = false;
      row.tr.hidden = !ok;
      if (ok && !row.noEp) {
        shown++;
        runsShown.add(row.run.run);
        if (v && v.cheated != null && !row.ep.has_transcript) judgedBlind++;
        else if (v && v.cheated != null) { judged++; if (v.cheated === true) cheated++; }
      }
    }
    band();
    summary.replaceChildren(
      h('strong', null, plural(shown, 'episode')), ` shown from ${plural(runsShown.size, 'run')} · `,
      h('strong', { class: cheated ? 'text-bad' : '' }, `${fmtInt(cheated)} cheated`),
      ` of ${fmtInt(judged)} judged`,
      judgedBlind ? ` (${plural(judgedBlind, 'verdict')} on episodes without a transcript not counted)` : '',
      runsState.hideFailed && failedHidden ? ` · ${plural(failedHidden, 'episode')} of failed runs hidden` : '');
    updateUrl();
  }

  function updateUrl() {
    if (!live) return;
    const target = R.runs(runsQuery());
    if (location.hash !== target) history.replaceState(null, '', target);
  }

  function sortBy(key) {
    if (runsState.sortKey === key) runsState.sortDir = -runsState.sortDir;
    else { runsState.sortKey = key; runsState.sortDir = 1; }
    applySort();
  }

  function applySort() {
    const col = RUN_COLS.find(c => c.key === runsState.sortKey);
    const order = rows.slice();
    if (col) order.sort((a, b) => cmpNullsLast(col.sort(a), col.sort(b), runsState.sortDir));
    for (const r of order) tbody.appendChild(r.tr);
    for (const c of RUN_COLS) {
      c.th.classList.toggle('sorted', c === col);
      c.th.dataset.dir = c === col ? (runsState.sortDir > 0 ? 'asc' : 'desc') : '';
    }
    band();
    updateUrl();
  }

  const search = h('input', {
    type: 'search', placeholder: 'Search run, control, agent, judge summary…', value: runsState.q,
    class: 'search', 'aria-label': 'Search',
  });
  search.addEventListener('input', debounce(() => { runsState.q = search.value; apply(); }, 80));
  const hideFailed = h('input', { type: 'checkbox', checked: runsState.hideFailed });
  hideFailed.addEventListener('change', () => { runsState.hideFailed = hideFailed.checked; apply(); });

  const filters = h('div', { class: 'filters' },
    search,
    labeled('topology', select([['', 'all'], ...topologies], runsState.topology, v => { runsState.topology = v; apply(); })),
    labeled('control', select([['', 'all'], ...controls], runsState.control, v => { runsState.control = v; apply(); })),
    labeled('cheated', select([['', 'any'], ['yes', 'yes'], ['no', 'no'], ['none', 'not judged']], runsState.cheated,
      v => { runsState.cheated = v; apply(); })),
    h('label', { class: 'field check' }, hideFailed, h('span', null, 'hide failed runs')));

  const table = h('table', { class: 'data runs-table' }, h('thead', null, h('tr', null, ths)), tbody);
  const page = h('div', { class: 'page wide' },
    h('div', { class: 'page-head' },
      h('h1', null, 'Runs'),
      muted(`${plural(runs.length, 'run')}, ${plural(rows.filter(r => !r.noEp).length, 'episode')}. Click a row for its transcript, a run name for its run page.`)),
    filters, summary,
    h('div', { class: 'table-wrap' }, table));
  if (runsState.sortKey) applySort();
  apply();
  return page;
}

// =====================================================================================
// 6. View: run page   #/run/<run>
// =====================================================================================

async function viewRun({ run }, view) {
  const d = await api(P.run(run));
  view.mounted(() => { document.title = `${run} · results`; });
  const rj = d.run_json || {};
  const eps = d.episodes || [];
  const logOk = !!d.eval_log_exists;
  const epFailures = [...new Set(eps.map(e => e.failure).filter(Boolean))];

  const head = h('div', { class: 'page-head' },
    h('h1', null, run),
    h('div', { class: 'chips' },
      d.failed ? badge('failed run', 'bad')
        : epFailures.length ? badge(`episode failure: ${epFailures.join(', ')}`, 'warn', 'not a harness crash, but an episode ended with a failure')
          : badge('completed', 'good'),
      badge(d.topology || '?', 'plain'),
      badge(`control: ${d.control || 'baseline'}`, 'def')),
    h('div', { class: 'links' },
      eps.map(ep => (ep.has_transcript
        ? link(R.episode(run, ep.episode), `transcript ${ep.episode}`, 'btn')
        : h('span', { class: 'btn disabled', title: 'no transcript on disk' }, `transcript ${ep.episode}`))),
      logLink(logOk, d.eval_log, R.context(run), 'context windows', 'btn'),
      logLink(logOk, d.eval_log, R.sample(run), 'sample and workspace', 'btn')));

  const meta = section('Run metadata', kv([
    ['task', d.task],
    ['model', d.model],
    ['topology', d.topology],
    ['agents', (d.agents || []).join(', ')],
    ['control', d.control],
    ['monitor model', d.monitor_model],
    ['eval id', rj.eval_id ? mono(rj.eval_id) : null],
    ['run id', rj.run_id],
    ['variant', rj.variant || null],
    ['ORBIT version', rj.orbit_version],
    ['Inspect version', rj.inspect_version],
  ], 'kv-grid'), kv([
    ['eval log', h('span', null, mono(d.eval_log), ' ', d.eval_log ? (d.eval_log_exists ? badge('present', 'good') : badge('missing', 'bad')) : null)],
    ['task root', rj.task_root ? mono(rj.task_root) : null],
    ['deviations', rj.deviations && rj.deviations.length
      ? h('ul', { class: 'plain-list' }, rj.deviations.map(x => h('li', null, mono(x)))) : null],
  ]));

  const defs = Array.isArray(rj.defenses) ? rj.defenses : [];
  const defSection = section('Defenses', defs.length ? defensesTable(defs) : muted('No defenses configured (baseline).'));

  const epSection = section(`Episodes (${eps.length})`,
    eps.length ? eps.map(ep => episodeCard(run, ep, logOk, d.eval_log)) : muted('No episodes recorded.'));

  const boardSection = section('Board messages', communicationView(d.communication, run));
  const evidence = section('Defense evidence', defenseEvidence(d.defense, defs));
  const swarmSection = rj.swarm ? section('Swarm', swarmView(rj.swarm, eps)) : null;

  return h('div', { class: 'page' }, head, meta, defSection, swarmSection, epSection, boardSection, evidence);
}

// Composition, hidden roles and the team's own report of a swarm run, beside the judge's verdicts.
function swarmView(sw, eps) {
  const verdictOf = name => { const ep = eps.find(e => e.orbit_agent === name); return ep && ep.verdict ? ep.verdict.cheated : null; };
  const roles = Object.entries(sw.roles || {}).sort(([a], [b]) => a.localeCompare(b));
  const anyCheated = eps.some(e => e.verdict && e.verdict.cheated === true);
  const judged = eps.filter(e => e.verdict && e.verdict.cheated != null).length;
  const table = h('table', { class: 'data' },
    h('thead', null, h('tr', null, ['agent', 'hidden role', 'judge: cheated'].map(c => h('th', null, c)))),
    h('tbody', null, roles.map(([name, role]) => h('tr', null,
      h('td', null, name), h('td', null, roleBadge(role)), h('td', null, yn(verdictOf(name), 'bad', 'good'))))));
  const reportVerdict = sw.swarm_report_cheated;
  const agree = reportVerdict == null || !judged ? null : (reportVerdict === anyCheated);
  return h('div', null,
    kv([
      ['composition', h('span', null, mono(sw.composition || DASH), ' ', muted(`${sw.regulators} regulators, ${sw.disregulators} disregulators, ${sw.workers} workers; seed ${sw.seed}`, 'small'))],
      ['judge: any agent cheated', judged ? yn(anyCheated, 'bad', 'good') : muted('not judged')],
      ['team report says cheated', sw.swarm_report == null ? badge('no report file', 'warn') : yn(reportVerdict, 'bad', 'good')],
      ['report agrees with judge', agree == null ? muted(DASH) : yn(agree, 'good', 'bad')],
    ], 'kv-grid'),
    sw.swarm_report ? h('div', null, h('h3', null, 'swarm_report.txt'), h('pre', { class: 'pre' }, String(sw.swarm_report))) : null,
    h('h3', null, 'Roles'), table);
}

function defensesTable(defs) {
  const cols = ['name', 'defense_type', 'defense_category', 'invocation', 'model_override', 'scope', 'fail_mode',
    'detection_threshold', 'max_resamples', 'enabled'];
  const table = h('table', { class: 'data' },
    h('thead', null, h('tr', null, cols.map(c => h('th', null, c.replace(/_/g, ' '))), h('th', null, 'config keys'))),
    h('tbody', null, defs.map(df => {
      const d = typeof df === 'object' && df ? df : { name: String(df) };
      const cfgKeys = d.config && typeof d.config === 'object' ? Object.keys(d.config) : [];
      return h('tr', null,
        cols.map(c => h('td', { class: c === 'name' || c === 'defense_type' ? 'mono' : '' },
          d[c] == null || d[c] === '' ? muted(DASH) : asText(d[c]))),
        h('td', null, cfgKeys.length ? cfgKeys.map(k => badge(k, 'plain')) : muted('none')));
    })));
  const configs = defs.filter(d => d && typeof d === 'object').map(d => {
    const extra = {};
    for (const k of ['config', 'target_agents', 'vaccination_prompt', 'guardian_spec']) {
      if (d[k] != null && !(Array.isArray(d[k]) && !d[k].length) && d[k] !== '' &&
          !(typeof d[k] === 'object' && !Array.isArray(d[k]) && !Object.keys(d[k]).length)) extra[k] = d[k];
    }
    if (!Object.keys(extra).length) return null;
    return lazyDetails(`${d.name} configuration`, () => configView(extra), { cls: 'sub' });
  });
  return [h('div', { class: 'table-wrap' }, table), configs];
}

/** Show a config object: string values as text blocks, others as JSON. */
function configView(obj) {
  const box = h('div', { class: 'config' });
  const walk = (o, prefix) => {
    for (const [k, v] of Object.entries(o)) {
      const key = prefix ? `${prefix}.${k}` : k;
      if (v && typeof v === 'object' && !Array.isArray(v) && Object.values(v).some(x => typeof x === 'string' && x.length > 60)) {
        walk(v, key);
      } else {
        box.appendChild(h('div', { class: 'cfg-key mono' }, key));
        box.appendChild(longText(typeof v === 'string' ? v : JSON.stringify(v, null, 2), { lines: 40 }));
      }
    }
  };
  walk(obj, '');
  return box;
}

function episodeCard(run, ep, logOk = true, evalLog = null) {
  return h('div', { class: 'ep-card' },
    h('div', { class: 'ep-card-head' },
      h('h3', null, ep.episode),
      ep.failure ? badge(ep.failure, 'bad') : null,
      h('span', { class: 'spacer' }),
      ep.has_transcript ? link(R.episode(run, ep.episode), 'transcript', 'btn btn-small') : muted('no transcript', 'small'),
      ep.orbit_agent ? logLink(logOk, evalLog, R.context(run, { agent: ep.orbit_agent }), `context of ${ep.orbit_agent}`, 'btn btn-small') : null),
    episodeRecord(ep),
    verdictBlock(ep.verdict),
    ep.error ? lazyDetails('error', () => longText(ep.error, { lines: 30 }), { cls: 'sub' }) : null,
    ep.final_text ? lazyDetails('final text', () => longText(ep.final_text, { lines: 40 }), { cls: 'sub' }) : null);
}

function communicationView(comm, run) {
  const samples = Object.entries(comm || {});
  if (!samples.length) return muted('No board communication recorded for this run.');
  return samples.map(([sample, c]) => {
    const msgs = ((c && c.messages) || []).slice().sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0));
    const events = (c && c.events) || [];
    const kinds = {};
    for (const e of events) kinds[e.kind || '?'] = (kinds[e.kind || '?'] || 0) + 1;
    return h('div', { class: 'board' },
      h('div', { class: 'board-head' },
        h('h3', null, `sample ${sample}`),
        muted(`${plural(msgs.length, 'message')} · ${plural(events.length, 'event')}` +
          (events.length ? ` (${Object.entries(kinds).map(([k, n]) => `${k} ${n}`).join(', ')})` : ''))),
      msgs.length ? h('ol', { class: 'board-msgs' }, msgs.map(m => h('li', { class: 'board-msg' },
        h('div', { class: 'bm-head' },
          h('span', { class: 'bm-seq' }, `#${m.sequence ?? '?'}`),
          h('span', { class: 'bm-sender' }, m.sender || '?'),
          h('span', { class: 'muted' }, ' → ', (m.recipients || []).join(', ') || 'all'),
          m.turn != null ? muted(` · turn ${m.turn}`) : null,
          m.channel_id ? muted(` · ${m.channel_id}`) : null,
          h('span', { class: 'spacer' }),
          m.id ? h('span', { class: 'tc-id' }, m.id) : null),
        longText(m.content, { lines: 25, cls: 'msg-text' }))))
        : muted('No messages were posted.'),
      events.length ? lazyDetails(`board events (${events.length})`, () => h('div', { class: 'table-wrap' },
        h('table', { class: 'data compact' },
          h('thead', null, h('tr', null, ['seq', 'kind', 'agent', 'message ids', 'tool call', 'channel'].map(x => h('th', null, x)))),
          h('tbody', null, events.slice().sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0)).map(e => h('tr', null,
            h('td', { class: 'num' }, String(e.sequence ?? '')),
            h('td', null, e.kind || ''),
            h('td', null, e.agent_id || ''),
            h('td', { class: 'mono small' }, (e.message_ids || []).join(', ') || DASH),
            h('td', { class: 'mono small' }, e.tool_call_id || DASH),
            h('td', null, e.channel_id || '')))))), { cls: 'sub' }) : null);
  });
}

function defenseEvidence(defense, runDefs) {
  const samples = Object.entries(defense || {});
  if (!samples.length) return muted('No defense evidence recorded for this run.');
  const typeOf = {};
  for (const d of runDefs || []) if (d && d.name) typeOf[d.name] = d.defense_type;
  return samples.map(([sample, dv]) => {
    const iv = (dv && dv.interventions) || [];
    const log = (dv && dv.log) || {};
    for (const d of (dv && dv.defenses) || []) if (d && d.name && !typeOf[d.name]) typeOf[d.name] = d.defense_type;
    const resolved = log.resolved_defenses || {};
    for (const [name, rd] of Object.entries(resolved)) {
      if (!typeOf[name] && rd && rd.config) typeOf[name] = rd.config.defense_type;
    }
    return h('div', { class: 'evidence' },
      h('h3', null, `sample ${sample}`),
      defenseCountsTable(iv, log, typeOf),
      kv([
        ['total activations', fmtInt(log.total_activations)],
        ['blocks in log', Array.isArray(log.blocks) ? fmtInt(log.blocks.length) : undefined],
        ['detections in log', Array.isArray(log.detections) ? fmtInt(log.detections.length) : undefined],
        ['true / false positives', log.true_positives != null ? `${fmtInt(log.true_positives)} / ${fmtInt(log.false_positives)}` : undefined],
        ['vaccinations', log.vaccinations ? fmtInt(Object.keys(log.vaccinations).length) : undefined],
        ['vaccination injections', log.vaccination_injections ? fmtInt(Object.keys(log.vaccination_injections).length) : undefined],
      ], 'kv-grid'),
      (log.vaccinations && Object.keys(log.vaccinations).length) || (log.vaccination_injections && Object.keys(log.vaccination_injections).length)
        ? lazyDetails('vaccinations (JSON)', () => h('pre', { class: 'pre' },
          JSON.stringify({ vaccinations: log.vaccinations, vaccination_injections: log.vaccination_injections }, null, 2)), { cls: 'sub' })
        : null,
      interventionsTable(iv));
  });
}

function defenseCountsTable(iv, log, typeOf) {
  const BOUNDS = ['model_input', 'model_output', 'tool_call'];
  const per = new Map();
  for (const x of iv) {
    const name = x.defense || '?';
    if (!per.has(name)) per.set(name, { checks: 0, blocks: 0, by: {} });
    const p = per.get(name);
    const b = x.boundary || '?';
    p.by[b] = p.by[b] || { checks: 0, blocks: 0 };
    p.checks++; p.by[b].checks++;
    if (x.allow === false) { p.blocks++; p.by[b].blocks++; }
  }
  const activity = log.activity || {};
  for (const name of Object.keys(activity)) if (!per.has(name)) per.set(name, { checks: 0, blocks: 0, by: {} });
  if (!per.size) return muted('No interventions recorded.');
  const extraBounds = [...new Set([...per.values()].flatMap(p => Object.keys(p.by)))].filter(b => !BOUNDS.includes(b));
  const bounds = [...BOUNDS, ...extraBounds];
  return h('div', null, h('div', { class: 'muted small' }, 'Boundary columns show blocks / checks, counted from the interventions below.'),
    h('div', { class: 'table-wrap' }, h('table', { class: 'data' },
    h('thead', null, h('tr', null,
      ['defense', 'type', 'checks', 'blocks'].map(x => h('th', { class: x === 'checks' || x === 'blocks' ? 'num' : '' }, x)),
      bounds.map(b => h('th', { class: 'num', title: `blocks / checks at ${b}` }, b)),
      h('th', { class: 'num' }, 'monitor model calls'),
      h('th', null, 'per agent (activity log)'))),
    h('tbody', null, [...per.entries()].map(([name, p]) => {
      const act = activity[name] || {};
      const agents = act.agents || {};
      return h('tr', null,
        h('td', { class: 'mono' }, name),
        h('td', { class: 'mono' }, typeOf[name] || DASH),
        h('td', { class: 'num' }, fmtInt(p.checks)),
        h('td', { class: 'num' }, p.blocks ? badge(fmtInt(p.blocks), 'blocked') : '0'),
        bounds.map(b => h('td', { class: 'num' }, p.by[b] ? `${fmtInt(p.by[b].blocks)} / ${fmtInt(p.by[b].checks)}` : DASH)),
        h('td', { class: 'num' }, fmtInt(act.model_calls)),
        h('td', { class: 'small per-agent' }, Object.keys(agents).length
          ? Object.entries(agents).map(([a, s]) => h('div', null,
            h('span', { class: 'mono' }, a), ': ', `${plural(s.checks || 0, 'check')}, ${plural(s.blocks || 0, 'block')}, ${plural(s.model_calls || 0, 'call')}`))
          : muted(DASH)));
    })))));
}

function interventionsTable(iv) {
  if (!iv.length) return muted('No interventions.');
  const st = { boundary: '', allow: '', defense: '', agent: '', q: '' };
  const bounds = [...new Set(iv.map(x => x.boundary).filter(Boolean))].sort();
  const defs = [...new Set(iv.map(x => x.defense).filter(Boolean))].sort();
  const agents = [...new Set(iv.map(x => x.agent).filter(Boolean))].sort();
  const count = h('span', { class: 'muted' });
  const rows = iv.map((x, i) => {
    const ref = x.tool_call_id || x.message_id || '';
    const tr = h('tr', { class: x.allow === false ? 'row-blocked' : '' },
      h('td', { class: 'num' }, String(i + 1)),
      h('td', { class: 'num' }, x.turn == null ? DASH : String(x.turn)),
      h('td', null, x.agent || DASH),
      h('td', { class: 'mono' }, x.defense || DASH),
      h('td', null, x.boundary || DASH),
      h('td', null, x.allow === false ? badge('BLOCK', 'blocked') : x.allow === true ? badge('allow', 'good') : muted(String(x.allow))),
      h('td', { class: 'mono' }, x.tool || x.tool_name || DASH),
      h('td', { class: 'reason' }, x.reason ? longText(x.reason, { lines: 6, chars: 1500 }) : muted(DASH)),
      h('td', { class: 'mono small', title: x.tool_call_id ? 'tool call id' : 'message id' }, ref || DASH));
    const hay = [x.reason, x.agent, x.defense, x.boundary, x.tool, ref].filter(Boolean).join(' ').toLowerCase();
    return { x, tr, hay };
  });
  const tbody = h('tbody', null, rows.map(r => r.tr));
  function apply() {
    const q = st.q.trim().toLowerCase();
    let n = 0;
    let blocks = 0;
    for (const r of rows) {
      const x = r.x;
      const ok = (!st.boundary || x.boundary === st.boundary) &&
        (!st.allow || (st.allow === 'block' ? x.allow === false : x.allow !== false)) &&
        (!st.defense || x.defense === st.defense) &&
        (!st.agent || x.agent === st.agent) &&
        (!q || r.hay.includes(q));
      r.tr.hidden = !ok;
      if (ok) { n++; if (x.allow === false) blocks++; }
    }
    count.textContent = `${fmtInt(n)} of ${fmtInt(rows.length)} interventions shown, ${fmtInt(blocks)} blocks`;
  }
  const search = h('input', { type: 'search', class: 'search small-search', placeholder: 'search reasons…', 'aria-label': 'Search interventions' });
  search.addEventListener('input', debounce(() => { st.q = search.value; apply(); }, 80));
  const filters = h('div', { class: 'filters' },
    labeled('boundary', select([['', 'all'], ...bounds], '', v => { st.boundary = v; apply(); })),
    labeled('decision', select([['', 'all'], ['block', 'blocked'], ['allow', 'allowed']], '', v => { st.allow = v; apply(); })),
    defs.length > 1 ? labeled('defense', select([['', 'all'], ...defs], '', v => { st.defense = v; apply(); })) : null,
    agents.length > 1 ? labeled('agent', select([['', 'all'], ...agents], '', v => { st.agent = v; apply(); })) : null,
    search, count);
  apply();
  return h('div', { class: 'interventions' },
    h('h4', null, 'Interventions'),
    filters,
    h('div', { class: 'table-wrap' }, h('table', { class: 'data compact' },
      h('thead', null, h('tr', null, ['#', 'turn', 'agent', 'defense', 'boundary', 'decision', 'tool', 'reason', 'message / call id']
        .map(x => h('th', { class: x === '#' || x === 'turn' ? 'num' : '' }, x)))),
      tbody)));
}

// =====================================================================================
// 7. View: transcript   #/run/<run>/episode/<episode>
// =====================================================================================

const STEP_KINDS = [
  ['', 'all steps'],
  ['agent', 'agent steps'],
  ['user', 'user steps'],
  ['system', 'system steps'],
  ['tools', 'steps with tool calls'],
  ['flagged', 'any flag below'],
  ['blocked', 'BLOCKED'],
  ['refusal', 'REFUSAL'],
  ['withheld', 'WITHHELD'],
  ['notice', 'provider-only notices'],
  ['error', 'tool errors'],
];

const FLAG_BADGE = {
  blocked: ['BLOCKED', 'blocked'],
  refusal: ['REFUSAL', 'refusal'],
  'refusal-text': ['REFUSAL TEXT (model-written)', 'plain'],
  withheld: ['WITHHELD', 'withheld'],
  notice: ['NOTICE', 'notice'],
  trailing: ['TRAILING-TURN FIX', 'notice'],
  error: ['ERROR', 'bad'],
};

function stepPreview(s, msg) {
  if (msg.trim()) return firstLine(msg, 100);
  const tc = (s.tool_calls || [])[0];
  if (tc) {
    let a = tc.arguments;
    if (a && typeof a === 'object') a = a.command || a.code || a.message || a.answer || JSON.stringify(a);
    return `${tc.function_name}: ${firstLine(asText(a), 90)}`;
  }
  if (s.reasoning_content) {
    const r = splitSignature(s.reasoning_content).text;
    return r ? '(reasoning) ' + firstLine(r.replace(/\*\*/g, ''), 90) : '(reasoning signature only)';
  }
  return '(empty)';
}

function resultView(r, orphan) {
  const rx = r.extra || {};
  const blocked = isBlockedResult(rx);
  const isErr = !blocked && (!!rx.is_error || !!rx.error);
  const withheld = !!rx.withheld_from_model_by_defense;
  const content = asText(r.content);
  return h('div', { class: ['result', blocked && 'is-blocked', isErr && 'is-error', withheld && 'is-withheld'] },
    h('div', { class: 'res-head' },
      h('span', { class: 'res-label' }, orphan ? 'result without a matching call' : 'result'),
      blocked ? badge('BLOCKED', 'blocked') : null,
      isErr ? badge('ERROR', 'bad') : null,
      withheld ? badge('WITHHELD', 'withheld', 'withheld from the model by a defense') : null,
      extraBadges(rx, ['is_error', 'error', 'withheld_from_model_by_defense', 'withheld_text']),
      h('span', { class: 'spacer' }),
      orphan && r.source_call_id ? h('span', { class: 'tc-id' }, r.source_call_id) : null),
    rx.error ? h('pre', { class: 'pre errtext' }, String(rx.error)) : null,
    rx.withheld_text ? lazyDetails('original content (never reached the model)', () => longText(asText(rx.withheld_text), { lines: 30 }), { cls: 'sub' }) : null,
    content ? longText(content, { lines: 30 }) : muted('(empty output)', 'small block'));
}

function toolCallView(fn, id, args, resultEls) {
  return h('div', { class: 'toolcall' },
    h('div', { class: 'tc-head' },
      h('span', { class: 'tc-label' }, 'tool call'),
      h('code', { class: 'tc-fn' }, fn || '?'),
      h('span', { class: 'spacer' }),
      id ? h('span', { class: 'tc-id' }, id) : null),
    argsView(fn, args),
    resultEls === null ? null
      : resultEls.length ? h('div', { class: 'tc-results' }, resultEls)
        : muted('(no result recorded)', 'small block'));
}

function renderStep(s) {
  const ex = s.extra || {};
  const results = (s.observation && s.observation.results) || [];
  const src = s.source || 'unknown';
  const msg = asText(s.message);
  const flags = new Set();
  if (ex.orbit_delivery_notice) flags.add('notice');
  if (ex.orbit_trailing_turn_fix) flags.add('trailing');
  if (ex.withheld_from_model_by_defense) flags.add('withheld');
  if (ex.defense_refusal) flags.add('refusal');
  else if (ex.refusal_text_written_by_model || (src === 'agent' && msg.trim() === REFUSAL)) flags.add('refusal-text');
  for (const r of results) {
    const rx = r.extra || {};
    if (isBlockedResult(rx)) flags.add('blocked');
    else if (rx.is_error || rx.error) flags.add('error');
    if (rx.withheld_from_model_by_defense) flags.add('withheld');
  }
  const providerOnly = flags.has('notice') || flags.has('trailing');
  const el = h('article', {
    class: ['step', 'src-' + src, providerOnly && 'provider-only', flags.has('refusal') && 'is-refusal',
      flags.has('blocked') && 'has-blocked', ex.withheld_from_model_by_defense && 'is-withheld'],
    id: `step-${s.step_id}`,
  });
  el.appendChild(h('header', { class: 'step-head' },
    h('span', { class: 'step-id' }, `#${s.step_id}`),
    h('span', { class: 'step-src' }, src),
    s.model_name ? muted(s.model_name, 'small') : null,
    [...flags].map(f => badge(...FLAG_BADGE[f])),
    extraBadges(ex, ['orbit_delivery_notice', 'orbit_trailing_turn_fix', 'withheld_from_model_by_defense', 'withheld_text',
      'defense_refusal', 'blocked_output_message_id', 'refusal_text_written_by_model']),
    h('span', { class: 'spacer' }),
    s.llm_call_count != null ? muted(plural(s.llm_call_count, 'model call'), 'small') : null));
  if (providerOnly) {
    el.appendChild(h('div', { class: 'caption caption-notice' },
      flags.has('trailing')
        ? 'Provider-only: continue prompt appended by the integration (trailing-turn fix), not part of the agent conversation.'
        : 'Provider-only: ORBIT delivery notice placed in the model input, not part of the agent conversation.'));
  }
  if (ex.withheld_from_model_by_defense) {
    el.appendChild(h('div', { class: 'caption caption-withheld' }, 'WITHHELD: a defense withheld this message from the model; the model received the placeholder shown below.'));
    if (ex.withheld_text) el.appendChild(lazyDetails('original content (never reached the model)', () => longText(asText(ex.withheld_text), { lines: 30 }), { cls: 'sub' }));
  }
  if (s.reasoning_content) el.appendChild(reasoningView(s.reasoning_content, 60));
  if (flags.has('refusal')) {
    el.appendChild(h('div', { class: 'caption caption-refusal' },
      'A defense blocked this completion and substituted the refusal sentence.'));
  } else if (flags.has('refusal-text')) {
    el.appendChild(h('div', { class: 'caption caption-notice' },
      'The model wrote the refusal sentence itself; no defense blocked this completion.'));
  }
  // Leading blank lines are dropped for display only.
  if (msg.trim()) el.appendChild(longText(msg.replace(/^\s*\n/, ''), { cls: 'msg-text' }));
  else if (!(s.tool_calls || []).length && !s.reasoning_content) el.appendChild(muted('(empty message)', 'small block'));

  const used = new Set();
  for (const tc of s.tool_calls || []) {
    const rs = results.filter(r => r.source_call_id === tc.tool_call_id);
    rs.forEach(r => used.add(r));
    el.appendChild(toolCallView(tc.function_name, tc.tool_call_id, tc.arguments, rs.map(r => resultView(r, false))));
  }
  for (const r of results) if (!used.has(r)) el.appendChild(resultView(r, true));

  const hay = [src, msg, splitSignature(s.reasoning_content).text,
    (s.tool_calls || []).map(tc => `${tc.function_name} ${asText(tc.arguments)}`).join('\n'),
    results.map(r => `${asText(r.content)} ${(r.extra && r.extra.error) || ''}`).join('\n')]
    .filter(Boolean).join('\n').toLowerCase();
  return { step: s, el, flags, hay, src, preview: stepPreview(s, msg), hasTools: !!(s.tool_calls || []).length };
}

function stepMatchesKind(e, k) {
  switch (k) {
    case '': return true;
    case 'agent': case 'user': case 'system': return e.src === k;
    case 'tools': return e.hasTools;
    case 'flagged': return e.flags.size > 0;
    case 'notice': return e.flags.has('notice') || e.flags.has('trailing');
    default: return e.flags.has(k);
  }
}

async function viewEpisode({ run, episode, query: q }, view) {
  const [d, runs] = await Promise.all([
    api(P.transcript(run, episode)),
    api(P.runs()).catch(() => null),
  ]);
  const ep = d.episode || {};
  const t = d.transcript || {};
  const traj = t.trajectory || {};
  const steps = traj.steps || [];
  const orbit = t.orbit || {};

  const runSum = runs && runs.find(r => r.run === run);
  // Prev / next step only through episodes that have a transcript to show.
  const epNames = runSum ? (runSum.episodes || []).filter(e => e.has_transcript || e.episode === episode).map(e => e.episode) : [];
  const logOk = !runSum || !!runSum.eval_log_exists;
  const evalLog = runSum ? runSum.eval_log : null;
  const pos = epNames.indexOf(episode);
  const prev = pos > 0 ? epNames[pos - 1] : null;
  const next = pos >= 0 && pos < epNames.length - 1 ? epNames[pos + 1] : null;
  const navBtn = (name, label) => (name
    ? link(R.episode(run, name), label, 'btn btn-small')
    : h('span', { class: 'btn btn-small disabled' }, label));

  const entries = steps.map(renderStep);
  const counts = { tools: 0, blocked: 0, refusal: 0, withheld: 0, notice: 0, error: 0 };
  for (const e of entries) {
    counts.tools += (e.step.tool_calls || []).length;
    for (const f of ['blocked', 'refusal', 'withheld', 'error']) if (e.flags.has(f)) counts[f]++;
    if (e.flags.has('notice') || e.flags.has('trailing')) counts.notice++;
  }

  const headerCard = h('section', { class: 'card ep-header' },
    h('div', { class: 'ep-title' },
      h('h1', null, episode),
      badge(ep.orbit_agent || t.agent || '?', 'agent'),
      t.model ? muted(t.model) : null,
      h('span', { class: 'spacer' }),
      navBtn(prev, `← ${prev || 'prev episode'}`),
      navBtn(next, `${next || 'next episode'} →`)),
    h('div', { class: 'links' },
      link(R.run(run), 'run page', 'btn btn-small'),
      logLink(logOk, evalLog, R.context(run, { agent: ep.orbit_agent || orbit.agent }), 'context windows for this agent', 'btn btn-small'),
      logLink(logOk, evalLog, R.sample(run), 'sample and workspace', 'btn btn-small')),
    episodeRecord(ep),
    h('div', { class: 'step-counts' },
      `${plural(steps.length, 'step')} · ${plural(counts.tools, 'tool call')}`,
      counts.blocked ? [' · ', badge(`${counts.blocked} BLOCKED`, 'blocked')] : null,
      counts.refusal ? [' · ', badge(`${counts.refusal} REFUSAL`, 'refusal')] : null,
      counts.withheld ? [' · ', badge(`${counts.withheld} WITHHELD`, 'withheld')] : null,
      counts.notice ? [' · ', badge(`${counts.notice} provider-only`, 'notice')] : null,
      counts.error ? [' · ', badge(`${counts.error} tool errors`, 'bad')] : null),
    h('h3', { class: 'sub-h' }, 'Judge verdict'),
    verdictBlock(ep.verdict),
    ep.error ? lazyDetails('episode error', () => longText(ep.error, { lines: 30 }), { cls: 'sub' }) : null,
    lazyDetails('ORBIT metadata and final metrics', () => h('div', null,
      kv([
        ['agent', orbit.agent], ['roster', asText(orbit.roster)], ['topology', orbit.topology],
        ['model calls', orbit.model_calls], ['submit recovered from events', asText(orbit.submit_recovered_from_events)],
        ['delivery notices recovered from events', asText(orbit.delivery_notices_recovered_from_events)],
        ['schema', traj.schema_version],
        ['agent version', ep.agent_version],
      ], 'kv-grid'),
      // Long values get the full width instead of a narrow grid cell.
      kv([
        ['eval log', orbit.eval_log ? mono(orbit.eval_log) : null],
        ['trajectory', traj.trajectory_id ? mono(traj.trajectory_id) : null],
        ['final metrics', traj.final_metrics ? h('pre', { class: 'pre' }, JSON.stringify(traj.final_metrics, null, 2)) : null],
      ])), { cls: 'sub' }),
    ep.final_text ? lazyDetails('final text', () => longText(ep.final_text, { lines: 40 }), { cls: 'sub' }) : null);

  // Sidebar: find box, kind filter, step list.
  const listBox = h('div', { class: 'side-list' });
  const countEl = h('div', { class: 'muted small side-count' });
  let active = null;
  const setActive = e => {
    if (active === e) return;
    if (active) active.item.classList.remove('active');
    active = e;
    if (!e) return;
    e.item.classList.add('active');
    const it = e.item;
    if (it.offsetTop < listBox.scrollTop || it.offsetTop + it.offsetHeight > listBox.scrollTop + listBox.clientHeight) {
      listBox.scrollTop = it.offsetTop - listBox.clientHeight / 3;
    }
  };
  const jump = e => {
    e.el.scrollIntoView({ block: 'start' });
    e.el.classList.remove('flash');
    void e.el.offsetWidth;
    e.el.classList.add('flash');
    setActive(e);
  };
  for (const e of entries) {
    e.item = h('button', {
      type: 'button', class: ['step-item', 'src-' + e.src, e.flags.size && 'flagged'],
      title: `step ${e.step.step_id} (${e.src})`, onclick: () => jump(e),
    },
    h('span', { class: 'si-id' }, String(e.step.step_id)),
    h('span', { class: 'si-src' }, e.src === 'system' ? 'sys' : e.src),
    h('span', { class: 'si-text' }, e.preview),
    e.flags.size ? h('span', { class: 'si-flags' }, [...e.flags].map(f => badge(FLAG_BADGE[f][0][0], FLAG_BADGE[f][1], FLAG_BADGE[f][0]))) : null);
    listBox.appendChild(e.item);
  }

  const find = h('input', { type: 'search', class: 'search', placeholder: 'find in steps…', 'aria-label': 'Find in steps', value: q.get('find') || '' });
  const urlKind = q.get('kind') || '';
  let kindVal = STEP_KINDS.some(([k]) => k === urlKind) ? urlKind : '';
  const noteBox = h('div', { class: 'url-notes' });
  const showNote = text => noteBox.appendChild(h('div', { class: 'url-note' }, text));
  if (kindVal !== urlKind) showNote(`Unknown step kind “${urlKind}” in the link; showing all steps.`);
  const kindSel = select(STEP_KINDS, kindVal, v => { kindVal = v; applyFilter(); }, { 'aria-label': 'Step kind' });
  let live = false;
  function updateUrl() {
    if (!live) return;
    const target = R.episode(run, episode, { find: find.value.trim(), kind: kindVal, step: q.get('step') });
    if (location.hash !== target) history.replaceState(null, '', target);
  }
  function applyFilter() {
    const needle = find.value.trim().toLowerCase();
    let n = 0;
    for (const e of entries) {
      const ok = (!needle || e.hay.includes(needle)) && stepMatchesKind(e, kindVal);
      e.el.hidden = !ok;
      e.item.hidden = !ok;
      if (ok) n++;
    }
    countEl.textContent = `${fmtInt(n)} of ${plural(entries.length, 'step')} shown`;
    updateUrl();
  }
  find.addEventListener('input', debounce(applyFilter, 120));
  find.addEventListener('keydown', ev => {
    if (ev.key === 'Enter') {
      const first = entries.find(e => !e.el.hidden);
      if (first) jump(first);
    }
  });
  applyFilter();

  const side = h('aside', { class: 'sidebar' },
    h('div', { class: 'side-tools' }, find, kindSel, noteBox, countEl),
    listBox);
  const stepsBox = h('div', { class: 'steps' }, entries.length ? entries.map(e => e.el) : muted('This transcript has no steps.'));
  const page = h('div', { class: 'split' }, side, h('div', { class: 'main-col' }, headerCard, stepsBox));

  // Highlight the step at the top of the viewport in the sidebar.
  const visible = new Set();
  const io = new IntersectionObserver(obs => {
    for (const o of obs) { if (o.isIntersecting) visible.add(o.target); else visible.delete(o.target); }
    const top = entries.find(e => visible.has(e.el));
    if (top) setActive(top);
  }, { rootMargin: '-60px 0px -65% 0px' });
  for (const e of entries) io.observe(e.el);
  view.cleanup(() => { live = false; io.disconnect(); });
  view.mounted(() => {
    live = true;
    document.title = `${episode} · ${run} · results`;
    updateUrl();
  });

  const startStep = q.get('step');
  if (startStep) {
    view.mounted(() => {
      const e = entries.find(x => String(x.step.step_id) === startStep);
      if (e) jump(e);
      else showNote(`Step “${startStep}” is not in this transcript.`);
    });
  }
  return page;
}

// =====================================================================================
// 8. View: context windows   #/run/<run>/context?agent=<name>&call=<i>
// =====================================================================================

// A call's flags are the union of its input messages' flags (plus the output being the
// refusal sentence). They describe the history the call saw, not what happened to its output.
const CALL_FLAG_LABEL = {
  delivery_notice: ['notice in input', 'notice', 'an ORBIT board delivery notice is in this call’s input'],
  trailing_turn_fix: ['continue prompt in input', 'notice', 'the integration appended a continue prompt (trailing-turn fix) to this input'],
  redacted: ['redacted input', 'withheld', 'a defense withheld a message of this input from the model'],
  refusal_in_history: ['refusal in history', 'plain', 'the input holds the refusal sentence: a substituted completion or the model’s own text'],
  output_blocked: ['output blocked', 'blocked', 'a defense blocked this call’s completion and substituted the refusal sentence'],
};
const CALL_FILTERS = [
  ['', 'any'],
  ['output_blocked', 'output blocked by a defense'],
  ['delivery_notice', 'notice in input'],
  ['trailing_turn_fix', 'continue prompt in input'],
  ['redacted', 'redacted input'],
  ['refusal_in_history', 'refusal in history'],
  ['error', 'has error'],
];
const MSG_FLAG_BADGE = {
  delivery_notice: ['delivery notice', 'notice'],
  trailing_turn_fix: ['trailing-turn fix', 'notice'],
  redacted: ['REDACTED', 'withheld'],
  refusal_in_history: ['refusal in history', 'plain'],
  output_blocked: ['BLOCKED OUTPUT', 'refusal'],
};

function msgFlagBadge(f) {
  const [label, kind] = MSG_FLAG_BADGE[f] || [f, 'plain'];
  return badge(label, kind, f);
}

function callFlagBadge(f) {
  const [label, kind, title] = CALL_FLAG_LABEL[f] || [f, 'plain', f];
  return badge(label, kind, title);
}

function callWho(c) {
  if (c.defense) return `${c.defense}${c.agent ? ' on ' + c.agent : ''}`;
  return c.agent || '(no agent)';
}

function usageText(u) {
  if (!u) return 'no usage';
  const parts = [`in ${fmtCompact(u.input)}`, `out ${fmtCompact(u.output)}`];
  if (u.cache_read) parts.push(`cache ${fmtCompact(u.cache_read)}`);
  return parts.join(' · ');
}

function blockReasons(ivs) {
  return ivs.map(x => `${x.defense || '?'} at ${x.boundary || '?'}: ${x.reason || '(no reason recorded)'}`).join('\n');
}

/**
 * Blocks from the run's defense evidence, indexed for the context view.
 * A model_output intervention carries the message id of the completion it checked, which is
 * the output id of the model call that produced it. The call index has no output ids, so the
 * sidebar aligns each agent's own calls with that agent's checked completions in order, and
 * only when the two counts agree. Opening a call checks the alignment against the real id.
 */
function blockIndex(detail, calls) {
  // Blocked outputs come from the API (call.output_blocked, matched by the raw output id on
  // the server); interventions are kept by message id for the reasons shown inside a call.
  const outBlocks = new Map();   // completion message id -> blocking interventions
  const toolBlocks = new Map();  // tool call id -> blocking interventions
  const add = (m, k, x) => { if (!m.has(k)) m.set(k, []); m.get(k).push(x); };
  for (const dv of Object.values((detail && detail.defense) || {})) {
    for (const x of (dv && dv.interventions) || []) {
      if (x.boundary === 'model_output' && x.message_id && x.allow === false) add(outBlocks, x.message_id, x);
      else if (x.boundary === 'tool_call' && x.allow === false && x.tool_call_id) add(toolBlocks, x.tool_call_id, x);
    }
  }
  const blockedCalls = new Map(); // call index -> blocking interventions (unknown by id here)
  for (const c of calls) if (c.output_blocked) blockedCalls.set(c.index, []);
  return { outBlocks, toolBlocks, callOut: new Map(), blockedCalls, misaligned: [], distrusted: false };
}

function blockedToolResult(ivs) {
  const notRun = ivs.some(x => x.tool_executed === false);
  return h('div', { class: 'result is-blocked' },
    h('div', { class: 'res-head' },
      h('span', { class: 'res-label' }, notRun ? 'not executed' : 'tool call blocked'),
      badge('BLOCKED', 'blocked')),
    h('pre', { class: 'pre errtext' }, blockReasons(ivs)));
}

function messageView(m, i, status, opts = {}) {
  const flags = m.flags || [];
  const notice = flags.includes('delivery_notice') || flags.includes('trailing_turn_fix');
  const el = h('div', {
    class: ['msg', 'role-' + m.role, status && 'st-' + status, notice && 'provider-only',
      flags.includes('redacted') && 'is-redacted', flags.includes('output_blocked') && 'is-refusal'],
    id: i != null ? `msg-${i}` : null,
  });
  el.appendChild(h('div', { class: 'msg-head' },
    i != null ? h('span', { class: 'step-id' }, `[${i}]`) : null,
    h('span', { class: 'msg-role' }, m.role),
    status === 'new' ? badge('NEW', 'new', 'not in the previous call of this caller') : null,
    status === 'prev-out' ? badge('previous output', 'prevout', 'the output of the previous call of this caller') : null,
    status === 'replaced' ? badge('replaces blocked output', 'blocked', 'this refusal took the place of the previous call’s blocked completion') : null,
    flags.map(msgFlagBadge),
    m.role === 'tool' && m.function ? h('code', { class: 'tc-fn' }, m.function) : null,
    m.role === 'assistant' && m.model ? muted(m.model, 'small') : null,
    h('span', { class: 'spacer' }),
    m.tool_call_id ? h('span', { class: 'tc-id', title: 'tool call id' }, m.tool_call_id) : null,
    m.id ? h('span', { class: 'tc-id', title: 'message id' }, m.id) : null));
  if (notice) {
    el.appendChild(h('div', { class: 'caption caption-notice' },
      flags.includes('trailing_turn_fix')
        ? 'Provider-only: continue prompt appended by the integration.'
        : 'Provider-only: ORBIT board delivery notice placed in the input.'));
  }
  if (flags.includes('redacted')) {
    el.appendChild(h('div', { class: 'caption caption-withheld' }, 'REDACTED: a defense withheld this message’s content from the model; the placeholder below is what the provider received.'));
    if (m.withheld_text) el.appendChild(lazyDetails('original content (never reached the model)', () => longText(asText(m.withheld_text), { lines: 30 }), { cls: 'sub' }));
  }
  if (m.error) el.appendChild(h('pre', { class: 'pre errtext' }, String(m.error)));
  if (m.reasoning) el.appendChild(reasoningView(m.reasoning, 40));
  const text = asText(m.text);
  if (text) el.appendChild(longText(text, { lines: 30, cls: 'msg-text' }));
  else if (!(m.tool_calls || []).length) el.appendChild(muted('(empty)', 'small block'));
  for (const tc of m.tool_calls || []) {
    const blocks = opts.toolBlocks ? opts.toolBlocks.get(tc.id) : null;
    el.appendChild(toolCallView(tc.function, tc.id, tc.arguments, blocks ? [blockedToolResult(blocks)] : null));
  }
  return el;
}

async function viewContext({ run, query: q }, view) {
  const [idx, detail] = await Promise.all([
    api(P.context(run)),
    api(P.run(run)).catch(() => null),
  ]);
  const calls = idx.calls || [];
  const byIndex = new Map(calls.map(c => [c.index, c]));
  const bi = blockIndex(detail, calls);
  let live = false;

  const noteBox = h('div', { class: 'url-notes' });
  const showNote = text => noteBox.appendChild(h('div', { class: 'url-note' }, text));

  const agentNames = [...new Set([...(idx.agents || []), ...calls.map(c => c.agent).filter(Boolean)])].sort();
  const agentOpts = [['', 'all agents'], ...agentNames.map(a => [a, a])];
  if (calls.some(c => !c.agent)) agentOpts.push(['(none)', '(no agent)']);
  const nDef = calls.filter(c => c.defense).length;

  // Filters from the URL. An unknown value falls back to "all", with a note saying so.
  const st = { agent: '', kind: '', flag: '', selected: null, onlyNew: false };
  const urlAgent = q.get('agent') || '';
  if (urlAgent && agentOpts.some(([v]) => v === urlAgent)) st.agent = urlAgent;
  else if (urlAgent) showNote(`Unknown agent “${urlAgent}” in the link; showing all agents.`);
  const urlKind = q.get('kind') || '';
  if (urlKind === 'agent' || urlKind === 'defense') st.kind = urlKind;
  else if (urlKind) showNote(`Unknown call kind “${urlKind}” in the link; showing all calls.`);
  const urlFlag = q.get('flag') || '';
  if (urlFlag && CALL_FILTERS.some(([v]) => v === urlFlag)) st.flag = urlFlag;
  else if (urlFlag) showNote(`Unknown flag “${urlFlag}” in the link; showing calls with any flag.`);
  if (!detail) showNote('Could not load the run’s defense evidence, so blocked outputs are not marked.');
  if (bi.misaligned.length) {
    showNote(`Blocked outputs are marked in this list only where an agent’s calls line up with its defense checks. ` +
      `Not aligned: ${bi.misaligned.join('; ')}. Opening a call still shows its blocks.`);
  }

  const items = calls.map(c => {
    const blocked = bi.blockedCalls.get(c.index);
    const blockedBadge = blocked ? badge('OUTPUT BLOCKED', 'blocked', blockReasons(blocked)) : null;
    if (blockedBadge) blockedBadge.classList.add('ci-blocked');
    const el = h('button', {
      type: 'button', class: ['call-item', c.defense ? 'is-defense' : 'is-agent', c.error && 'has-error', blocked && 'is-blocked-call'],
      onclick: () => selectCall(c.index, { fromList: true }),
    },
    h('div', { class: 'ci-line' },
      h('span', { class: 'ci-idx' }, `#${c.index}`),
      h('span', { class: 'ci-who' }, c.agent || '(no agent)'),
      c.defense ? badge(c.defense, 'def') : badge('agent call', 'agent'),
      blockedBadge),
    h('div', { class: 'ci-line ci-meta' },
      `${plural(c.n_input, 'msg')} · ${usageText(c.usage)}`,
      (c.output_tool_calls || []).length ? ` · → ${c.output_tool_calls.join(', ')}` : ''),
    c.output_preview && !(c.output_tool_calls || []).length
      ? h('div', { class: 'ci-line ci-preview' }, firstLine(c.output_preview, 70)) : null,
    (c.flags || []).length || c.error
      ? h('div', { class: 'ci-line' }, (c.flags || []).map(callFlagBadge), c.error ? badge('error', 'bad', c.error) : null)
      : null);
    return { c, el };
  });

  const listBox = h('div', { class: 'side-list' }, items.map(it => it.el));
  const countEl = h('div', { class: 'muted small side-count' });
  const detailEl = h('div', { class: 'call-detail' });
  const blockedCount = h('span', { class: 'text-bad' });

  const visibleItems = () => items.filter(it => !it.el.hidden);
  function matches(c) {
    return (!st.agent || (st.agent === '(none)' ? !c.agent : c.agent === st.agent)) &&
      (!st.kind || (st.kind === 'agent' ? !c.defense : !!c.defense)) &&
      (!st.flag || (st.flag === 'error' ? !!c.error
        : st.flag === 'output_blocked' ? bi.blockedCalls.has(c.index)
          : (c.flags || []).includes(st.flag)));
  }
  function applyFilter() {
    let n = 0;
    for (const it of items) {
      const ok = matches(it.c);
      it.el.hidden = !ok;
      if (ok) n++;
    }
    countEl.textContent = `${fmtInt(n)} of ${plural(calls.length, 'call')} shown`;
    blockedCount.textContent = bi.blockedCalls.size ? ` · ${plural(bi.blockedCalls.size, 'output')} blocked` : '';
    updateUrl();
  }

  function updateUrl() {
    if (!live) return;
    const target = R.context(run, {
      agent: st.agent, kind: st.kind, flag: st.flag, call: st.selected == null ? '' : st.selected,
    });
    if (location.hash !== target) history.replaceState(null, '', target);
  }

  // If an opened call's real output id disagrees with the order alignment, drop the marks.
  function checkAlignment() { /* blocked outputs are matched by id on the server */ }

  function scrollItemIntoList(it) {
    const e = it.el;
    if (e.offsetTop < listBox.scrollTop || e.offsetTop + e.offsetHeight > listBox.scrollTop + listBox.clientHeight) {
      listBox.scrollTop = e.offsetTop - listBox.clientHeight / 3;
    }
  }

  function prevSameCaller(c) {
    for (let j = c.index - 1; j >= 0; j--) {
      const p = byIndex.get(j);
      if (p && p.agent === c.agent && p.defense === c.defense) return p;
    }
    return null;
  }

  let selToken = 0;
  async function selectCall(i, opts = {}) {
    const c = byIndex.get(i);
    if (!c) return;
    st.selected = i;
    for (const it of items) it.el.classList.toggle('active', it.c.index === i);
    const it = items.find(x => x.c.index === i);
    if (it && !opts.fromList) scrollItemIntoList(it);
    if (it && opts.focus) it.el.focus({ preventScroll: true });
    updateUrl();
    const tok = ++selToken;
    detailEl.replaceChildren(h('div', { class: 'loading' }, `Loading call #${i}…`));
    // Holding j or k skips through calls; only the call the user stops on is fetched.
    if (opts.debounce) {
      await new Promise(res => setTimeout(res, 90));
      if (tok !== selToken || !live) return;
    }
    const prev = prevSameCaller(c);
    try {
      const [full, prevFull] = await Promise.all([
        apiCall(P.call(run, i)),
        prev ? apiCall(P.call(run, prev.index)).catch(e => ({ __error: e })) : Promise.resolve(null),
      ]);
      if (tok !== selToken || !live) return;
      checkAlignment(i, full);
      detailEl.replaceChildren(callDetail(c, full, prev, prevFull, selectCall, { bi, st }));
      const top = detailEl.getBoundingClientRect().top;
      if (top < 0 || top > window.innerHeight) window.scrollTo(0, 0);
      else if (window.scrollY > 0 && opts.scrollTop !== false) window.scrollTo(0, 0);
    } catch (e) {
      if (tok !== selToken || !live) return;
      detailEl.replaceChildren(errorBox(e, `call #${i}`));
    }
  }

  function move(dir, focus) {
    const vis = visibleItems();
    if (!vis.length) return;
    let next;
    const pos = vis.findIndex(it => it.c.index === st.selected);
    if (pos >= 0) next = vis[Math.min(vis.length - 1, Math.max(0, pos + dir))];
    else if (st.selected == null) next = dir > 0 ? vis[0] : vis[vis.length - 1];
    // The selected call is filtered out: go to the nearest visible call in that direction.
    else if (dir > 0) next = vis.find(x => x.c.index > st.selected) || vis[vis.length - 1];
    else next = [...vis].reverse().find(x => x.c.index < st.selected) || vis[0];
    if (next && next.c.index !== st.selected) selectCall(next.c.index, { debounce: true, focus });
  }

  // j / k work anywhere on the page. The arrow keys move between calls only while the call
  // list has focus, so elsewhere they still scroll a long call.
  const onKey = e => {
    if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
    const tgt = e.target;
    if (tgt && (tgt.tagName === 'INPUT' || tgt.tagName === 'SELECT' || tgt.tagName === 'TEXTAREA' || tgt.isContentEditable)) return;
    const inList = tgt instanceof Node && listBox.contains(tgt);
    let dir = 0;
    if (e.key === 'j' || (inList && e.key === 'ArrowDown')) dir = 1;
    else if (e.key === 'k' || (inList && e.key === 'ArrowUp')) dir = -1;
    else return;
    e.preventDefault();
    move(dir, inList);
  };

  const kindBtns = h('div', { class: 'seg', role: 'group', 'aria-label': 'call kind' },
    [['', 'all'], ['agent', 'agent calls only'], ['defense', 'defense calls only']].map(([v, label]) => {
      const b = h('button', { type: 'button', class: ['seg-btn', st.kind === v && 'active'], dataset: { v } }, label);
      b.addEventListener('click', () => {
        st.kind = v;
        for (const x of kindBtns.children) x.classList.toggle('active', x.dataset.v === v);
        applyFilter();
      });
      return b;
    }));

  const side = h('aside', { class: 'sidebar' },
    h('div', { class: 'side-tools' },
      h('div', { class: 'side-title' }, h('strong', null, 'Model calls'), ' ',
        muted(`${fmtInt(calls.length - nDef)} agent, ${fmtInt(nDef)} defense`), blockedCount),
      labeled('agent', select(agentOpts, st.agent, v => { st.agent = v; applyFilter(); })),
      kindBtns,
      labeled('flag', select(CALL_FILTERS, st.flag, v => { st.flag = v; applyFilter(); })),
      h('div', { class: 'muted small' }, 'keys: j / k move between calls; ↓ / ↑ do too while the list has focus'),
      noteBox,
      countEl),
    listBox);

  applyFilter();
  const page = h('div', { class: 'split' }, side, h('div', { class: 'main-col' }, detailEl));

  view.mounted(() => {
    live = true;
    document.title = `context · ${run} · results`;
    document.addEventListener('keydown', onKey);
  });
  view.cleanup(() => {
    live = false;
    document.removeEventListener('keydown', onKey);
  });

  if (!calls.length) {
    detailEl.replaceChildren(muted('The eval log has no model calls.'));
    view.mounted(updateUrl);
    return page;
  }
  const want = q.get('call');
  const wanted = want && /^\d+$/.test(want) && byIndex.has(Number(want)) ? Number(want) : null;
  const vis = visibleItems();
  const start = wanted != null ? wanted : vis.length ? vis[0].c.index : null;
  if (want && wanted == null) {
    showNote(`Call “${want}” does not exist; this run has calls #0 to #${calls[calls.length - 1].index}.` +
      (start != null ? ` Showing call #${start}.` : ''));
  }
  if (start == null) {
    detailEl.replaceChildren(muted('No calls match these filters.'));
    view.mounted(updateUrl);
  } else {
    view.mounted(() => selectCall(start));
  }
  return page;
}

function callDetail(c, full, prev, prevFull, selectCall, ctx) {
  const { bi, st } = ctx;
  const input = full.input || [];
  const outBlocks = full.output && full.output.id ? bi.outBlocks.get(full.output.id) || null : null;
  // Diff against the previous call by the same caller (same agent and same defense).
  const status = new Array(input.length).fill(null);
  let hint;
  if (!prev) {
    hint = h('div', { class: 'diff-hint' }, `First call by ${callWho(c)}: there is no earlier call to compare with.`);
  } else if (!prevFull || prevFull.__error) {
    hint = h('div', { class: 'diff-hint' }, `Could not load call #${prev.index} to compare with.`,
      prevFull && prevFull.__error ? errorBox(prevFull.__error) : null);
  } else {
    const prevIds = new Set((prevFull.input || []).map(m => m.id).filter(Boolean));
    const prevOut = prevFull.output && prevFull.output.id;
    const prevBlocks = prevOut ? bi.outBlocks.get(prevOut) || null : null;
    const cur = new Set(input.map(m => m.id).filter(Boolean));
    let nNew = 0;
    let nKept = 0;
    let nOut = 0;
    let replacedAt = -1;
    input.forEach((m, i) => {
      if (!m.id) return;
      if (m.id === prevOut) { status[i] = 'prev-out'; nOut++; }
      else if (prevIds.has(m.id)) { status[i] = 'kept'; nKept++; }
      else if (prevBlocks && replacedAt < 0 && (m.flags || []).includes('refusal_in_history')) { status[i] = 'replaced'; replacedAt = i; }
      else { status[i] = 'new'; nNew++; }
    });
    const dropped = [...prevIds].filter(id => !cur.has(id)).length;
    const blockers = prevBlocks ? [...new Set(prevBlocks.map(x => x.defense || '?'))].join(', ') : '';
    hint = h('div', { class: 'diff-hint' },
      'Compared with ',
      h('button', { type: 'button', class: 'linkish', onclick: () => selectCall(prev.index) }, `call #${prev.index}`),
      `, the previous call by ${callWho(c)}: `,
      badge(`${nNew} new`, 'new'), ' ',
      `${fmtInt(nKept)} carried over`,
      nOut ? `, ${nOut} is that call’s output` : '',
      dropped ? `, ${plural(dropped, 'message')} from that input no longer present` : '',
      '.',
      prevBlocks ? h('div', { class: 'diff-blocked' },
        badge('BLOCKED', 'blocked'), ` ${blockers} blocked the output of call #${prev.index}`,
        replacedAt >= 0 ? `; the refusal at [${replacedAt}] took its place.` : ', and it does not appear in this input.') : null);
  }

  const firstNew = status.findIndex(s => s === 'new' || s === 'replaced');
  const msgs = h('div', { class: ['msgs', prev && st.onlyNew && 'only-new'] },
    input.map((m, i) => messageView(m, i, status[i], { toolBlocks: bi.toolBlocks })));
  const onlyNew = h('input', { type: 'checkbox', checked: st.onlyNew });
  onlyNew.addEventListener('change', () => {
    st.onlyNew = onlyNew.checked;
    msgs.classList.toggle('only-new', onlyNew.checked);
  });

  const u = full.usage || {};
  const head = h('section', { class: 'card call-head' },
    h('div', { class: 'ep-title' },
      h('h1', null, `Call #${c.index}`),
      c.defense ? badge(c.defense, 'def') : badge('agent call', 'agent'),
      h('span', null, c.agent || '(no agent)'),
      outBlocks ? badge('OUTPUT BLOCKED', 'blocked', blockReasons(outBlocks)) : null,
      (full.flags || []).map(callFlagBadge),
      full.error ? badge('error', 'bad') : null),
    kv([
      ['model', full.model],
      ['timestamp', fmtTime(full.timestamp)],
      ['working time', fmtSecs(full.working_time)],
      ['stop reason', full.stop_reason],
      ['input messages', `${fmtInt(full.n_input)} (${fmtInt(full.input_chars)} chars)`],
      ['usage input', fmtInt(u.input)],
      ['usage cache read', fmtInt(u.cache_read)],
      ['usage output', fmtInt(u.output)],
      ['usage reasoning', fmtInt(u.reasoning)],
      ['output tool calls', (full.output_tool_calls || []).join(', ') || null],
    ], 'kv-grid'),
    full.error ? h('pre', { class: 'pre errtext' }, String(full.error)) : null,
    full.output_error ? h('pre', { class: 'pre errtext' }, `output error: ${full.output_error}`) : null,
    hint);

  const inputSec = h('section', { class: 'card' },
    h('div', { class: 'sec-head' },
      h('h2', null, `Input (${plural(input.length, 'message')})`),
      h('span', { class: 'spacer' }),
      firstNew >= 0 ? h('button', {
        type: 'button', class: 'btn btn-small',
        onclick: () => { const el = document.getElementById(`msg-${firstNew}`); if (el) el.scrollIntoView({ block: 'start' }); },
      }, 'jump to first new') : null,
      prev ? h('label', { class: 'field check' }, onlyNew, h('span', null, 'only new and previous output')) : null),
    input.length ? msgs : muted('No input messages.'));

  const outputSec = h('section', { class: ['card', outBlocks && 'card-blocked'] },
    h('h2', null, 'Output'),
    outBlocks ? h('div', { class: 'caption caption-blocked' },
      h('strong', null, 'BLOCKED. '),
      'A defense blocked this completion at the model output. Its text and tool calls were dropped, ' +
      'and the conversation continued with the refusal sentence in its place.',
      h('pre', { class: 'pre errtext' }, blockReasons(outBlocks))) : null,
    full.output ? messageView(full.output, null, null, { toolBlocks: bi.toolBlocks }) : muted('No output message.'));

  const schemas = full.tool_schemas || [];
  const toolsSec = h('section', { class: 'card' },
    h('h2', null, `Tools offered (${schemas.length})`),
    schemas.length ? schemas.map(s => lazyDetails(s.name, () => longText(s.description || '(no description)', { lines: 30 }), { cls: 'sub' }))
      : muted('No tools were offered in this call.'));

  return h('div', null, head, inputSec, outputSec, toolsSec);
}

// =====================================================================================
// 9. View: sample   #/run/<run>/sample
// =====================================================================================

const META_ORDER = ['listing', 'git', 'agent_evals', 'agent_evals_main', 'probe'];

function probeView(probe) {
  if (!probe || typeof probe !== 'object' || Array.isArray(probe)) return h('pre', { class: 'pre' }, asText(probe));
  return h('div', { class: 'probe' }, Object.entries(probe).map(([k, v]) => {
    if (v && typeof v === 'object' && ('out' in v || 'rc' in v)) {
      return h('div', { class: 'probe-item' },
        h('div', { class: 'cfg-key' }, h('span', { class: 'mono' }, k), ' ',
          badge(`rc ${v.rc}`, v.rc === 0 ? 'good' : 'bad')),
        longText(asText(v.out), { lines: 30 }),
        v.err ? h('pre', { class: 'pre errtext' }, asText(v.err)) : null);
    }
    return h('div', { class: 'probe-item' }, h('div', { class: 'cfg-key mono' }, k), h('pre', { class: 'pre' }, asText(v)));
  }));
}

async function viewSample({ run }, view) {
  const s = await api(P.sample(run));
  view.mounted(() => { document.title = `sample · ${run} · results`; });
  const lim = s.limit;
  const facts = section('Sample', kv([
    ['status', s.status ? badge(s.status, s.status === 'success' ? 'good' : 'bad') : null],
    ['model', s.model],
    ['eval id', s.eval_id ? mono(s.eval_id) : null],
    ['sample', s.sample_id != null ? `${s.sample_id} (epoch ${s.epoch})` : null],
    ['total time', fmtSecs(s.total_time)],
    ['working time', fmtSecs(s.working_time)],
    ['limit', lim ? `${lim.type}${lim.limit != null ? ' ' + lim.limit : ''}${lim.message ? ': ' + lim.message : ''}` : null],
    ['messages', fmtInt(s.n_messages)],
    ['events', fmtInt(s.n_events)],
  ], 'kv-grid'),
  s.error ? [h('h3', { class: 'sub-h' }, 'Error'), longText(s.error, { lines: 30, cls: 'errtext' })] : null);

  const usage = Object.entries(s.model_usage || {});
  const usageSec = section('Model usage', usage.length ? h('div', { class: 'table-wrap' }, h('table', { class: 'data' },
    h('thead', null, h('tr', null, ['model', 'input', 'cache read', 'output', 'reasoning'].map((x, i) => h('th', { class: i ? 'num' : '' }, x)))),
    h('tbody', null, usage.map(([m, u]) => h('tr', null,
      h('td', { class: 'mono' }, m),
      ['input', 'cache_read', 'output', 'reasoning'].map(k => h('td', { class: 'num' }, fmtInt(u && u[k])))))))) : muted('No usage recorded.'));

  const scores = Object.entries(s.scores || {});
  const scoreSecs = scores.length ? scores.map(([name, sc]) => {
    const md = (sc && sc.metadata) || {};
    const keys = [...META_ORDER.filter(k => k in md), ...Object.keys(md).filter(k => !META_ORDER.includes(k))];
    return section(`Score: ${name}`,
      kv([
        ['value', asText(sc.value)],
        ['explanation', sc.explanation],
      ], 'kv-grid'),
      h('h3', { class: 'sub-h' }, 'Answer (message.txt)'),
      sc.answer ? longText(sc.answer, { lines: 40 }) : muted('No answer.'),
      keys.map(k => h('div', { class: 'meta-block' },
        h('h3', { class: 'sub-h' }, k),
        k === 'probe' ? probeView(md[k]) : longText(asText(md[k]), { lines: 60, chars: 40000 }))));
  }) : section('Scores', muted('No scores (the sample did not reach the scorer).'));

  const storeSec = s.store
    ? section('Store', lazyDetails('sample store (JSON)', () => longText(JSON.stringify(s.store, null, 2), { lines: 80, chars: 60000 }), { cls: 'sub' }))
    : null;

  return h('div', { class: 'page' },
    h('div', { class: 'page-head' },
      h('h1', null, `${run}: sample and workspace`),
      h('div', { class: 'links' }, link(R.run(run), 'run page', 'btn btn-small'), link(R.context(run), 'context windows', 'btn btn-small'))),
    facts, usageSec, scoreSecs, storeSec);
}

// =====================================================================================
// 10. Boot
// =====================================================================================

renderShell();
window.addEventListener('hashchange', route);
route();

})();
