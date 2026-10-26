// Headless pass over a served static bundle of the results browser (make share-browser).
//
//   NODE_PATH=/opt/homebrew/lib/node_modules node guardian/checks/share_browser.js http://127.0.0.1:8798/
//
// Needs Playwright with Chromium. Exit 1 on any failure. Checks, each against the live page: every main view at
// 360, 400 and 1440 px loads with no console error, no page error, no response of 400 or more, no request outside
// the bundle (the page, data/, figures/) and no horizontal page scroll; the context page lists the calls and shows
// the static note for a selected call instead of an error box; the four theme states (system light and dark, an
// explicit data-theme of light or dark) compute the right tokens; a theme change after drawing redraws the raster.
'use strict';
const { chromium } = require('playwright');

const BASE = process.argv[2] || 'http://127.0.0.1:8798/';
const WIDTHS = [360, 400, 1440];
const failures = [];
const fail = m => { failures.push(m); console.log('  FAIL ' + m); };
const ready = page => page.waitForFunction(() => !document.querySelector('#main > .loading'), null, { timeout: 90000 });

async function main() {
  const runs = await (await fetch(BASE + 'data/runs.json')).json();
  const swarm = runs.find(r => !r.failed && r.swarm && r.swarm.composition) || runs.find(r => r.swarm) || runs[0];
  const withTranscript = runs.find(r => (r.episodes || []).some(e => e.has_transcript)) || runs[0];
  const episode = (withTranscript.episodes || []).find(e => e.has_transcript);
  const withLog = runs.find(r => r.eval_log_exists && !r.failed && !r.mock) || runs[0];
  const enc = encodeURIComponent;
  const views = [
    { name: 'summary', hash: '#/' },
    { name: 'runs', hash: '#/runs?all=1', runs: true },
    { name: 'swarm', hash: `#/swarm?run=${enc(swarm.run)}`, swarm: true },
    { name: 'run', hash: `#/run/${enc(withTranscript.run)}` },
    { name: 'transcript', hash: episode ? `#/run/${enc(withTranscript.run)}/episode/${enc(episode.episode)}` : null },
    { name: 'sample', hash: `#/run/${enc(withLog.run)}/sample` },
    { name: 'timeline', hash: `#/run/${enc(withLog.run)}/timeline`, timeline: true },
    { name: 'context', hash: `#/run/${enc(withLog.run)}/context?call=0`, context: true },
  ].filter(v => v.hash);
  const browser = await chromium.launch();
  for (const width of WIDTHS) {
    const ctx = await browser.newContext({ viewport: { width, height: 900 } });
    const page = await ctx.newPage();
    const errors = [], outside = [], statuses = [];
    page.on('pageerror', e => errors.push('pageerror: ' + e.message));
    page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
    page.on('request', r => { const u = r.url(); if (!(u === BASE || u.startsWith(BASE + '#') || u.startsWith(BASE + 'data/') || u.startsWith(BASE + 'figures/') || u.startsWith(BASE + 'index.html'))) outside.push(u); });
    page.on('response', r => { if (r.status() >= 400) statuses.push(r.status() + ' ' + r.url()); });
    for (const v of views) {
      await page.goto(BASE + v.hash, { waitUntil: 'networkidle' });
      await ready(page);
      if (v.swarm) await page.waitForFunction(() => document.querySelector('.adopt-svg') && document.querySelector('#explore-card canvas.raster'), null, { timeout: 120000 }).catch(() => fail(`${width}px ${v.name}: swarm charts did not draw`));
      if (v.timeline) await page.waitForFunction(() => document.querySelector('.timeline-svg'), null, { timeout: 90000 }).catch(() => fail(`${width}px ${v.name}: timeline did not draw`));
      if (v.context) {
        await page.waitForFunction(() => document.querySelector('.call-detail .static-note') || document.querySelector('.call-detail .error-box') || document.querySelector('.call-detail .call-view'), null, { timeout: 60000 }).catch(() => {});
        const c = await page.evaluate(() => ({ calls: document.querySelectorAll('.call-item').length, note: !!document.querySelector('.call-detail .static-note'), sideNote: [...document.querySelectorAll('.url-note')].some(n => n.textContent.includes('shared copy')) }));
        if (!c.calls) fail(`${width}px context: no calls listed`);
        if (!c.note) fail(`${width}px context: the selected call does not show the static note`);
        if (!c.sideNote) fail(`${width}px context: the sidebar lacks the static note`);
      }
      if (v.runs) {
        // open the first run: its first agent row must be visible, not under the agents table's own header
        await page.locator('tr.run-row .expander').first().click();
        await page.waitForTimeout(250);
        const r = await page.evaluate(() => {
          const row = document.querySelector('tr.detail-row .eps-table tbody tr');
          if (!row) return { missing: true };
          row.scrollIntoView({ block: 'center' });
          const b = row.getBoundingClientRect();
          const el = document.elementFromPoint(b.left + Math.min(40, b.width / 2), b.top + b.height / 2);
          return { covered: !(el && row.contains(el)), by: el ? el.tagName + '.' + el.className : 'nothing' };
        });
        if (r.missing) fail(`${width}px runs: no agent row after opening the first run`);
        else if (r.covered) fail(`${width}px runs: the first agent row is covered by ${r.by}`);
      }
      await page.waitForTimeout(250);
      const m = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: window.innerWidth, boxes: document.querySelectorAll('.error-box').length, h1: (document.querySelector('h1') || {}).textContent }));
      if (m.sw > m.iw) fail(`${width}px ${v.name}: horizontal page scroll (${m.sw} > ${m.iw})`);
      if (m.boxes) fail(`${width}px ${v.name}: ${m.boxes} error box(es)`);
      console.log(`  ${width}px ${v.name}: ok (${m.h1})`);
    }
    for (const e of errors) fail(`${width}px: ${e}`);
    for (const u of outside) fail(`${width}px: request outside the bundle: ${u}`);
    for (const s of statuses) fail(`${width}px: ${s}`);
    await ctx.close();
  }
  // theme states and a theme change after drawing
  for (const [scheme, attr, wantDark] of [['light', null, false], ['dark', null, true], ['light', 'dark', true], ['dark', 'light', false]]) {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: scheme });
    const page = await ctx.newPage();
    await page.goto(BASE + `#/swarm?run=${enc(swarm.run)}`, { waitUntil: 'networkidle' });
    await ready(page);
    await page.waitForFunction(() => document.querySelector('#explore-card canvas.raster'), null, { timeout: 120000 });
    const sample = () => page.evaluate(() => {
      const c = document.querySelector('#explore-card canvas.raster');
      const d = c.getContext('2d').getImageData(0, 0, c.width, Math.min(c.height, 40)).data;
      let r = 0, g = 0, b = 0, n = 0;
      for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2]; n++; }
      const cs = getComputedStyle(document.documentElement);
      return { raster: [r / n, g / n, b / n].map(Math.round).join(','), bg: cs.getPropertyValue('--bg').trim(), scheme: cs.colorScheme, body: getComputedStyle(document.body).backgroundColor };
    });
    if (attr) { await page.evaluate(a => document.documentElement.setAttribute('data-theme', a), attr); await page.waitForTimeout(100); await ready(page); await page.waitForFunction(() => document.querySelector('#explore-card canvas.raster'), null, { timeout: 120000 }); }
    const s = await sample();
    const isDark = s.bg === '#15161a';
    if (isDark !== wantDark) fail(`theme os=${scheme} data-theme=${attr}: --bg ${s.bg} (expected ${wantDark ? 'dark' : 'light'})`);
    if ((s.scheme === 'dark') !== wantDark) fail(`theme os=${scheme} data-theme=${attr}: color-scheme ${s.scheme}`);
    console.log(`  theme os=${scheme} data-theme=${attr}: --bg ${s.bg}, color-scheme ${s.scheme}, body ${s.body}, raster ${s.raster}`);
    if (!attr) {
      // flip the viewer's choice after drawing: the raster must be drawn again in the other palette
      const before = s.raster;
      await page.evaluate(a => document.documentElement.setAttribute('data-theme', a), wantDark ? 'light' : 'dark');
      await page.waitForTimeout(100); await ready(page);
      await page.waitForFunction(() => document.querySelector('#explore-card canvas.raster'), null, { timeout: 120000 });
      const after = (await sample()).raster;
      if (after === before) fail(`theme change after drawing (os=${scheme}): raster colours unchanged (${before})`);
      else console.log(`  theme change after drawing (os=${scheme}): raster ${before} -> ${after}`);
    }
    await ctx.close();
  }
  await browser.close();
  console.log(failures.length ? `${failures.length} failure(s)` : 'share_browser: ALL PASS');
  process.exit(failures.length ? 1 : 0);
}
main().catch(e => { console.error(e); process.exit(1); });
