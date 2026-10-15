const $ = (s) => document.querySelector(s);
const els = {
  grid: $('#sample-grid'), details: $('#sample-details'), source: $('#source-link'),
  file: $('#video-file'), run: $('#run-button'), cancel: $('#cancel-button'),
  card: $('#status-card'), status: $('#status-heading'), message: $('#status-message'),
  progress: $('#progress'), elapsed: $('#elapsed'), video: $('#video'),
  empty: $('.empty-state'), mode: $('#feed-mode'), name: $('#feed-name'), led: $('#led'), time: $('#timecode'),
  viewOrig: $('#view-original'), viewOver: $('#view-overlay'),
  scan: $('#scan'), results: $('#results'), placeholder: $('#results-placeholder'),
  log: $('#process-log'), hint: $('#next-hint'), summary: $('#summary'),
  steps: [$('#step1'), $('#step2'), $('#step3')]
};
const events = [];
const profileEl = $('#profile');
async function loadProfiles() {
  const {profiles, default: fallback} = await api('/api/profiles');
  profileEl.replaceChildren(...profiles.map(p => Object.assign(document.createElement('option'),
    {value: p.name, textContent: p.label, title: p.note})));
  const saved = (() => { try { return localStorage.getItem('aeris-profile'); } catch { return null; } })();
  profileEl.value = profiles.some(p => p.name === saved) ? saved : fallback;
  const note = () => { profileEl.title = profiles.find(p => p.name === profileEl.value)?.note || ''; };
  profileEl.addEventListener('change', () => {
    note(); try { localStorage.setItem('aeris-profile', profileEl.value); } catch { /* private mode */ }
  });
  note();
}
const RECOMMENDED = 'a380';
const LABEL = {idle: 'Idle', preparing: 'Preparing', ready: 'Ready to run', running: 'Analysing',
  complete: 'Complete', failed: 'Failed', cancelled: 'Cancelled', error: 'Problem'};
let samples = [];
let selectedSlug = '';
let lastStatus = '';
let lastEvent = '';
let lastVideoKey = '';
let lastRunDir = '';
let showingOverlay = true;
let errorText = '';
let busySince = 0;
let current = null;
let autoRun = false;
let summaryData = null;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let data = {};
  try { data = await response.json(); } catch { /* non-JSON error body */ }
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
const post = (path, data = {}) => api(path, {
  method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)
});
const goTo = (el) => el.scrollIntoView({behavior: 'smooth', block: 'start'});

function buildSamples() {
  els.grid.replaceChildren(...samples.map(s => {
    const b = document.createElement('button');
    b.className = 'sample'; b.type = 'button'; b.dataset.slug = s.slug;
    b.setAttribute('aria-pressed', 'false');
    b.innerHTML = '<b></b><small></small>';
    b.querySelector('b').textContent = s.title;
    b.querySelector('small').textContent = s.description;
    if (s.slug === RECOMMENDED) {
      const r = document.createElement('span'); r.className = 'rec'; r.textContent = 'START HERE'; b.append(r);
    }
    b.addEventListener('click', () => choose(s.slug));
    b.addEventListener('mouseenter', () => describe(s.slug));
    b.addEventListener('focus', () => describe(s.slug));
    return b;
  }));
}
function describe(slug) {
  const s = samples.find(x => x.slug === slug);
  els.details.textContent = s
    ? `${s.description}. What the smoke run found: ${s.observation} Credit: ${s.author}, ${s.license}.`
    : 'Hover a sample to see what it shows.';
  els.source.style.visibility = s ? 'visible' : 'hidden';
  if (s) els.source.href = s.page;
}
async function choose(slug) {
  selectedSlug = slug; errorText = ''; summaryData = null;
  describe(slug);
  try { render(await post('/api/select', {slug})); } catch (e) { fail(e); }
}
function fail(error) { autoRun = false; errorText = error.message; if (current) render(current); else paintError(); }
function paintError() {
  els.card.dataset.state = 'error'; els.status.textContent = LABEL.error; els.message.textContent = errorText;
}

function setVideo(src, key) {
  if (key === lastVideoKey) return;
  lastVideoKey = key;
  els.video.pause(); els.video.src = src; els.video.hidden = false; els.empty.hidden = true; els.video.load();
}
function fmtElapsed(ms) {
  const t = Math.floor(ms / 1000);
  return `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}`;
}
function plural(n, one, many) { return `${n} ${n === 1 ? one : many}`; }
function numberOrNull(value) {
  if (value === '' || value == null) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function render(state) {
  current = state;
  const busy = state.status === 'running' || state.status === 'preparing';
  if (busy && !busySince) busySince = Date.now();
  if (!busy) busySince = 0;

  // one-click demo: start the run as soon as the sample is ready
  if (autoRun && state.status === 'ready' && state.selected_name) {
    autoRun = false;
    startLive('selected');
  }

  if (!selectedSlug && state.selected_name) {
    selectedSlug = samples.find(s => s.title === state.selected_name)?.slug || '';
  }
  const shown = errorText ? 'error' : state.status;
  els.card.dataset.state = shown;
  els.led.dataset.state = shown;
  els.status.textContent = LABEL[shown] || shown;
  els.message.textContent = errorText || state.message;
  els.progress.hidden = !busy;
  els.run.disabled = !state.selected_name || busy;
  $('#export-button').disabled = !state.selected_name || busy;
  els.cancel.hidden = state.status !== 'running';
  els.file.disabled = busy;
  els.grid.querySelectorAll('.sample').forEach(b => {
    b.disabled = busy;
    b.setAttribute('aria-pressed', String(b.dataset.slug === selectedSlug));
  });
  if (state.status !== lastEvent) {
    lastEvent = state.status;
    const t = new Date().toTimeString().slice(0, 8);
    events.push(`[${t}] ${(LABEL[state.status] || state.status).toLowerCase()}: ${state.message}`);
    if (events.length > 40) events.shift();
  }
  const out = events.join('\n') + (state.log ? `\n\n${state.log}` : '');
  if (els.log.textContent !== out) { els.log.textContent = out || 'No process output yet.'; els.log.scrollTop = els.log.scrollHeight; }
  els.scan.hidden = state.status !== 'running';

  const justFinished = state.status === 'complete' && lastStatus !== 'complete';
  if (justFinished) showingOverlay = true;
  lastStatus = state.status;

  const src = showingOverlay && state.overlay_url ? state.overlay_url : state.video_url;
  const key = `${state.selected_name || ''}|${state.run_dir || ''}|${showingOverlay ? 'o' : 'v'}|${src || ''}`;
  if (src) {
    setVideo(`${src}?v=${encodeURIComponent(key)}`, key);
    els.mode.textContent = showingOverlay && state.overlay_url ? 'Overlay' : 'Original';
    els.name.textContent = state.selected_name || '';
  } else {
    els.video.pause(); els.video.removeAttribute('src'); els.video.hidden = true; els.empty.hidden = false;
    els.mode.textContent = 'Standby'; els.name.textContent = 'no source'; lastVideoKey = '';
  }
  const hasOverlay = Boolean(state.overlay_url);
  els.viewOrig.disabled = !state.video_url;
  els.viewOver.disabled = !hasOverlay;
  els.viewOrig.setAttribute('aria-pressed', String(Boolean(state.video_url) && !(showingOverlay && hasOverlay)));
  els.viewOver.setAttribute('aria-pressed', String(showingOverlay && hasOverlay));

  // guided stages
  const hasVideo = Boolean(state.selected_name);
  const hasResult = Boolean(state.metrics) && state.status === 'complete';
  const s1 = hasVideo ? 'done' : 'active';
  const s2 = !hasVideo ? 'locked' : hasResult ? 'done' : 'active';
  const s3 = hasResult ? 'active' : 'locked';
  [s1, s2, s3].forEach((v, i) => { els.steps[i].dataset.state = v; });
  els.hint.textContent = errorText ? 'Something went wrong. Read the status below, then try again.'
    : !hasVideo ? 'Pick a source in the bar above the video, or press “Run a one-click demo”.'
    : busy ? 'Working… this page updates by itself.'
    : hasResult ? 'Done. Summary and data are in the findings tabs under the video.'
    : 'Source ready. Press “Run analysis” in step 2.';

  els.results.hidden = !hasResult;
  els.placeholder.hidden = hasResult;
  if (state.metrics) {
    const c = state.metrics.counts || {};
    $('#frames-count').textContent = c.frames_processed ?? '—';
    $('#detections-count').textContent = c.detections_at_or_above_min_score ?? '—';
    $('#tracks-count').textContent = c.tracks_created ?? '—';
    const reasons = Object.values(state.metrics.truth_dependent || {}).filter(i => !i.available).map(i => i.reason);
    $('#metric-reason').textContent = reasons.length
      ? `Accuracy scores are not shown because: ${[...new Set(reasons)].join('; ')}.`
      : 'Truth-dependent metrics require complete annotations.';
    $('#output-path').textContent = state.run_dir || '';
    writeSummary(state);
    if (state.run_dir !== lastRunDir) { lastRunDir = state.run_dir; loadTracks(); }
  } else lastRunDir = '';
  if (justFinished && state.metrics) setTimeout(() => goTo($('#workspace')), 400);
}

function writeSummary(state) {
  const c = state.metrics.counts || {};
  const frames = c.frames_processed ?? 0, dets = c.detections_at_or_above_min_score ?? 0, tracks = c.tracks_created ?? 0;
  const seconds = summaryData?.seconds;
  const dur = seconds ? ` (${seconds.toFixed(1)} s of video)` : '';
  let lead;
  if (!dets) lead = `AERIS analysed ${plural(frames, 'frame', 'frames')}${dur} and found no aircraft above its detection threshold.`;
  else lead = `AERIS analysed ${plural(frames, 'frame', 'frames')}${dur}, recorded ${plural(dets, 'detection', 'detections')}, and created ${plural(tracks, 'track ID', 'track IDs')}.`;
  const note = !dets ? 'That does not mean no aircraft are present — the default detector can miss small or distant ones.'
    : 'These are observations from one run. Without labelled ground truth, accuracy is not measured.';
  els.summary.innerHTML = '';
  els.summary.append(lead, Object.assign(document.createElement('small'), {textContent: note}));
}

/* ---- results: tracks.csv → timeline, paths, table ---- */
function parseCsv(text) {
  const lines = text.trim().split('\n'); const head = lines.shift().split(',');
  return lines.map(l => { const v = l.split(','); return Object.fromEntries(head.map((h, i) => [h, v[i]])); });
}
async function loadTracks() {
  const timeline = $('#timeline'), tbody = $('#track-table tbody');
  try {
    const res = await fetch('/api/artifact/tracks.csv');
    if (!res.ok) throw new Error('tracks.csv unavailable');
    const rows = parseCsv(await res.text());
    const by = new Map();
    rows.forEach(r => { if (!by.has(r.track_id)) by.set(r.track_id, []); by.get(r.track_id).push(r); });
    const times = rows.map(r => numberOrNull(r.timestamp_s)).filter(x => x !== null);
    const tmin = times.reduce((min, t) => Math.min(min, t), Infinity);
    const tmax = times.reduce((max, t) => Math.max(max, t), -Infinity);
    const firstTime = times.length ? tmin : 0;
    const lastTime = times.length ? tmax : 0;
    const span = Math.max(lastTime - firstTime, 0.1);
    summaryData = {seconds: times.length ? lastTime - firstTime : null};
    if (current?.metrics) writeSummary(current);
    $('#timeline-axis').textContent = `${firstTime.toFixed(1)} – ${lastTime.toFixed(1)} s`;
    const lanes = document.createDocumentFragment();
    for (const [id, rs] of by) {
      const lane = document.createElement('div'); lane.className = 'lane';
      const bar = document.createElement('div'); bar.className = 'lane-bar';
      rs.forEach(r => {
        const i = document.createElement('i');
        i.style.left = `${((Number(r.timestamp_s) - firstTime) / span) * 100}%`;
        i.style.width = `${Math.max(100 / (rows.length || 1), 0.6)}%`;
        if (r.observed === '0' || r.lifecycle === 'lost') i.className = 'pred';
        else if (r.lifecycle === 'tentative') i.className = 'tent';
        bar.append(i);
      });
      lane.append(Object.assign(document.createElement('span'), {textContent: id}), bar);
      lanes.append(lane);
    }
    timeline.replaceChildren(lanes);
    if (!by.size) timeline.innerHTML = '<p class="muted">No tracks were created in this run.</p>';
    const tableRows = document.createDocumentFragment();
    for (const [id, rs] of by) {
      const tr = document.createElement('tr');
      const speeds = rs.map(r => numberOrNull(r.speed_px_s)).filter(x => x !== null);
      const mean = speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : null;
      const confirmed = rs.filter(r => r.lifecycle === 'confirmed').length;
      [id, `${(+rs[0].timestamp_s).toFixed(1)} s`, `${(+rs.at(-1).timestamp_s).toFixed(1)} s`, rs.length,
       confirmed, mean === null ? 'n/a' : `${mean.toFixed(1)} px/s`]
        .forEach(v => tr.append(Object.assign(document.createElement('td'), {textContent: v})));
      tableRows.append(tr);
    }
    tbody.replaceChildren(tableRows);
    drawPaths(by);
  } catch (e) {
    const message = document.createElement('p');
    message.className = 'muted'; message.textContent = e.message;
    timeline.replaceChildren(message);
    tbody.replaceChildren();
  }
}
function drawPaths(by) {
  const cv = $('#paths'), ctx = cv.getContext('2d');
  const all = [...by.values()].flat();
  const W = all.reduce((max, r) => Math.max(max, numberOrNull(r.x_max_px) || 0), Math.max(els.video.videoWidth || 0, 1));
  const H = all.reduce((max, r) => Math.max(max, numberOrNull(r.y_max_px) || 0), Math.max(els.video.videoHeight || 0, 1));
  cv.height = Math.round(cv.width * H / W);
  $('#paths-axis').textContent = `${Math.round(W)} × ${Math.round(H)} px`;
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.strokeStyle = '#10231a'; ctx.lineWidth = 1;
  for (let x = 0; x <= cv.width; x += cv.width / 8) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, cv.height); ctx.stroke(); }
  for (let y = 0; y <= cv.height; y += cv.height / 6) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cv.width, y); ctx.stroke(); }
  const sx = cv.width / W, sy = cv.height / H;
  const colors = ['#00ff41', '#ffb02e', '#4fd8ff', '#ff6b9d', '#b79cff'];
  [...by.values()].forEach((rs, k) => {
    const points = rs.map(r => ({x: numberOrNull(r.cx_px), y: numberOrNull(r.cy_px),
      predicted: r.observed === '0' || r.lifecycle === 'lost'}));
    const valid = points.filter(p => p.x !== null && p.y !== null);
    if (!valid.length) return;
    ctx.strokeStyle = colors[k % colors.length]; ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = 2;
    let previous = null;
    for (const point of points) {
      if (point.x === null || point.y === null) { previous = null; continue; }
      if (previous) {
        ctx.setLineDash(point.predicted || previous.predicted ? [5, 4] : []);
        ctx.beginPath(); ctx.moveTo(previous.x * sx, previous.y * sy);
        ctx.lineTo(point.x * sx, point.y * sy); ctx.stroke();
      }
      previous = point;
    }
    ctx.setLineDash([]);
    const last = valid.at(-1);
    ctx.beginPath(); ctx.arc(last.x * sx, last.y * sy, 4, 0, 7); ctx.fill();
  });
}

/* ---- wiring ---- */
async function refresh() { try { render(await api('/api/state')); } catch (e) { errorText = e.message; paintError(); } }
async function useFile(file) {
  if (file.size > 500_000_000) { fail(new Error('Video exceeds 500 MB.')); return; }
  errorText = ''; selectedSlug = ''; describe('');
  goTo($('#workspace'));
  els.card.dataset.state = 'preparing'; els.status.textContent = LABEL.preparing; els.message.textContent = 'Copying video locally…';
  try { render(await api('/api/upload', {method: 'POST', headers: {'X-Filename': encodeURIComponent(file.name)}, body: file})); }
  catch (e) { fail(e); }
}
els.file.addEventListener('change', async () => {
  const file = els.file.files?.[0]; if (!file) return;
  await useFile(file); els.file.value = '';
});
$('#hero-own').addEventListener('click', () => els.file.click());
$('#choose-file').addEventListener('click', () => els.file.click());
$('#hero-demo').addEventListener('click', async () => {
  goTo($('#workspace')); autoRun = true;
  if (current?.selected_name) { autoRun = false; startLive('selected'); } else choose(RECOMMENDED);
});

$('#again').addEventListener('click', () => goTo($('#workspace')));
els.run.addEventListener('click', () => startLive('selected'));
$('#export-button').addEventListener('click', async () => { errorText = ''; try { render(await post('/api/run', {profile: profileEl.value})); } catch (e) { fail(e); } });
els.cancel.addEventListener('click', async () => { autoRun = false; try { render(await post('/api/cancel')); } catch (e) { fail(e); } });
els.viewOrig.addEventListener('click', () => { showingOverlay = false; if (current) render(current); });
els.viewOver.addEventListener('click', () => { showingOverlay = true; if (current) render(current); });
const mmss = (t) => `${String(Math.floor(t / 60)).padStart(2, '0')}:${(t % 60).toFixed(1).padStart(4, '0')}`;
els.video.addEventListener('timeupdate', () => { els.time.textContent = `${mmss(els.video.currentTime)} / ${mmss(els.video.duration || 0)}`; });
els.video.addEventListener('loadedmetadata', () => { els.time.textContent = `${mmss(0)} / ${mmss(els.video.duration || 0)}`; });
document.querySelectorAll('#tabs [data-tab]').forEach(tab => tab.addEventListener('click', () => {
  document.querySelectorAll('#tabs [data-tab]').forEach(t => t.setAttribute('aria-selected', String(t === tab)));
  document.querySelectorAll('.tab-panel').forEach(p => { p.hidden = p.dataset.panel !== tab.dataset.tab; });
}));
setInterval(() => { els.elapsed.textContent = busySince ? `ELAPSED ${fmtElapsed(Date.now() - busySince)}` : ''; }, 500);

(async function init() {
  try {
    samples = (await api('/api/samples')).samples;
    await loadProfiles();
    buildSamples();
    await refresh();
  } catch (e) { errorText = e.message; paintError(); }
  setInterval(refresh, 1200);
})();

// --- live mode: server-side loop streams annotated MJPEG; state comes from /api/state.live ---
const live = {view: $('#live-view'), stop: $('#live-stop'), status: $('#live-status')};
async function startLive(source) {
  errorText = '';
  live.status.textContent = 'Loading detector…';
  try {
    await post('/api/live/start', {source, profile: profileEl.value});
    live.view.src = `/api/live/stream?t=${Date.now()}`;
  } catch (e) { live.status.textContent = ''; fail(e); }
}
function paintLive(state) {
  const l = state.live, on = !!l && ['starting', 'running'].includes(l.state);
  live.stop.hidden = !on;
  live.view.hidden = !(on || (l && !state.video_url));
  if (!live.view.hidden) { els.video.hidden = true; els.empty.hidden = true; }
  live.status.textContent = !l ? '' : l.state === 'running'
    ? `LIVE · ${l.fps ? l.fps.toFixed(1) : '—'} fps · ${l.latency_ms ? l.latency_ms.toFixed(0) : '—'} ms/frame · ${l.detections} det · ${l.tracks} tracks · ${l.dropped} dropped`
    : `${l.state}: ${l.message}`;
  paintPanel(l);
}
const panelEl = $('#live-panel');
function scoreHtml(l) {
  const m = l.score && l.score.available && l.score.models.constant_velocity;
  if (!m) return '<div class="mono-dim">prediction error: waiting for forecasts to come due…</div>';
  const rows = Object.entries(m).filter(([, h]) => h.scored).map(([h, v]) =>
    `<tr><td>+${h}s</td><td>${v.median_error_px.toFixed(1)} px</td><td>${v.median_direction_error_deg == null ? '—' : v.median_direction_error_deg.toFixed(0) + '°'}</td><td>n=${v.scored}</td></tr>`).join('');
  return `<div class="lp-track"><b>Prediction error</b> <span class="mono-dim">last 60 s, vs later detections (not ground truth)</span>` +
    `<table><thead><tr><td></td><td>median pos</td><td>median dir</td><td></td></tr></thead><tbody>${rows}</tbody></table></div>`;
}
function paintPanel(l) {
  const rows = (l && l.state === 'running' && l.panel) || [];
  if (!rows.length) { panelEl.innerHTML = '<span class="muted">No confirmed tracks yet.</span>' + (l && l.score ? scoreHtml(l) : ''); return; }
  const f1 = n => (n == null ? '—' : n.toFixed(0));
  const safe = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const bearing = b => !b ? 'no current measurement' : b.azimuth_deg == null
    ? `unavailable: ${safe(b.availability_reason)}`
    : `${b.azimuth_deg.toFixed(1)}° az · ${b.elevation_deg.toFixed(1)}° el`;
  panelEl.innerHTML = rows.map(r => {
    const fc = r.forecast.map(f => `<tr><td>+${f.horizon_s}s</td><td>${f1(f.x)}, ${f1(f.y)}</td><td>±${f.sigma_px.toFixed(0)} px</td></tr>`).join('');
    const world = (r.forecast_bearings || []).map(b =>
      `<tr><td>+${b.horizon_s}s</td><td>${bearing(b)}</td></tr>`).join('');
    return `<div class="lp-track"><b>${r.id}</b> <span class="mono-dim">${r.lifecycle} · ${r.observations} obs · ` +
      `${r.speed_px_s == null ? 'speed —' : r.speed_px_s.toFixed(0) + ' px/s'} · ` +
      `${r.heading_deg == null ? 'heading unresolved' : r.heading_deg.toFixed(0) + '°'}</span>` +
      (fc ? `<table><tbody>${fc}</tbody></table>` : '<div class="mono-dim">warming up (not enough history)</div>') +
      `<div class="mono-dim">Surveyed-site bearing (unvalidated angular accuracy): ${bearing(r.observed_bearing)}</div>` +
      (world ? `<table><tbody>${world}</tbody></table>` : '') + '</div>';
  }).join('') + scoreHtml(l);
}
$('#live-camera').addEventListener('click', () => startLive('camera:0'));
$('#live-url-go').addEventListener('click', () => { const u = $('#live-url').value.trim(); if (u) startLive(u); });
live.stop.addEventListener('click', async () => { try { await post('/api/live/stop'); } catch (e) { fail(e); } });
setInterval(() => { if (current) paintLive(current); }, 500);
