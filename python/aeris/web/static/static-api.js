// Hosted, pre-recorded build: answers /api/* from files under /demo/<slug>/ instead of a server.
(() => {
  const real = window.fetch.bind(window);
  const NO = 'Uploads, cameras and streams are off on the hosted page. See “Run it yourself” below.';
  const blank = {status: 'idle', message: 'Choose a recorded sample.', selected_name: null, source_page: null,
    run_dir: null, log: '', video_url: null, overlay_url: null, metrics: null, live: null};
  let state = {...blank}, slug = null, metrics = null, manifest = null;
  const json = (v, status = 200) => new Response(JSON.stringify(v), {status, headers: {'Content-Type': 'application/json'}});
  const load = () => manifest ??= real('/demo/manifest.json').then(r => r.json());
  const goDocs = (e) => { e.preventDefault(); e.stopImmediatePropagation(); location.hash = 'run-it-yourself'; };
  addEventListener('DOMContentLoaded', () => ['hero-own', 'choose-file'].forEach(id =>
    document.getElementById(id)?.addEventListener('click', goDocs, true)));

  window.fetch = async (input, opts = {}) => {
    const path = (typeof input === 'string' ? input : input.url).split('?')[0];
    if (!path.startsWith('/api/')) return real(input, opts);
    const m = await load(), body = () => JSON.parse(opts.body || '{}');
    switch (path) {
      case '/api/samples': return json({samples: m.samples});
      case '/api/profiles': return json({default: 'fast-zoom', profiles: [{name: 'fast-zoom', label: 'Fast + zoom', note: m.profile_note}]});
      case '/api/state': return json(state);
      case '/api/select': {
        const s = m.samples.find(x => x.slug === body().slug);
        if (!s) return json({error: 'Unknown sample.'}, 400);
        slug = s.slug; metrics = await (await real(`/demo/${slug}/metrics.json`)).json();
        state = {...blank, status: 'ready', message: 'Pre-recorded sample ready. “Run analysis” shows what AERIS produced offline.',
          selected_name: s.title, source_page: s.page, video_url: `/demo/${slug}/original.mp4`};
        return json(state);
      }
      case '/api/run': case '/api/live/start':
        if (path === '/api/live/start' && body().source !== 'selected') return json({error: NO}, 400);
        if (!slug) return json({error: 'Choose a sample first.'}, 400);
        state = {...state, status: 'complete', message: 'Pre-recorded result, not running live.',
          run_dir: `pre-recorded: ${slug}`, overlay_url: `/demo/${slug}/overlay.mp4`, metrics};
        return json(state);
      case '/api/live/stop': case '/api/cancel': return json(state);
      case '/api/upload': return json({error: NO}, 400);
      default:
        if (path.startsWith('/api/artifact/') && slug) return real(`/demo/${slug}/${path.split('/').pop()}`);
        return json({error: 'Not available on the hosted page.'}, 404);
    }
  };
})();
