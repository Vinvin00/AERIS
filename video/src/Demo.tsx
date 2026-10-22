import { useMemo } from "react";
import {
  AbsoluteFill, Easing, OffthreadVideo, Sequence, interpolate, spring, staticFile,
  useCurrentFrame, useVideoConfig,
} from "remotion";
import { loadFont as loadInter } from "@remotion/google-fonts/Inter";
import { loadFont as loadMono } from "@remotion/google-fonts/JetBrainsMono";
import { Box, CLIPS, FPS, INTRO_SHOTS, OPEN, OUTRO, SCENES, Scene } from "./scenes";

const { fontFamily: SANS } = loadInter("normal", { weights: ["300", "400", "600"] });
const { fontFamily: MONO } = loadMono("normal", { weights: ["400", "500"] });
const ACCENT = "#8CFFC1"; // detection + track
const PREDICT = "#FFC46B"; // forecast
const SRC_W = 1920;
const SRC_H = 1080;
const TRAIL = 40; // frames of history drawn behind each track
const ease = Easing.bezier(0.45, 0, 0.2, 1);
const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

const fadeEdges = (f: number, total: number, d = 10) => interpolate(f, [0, d, total - d, total], [0, 1, 1, 0], clamp);
const fmt = (n: number) => Math.round(n).toLocaleString("en-US");

// Camera: smoothed centroid of the boxes on screen, so the push-in drifts toward the aircraft.
function useFocus(scene: Scene) {
  return useMemo(() => {
    const start = Math.round(scene.from * FPS);
    const raw = Array.from({ length: Math.round(scene.dur * FPS) }, (_, i) => {
      const boxes = scene.clip.frames[start + i] ?? [];
      if (!boxes.length) return null;
      const xs = boxes.flatMap((b) => [b[1], b[3]]);
      const ys = boxes.flatMap((b) => [b[2], b[4]]);
      return [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2];
    });
    const R = 24;
    return raw.map((_, i) => {
      const near = raw.slice(Math.max(0, i - R), i + R + 1).filter(Boolean) as number[][];
      if (!near.length) return [SRC_W / 2, SRC_H / 2];
      return [near.reduce((s, p) => s + p[0], 0) / near.length, near.reduce((s, p) => s + p[1], 0) / near.length];
    });
  }, [scene]);
}

const Grain = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{ opacity: 0.09, mixBlendMode: "overlay", pointerEvents: "none" }}>
      <svg width="100%" height="100%">
        <filter id="g"><feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves={2} seed={frame % 12} /></filter>
        <rect width="100%" height="100%" filter="url(#g)" />
      </svg>
    </AbsoluteFill>
  );
};

const Vignette = () => (
  <AbsoluteFill style={{ background: "radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,0.55) 100%)" }} />
);

function Brackets({ x0, y0, x1, y1, p, label }: { x0: number; y0: number; x1: number; y1: number; p: number; label?: string }) {
  // p: 0 -> 1 lock-on progress; brackets close in from 1.7x and fade up
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const k = interpolate(p, [0, 1], [1.7, 1]);
  const w = ((x1 - x0) / 2) * k, h = ((y1 - y0) / 2) * k;
  const L = Math.max(8, Math.min(36, Math.min(w, h) * 0.55));
  const [a, b, c, d] = [cx - w, cy - h, cx + w, cy + h];
  const path = `M${a},${b + L}V${b}H${a + L} M${c - L},${b}H${c}V${b + L} M${c},${d - L}V${d}H${c - L} M${a + L},${d}H${a}V${d - L}`;
  return (
    <g opacity={p}>
      <path d={path} stroke={ACCENT} strokeWidth={3} fill="none" style={{ filter: `drop-shadow(0 0 6px ${ACCENT}88)` }} />
      {label && (
        <text x={a} y={b - 10} fill={ACCENT} fontFamily={MONO} fontSize={17} fontWeight={500} letterSpacing={1.2}
          style={{ paintOrder: "stroke", stroke: "rgba(0,0,0,0.45)", strokeWidth: 3 }}>{label}</text>
      )}
    </g>
  );
}

// Past observed centers of one track, fading with age.
function Trail({ pts, opacity }: { pts: { x: number; y: number; age: number }[]; opacity: number }) {
  return (
    <g opacity={opacity}>
      {pts.slice(1).map((p, i) => {
        const q = pts[i];
        return <line key={i} x1={q.x} y1={q.y} x2={p.x} y2={p.y} stroke={ACCENT} strokeLinecap="round"
          strokeWidth={interpolate(p.age, [0, TRAIL], [3.5, 1])} strokeOpacity={interpolate(p.age, [0, TRAIL], [0.9, 0])} />;
      })}
    </g>
  );
}

// AERIS's 1 s forecast: dashed arrow flowing forward, ring at the predicted position.
function Forecast({ x, y, fx, fy, p, frame, label }: { x: number; y: number; fx: number; fy: number; p: number; frame: number; label: boolean }) {
  const len = Math.hypot(fx - x, fy - y);
  if (len < 4) return null;
  const ux = (fx - x) / len, uy = (fy - y) / len;
  const tipX = x + (fx - x) * p, tipY = y + (fy - y) * p;
  const head = 15;
  return (
    <g style={{ filter: `drop-shadow(0 0 5px ${PREDICT}99)` }}>
      <line x1={x} y1={y} x2={tipX} y2={tipY} stroke={PREDICT} strokeWidth={3.5} strokeDasharray="10 7"
        strokeDashoffset={-frame * 1.6} strokeLinecap="round" />
      <path d={`M${tipX - ux * head - uy * head * 0.55},${tipY - uy * head + ux * head * 0.55} L${tipX},${tipY} L${tipX - ux * head + uy * head * 0.55},${tipY - uy * head - ux * head * 0.55}`}
        stroke={PREDICT} strokeWidth={3.5} fill="none" opacity={p} />
      <circle cx={fx} cy={fy} r={11 + 3 * Math.sin(frame / 4)} stroke={PREDICT} strokeWidth={2} fill="none" opacity={p * 0.8} />
      {label && p > 0.9 && (
        <text x={fx + 14} y={fy + 5} fill={PREDICT} fontFamily={MONO} fontSize={15} fontWeight={500}
          style={{ paintOrder: "stroke", stroke: "rgba(0,0,0,0.45)", strokeWidth: 3 }}>+1.0 s</text>
      )}
    </g>
  );
}

function Counter({ value, frame, start, decimals = 0 }: { value: number; frame: number; start: number; decimals?: number }) {
  const v = value * interpolate(frame, [start, start + 24], [0, 1], { ...clamp, easing: ease });
  return <>{decimals ? v.toFixed(decimals) : fmt(v)}</>;
}

function Panel({ scene, frame, at, vertical }: { scene: Scene; frame: number; at: number; vertical: boolean }) {
  const s = scene.clip.stats;
  const p = spring({ frame: frame - at, fps: FPS, config: { damping: 200 } });
  const row = (k: string, v: React.ReactNode, color = "white") => (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 40, padding: "7px 0", borderTop: "1px solid rgba(255,255,255,0.12)" }}>
      <span style={{ color: "rgba(255,255,255,0.6)" }}>{k}</span><span style={{ color }}>{v}</span>
    </div>
  );
  return (
    <div style={{
      position: "absolute", [vertical || scene.panel === "right" ? "right" : "left"]: vertical ? 64 : 96,
      top: vertical ? 196 : 190, width: vertical ? 480 : 430,
      padding: vertical ? "14px 18px" : "20px 24px", background: "rgba(6,10,16,0.55)", backdropFilter: "blur(14px)",
      border: "1px solid rgba(255,255,255,0.14)", borderRadius: 6, fontFamily: MONO, fontSize: vertical ? 18 : 17,
      letterSpacing: 0.5, opacity: p, transform: `translateX(${(1 - p) * (scene.panel === "left" && !vertical ? -30 : 30)}px)`,
    }}>
      <div style={{ color: PREDICT, letterSpacing: 3, marginBottom: 10 }}>PREDICTION · THIS CLIP</div>
      {row("Horizon", "1.0 s")}
      {row("Median position error", <><Counter value={s.fcErrPx} frame={frame} start={at + 6} /> px</>, PREDICT)}
      {row("Median heading error", <><Counter value={s.fcDirDeg} frame={frame} start={at + 10} decimals={1} />°</>, PREDICT)}
      {row("Forecasts scored", <Counter value={s.fcScored} frame={frame} start={at + 14} />)}
      {row("Inference", <><Counter value={s.inferMs} frame={frame} start={at + 18} /> ms/frame</>)}
      <div style={{ color: "rgba(255,255,255,0.45)", fontFamily: SANS, fontSize: vertical ? 14 : 13, marginTop: 10, lineHeight: 1.4 }}>
        Scored against where AERIS later detected the aircraft, not hand-labelled truth.
      </div>
    </div>
  );
}

function Shot({ scene }: { scene: Scene }) {
  const frame = useCurrentFrame();
  const { width: W, height: H, fps } = useVideoConfig();
  const vertical = H > W;
  const n = Math.round(scene.dur * fps);
  const start = Math.round(scene.from * fps);
  const focus = useFocus(scene);
  const t = Math.min(frame, n - 1);
  const data = scene.clip.frames;

  // camera
  const cover = Math.max(W / SRC_W, H / SRC_H);
  const z = interpolate(t, [0, n - 1], scene.zoom, { easing: ease });
  const s = (vertical ? cover * scene.vZoom : cover) * z;
  const follow = interpolate(t, [0, n - 1], [0.35, 0.8]);
  const px = SRC_W / 2 + (focus[t][0] - SRC_W / 2) * follow;
  const py = SRC_H / 2 + (focus[t][1] - SRC_H / 2) * follow;
  const clampAxis = (pos: number, src: number, out: number) =>
    src * s >= out ? Math.min(0, Math.max(out - src * s, pos)) : (out - src * s) / 2;
  const tx = clampAxis((scene.bare ? W * 0.72 : W / 2) - px * s, SRC_W, W); // intro: aircraft in the right third
  const ty = clampAxis(H / 2 - py * s, SRC_H, H);
  const X = (x: number) => tx + x * s, Y = (y: number) => ty + y * s;
  const fills = SRC_W * s >= W && SRC_H * s >= H;

  // timing
  const lockF = Math.round(scene.lockAt * fps);
  const lockP = spring({ frame: frame - lockF, fps, config: { damping: 18, stiffness: 120 } });
  const sweep = interpolate(frame, [lockF - 4, lockF + 14], [-0.1, 1.1], clamp);
  const trailF = scene.trailAt == null ? Infinity : Math.round(scene.trailAt * fps);
  const predF = scene.predictAt == null ? Infinity : Math.round(scene.predictAt * fps);
  const trailP = interpolate(frame, [trailF, trailF + 15], [0, 1], clamp);

  const ids = useMemo(() => {
    const m = new Map<string, number>();
    for (let i = 0; i < n; i++) for (const b of data[start + i] ?? []) if (!m.has(b[0])) m.set(b[0], m.size + 1);
    return m;
  }, [data, n, start]);
  const firstSeen = useMemo(() => {
    const m = new Map<string, number>();
    for (let i = 0; i < n; i++) for (const b of data[start + i] ?? []) if (!m.has(b[0])) m.set(b[0], i);
    return m;
  }, [data, n, start]);

  const boxes: Box[] = data[start + t] ?? [];
  const trailOf = (id: string) => {
    const pts: { x: number; y: number; age: number }[] = [];
    for (let k = Math.min(TRAIL, t); k >= 0; k--) {
      const b = (data[start + t - k] ?? []).find((q) => q[0] === id);
      if (b) pts.push({ x: X(b[5]), y: Y(b[6]), age: k });
    }
    return pts;
  };
  const label = (b: Box) => {
    if (scene.labels === "none") return undefined;
    const id = `T${ids.get(b[0])}`;
    if (scene.labels === "id" || frame < trailF) return id;
    return `${id} · ${b[7] ?? "–"} px/s${b[8] == null ? "" : ` · ${String(b[8]).padStart(3, "0")}°`}`;
  };

  const grade = "contrast(1.12) saturate(1.08) brightness(0.94)";
  const video = (style: React.CSSProperties) => (
    <OffthreadVideo src={staticFile(scene.video)} startFrom={start} muted style={style} />
  );
  const titleIn = spring({ frame: frame - lockF - 10, fps, config: { damping: 200 } });
  const titleOut = interpolate(frame, [n - 22, n - 8], [1, 0], clamp);
  const hud = interpolate(frame, [lockF, lockF + 12], [0, 1], clamp);
  const pad = vertical ? 64 : 96;
  const bar = vertical ? 0 : 92; // letterbox
  const srcT = (scene.from + t / fps).toFixed(2).padStart(5, "0");
  const arrowsNow = frame >= predF ? boxes.filter((b) => b[9] != null).length : 0;

  return (
    <AbsoluteFill style={{ backgroundColor: "black", opacity: scene.bare ? interpolate(frame, [n - 9, n], [1, 0], clamp) : fadeEdges(frame, n, 9) }}>
      {!fills && video({ position: "absolute", width: "100%", height: "100%", objectFit: "cover", filter: `blur(50px) brightness(0.35) ${grade}`, transform: "scale(1.2)" })}
      {video({
        position: "absolute", left: tx, top: ty, width: SRC_W * s, height: SRC_H * s, filter: grade,
        ...(fills ? {} : { WebkitMaskImage: "linear-gradient(transparent, black 14%, black 86%, transparent)" }),
      })}
      <svg width={W} height={H} style={{ position: "absolute" }}>
        {frame >= trailF && boxes.map((b) => <Trail key={`tr${b[0]}`} pts={trailOf(b[0])} opacity={trailP} />)}
        {frame >= predF && boxes.map((b) => {
          if (b[9] == null || b[10] == null) return null;
          const p = interpolate(frame, [predF, predF + 14], [0, 1], { ...clamp, easing: ease });
          return <Forecast key={`fc${b[0]}`} x={X(b[5])} y={Y(b[6])} fx={X(b[9])} fy={Y(b[10])} p={p} frame={frame}
            label={scene.labels !== "none"} />;
        })}
        {frame >= lockF - 2 && boxes.map((b) => {
          const own = spring({ frame: t - Math.max(lockF, firstSeen.get(b[0]) ?? 0), fps, config: { damping: 18, stiffness: 120 } });
          return <Brackets key={b[0]} x0={X(b[1])} y0={Y(b[2])} x1={X(b[3])} y1={Y(b[4])} p={Math.min(lockP, own)} label={label(b)} />;
        })}
      </svg>
      {sweep > -0.1 && sweep < 1.1 && (
        <AbsoluteFill style={{
          background: `linear-gradient(90deg, transparent ${sweep * 100 - 12}%, ${ACCENT}33 ${sweep * 100 - 1}%, ${ACCENT}cc ${sweep * 100}%, transparent ${sweep * 100 + 0.4}%)`,
        }} />
      )}
      <Vignette />
      <Grain />
      {bar > 0 && <>
        <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: bar, background: "black" }} />
        <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: bar, background: "black" }} />
      </>}
      {!scene.bare && <div style={{ position: "absolute", inset: 0, opacity: hud, fontFamily: MONO, color: "rgba(255,255,255,0.85)", fontSize: vertical ? 22 : 18, letterSpacing: 2 }}>
        <div style={{ position: "absolute", left: pad, top: bar + (vertical ? 150 : 34) }}>
          AERIS <span style={{ color: ACCENT }}>●</span> TRACK VIEW
        </div>
        <div style={{ position: "absolute", right: pad, top: bar + (vertical ? 150 : 34), textAlign: "right" }}>
          T+{srcT}s · 30 FPS · {scene.clip.stats.inferMs} MS/FRAME
        </div>
        <div style={{ position: "absolute", right: pad, bottom: bar + (vertical ? 170 : 34), textAlign: "right", lineHeight: 1.7 }}>
          TARGETS <span style={{ color: ACCENT }}>{String(boxes.length).padStart(2, "0")}</span>
          {frame >= predF && <><br />FORECASTS <span style={{ color: PREDICT }}>{String(arrowsNow).padStart(2, "0")}</span></>}
        </div>
      </div>}
      {!scene.bare && scene.panel && frame >= predF && <Panel scene={scene} frame={frame} at={predF + 6} vertical={vertical} />}
      {!scene.bare && <div style={{
        position: "absolute", left: pad, right: pad, bottom: bar + (vertical ? 260 : 74),
        opacity: titleIn * titleOut, transform: `translateY(${(1 - titleIn) * 24}px)`,
      }}>
        <div style={{ fontFamily: MONO, color: scene.panel ? PREDICT : ACCENT, fontSize: vertical ? 26 : 20, letterSpacing: 4, marginBottom: 14 }}>{scene.kicker}</div>
        <div style={{ fontFamily: SANS, color: "white", fontWeight: 600, fontSize: vertical ? 64 : 54, lineHeight: 1.1, letterSpacing: -1, textShadow: "0 2px 24px rgba(0,0,0,0.5)", maxWidth: vertical ? undefined : 1100 }}>
          {scene.title}
        </div>
      </div>}
      <div style={{ position: "absolute", left: pad, bottom: vertical ? 90 : 34, fontFamily: SANS, fontSize: vertical ? 20 : 15, color: "rgba(255,255,255,0.55)" }}>
        {scene.credit}
      </div>
    </AbsoluteFill>
  );
}

// Shared title card: intro and outro use the same look.
function Card({ seconds, children }: { seconds: number; children: (up: (at: number) => React.CSSProperties, frame: number, v: boolean) => React.ReactNode }) {
  const frame = useCurrentFrame();
  const { width: W, height: H, fps } = useVideoConfig();
  const v = H > W;
  const up = (at: number): React.CSSProperties => ({
    opacity: interpolate(frame, [at, at + 15], [0, 1], clamp),
    transform: `translateY(${interpolate(frame, [at, at + 20], [16, 0], { ...clamp, easing: ease })}px)`,
  });
  const spacing = interpolate(frame, [0, seconds * fps], [v ? 18 : 30, v ? 26 : 42]);
  return (
    <AbsoluteFill style={{ backgroundColor: "#05080d", justifyContent: "center", alignItems: "center", textAlign: "center", padding: v ? 80 : 200, opacity: fadeEdges(frame, seconds * fps, 12) }}>
      <div style={{ ...up(4), fontFamily: SANS, fontWeight: 600, color: "white", fontSize: v ? 150 : 140, letterSpacing: spacing, marginRight: -spacing }}>AERIS</div>
      <div style={{ ...up(14), width: 120, height: 4, background: ACCENT, margin: "26px 0 34px" }} />
      {children(up, frame, v)}
      <Grain />
    </AbsoluteFill>
  );
}

// Intro: title over real footage, boxes locking on behind it (left-aligned, editorial).
function OpenOverFootage({ shot }: { shot: Scene }) {
  const frame = useCurrentFrame();
  const up = (at: number): React.CSSProperties => ({
    opacity: interpolate(frame, [at, at + 15], [0, 1], clamp),
    transform: `translateY(${interpolate(frame, [at, at + 20], [18, 0], { ...clamp, easing: ease })}px)`,
  });
  const out = interpolate(frame, [OPEN * FPS - 12, OPEN * FPS], [1, 0], clamp);
  const spacing = interpolate(frame, [0, OPEN * FPS], [26, 38]);
  return (
    <AbsoluteFill>
      <Shot scene={shot} />
      <AbsoluteFill style={{ background: "linear-gradient(90deg, rgba(3,6,10,0.92) 0%, rgba(3,6,10,0.65) 38%, rgba(3,6,10,0) 70%)", opacity: out }} />
      <div style={{ position: "absolute", left: 120, top: 0, bottom: 0, display: "flex", flexDirection: "column", justifyContent: "center", opacity: out }}>
        <div style={{ ...up(4), fontFamily: MONO, color: ACCENT, fontSize: 22, letterSpacing: 6, marginBottom: 18 }}>DETECT · TRACK · <span style={{ color: PREDICT }}>PREDICT</span></div>
        <div style={{ ...up(10), fontFamily: SANS, fontWeight: 600, color: "white", fontSize: 168, letterSpacing: spacing, lineHeight: 1 }}>AERIS</div>
        <div style={{ ...up(18), width: 120, height: 4, background: ACCENT, margin: "30px 0 30px" }} />
        <div style={{ ...up(26), fontFamily: SANS, color: "rgba(255,255,255,0.9)", fontSize: 36, fontWeight: 300, lineHeight: 1.3 }}>
          Aircraft detection, tracking and<br />trajectory prediction from one camera.
        </div>
      </div>
    </AbsoluteFill>
  );
}

const Open = () => <OpenOverFootage shot={INTRO_SHOTS.hero} />;

const totals = CLIPS.reduce((a, c) => ({ frames: a.frames + c.stats.frames, detections: a.detections + c.stats.detections, arrows: a.arrows + c.stats.arrows }), { frames: 0, detections: 0, arrows: 0 });

const Outro = () => (
  <Card seconds={OUTRO}>
    {(up, frame, v) => <>
      <div style={{ ...up(22), fontFamily: SANS, color: "white", fontSize: v ? 44 : 38, fontWeight: 400, maxWidth: 1200, lineHeight: 1.3 }}>
        Aircraft detection, tracking and trajectory forecasting from a single camera.
      </div>
      <div style={{ ...up(34), display: "flex", flexDirection: v ? "column" : "row", gap: v ? 28 : 90, marginTop: 50 }}>
        {[["FRAMES ANALYSED", totals.frames, ACCENT], ["DETECTIONS", totals.detections, ACCENT], ["FORECASTS DRAWN", totals.arrows, PREDICT]].map(([k, n, c], i) => (
          <div key={k as string}>
            <div style={{ fontFamily: SANS, fontWeight: 600, fontSize: v ? 72 : 64, color: c as string, letterSpacing: -1 }}>
              <Counter value={n as number} frame={frame} start={38 + i * 6} />
            </div>
            <div style={{ fontFamily: MONO, fontSize: v ? 20 : 16, letterSpacing: 3, color: "rgba(255,255,255,0.6)" }}>{k}</div>
          </div>
        ))}
      </div>
      <div style={{ ...up(60), fontFamily: SANS, color: "rgba(255,255,255,0.55)", fontSize: v ? 24 : 20, marginTop: 46, maxWidth: 1100, lineHeight: 1.5 }}>
        Totals across the three runs shown. Every box and arrow is real AERIS output on public footage, run at 30 fps.
        Research prototype, not a benchmark.
      </div>
      <div style={{ ...up(70), fontFamily: MONO, color: ACCENT, fontSize: v ? 28 : 24, marginTop: 30, letterSpacing: 1 }}>github.com/Vinvin00/AERIS</div>
    </>}
  </Card>
);

export const Demo = () => {
  let at = Math.round(OPEN * FPS);
  return (
    <AbsoluteFill style={{ backgroundColor: "black" }}>
      <Sequence durationInFrames={Math.round(OPEN * FPS)}><Open /></Sequence>
      {SCENES.map((scene) => {
        const from = at;
        at += Math.round(scene.dur * FPS);
        return <Sequence key={scene.video} from={from} durationInFrames={Math.round(scene.dur * FPS)}><Shot scene={scene} /></Sequence>;
      })}
      <Sequence from={at} durationInFrames={Math.round(OUTRO * FPS)}><Outro /></Sequence>
    </AbsoluteFill>
  );
};
