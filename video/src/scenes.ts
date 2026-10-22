import formation from "./data/formation.json";
import redArrows from "./data/red-arrows.json";
import airForceOne from "./data/air-force-one.json";

// One observed AERIS track on a frame, source pixels (1920x1080):
// [track_id, x_min, y_min, x_max, y_max, cx, cy, speed_px_s, heading_deg|null, forecast_x|null, forecast_y|null]
// The forecast point is the 1 s arrow AERIS's own overlay draws (gated, ADR 026), or null when it draws none.
export type Box = [string, number, number, number, number, number, number, number | null, number | null, number | null, number | null];
export type Stats = { frames: number; detections: number; arrows: number; inferMs: number; fcErrPx: number; fcDirDeg: number; fcScored: number };
export type Clip = { frames: Box[][]; stats: Stats };

export type Scene = {
  video: string;
  clip: Clip;
  from: number; // start in source seconds
  dur: number; // seconds on screen
  lockAt: number; // seconds into the scene when boxes switch on
  trailAt: number | null; // when trajectory trails start drawing
  predictAt: number | null; // when forecast arrows start drawing
  zoom: [number, number];
  vZoom: number; // 9:16 framing: 1 = fill height, lower = show more width over a blurred fill
  labels: "full" | "id" | "none";
  panel: false | "left" | "right"; // prediction numbers panel (16:9 side)
  kicker: string;
  title: string;
  credit: string;
  bare?: boolean; // footage + boxes only: no HUD, panel or title (used under the intro)
};

export const FPS = 30;
export const OPEN = 3.5;
export const OUTRO = 6;
const F = formation as Clip, R = redArrows as Clip, A = airForceOne as Clip;
export const CLIPS = [F, R, A];

// Windows chosen from per-frame checks of tracks.csv (see README.md).
export const SCENES: Scene[] = [
  {
    video: "formation.mp4", clip: F, from: 6.5, dur: 3.5, lockAt: 0.3, trailAt: 0.6, predictAt: 1.6,
    zoom: [1.05, 1.22], vZoom: 0.62, labels: "full", panel: "left",
    kicker: "01 — TRACK + PREDICT", title: "Four jets. Four locks. Every frame.",
    credit: "Footage: Lucastphotography, CC BY 4.0, Wikimedia Commons",
  },
  {
    video: "red-arrows.mp4", clip: R, from: 8.57, dur: 3.8, lockAt: 0.3, trailAt: 0.6, predictAt: 1.2,
    zoom: [1.08, 1.25], vZoom: 0.62, labels: "none", panel: "left",
    kicker: "02 — PREDICT AT SCALE", title: "Nine jets, nine predicted headings.",
    credit: "Footage: Jo Brodie, CC0, Wikimedia Commons",
  },
  {
    video: "air-force-one.mp4", clip: A, from: 0, dur: 4.1, lockAt: 0.5, trailAt: 0.8, predictAt: 1.3,
    zoom: [1.05, 1.35], vZoom: 1.0, labels: "id", panel: "right",
    kicker: "03 — PREDICT", title: "Where it will be one second from now.",
    credit: "Footage: Designism, CC0, Wikimedia Commons",
  },
];

// Footage under the intro: runs 3.0-6.5 s, straight into scene 1 (6.5-10 s).
export const INTRO_SHOTS: Record<string, Scene> = {
  hero: {
    video: "formation.mp4", clip: F, from: 3.0, dur: OPEN, lockAt: 1.0, trailAt: 1.4, predictAt: null,
    zoom: [1.5, 1.38], vZoom: 1, labels: "none", panel: false, kicker: "", title: "", bare: true,
    credit: "Footage: Lucastphotography, CC BY 4.0, Wikimedia Commons",
  },
};

export const TOTAL = Math.round((OPEN + SCENES.reduce((s, x) => s + x.dur, 0) + OUTRO) * FPS);
