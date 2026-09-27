"""AERIS-007 measured baseline on AOT: frozen selection, detector run, frozen gate.

The benchmark definition is `configs/evaluation/aot_benchmark.json`, committed before the
detector ran. Steps, each a subcommand:

    python -m aeris.evaluation.baseline select   # frame lists from labels only
    python -m aeris.evaluation.baseline fetch    # development + validation frames (not holdout)
    python -m aeris.evaluation.baseline run --threads 1 --run-id baseline
    python -m aeris.evaluation.baseline run --threads 4 --run-id repeat
    python -m aeris.evaluation.baseline report   # threshold, metrics, gate, manifest

Frames, detections and reports stay out of Git (`data/raw/aot/`, `benchmarks/runs/`).
`report` writes `benchmarks/baseline_manifest.json` and `configs/evaluation/regression.json`
with the digests needed to reproduce and check them.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from aeris.data import aot
from aeris.data.contracts import AnnotationRecord, DetectionRecord
from aeris.evaluation._common import WIDTH_BINS, sequence_bootstrap
from aeris.evaluation.detection import evaluate_detection

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "evaluation" / "aot_benchmark.json"
MANIFEST_PATH = ROOT / "benchmarks" / "baseline_manifest.json"
REGRESSION_PATH = ROOT / "configs" / "evaluation" / "regression.json"
DATA_DIR = ROOT / "data" / "raw" / "aot"
RUNS_DIR = ROOT / "benchmarks" / "runs" / "aeris-007"
SPLITS = ("development", "validation", "holdout")
EVALUATED = ("development", "validation")
GRID = tuple(round(0.01 * i, 2) for i in range(1, 100))
LOCKS = ("requirements/data-linux-x86_64-py311.txt", "requirements/eval-linux-x86_64-py311.txt",
         "requirements/detector-linux-x86_64-py311.txt")


class BaselineError(ValueError):
    """The benchmark inputs or outputs are inconsistent with the frozen definition."""


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def digest(value) -> str:
    """SHA-256 of canonical JSON (sorted keys, compact, no NaN)."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _order(salt: str, text: str) -> str:
    return hashlib.sha256((salt + text).encode()).hexdigest()


def capture_day(time_ns: int) -> str:
    return datetime.datetime.fromtimestamp(time_ns / 1e9, datetime.timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------- selection (labels only)

def split_days(days, salt: str) -> dict[str, str]:
    """Day -> split under the frozen allocation (holdout n//4, validation to n//2)."""
    ordered = sorted(set(days), key=lambda day: _order(salt, day))
    n = len(ordered)
    if n < 4:
        raise BaselineError(f"{n} capture days cannot fill three splits")
    return {day: ("holdout" if i < n // 4 else "validation" if i < n // 2 else "development")
            for i, day in enumerate(ordered)}


def flight_days(index: dict[str, dict]) -> dict[str, str]:
    """flight_id -> capture day of its earliest frame over all its cameras."""
    first: dict[str, int] = {}
    for entry in index.values():
        first[entry["flight_id"]] = min(first.get(entry["flight_id"], entry["first_time_ns"]),
                                        entry["first_time_ns"])
    return {flight: capture_day(t) for flight, t in first.items()}


def choose_sequences(index: dict[str, dict], flight_day: dict[str, str], days: dict[str, str],
                     split: str, count: int, salt: str) -> list[str]:
    """Round-robin over the split's days (hash order), sequences in hash order within a day."""
    by_day: dict[str, list[str]] = defaultdict(list)
    for sequence, entry in index.items():
        day = flight_day[entry["flight_id"]]
        if days[day] == split:
            by_day[day].append(sequence)
    queues = {day: sorted(ids, key=lambda f: _order(salt, f)) for day, ids in by_day.items()}
    order = sorted(queues, key=lambda day: _order(salt, day))
    chosen: list[str] = []
    while len(chosen) < count and any(queues.values()):
        for day in order:
            if queues[day] and len(chosen) < count:
                chosen.append(queues[day].pop(0))
    return chosen


def choose_frames(frames: list[aot.AotFrame], sequences: list[str], stride: int,
                  caps: dict[str, int], salt: str) -> list[aot.AotFrame]:
    """Per sequence, the first `caps[kind]` candidate frames of each kind in hash order."""
    by_sequence: dict[str, list[aot.AotFrame]] = defaultdict(list)
    for frame in frames:
        if frame.frame % stride == 0:
            by_sequence[frame.sequence_id].append(frame)
    chosen = []
    for sequence in sequences:
        picked = []
        for kind in ("scored", "empty", "other"):
            pool = [f for f in by_sequence[sequence] if aot.frame_kind(f) == kind]
            pool.sort(key=lambda f: _order(salt, f"{sequence}:{f.frame}"))
            picked += pool[:caps[kind]]
        chosen += sorted(picked, key=lambda f: f.frame)
    return chosen


def frame_entry(frame: aot.AotFrame, split: str) -> dict:
    return {"split": split, "sequence_id": frame.sequence_id, "part": frame.part,
            "camera": frame.camera, "flight_id": frame.flight_id, "frame": frame.frame,
            "time_ns": frame.time_ns, "img_name": frame.img_name,
            "key": frame.key, "kind": aot.frame_kind(frame),
            "objects": [[o.object_id, o.category, *o.box, o.range_m] for o in frame.objects]}


def entry_frame(entry: dict) -> aot.AotFrame:
    objects = [aot.AotObject(item[0], item[1], tuple(item[2:6]), item[6]) for item in entry["objects"]]
    return aot.AotFrame(entry["part"], entry["flight_id"], entry["frame"], entry["time_ns"],
                        entry["img_name"], objects)


def select(config: dict, data_dir: Path = DATA_DIR) -> dict:
    """The frozen frame lists, derived from the pinned ground truth alone."""
    paths = {}
    for part, pin in config["source"]["groundtruth"].items():
        path = data_dir / part / "groundtruth.csv"
        if file_sha256(path) != pin["sha256"]:
            raise BaselineError(f"{path}: SHA-256 differs from the pinned ground truth")
        paths[part] = path
    salt = config["split"]["salt"]
    index = aot.sequence_index(paths)
    flight_day = flight_days(index)
    days = split_days(flight_day.values(), salt)
    sel = config["selection"]
    sequences = {split: choose_sequences(index, flight_day, days, split,
                                         sel["sequences_per_split"][split], salt)
                 for split in SPLITS}
    wanted = {q for ids in sequences.values() for q in ids}
    frames: list[aot.AotFrame] = []
    for part, path in paths.items():
        flights = [index[q]["flight_id"] for q in wanted if index[q]["part"] == part]
        frames += list(aot.read_frames(path, part, flights).values())
    result = {"benchmark_id": config["benchmark_id"], "days": days, "sequences": sequences,
              "sequence_day": {q: flight_day[index[q]["flight_id"]] for q in wanted},
              "first_time_ns": {q: index[q]["first_time_ns"] for q in wanted}, "frames": {}}
    for split in SPLITS:
        chosen = choose_frames(frames, sequences[split], sel["frame_stride"],
                               sel["per_sequence_caps"], salt)
        result["frames"][split] = [frame_entry(f, split) for f in chosen]
    return result


def selection_summary(selection: dict) -> dict:
    """Counts per split: frames by kind, scored aircraft by width bin, ignored objects."""
    summary = {}
    for split, entries in selection["frames"].items():
        counts: Counter = Counter()
        for entry in entries:
            frame = entry_frame(entry)
            counts["frames"] += 1
            counts[f"frames_{entry['kind']}"] += 1
            for item in frame.objects:
                if item.scored:
                    counts["scored_aircraft"] += 1
                    counts[f"scored_{item.category.lower()}"] += 1
                    counts[f"scored_width_{_bin(item.width)}"] += 1
                elif item.category in aot.AIRCRAFT:
                    counts["nominal_aircraft_ignored"] += 1
                elif item.category in aot.UNKNOWN:
                    counts["unknown_objects_ignored"] += 1
                else:
                    counts["not_aircraft_left_out"] += 1
        sequences = selection["sequences"][split]
        summary[split] = {"sequences": len(sequences),
                          "flights": len({q.rsplit("-", 1)[0] for q in sequences}),
                          "days": len({selection["sequence_day"][q] for q in sequences}),
                          "frames_digest": digest(entries), **dict(sorted(counts.items()))}
    return summary


def _bin(width: float) -> str:
    return next(name for name, upper in WIDTH_BINS if width < upper)


# ---------------------------------------------------------------- retrieval

def image_path(entry: dict, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "images" / entry["key"]


def fetch_frames(entries: list[dict], pins: dict[str, dict], data_dir: Path, limit: int,
                 workers: int = 6, fetcher=aot.fetch) -> dict[str, dict]:
    """Download (or reuse pinned) frames; key -> {sha256, md5, size, etag}."""
    def one(entry):
        pin = pins.get(entry["key"], {})
        facts = fetcher(aot.image_url(entry["key"]), image_path(entry, data_dir),
                        expected_sha256=pin.get("sha256"), limit=limit)
        return entry["key"], {"sha256": facts["sha256"], "md5": facts["md5"],
                              "size": facts["size"]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, entries))


# ---------------------------------------------------------------- records and detector run

def annotations(entries: list[dict], selection: dict, fetched: dict[str, dict]) -> list[AnnotationRecord]:
    return [aot.annotation(entry_frame(e), first_time_ns=selection["first_time_ns"][e["sequence_id"]],
                           image_path=e["key"], image_sha256=fetched[e["key"]]["sha256"])
            for e in entries]


def run_detector(entries: list[dict], truth: list[AnnotationRecord], data_dir: Path, threads: int,
                 unmapped_floor: float = 0.05):
    """Detections (contract records) and unmapped-class candidates for each frame."""
    import cv2

    from aeris.detection import baseline as det

    config = det.load_config()
    settings = det.settings_from_config(config)
    onnx = det.require_artifact(ROOT / config["export"]["aeris_export"]["path"],
                                config["export"]["aeris_export"]["sha256"])
    model = det.OnnxModel(onnx, settings, threads=threads)
    prov = det.provenance(config, f"onnxruntime CPUExecutionProvider threads={threads}")
    detections, unmapped, seconds = [], [], []
    for entry, record in zip(entries, truth):
        start = time.perf_counter()
        path = image_path(entry, data_dir)
        if file_sha256(path) != record.frame.image_checksum:
            raise BaselineError(f"{path}: SHA-256 differs from the recorded frame")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape != (aot.HEIGHT, aot.WIDTH, 3):
            raise BaselineError(f"{path}: not a {aot.WIDTH}x{aot.HEIGHT} image")
        result = det.detect(model, image, settings)
        detections += det.to_records(result.detections, record.frame, result.letterbox,
                                     settings, prov)
        boxes = result.letterbox.to_original(result.candidates.boxes_input)
        for box, score, source in zip(boxes, result.candidates.scores,
                                      result.candidates.source_classes):
            if int(source) not in settings.class_map and float(score) >= unmapped_floor:
                unmapped.append({"sequence_id": record.frame.sequence_id,
                                 "frame_index": record.frame.frame_index,
                                 "coco_class": det.COCO_CLASSES[int(source)],
                                 "score": float(score), "bbox": [float(v) for v in box]})
        seconds.append(time.perf_counter() - start)
    meta = {"threads": threads, "model_provenance": prov, "providers": model.providers,
            "frames": len(entries), "wall_s_total": sum(seconds),
            "wall_s_per_frame_median": sorted(seconds)[len(seconds) // 2] if seconds else None,
            "unmapped_floor": unmapped_floor}
    return detections, unmapped, meta


def write_jsonl(path: Path, records) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join((r.to_json() if hasattr(r, "to_json") else
                    json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False)) + "\n"
                   for r in records)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def read_detections(path: Path) -> list[DetectionRecord]:
    return [DetectionRecord.from_json(line) for line in path.read_text().splitlines() if line]


# ---------------------------------------------------------------- evaluation

def select_threshold(truth, detections, grid=GRID) -> tuple[float | None, list[dict]]:
    """Highest-F1 grid threshold (ties to the higher one); None if no true positive."""
    table, best = [], None
    for threshold in grid:
        report = evaluate_detection(truth, detections, score_threshold=threshold,
                                    check_versions=False)
        row = {"score_threshold": threshold, "f1": report["f1"], "precision": report["precision"],
               "recall": report["recall"], "true_positives": report["true_positives"]}
        table.append(row)
        if report["true_positives"] and report["f1"] is not None:
            if best is None or report["f1"] >= best["f1"]:
                best = row
    return (best["score_threshold"] if best else None), table


def gated_metrics(report: dict) -> dict[str, float | None]:
    values = {"ap50": report["ap50"], "ap_50_95": report["ap_50_95"],
              "precision": report["precision"], "recall": report["recall"],
              "false_positives_per_negative_frame":
                  report["negative_frames"]["false_positives_per_negative_frame"]}
    for name, _ in WIDTH_BINS:
        values[f"width_bin_recall {name}"] = report["width_bins"][name]["recall"]
    return values


def _restrict(truth: list[AnnotationRecord], category_ids: set[str]) -> list[AnnotationRecord]:
    """Truth where only objects whose track id starts with a listed category stay scored."""
    out = []
    for record in truth:
        objects = []
        for item in record.objects:
            keep = aot.object_category(item.annotation_id) in category_ids
            objects.append(item if keep or item.ignore else dataclasses.replace(item, ignore=True))
        out.append(dataclasses.replace(record, objects=tuple(objects)))
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def missed_with_other_class(truth, detections, unmapped, threshold: float, iou: float = 0.5) -> dict:
    """Diagnostic (not a COCO metric): scored aircraft with no aircraft detection at IoU >= 0.5
    and score >= threshold, and how many of those an unmapped COCO class covers instead."""
    dets = defaultdict(list)
    for d in detections:
        if d.score >= threshold:
            dets[(d.sequence_id, d.frame_index)].append(
                (d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max))
    other = defaultdict(list)
    for u in unmapped:
        if u["score"] >= threshold:
            other[(u["sequence_id"], u["frame_index"])].append((u["coco_class"], u["bbox"]))
    missed, covered = 0, Counter()
    for record in truth:
        key = (record.frame.sequence_id, record.frame.frame_index)
        for item in record.objects:
            if item.ignore:
                continue
            box = (item.bbox.x_min, item.bbox.y_min, item.bbox.x_max, item.bbox.y_max)
            if any(_iou(box, d) >= iou for d in dets[key]):
                continue
            missed += 1
            hits = [c for c, b in other[key] if _iou(box, b) >= iou]
            if hits:
                covered[Counter(hits).most_common(1)[0][0]] += 1
    return {"definition": "scored aircraft with no aircraft detection at IoU >= 0.5 and score >= the "
                          "frozen threshold; covered = an unmapped COCO class candidate at or above the "
                          "threshold overlaps it at IoU >= 0.5 (most frequent class counted)",
            "missed": missed, "covered_by_unmapped_class": dict(covered.most_common())}


def per_frame_counts(truth, detections, threshold: float) -> dict[tuple, dict]:
    """TP/FP/FN, negative-frame FP and width-bin support/matches for every frame."""
    by_frame = defaultdict(list)
    for d in detections:
        by_frame[(d.sequence_id, d.frame_index)].append(d)
    out = {}
    for record in truth:
        key = (record.frame.sequence_id, record.frame.frame_index)
        report = evaluate_detection([record], by_frame[key], score_threshold=threshold,
                                    check_versions=False)
        bins = {name: (report["width_bins"][name]["support"],
                       round((report["width_bins"][name]["recall"] or 0.0)
                             * report["width_bins"][name]["support"]))
                for name, _ in WIDTH_BINS}
        out[key] = {"tp": report["true_positives"], "fp": report["false_positives"],
                    "fn": report["false_negatives"], "negative": record.is_negative, "bins": bins}
    return out


def day_bootstrap(truth, detections, counts, day_of, threshold: float, boot: dict) -> dict:
    """Capture-day percentile intervals for the gated metrics on one split."""
    groups = defaultdict(list)
    for key in counts:
        groups[day_of[key[0]]].append(key)

    def ratio(num, den):
        return None if den == 0 else num / den

    def stat(fn):
        return lambda keys: fn([counts[k] for k in keys])

    statistics = {
        "precision": stat(lambda c: ratio(sum(x["tp"] for x in c), sum(x["tp"] + x["fp"] for x in c))),
        "recall": stat(lambda c: ratio(sum(x["tp"] for x in c), sum(x["tp"] + x["fn"] for x in c))),
        "false_positives_per_negative_frame": stat(lambda c: ratio(
            sum(x["fp"] for x in c if x["negative"]), sum(x["negative"] for x in c)))}
    for name, _ in WIDTH_BINS:
        statistics[f"width_bin_recall {name}"] = stat(lambda c, n=name: ratio(
            sum(x["bins"][n][1] for x in c), sum(x["bins"][n][0] for x in c)))
    out = {}
    for name, fn in statistics.items():
        safe = (lambda f: (lambda keys: -1.0 if f(keys) is None else f(keys)))(fn)
        result = sequence_bootstrap(groups, safe, level=boot["level"],
                                    resamples=boot["resamples_threshold_metrics"], seed=boot["seed"])
        if result["low"] is not None and result["low"] < 0:
            result.update(low=None, high=None, availability_reason={
                "low": "a resample had no denominator", "high": "a resample had no denominator"})
        out[name] = result
    truth_by = {(r.frame.sequence_id, r.frame.frame_index): r for r in truth}
    dets_by = defaultdict(list)
    for d in detections:
        dets_by[(d.sequence_id, d.frame_index)].append(d)

    def ap_stat(which):
        def fn(keys):
            seen: Counter = Counter()
            t_list, d_list = [], []
            for key in keys:
                seen[key] += 1
                suffix = f"~{seen[key]}"
                record = truth_by[key]
                frame = dataclasses.replace(record.frame, sequence_id=record.frame.sequence_id + suffix)
                t_list.append(dataclasses.replace(record, frame=frame))
                d_list += [dataclasses.replace(d, sequence_id=d.sequence_id + suffix)
                           for d in dets_by[key]]
            value = evaluate_detection(t_list, d_list, check_versions=False)[which]
            return -1.0 if value is None else value
        return fn
    for which in ("ap50", "ap_50_95"):
        result = sequence_bootstrap(groups, ap_stat(which), level=boot["level"],
                                    resamples=boot["resamples_ap"], seed=boot["seed"])
        out[which] = result
    return out


def tolerance(metric: str, baseline: dict, repeat_delta: float, floors: dict,
              report: dict, min_support: dict) -> dict:
    """Frozen gate entry for one metric (see regression_rule in the benchmark config)."""
    lower_is_better = metric == "false_positives_per_negative_frame"
    if metric in ("ap50", "ap_50_95"):
        floor, support = floors["ap"], report["support"]["ground_truth"]
        needed = min_support["objects"]
    elif metric == "precision":
        support = (report["true_positives"] or 0) + (report["false_positives"] or 0)
        floor, needed = (1 / support if support else None), min_support["objects"]
    elif metric == "recall":
        support = report["support"]["ground_truth"]
        floor, needed = (1 / support if support else None), min_support["objects"]
    elif metric == "false_positives_per_negative_frame":
        support = report["negative_frames"]["count"]
        floor, needed = (1 / support if support else None), min_support["negative_frames"]
    else:
        name = metric.split(" ", 1)[1]
        support = report["width_bins"][name]["support"]
        floor, needed = (1 / support if support else None), min_support["objects"]
    entry = {"metric": metric, "direction": "lower_is_better" if lower_is_better else "higher_is_better",
             "baseline": baseline[metric], "support": support}
    if baseline[metric] is None or support < needed or floor is None:
        entry.update(gated=False, tolerance_abs=None,
                     reason=f"support {support} < minimum {needed} or metric unavailable; "
                            "reported as not gated, never as a pass")
        return entry
    entry.update(gated=True, repeat_delta_abs=repeat_delta, resolution_floor=floor,
                 tolerance_abs=max(2 * repeat_delta, floor))
    return entry


def gate_check(gate: dict, candidate: dict) -> dict:
    """Apply frozen gate entries to a candidate's metrics (for AERIS-015 and tests)."""
    failures, unavailable = [], []
    for entry in gate["gates"]:
        if not entry["gated"]:
            continue
        value = candidate.get(entry["metric"])
        if value is None:
            unavailable.append(entry["metric"])
            continue
        worse = (value - entry["baseline"] if entry["direction"] == "lower_is_better"
                 else entry["baseline"] - value)
        if worse > entry["tolerance_abs"]:
            failures.append({"metric": entry["metric"], "baseline": entry["baseline"],
                             "candidate": value, "worse_by": worse,
                             "tolerance_abs": entry["tolerance_abs"]})
    status = "blocked" if unavailable else ("fail" if failures else "pass")
    return {"status": status, "failures": failures, "unavailable": unavailable}


# ---------------------------------------------------------------- provenance

def environment() -> dict:
    import cv2
    import numpy
    import onnxruntime

    from aeris.evaluation._common import evaluator_versions

    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"python": sys.version.split()[0], "platform": platform.platform(), "cpu": cpu,
            "numpy": numpy.__version__, "opencv": cv2.__version__,
            "onnxruntime": onnxruntime.__version__, "evaluators": evaluator_versions(),
            "locks": {name: file_sha256(ROOT / name) for name in LOCKS}}


def code_revision() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=False).stdout.strip()
    return {"commit": git("rev-parse", "HEAD") or None,
            "dirty_paths": [line[3:] for line in git("status", "--porcelain").splitlines()]}


# ---------------------------------------------------------------- commands

def _paths(run_id: str) -> Path:
    return RUNS_DIR / run_id


def cmd_select(config: dict) -> dict:
    selection = select(config)
    write_json(RUNS_DIR / "selection.json", selection)
    return selection_summary(selection)


def cmd_fetch(config: dict, workers: int) -> dict:
    selection = load_json(RUNS_DIR / "selection.json")
    pins_path = RUNS_DIR / "fetched.json"
    pins = load_json(pins_path) if pins_path.is_file() else {}
    entries = [e for s in EVALUATED for e in selection["frames"][s]]
    fetched = fetch_frames(entries, pins, DATA_DIR, config["budget"]["per_frame_bytes_max"], workers)
    write_json(pins_path, fetched)
    return {"frames": len(fetched), "bytes": sum(v["size"] for v in fetched.values())}


def cmd_run(config: dict, run_id: str, threads: int) -> dict:
    selection = load_json(RUNS_DIR / "selection.json")
    fetched = load_json(RUNS_DIR / "fetched.json")
    out, meta = _paths(run_id), {"run_id": run_id, "code": code_revision(), "splits": {}}
    for split in EVALUATED:
        entries = selection["frames"][split]
        truth = annotations(entries, selection, fetched)
        truth_digest = write_jsonl(RUNS_DIR / f"{split}.annotations.jsonl", truth)
        detections, unmapped, info = run_detector(entries, truth, DATA_DIR, threads)
        meta["splits"][split] = {**info, "annotations_sha256": truth_digest,
                                 "detections_sha256": write_jsonl(out / f"{split}.detections.jsonl", detections),
                                 "unmapped_sha256": write_jsonl(out / f"{split}.unmapped.jsonl", unmapped),
                                 "detections": len(detections)}
    write_json(out / "run.json", meta)
    return meta


def _load_run(run_id: str, split: str):
    out = _paths(run_id)
    return (read_detections(out / f"{split}.detections.jsonl"),
            [json.loads(line) for line in (out / f"{split}.unmapped.jsonl").read_text().splitlines() if line])


def _truth(split: str) -> list[AnnotationRecord]:
    path = RUNS_DIR / f"{split}.annotations.jsonl"
    return [AnnotationRecord.from_json(line) for line in path.read_text().splitlines() if line]


def compare_runs(a: list[DetectionRecord], b: list[DetectionRecord]) -> dict:
    """Detection-level repeatability between two runs over the same frames."""
    def group(ds):
        return {k: sorted(v, key=lambda d: -d.score) for k, v in _by_frame(ds).items()}
    ga, gb = group(a), group(b)
    count_mismatch = sum(len(ga.get(k, [])) != len(gb.get(k, [])) for k in set(ga) | set(gb))
    score_diff = box_diff = 0.0
    for key in set(ga) & set(gb):
        for x, y in zip(ga[key], gb[key]):
            score_diff = max(score_diff, abs(x.score - y.score))
            box_diff = max(box_diff, max(abs(p - q) for p, q in zip(
                (x.bbox.x_min, x.bbox.y_min, x.bbox.x_max, x.bbox.y_max),
                (y.bbox.x_min, y.bbox.y_min, y.bbox.x_max, y.bbox.y_max))))
    return {"detections": [len(a), len(b)], "frames_with_count_mismatch": count_mismatch,
            "max_abs_score_difference": score_diff, "max_abs_box_difference_px": box_diff}


def _by_frame(detections):
    out = defaultdict(list)
    for d in detections:
        out[(d.sequence_id, d.frame_index)].append(d)
    return out


def cmd_report(config: dict) -> dict:
    selection = load_json(RUNS_DIR / "selection.json")
    fetched = load_json(RUNS_DIR / "fetched.json")
    runs = {rid: load_json(_paths(rid) / "run.json") for rid in ("baseline", "repeat")}
    truth = {s: _truth(s) for s in EVALUATED}
    for s in EVALUATED:
        recorded = {runs[rid]["splits"][s]["annotations_sha256"] for rid in runs}
        if recorded != {file_sha256(RUNS_DIR / f"{s}.annotations.jsonl")}:
            raise BaselineError(f"{s}: annotations differ between runs or changed since them")
    ev = config["evaluation"]
    base = {s: _load_run("baseline", s) for s in EVALUATED}
    rep = {s: _load_run("repeat", s) for s in EVALUATED}
    threshold, table = select_threshold(truth["validation"], base["validation"][0])
    threshold_repeat, _ = select_threshold(truth["validation"], rep["validation"][0])
    if threshold is None:
        raise BaselineError("no validation threshold gives a true positive; gate blocked")
    kwargs = {"iou_threshold": ev["iou_threshold"], "max_detections_per_frame": ev["max_detections_per_frame"]}
    reports = {s: evaluate_detection(truth[s], base[s][0], score_threshold=threshold, **kwargs)
               for s in EVALUATED}
    repeat_dev = evaluate_detection(truth["development"], rep["development"][0],
                                    score_threshold=threshold, **kwargs)
    values = gated_metrics(reports["development"])
    repeat_values = gated_metrics(repeat_dev)
    deltas = {m: (abs(values[m] - repeat_values[m])
                  if values[m] is not None and repeat_values[m] is not None else 0.0) for m in values}
    rule = config["regression_rule"]
    gates = [tolerance(m, values, deltas[m], rule["resolution_floor"], reports["development"],
                       rule["min_support"]) for m in values]
    day_of = selection["sequence_day"]
    counts = per_frame_counts(truth["development"], base["development"][0], threshold)
    intervals = day_bootstrap(truth["development"], base["development"][0], counts, day_of,
                              threshold, ev["bootstrap"])
    per_category = {}
    for name in sorted(aot.AIRCRAFT):
        restricted = evaluate_detection(_restrict(truth["development"], {name}), base["development"][0],
                                        score_threshold=threshold, check_versions=False, **kwargs)
        per_category[name] = {"support": restricted["support"]["ground_truth"],
                              "recall": restricted["recall"], "ap50": restricted["ap50"],
                              "width_bins": restricted["width_bins"]}
    diagnostics = {"missed_aircraft": missed_with_other_class(
        truth["development"], base["development"][0], base["development"][1], threshold),
        "per_category_development": per_category,
        "negative_frames_with_birds_left_out": sum(
            1 for e in selection["frames"]["development"] if e["kind"] == "other"
            and all(o[1] in aot.NOT_AIRCRAFT for o in e["objects"]))}
    repeatability = {s: compare_runs(base[s][0], rep[s][0]) for s in EVALUATED}
    repeatability["validation_threshold_repeat"] = threshold_repeat
    repeatability["development_metric_abs_differences"] = deltas
    results = {"threshold_selection": {"rule": ev["threshold_rule"], "selected": threshold,
                                       "table": table},
               "reports": reports, "repeat_development_report": repeat_dev,
               "repeatability": repeatability, "intervals_development": intervals,
               "diagnostics": diagnostics}
    write_json(RUNS_DIR / "results.json", results)
    results_sha = file_sha256(RUNS_DIR / "results.json")
    summary = selection_summary(selection)
    manifest = build_manifest(config, selection, summary, fetched, runs, results, results_sha)
    write_json(MANIFEST_PATH, manifest)
    regression = build_regression(config, threshold, values, gates, intervals, manifest)
    write_json(REGRESSION_PATH, regression)
    return {"threshold": threshold, "development": values, "gates": gates,
            "repeatability": repeatability}


def build_manifest(config, selection, summary, fetched, runs, results, results_sha) -> dict:
    from aeris.detection import baseline as det

    detector = det.load_config()
    frames = {}
    for split in SPLITS:
        rows = []
        for e in selection["frames"][split]:
            row = {"key": e["key"], "sequence_id": e["sequence_id"], "frame": e["frame"],
                   "kind": e["kind"]}
            if split in EVALUATED:
                row.update(fetched[e["key"]])
            rows.append(row)
        frames[split] = rows
    dev = results["reports"]["development"]
    val = results["reports"]["validation"]
    return {
        "schema_version": 1, "benchmark_id": config["benchmark_id"], "task": "AERIS-007",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "definition": {"path": "configs/evaluation/aot_benchmark.json",
                       "sha256": file_sha256(CONFIG_PATH)},
        "source": config["source"],
        "splits": {s: {"days": sorted(d for d, v in selection["days"].items() if v == s),
                       "sequences": selection["sequences"][s], **summary[s]} for s in SPLITS},
        "holdout": {"sealed": True, "frames_downloaded": False,
                    "note": "Frame list frozen by digest; not fetched or evaluated in AERIS-007."},
        "frames": frames,
        "retrieval": {"images": "GET <bucket_url>/<key>; check the S3 ETag (MD5) and the SHA-256 here",
                      "commands": ["python -m aeris.evaluation.baseline select",
                                   "python -m aeris.evaluation.baseline fetch",
                                   "python -m aeris.evaluation.baseline run --threads 1 --run-id baseline",
                                   "python -m aeris.evaluation.baseline run --threads 4 --run-id repeat",
                                   "python -m aeris.evaluation.baseline report"]},
        "model": {"detector_config": {"path": "configs/detector/baseline.json",
                                      "sha256": file_sha256(ROOT / "configs/detector/baseline.json")},
                  "onnx": detector["export"]["aeris_export"],
                  "weights_sha256": detector["model"]["weights"]["sha256"],
                  "class_mapping_version": detector["class_mapping"]["version"]},
        "environment": environment(),
        "runs": {rid: {"code": r["code"], "splits": {s: {k: v for k, v in r["splits"][s].items()
                                                         if k != "model_provenance"}
                                                     for s in EVALUATED},
                       "model_provenance": r["splits"]["development"]["model_provenance"]}
                 for rid, r in runs.items()},
        "results": {"path": "benchmarks/runs/aeris-007/results.json", "sha256": results_sha,
                    "validation_threshold": results["threshold_selection"]["selected"],
                    "development": _headline(dev), "validation": _headline(val),
                    "repeatability": results["repeatability"],
                    "intervals_development": {k: {"low": v["low"], "high": v["high"],
                                                  "sequences": v["sequences"],
                                                  "resamples": v["resamples"]}
                                              for k, v in results["intervals_development"].items()},
                    "diagnostics": results["diagnostics"]},
        "pretraining_overlap": "YOLOX-S weights were trained on COCO 2017; AOT frames are not part "
                               "of COCO, so no image overlap is known. Domain overlap: COCO has an "
                               "'airplane' class and no helicopter class.",
    }


def _headline(report: dict) -> dict:
    return {"frames": report["frames"], "support": report["support"], "ap50": report["ap50"],
            "ap_50_95": report["ap_50_95"], "score_threshold": report["score_threshold"],
            "precision": report["precision"], "recall": report["recall"], "f1": report["f1"],
            "true_positives": report["true_positives"], "false_positives": report["false_positives"],
            "false_negatives": report["false_negatives"],
            "false_positives_per_complete_frame": report["false_positives_per_complete_frame"],
            "negative_frames": report["negative_frames"], "width_bins": report["width_bins"]}


def build_regression(config, threshold, values, gates, intervals, manifest) -> dict:
    rule = config["regression_rule"]
    return {
        "schema_version": 1, "regression_id": "aeris-007-regression-v1",
        "benchmark_id": config["benchmark_id"], "split": rule["split"],
        "development_frames_digest": manifest["splits"]["development"]["frames_digest"],
        "frozen_at": manifest["created_at"],
        "frozen_before_candidates": "Set from the baseline and its repeat run only; no candidate "
                                    "model had been evaluated when this file was written.",
        "protocol": {"evaluator": config["evaluation"]["evaluator"],
                     "iou_threshold": config["evaluation"]["iou_threshold"],
                     "score_threshold": threshold,
                     "score_threshold_source": "validation split, " + config["evaluation"]["threshold_rule"],
                     "max_detections_per_frame": config["evaluation"]["max_detections_per_frame"],
                     "detector_settings": "configs/detector/baseline.json (unchanged for the baseline)"},
        "baseline": values,
        "gates": gates,
        "tolerance_rationale": rule["tolerance"] + " " + rule["resolution_floor"]["ap_reason"] + " "
                               + rule["resolution_floor"]["count_metric_reason"] + " " + rule["support_rule"],
        "intervals_development": {k: [v["low"], v["high"]] for k, v in intervals.items()},
        "interval_note": rule["interval_note"],
        "holdout": {"sealed": True, "frames_digest": manifest["splits"]["holdout"]["frames_digest"],
                    "policy": config["split"]["holdout_policy"]},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("select")
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--workers", type=int, default=6)
    run = sub.add_parser("run")
    run.add_argument("--threads", type=int, required=True)
    run.add_argument("--run-id", required=True, choices=("baseline", "repeat"))
    sub.add_parser("report")
    args = parser.parse_args(argv)
    config = load_json(CONFIG_PATH)
    try:
        if args.command == "select":
            result = cmd_select(config)
        elif args.command == "fetch":
            result = cmd_fetch(config, args.workers)
        elif args.command == "run":
            result = cmd_run(config, args.run_id, args.threads)
        else:
            result = cmd_report(config)
    except (BaselineError, aot.AotError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
