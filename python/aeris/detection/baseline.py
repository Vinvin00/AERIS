"""YOLOX-Tiny COCO baseline: letterbox preprocessing, grid decoding, NMS and class mapping.

The exported graph returns raw, undecoded head output ``(1, 3549, 85)``:
``cx, cy, w, h`` (log-space, per stride), objectness and 80 COCO class scores. Boxes
returned here are original-frame ``xyxy`` pixels (contract v1); letterbox padding
and scale are inverted. Only COCO ``airplane`` is mapped to ``aeris-aircraft-v1``.
Runners (ONNX Runtime, OpenCV DNN) only execute the graph and share this pre/post code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from aeris.data.contracts import BBox, DetectionRecord

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "detector" / "baseline.json"
STRIDES = (8, 16, 32)


class DetectorError(ValueError):
    """Model, config or input violates the baseline contract."""


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_model_path(config: dict, root: Path = ROOT) -> Path:
    """Return the weight path only if it exists and matches the pinned SHA-256."""
    path = root / config["model"]["path"]
    if not path.is_file():
        raise DetectorError(f"model file missing: {path}")
    if sha256_file(path) != config["model"]["sha256"]:
        raise DetectorError("model SHA-256 differs from configs/detector/baseline.json")
    return path


def preprocess(image_bgr: np.ndarray, size: int = 416, pad_value: int = 114):
    """Aspect-preserving resize to the top-left of a padded square; BGR, 0-255, CHW float32.

    Returns ``(tensor[1,3,size,size], ratio)``; original = model_pixel / ratio.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise DetectorError("expected HxWx3 uint8 BGR image")
    height, width = image_bgr.shape[:2]
    ratio = min(size / height, size / width)
    new_w, new_h = max(1, int(width * ratio)), max(1, int(height * ratio))
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    canvas[:new_h, :new_w] = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32)[None]
    return tensor, ratio


def _grids(size: int):
    grids, strides = [], []
    for stride in STRIDES:
        cells = size // stride
        xs, ys = np.meshgrid(np.arange(cells), np.arange(cells))
        grids.append(np.stack((xs, ys), 2).reshape(-1, 2))
        strides.append(np.full((cells * cells, 1), stride))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


def decode_head(raw: np.ndarray, size: int = 416) -> np.ndarray:
    """Turn raw head output into model-pixel ``cx, cy, w, h`` plus scores (copy)."""
    grid, stride = _grids(size)
    if raw.shape != (1, len(grid), 85):
        raise DetectorError(f"unexpected output shape {raw.shape}")
    out = raw[0].astype(np.float32, copy=True)
    out[:, :2] = (out[:, :2] + grid) * stride
    out[:, 2:4] = np.exp(out[:, 2:4]) * stride
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy NMS on xyxy boxes; returns kept indices, highest score first."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if not rest.size:
            break
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        order = rest[inter / np.maximum(area_i + area_r - inter, 1e-12) <= iou_threshold]
    return keep


def postprocess(raw: np.ndarray, ratio: float, width: int, height: int, config: dict):
    """Decode, filter to mapped classes, class-aware NMS, invert letterbox, clip.

    Returns ``[(xyxy, mapped_class_id, score, coco_class_index), ...]`` by descending score.
    """
    post = config["postprocess"]
    out = decode_head(raw, config["input"]["size"])
    class_map = {int(k): v["aeris_class_id"] for k, v in config["class_map"].items()}
    scores_all = out[:, 4:5] * out[:, 5:]
    results = []
    for coco_index, mapped in class_map.items():
        scores = scores_all[:, coco_index]
        idx = np.flatnonzero(scores >= post["score_floor"])
        if not idx.size:
            continue
        cx, cy, w, h = (out[idx, k] / ratio for k in range(4))
        boxes = np.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), 1)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, kept_scores = boxes[valid], scores[idx][valid]
        for k in nms(boxes, kept_scores, post["nms_iou"]):
            results.append((boxes[k].tolist(), mapped, float(min(kept_scores[k], 1.0)), coco_index))
    results.sort(key=lambda r: -r[2])
    return results[:post["max_detections"]]


class OnnxRuntimeRunner:
    """ONNX Runtime, CPUExecutionProvider only (no silent provider fallback)."""
    name = "onnxruntime-cpu"

    def __init__(self, model_path: Path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise DetectorError(f"unexpected providers {self.session.get_providers()}")
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: tensor})[0]


class OpenCvDnnRunner:
    """Independent CPU runtime (OpenCV DNN) executing the same exported graph."""
    name = "opencv-dnn-cpu"

    def __init__(self, model_path: Path):
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        self.net.setInput(tensor)
        return self.net.forward()


def detect(image_bgr: np.ndarray, runner, config: dict, *, sequence_id: str, frame_index: int,
           timestamp_s: float | None = None) -> list[DetectionRecord]:
    """Run one frame and return contract-v1 detections in original-frame pixels."""
    height, width = image_bgr.shape[:2]
    tensor, ratio = preprocess(image_bgr, config["input"]["size"], config["input"]["pad_value"])
    provenance = f"{config['model']['name']}@{config['model']['sha256'][:12]}/{runner.name}"
    reason = {} if timestamp_s is not None else {"timestamp_s": "still image"}
    return [
        DetectionRecord(1, sequence_id, frame_index, timestamp_s, width, height, BBox(*box),
                        class_id, config["class_mapping_version"], score, provenance, dict(reason))
        for box, class_id, score, _ in postprocess(runner(tensor), ratio, width, height, config)
    ]


def agreement_report(image_paths: list[Path], config: dict, *, timing_runs: int = 20) -> dict:
    """Compare ONNX Runtime and OpenCV DNN on fixed images against declared tolerances.

    Timings are single-machine feasibility numbers (warmup excluded), not a benchmark.
    """
    import time
    model = verified_model_path(config)
    runners = {r.name: r for r in (OnnxRuntimeRunner(model), OpenCvDnnRunner(model))}
    tol, images = config["agreement"], []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise DetectorError(f"cannot decode {path}")
        tensor, ratio = preprocess(image, config["input"]["size"], config["input"]["pad_value"])
        raw = {name: run(tensor) for name, run in runners.items()}
        found = {name: postprocess(r, ratio, image.shape[1], image.shape[0], config) for name, r in raw.items()}
        a, b = (found[name] for name in runners)
        paired = len(a) == len(b)
        box_diff = max((max(abs(x - y) for x, y in zip(p[0], q[0])) for p, q in zip(a, b)), default=0.0)
        score_diff = max((abs(p[2] - q[2]) for p, q in zip(a, b)), default=0.0)
        raw_diff = float(np.abs(raw["onnxruntime-cpu"] - raw["opencv-dnn-cpu"]).max())
        images.append({"file": path.name, "sha256": sha256_file(path), "width": image.shape[1],
                       "height": image.shape[0], "detections": len(a), "same_count": paired,
                       "raw_max_abs_diff": raw_diff, "box_max_diff_px": box_diff,
                       "score_max_diff": score_diff,
                       "within_tolerance": bool(paired and raw_diff <= tol["raw_max_abs"]
                                                and box_diff <= tol["box_px"]
                                                and score_diff <= tol["score_abs"])})
    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    tensor, _ = preprocess(first, config["input"]["size"], config["input"]["pad_value"])
    latency = {}
    for name, run in runners.items():
        run(tensor)
        times = []
        for _ in range(timing_runs):
            start = time.perf_counter()
            run(tensor)
            times.append((time.perf_counter() - start) * 1e3)
        latency[name] = {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
                         "runs": timing_runs, "warmup_runs": 1, "scope": "model forward only, one image"}
    import onnxruntime
    return {"model_sha256": config["model"]["sha256"], "tolerances": tol, "images": images,
            "latency": latency, "all_within_tolerance": all(i["within_tolerance"] for i in images),
            "versions": {"onnxruntime": onnxruntime.__version__, "opencv": cv2.__version__,
                         "numpy": np.__version__}}


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Write a runtime agreement report for the baseline.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    text = json.dumps(agreement_report(args.images, load_config()), indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
