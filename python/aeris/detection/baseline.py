"""Pretrained YOLOX-S baseline (AERIS-006).

Letterbox preprocessing, raw-output decoding, per-class NMS, COCO-to-AERIS class mapping
and the exact inverse of the resize into original-frame pixel coordinates, plus ONNX
Runtime CPU inference. Every setting comes from `configs/detector/baseline.json`. The
decode and NMS here run on raw head outputs, so a PyTorch model and its ONNX export
share one postprocessing path and can be compared on raw outputs. Nothing is trained or
tuned on AERIS data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from aeris.data.contracts import SCHEMA_VERSION, BBox, DetectionRecord, FrameRecord

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "detector" / "baseline.json"
# COCO 2017 contiguous class names in model output order (yolox/data/datasets/coco_classes.py).
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


class DetectorError(ValueError):
    """An input, configuration or model artifact cannot be used as supplied."""


@dataclass(frozen=True)
class Settings:
    """The parts of the baseline config that change numbers."""

    input_height: int
    input_width: int
    pad_value: int
    strides: tuple[int, ...]
    num_classes: int
    score_threshold: float
    nms_iou_threshold: float
    class_map: dict
    class_mapping_version: str

    @property
    def anchors(self) -> int:
        return sum((self.input_height // s) * (self.input_width // s) for s in self.strides)


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def settings_from_config(config: dict) -> Settings:
    """Extract and check the numeric settings; refuse anything the code does not implement."""
    try:
        tensor = config["input"]["tensor"]
        post = config["postprocess"]
        mapping = config["class_mapping"]
        _, channels, height, width = tensor["shape"]
        settings = Settings(
            input_height=height, input_width=width,
            pad_value=config["input"]["pad"]["value"],
            strides=tuple(config["output"]["strides"]),
            num_classes=config["output"]["shape"][2] - 5,
            score_threshold=post["score_threshold"],
            nms_iou_threshold=post["nms"]["iou_threshold"],
            class_map={int(k): v for k, v in mapping["map"].items()},
            class_mapping_version=mapping["version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DetectorError(f"config: missing or malformed field ({exc})") from exc
    if (channels != 3 or config["input"]["channel_order"] != "BGR"
            or tensor["layout"] != "NCHW" or tensor["dtype"] != "float32"):
        raise DetectorError("config: only 3-channel BGR float32 NCHW input is implemented")
    if post["class_selection"].split()[0] != "argmax" or post["nms"]["kind"] != "greedy, per class":
        raise DetectorError("config: only argmax class selection with per-class greedy NMS")
    if post["max_detections_per_frame"] is not None or post["clip_to_frame"] is not True:
        raise DetectorError("config: a per-frame cap or unclipped boxes are not implemented")
    if any(height % s or width % s for s in settings.strides):
        raise DetectorError("config: input size must be divisible by every stride")
    if config["output"]["shape"][1] != settings.anchors:
        raise DetectorError("config: output anchor count does not match the strides")
    if not all(0 <= k < settings.num_classes for k in settings.class_map):
        raise DetectorError("config: class map uses a source class outside the model")
    return settings


@dataclass(frozen=True)
class Letterbox:
    """The aspect-preserving resize and bottom/right pad applied to one frame."""

    width: int
    height: int
    resized_width: int
    resized_height: int

    def to_original(self, boxes: np.ndarray) -> np.ndarray:
        """Map xyxy edge coordinates from model input pixels to original-frame pixels.

        cv2.resize maps pixel edges linearly (x_resized = x * resized_width / width), so
        dividing by the per-axis ratio that was actually applied is the exact inverse. The
        pad sits bottom/right and adds no offset. Upstream YOLOX divides both axes by the
        requested scale r instead, which differs by under one resized pixel when
        width * r is not an integer.
        """
        out = np.array(boxes, dtype=np.float64, copy=True).reshape(-1, 4)
        out[:, [0, 2]] *= self.width / self.resized_width
        out[:, [1, 3]] *= self.height / self.resized_height
        return out


def preprocess(image: np.ndarray, settings: Settings) -> tuple[np.ndarray, Letterbox]:
    """YOLOX non-legacy ValTransform: resize, pad with 114, BGR 0-255 float32 NCHW."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or image.shape[0] < 1 or image.shape[1] < 1):
        raise DetectorError("image: expected an HxWx3 uint8 BGR array")
    height, width = image.shape[:2]
    ratio = min(settings.input_height / height, settings.input_width / width)
    resized_width, resized_height = int(width * ratio), int(height * ratio)
    if resized_width < 1 or resized_height < 1:
        raise DetectorError("image: too elongated to letterbox into the model input")
    padded = np.full((settings.input_height, settings.input_width, 3), settings.pad_value,
                     dtype=np.uint8)
    padded[:resized_height, :resized_width] = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
    return tensor, Letterbox(width, height, resized_width, resized_height)


def _grid(settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    cells, steps = [], []
    for stride in settings.strides:
        ys, xs = np.meshgrid(np.arange(settings.input_height // stride),
                             np.arange(settings.input_width // stride), indexing="ij")
        cells.append(np.stack((xs.ravel(), ys.ravel()), axis=1))
        steps.append(np.full(xs.size, stride))
    return np.concatenate(cells).astype(np.float64), np.concatenate(steps).astype(np.float64)


def _raw_rows(raw: np.ndarray, settings: Settings) -> np.ndarray:
    raw = np.asarray(raw)
    if raw.ndim == 3:
        if raw.shape[0] != 1:
            raise DetectorError("raw: batch size must be 1")
        raw = raw[0]
    if raw.shape != (settings.anchors, settings.num_classes + 5):
        raise DetectorError(f"raw: expected shape ({settings.anchors}, {settings.num_classes + 5}), "
                            f"got {raw.shape}")
    if not np.isfinite(raw).all():
        raise DetectorError("raw: non-finite model output")
    return raw


def decode(raw: np.ndarray, settings: Settings, rows: np.ndarray | None = None) -> np.ndarray:
    """cx, cy, w, h in padded-input pixels for the given anchor rows (all rows by default)."""
    raw = _raw_rows(raw, settings)
    cells, steps = _grid(settings)
    if rows is not None:
        raw, cells, steps = raw[rows], cells[rows], steps[rows]
    reg = raw[:, :4].astype(np.float64)
    with np.errstate(over="ignore"):
        size = np.exp(reg[:, 2:4]) * steps[:, None]
    boxes = np.concatenate([(reg[:, :2] + cells) * steps[:, None], size], axis=1)
    if not np.isfinite(boxes).all():
        raise DetectorError("raw: box size overflows when decoded")
    return boxes


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    half = boxes[:, 2:4] / 2
    return np.concatenate([boxes[:, :2] - half, boxes[:, :2] + half], axis=1)


def box_iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box with each row of `others`; 0 where the union is empty."""
    inter = (np.clip(np.minimum(box[2], others[:, 2]) - np.maximum(box[0], others[:, 0]), 0, None)
             * np.clip(np.minimum(box[3], others[:, 3]) - np.maximum(box[1], others[:, 1]), 0, None))
    union = ((box[2] - box[0]) * (box[3] - box[1])
             + (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1]) - inter)
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _order(scores: np.ndarray) -> np.ndarray:
    """Score descending, ties by position (as a stable sort)."""
    return np.lexsort((np.arange(len(scores)), -scores))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS as torchvision.ops.nms: suppress IoU > threshold; kept indices by score."""
    order = _order(scores)
    suppressed = np.zeros(len(scores), dtype=bool)
    keep = []
    for position, index in enumerate(order):
        if suppressed[index]:
            continue
        keep.append(index)
        rest = order[position + 1:]
        rest = rest[~suppressed[rest]]
        suppressed[rest[box_iou(boxes[index], boxes[rest]) > iou_threshold]] = True
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes, scores, classes, iou_threshold: float) -> np.ndarray:
    """Per-class NMS; kept indices by score descending, ties by index."""
    keep = [members[nms(boxes[members], scores[members], iou_threshold)]
            for members in (np.flatnonzero(classes == c) for c in np.unique(classes))]
    keep = np.concatenate(keep) if keep else np.zeros(0, dtype=np.int64)
    return keep[np.lexsort((keep, -scores[keep]))]


@dataclass(frozen=True)
class Candidates:
    """Post-NMS detections over all source classes, before class mapping (diagnostics)."""

    anchors: np.ndarray       # anchor row index in the raw output
    boxes_input: np.ndarray   # xyxy, padded-input pixels
    scores: np.ndarray        # objectness * max class probability
    source_classes: np.ndarray


def candidates(raw: np.ndarray, settings: Settings) -> Candidates:
    """Upstream yolox.utils.postprocess on raw outputs: argmax, threshold, per-class NMS."""
    raw = _raw_rows(raw, settings)
    probs = raw[:, 5:5 + settings.num_classes]
    source = np.argmax(probs, axis=1)
    scores = raw[:, 4].astype(np.float64) * probs[np.arange(len(raw)), source].astype(np.float64)
    rows = np.flatnonzero(scores >= settings.score_threshold)
    boxes = cxcywh_to_xyxy(decode(raw, settings, rows))
    keep = batched_nms(boxes, scores[rows], source[rows], settings.nms_iou_threshold)
    return Candidates(rows[keep], boxes[keep], scores[rows][keep], source[rows][keep])


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]  # xyxy, half-open original-frame pixels
    score: float
    class_id: int
    source_class: int
    anchor: int


def postprocess(raw: np.ndarray, letterbox: Letterbox, settings: Settings) -> list[Detection]:
    """Mapped detections in original-frame pixels, clipped to the frame, by score."""
    return map_candidates(candidates(raw, settings), letterbox, settings)


def map_candidates(found: Candidates, letterbox: Letterbox, settings: Settings) -> list[Detection]:
    """Keep mapped source classes, invert the letterbox, clip, drop boxes left empty."""
    mapped = np.isin(found.source_classes, list(settings.class_map))
    boxes = letterbox.to_original(found.boxes_input[mapped])
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, letterbox.width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, letterbox.height)
    result = []
    for box, score, source, anchor in zip(boxes, found.scores[mapped],
                                          found.source_classes[mapped], found.anchors[mapped]):
        if box[0] < box[2] and box[1] < box[3]:
            result.append(Detection(tuple(float(v) for v in box), float(score),
                                    settings.class_map[int(source)], int(source), int(anchor)))
    return result


def to_records(detections: list[Detection], frame: FrameRecord, letterbox: Letterbox,
               settings: Settings, model_provenance: str) -> list[DetectionRecord]:
    """Contract-v1 detection records for one frame."""
    if (frame.width, frame.height) != (letterbox.width, letterbox.height):
        raise DetectorError("frame: width/height differ from the preprocessed image")
    reasons = ({"timestamp_s": frame.availability_reason.get("timestamp_s", "frame has no timestamp")}
               if frame.timestamp_s is None else {})
    return [DetectionRecord(
        schema_version=SCHEMA_VERSION, sequence_id=frame.sequence_id,
        frame_index=frame.frame_index, timestamp_s=frame.timestamp_s,
        width=frame.width, height=frame.height, bbox=BBox(*d.bbox), class_id=d.class_id,
        class_mapping_version=settings.class_mapping_version, score=d.score,
        model_provenance=model_provenance, availability_reason=reasons) for d in detections]


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_artifact(path: Path | str, sha256: str | None) -> Path:
    """The artifact path if it exists and matches the recorded SHA-256."""
    path = Path(path)
    if not path.is_file():
        raise DetectorError(f"artifact missing: {path}")
    if sha256 is None:
        raise DetectorError(f"artifact has no recorded SHA-256: {path}")
    actual = sha256_file(path)
    if actual != sha256:
        raise DetectorError(f"artifact SHA-256 mismatch for {path}: {actual} != {sha256}")
    return path


class OnnxModel:
    """An ONNX model on ONNX Runtime's CPU execution provider only."""

    def __init__(self, path: Path | str, settings: Settings, *, threads: int = 1):
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.providers = self.session.get_providers()
        if self.providers != ["CPUExecutionProvider"]:
            raise DetectorError(f"onnxruntime: unexpected providers {self.providers}")
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        expected_in = [1, 3, settings.input_height, settings.input_width]
        expected_out = [1, settings.anchors, settings.num_classes + 5]
        if len(inputs) != 1 or inputs[0].shape != expected_in or inputs[0].type != "tensor(float)":
            raise DetectorError(f"onnx: expected one float input {expected_in}")
        if len(outputs) != 1 or outputs[0].shape != expected_out:
            raise DetectorError(f"onnx: expected one output {expected_out}")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.version = onnxruntime.__version__

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: tensor})[0]


@dataclass(frozen=True)
class FrameResult:
    detections: list[Detection]
    candidates: Candidates
    letterbox: Letterbox
    raw: np.ndarray


def detect(model, image: np.ndarray, settings: Settings) -> FrameResult:
    """Run any raw-output model (callable on the input tensor) on one BGR frame."""
    tensor, letterbox = preprocess(image, settings)
    raw = np.asarray(model(tensor))
    found = candidates(raw, settings)
    return FrameResult(map_candidates(found, letterbox, settings), found, letterbox, raw)


def provenance(config: dict, runtime: str) -> str:
    weights = config["model"]["weights"]
    return (f"{config['config_id']}; {config['model']['name']} YOLOX@"
            f"{config['model']['code']['commit'][:12]} weights sha256:{weights['sha256']}; "
            f"{runtime}")


def still_frame(path: Path, image: np.ndarray) -> FrameRecord:
    """A contract frame for an unlabelled still image (no timestamp, no camera)."""
    return FrameRecord(SCHEMA_VERSION, path.stem.replace(" ", "_"), 0, None, "unavailable",
                       int(image.shape[1]), int(image.shape[0]), str(path), sha256_file(path), None,
                       "unlabeled", {"timestamp_s": "still image", "camera_id": "not recorded"})


def _overlay(image: np.ndarray, result: FrameResult, settings: Settings, floor: float) -> np.ndarray:
    """Mapped detections in green; unmapped classes at or above `floor` in grey."""
    canvas = image.copy()
    boxes = result.letterbox.to_original(result.candidates.boxes_input)
    for box, score, source in zip(boxes, result.candidates.scores, result.candidates.source_classes):
        if score >= floor and int(source) not in settings.class_map:
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (160, 160, 160), 1)
            cv2.putText(canvas, f"{COCO_CLASSES[source]} {score:.2f}", (x0, max(y0 - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    for det in result.detections:
        x0, y0, x1, y1 = (int(round(v)) for v in det.bbox)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(canvas, f"aircraft {det.score:.2f}", (x0, max(y0 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return canvas


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the ONNX baseline on still images.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("detect", help="write contract detections and diagnostics per image")
    run.add_argument("images", nargs="+", type=Path)
    run.add_argument("--onnx", type=Path, required=True)
    run.add_argument("--onnx-sha256", required=True, help="expected SHA-256 of the ONNX file")
    run.add_argument("--out", type=Path, required=True, help="output directory (not in Git)")
    run.add_argument("--overlay-floor", type=float, default=None,
                     help="also draw an overlay; unmapped classes are drawn at or above this score")
    args = parser.parse_args(argv)
    config = load_config()
    settings = settings_from_config(config)
    try:
        model = OnnxModel(require_artifact(args.onnx, args.onnx_sha256), settings)
    except DetectorError as exc:
        print(exc, file=sys.stderr)
        return 2
    label = provenance(config, f"onnx sha256:{args.onnx_sha256}; onnxruntime {model.version} CPU")
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"{path}: not a readable image", file=sys.stderr)
            return 2
        result = detect(model, image, settings)
        records = to_records(result.detections, still_frame(path, image), result.letterbox,
                             settings, label)
        stem = str(args.out / path.stem)
        Path(stem + ".detections.jsonl").write_text(
            "".join(r.to_json() + "\n" for r in records), encoding="utf-8")
        found = result.candidates
        boxes = result.letterbox.to_original(found.boxes_input)
        Path(stem + ".candidates.json").write_text(json.dumps({
            "image": str(path), "image_sha256": sha256_file(path), "model_provenance": label,
            "letterbox": vars(result.letterbox),
            "candidates": [{"anchor": int(a), "source_class": int(c), "name": COCO_CLASSES[c],
                            "score": float(s), "bbox_original_unclipped": [float(v) for v in box]}
                           for a, c, s, box in zip(found.anchors, found.source_classes,
                                                   found.scores, boxes)]}, indent=1) + "\n")
        if args.overlay_floor is not None:
            cv2.imwrite(stem + ".overlay.jpg", _overlay(image, result, settings, args.overlay_floor))
        print(f"{path}: {len(records)} aircraft, {len(found.scores)} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
