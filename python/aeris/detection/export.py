"""PyTorch reference, ONNX export and PyTorch/ONNX parity for the YOLOX-S baseline (AERIS-006).

Needs torch, torchvision and YOLOX built from the pinned commit
(`requirements/detector-torch-linux-x86_64-py311.txt`). The torch-free parts of the baseline
live in `aeris.detection.baseline`, which both paths share: preprocessing, decoding, NMS,
class mapping and the inverse letterbox. So parity is measured on raw head outputs and on
the final detections that the same postprocessing makes from each.

    python -m aeris.detection.export parity --out DIR
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from aeris.detection import baseline as b
from aeris.evaluation._common import record_mismatches

AERIS_ONNX = b.ROOT / "models" / "onnx" / "yolox_s_aeris_opset17.onnx"
SAMPLE_DIR = b.ROOT / "data" / "raw" / "detector_samples"


def reference_versions() -> dict:
    """Installed torch/torchvision versions and the YOLOX commit pip recorded."""
    dist = importlib.metadata.distribution("yolox")
    direct = dist.read_text("direct_url.json")
    info = json.loads(direct) if direct else {}
    return {"torch": importlib.metadata.version("torch"),
            "torchvision": importlib.metadata.version("torchvision"),
            "yolox": {"version": dist.version, "url": info.get("url"),
                      "commit": info.get("vcs_info", {}).get("commit_id"),
                      "changed_files": record_mismatches(dist)}}


def require_pinned_yolox(config: dict) -> dict:
    """The versions record, if YOLOX is the pinned commit with unmodified installed files."""
    versions = reference_versions()
    pinned = config["model"]["code"]["commit"]
    problems = []
    if versions["yolox"]["commit"] != pinned:
        problems.append(f"YOLOX commit {versions['yolox']['commit']} != pinned {pinned}")
    if versions["yolox"]["changed_files"]:
        problems.append(f"installed YOLOX files differ from RECORD: {versions['yolox']['changed_files']}")
    if problems:
        raise b.DetectorError("; ".join(problems))
    return versions


class TorchModel:
    """Upstream YOLOX-S in PyTorch on CPU, returning raw head outputs (no decode)."""

    def __init__(self, config: dict, settings: b.Settings, checkpoint: Path | str | None = None):
        import torch
        from yolox.exp import get_exp

        self.versions = require_pinned_yolox(config)
        weights = config["model"]["weights"]
        path = b.require_artifact(checkpoint or b.ROOT / weights["path"], weights["sha256"])
        exp = get_exp(None, "yolox-s")
        expected = config["model"]["code"]["exp_values"]
        actual = {"depth": exp.depth, "width": exp.width, "num_classes": exp.num_classes,
                  "act": exp.act}
        if actual != expected or tuple(exp.test_size) != (settings.input_height, settings.input_width):
            raise b.DetectorError(f"YOLOX exp {actual} {exp.test_size} differs from the config")
        model = exp.get_model()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"] if "model" in state else state)
        model.eval()
        model.head.decode_in_inference = False
        self.module = model
        self.settings = settings

    def __call__(self, tensor: np.ndarray, *, decode: bool = False) -> np.ndarray:
        import torch

        self.module.head.decode_in_inference = decode
        try:
            with torch.no_grad():
                return self.module(torch.from_numpy(tensor)).numpy()
        finally:
            self.module.head.decode_in_inference = False


def export_onnx(model: TorchModel, path: Path | str, opset: int) -> Path:
    """Export as upstream tools/export_onnx.py does: SiLU as x * sigmoid(x), no decode."""
    import onnx
    import torch
    from torch import nn
    from yolox.models.network_blocks import SiLU
    from yolox.utils import replace_module

    module = replace_module(copy.deepcopy(model.module), nn.SiLU, SiLU)
    module.head.decode_in_inference = False
    module.eval()
    dummy = torch.zeros(1, 3, model.settings.input_height, model.settings.input_width)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(module, (dummy,), str(path), input_names=["images"],
                          output_names=["output"], opset_version=opset, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    return path


def parity_images(dog_path: Path) -> list[tuple[str, np.ndarray, str]]:
    """The fixed parity sample: (id, BGR image, provenance)."""
    dog = cv2.imread(str(dog_path), cv2.IMREAD_COLOR)
    if dog is None:
        raise b.DetectorError(f"cannot read {dog_path}")
    rng = np.random.default_rng(6)
    wide = cv2.resize(dog[144:432], (1000, 333), interpolation=cv2.INTER_AREA)
    return [
        ("yolox-dog", dog, f"real photo, SHA-256 {b.sha256_file(dog_path)}"),
        ("yolox-dog-wide", wide, "rows 144-431 of yolox-dog resized to 1000x333 (INTER_AREA): "
                                 "floor(333 r) is not an integer multiple"),
        ("gray-720p", np.full((720, 1280, 3), 114, np.uint8), "uniform 114, 1280x720"),
        ("noise-720p", rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8),
         "uniform noise, numpy default_rng(6), 1280x720"),
    ]


def _detections(found: b.Candidates, letterbox: b.Letterbox) -> dict:
    boxes = letterbox.to_original(found.boxes_input)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, letterbox.width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, letterbox.height)
    return {(int(a), int(c)): (box, float(s), bi)
            for a, c, s, box, bi in zip(found.anchors, found.source_classes, found.scores, boxes,
                                        found.boxes_input)}


def _exempt(key, item, other: dict, settings: b.Settings, tol: float) -> bool:
    """An unmatched detection passes only if its own score is within `tol` of the score
    threshold, or its IoU with a same-class box kept by the other output is within `tol`
    of the NMS threshold (configs/detector/baseline.json parity_tolerances)."""
    if abs(item[1] - settings.score_threshold) <= tol:
        return True
    same = [v[2] for k, v in other.items() if k[1] == key[1]]
    if not same:
        return False
    ious = b.box_iou(item[2], np.array(same))
    return bool(np.any(np.abs(ious - settings.nms_iou_threshold) <= tol))


def compare(raw_ref: np.ndarray, raw_test: np.ndarray, letterbox: b.Letterbox,
            settings: b.Settings, tolerances: dict) -> dict:
    """Raw-output and final-detection agreement of one frame under the declared tolerances."""
    ref, test = raw_ref.reshape(-1, raw_ref.shape[-1]), raw_test.reshape(-1, raw_test.shape[-1])
    reg = float(np.max(np.abs(ref[:, :4].astype(np.float64) - test[:, :4])))
    prob = float(np.max(np.abs(ref[:, 4:].astype(np.float64) - test[:, 4:])))
    det_ref = _detections(b.candidates(ref, settings), letterbox)
    det_test = _detections(b.candidates(test, settings), letterbox)
    matched = sorted(set(det_ref) & set(det_test))
    box_diff = max((float(np.max(np.abs(det_ref[k][0] - det_test[k][0]))) for k in matched),
                   default=0.0)
    score_diff = max((abs(det_ref[k][1] - det_test[k][1]) for k in matched), default=0.0)
    boundary = tolerances["boundary_abs"]
    unmatched = []
    for side, mine, other in (("reference_only", det_ref, det_test),
                              ("test_only", det_test, det_ref)):
        for key in sorted(set(mine) - set(other)):
            unmatched.append({"side": side, "anchor": key[0], "source_class": key[1],
                              "score": mine[key][1],
                              "exempt": _exempt(key, mine[key], other, settings, boundary)})
    checks = {"raw_regression_abs": reg <= tolerances["raw_regression_abs"],
              "raw_probability_abs": prob <= tolerances["raw_probability_abs"],
              "final_box_coordinate_abs_px": box_diff <= tolerances["final_box_coordinate_abs_px"],
              "final_score_abs": score_diff <= tolerances["final_score_abs"],
              "unmatched_all_exempt": all(u["exempt"] for u in unmatched)}
    return {"raw_regression_max_abs": reg, "raw_probability_max_abs": prob,
            "detections_reference": len(det_ref), "detections_test": len(det_test),
            "matched": len(matched), "final_box_max_abs_px": box_diff,
            "final_score_max_abs": score_diff, "unmatched": unmatched, "checks": checks,
            "passed": all(checks.values())}


def tolerances_from_config(config: dict) -> dict:
    tol = config["parity_tolerances"]
    return {key: tol[key] for key in ("raw_regression_abs", "raw_probability_abs",
                                      "final_box_coordinate_abs_px", "final_score_abs",
                                      "boundary_abs")}


def run_parity(config: dict, settings: b.Settings, torch_model: TorchModel,
               onnx_models: dict, images: list) -> dict:
    """Compare the PyTorch reference with each ONNX model on the fixed sample."""
    tolerances = tolerances_from_config(config)
    report = {"tolerances": tolerances, "reference": torch_model.versions, "frames": []}
    for image_id, image, provenance in images:
        tensor, letterbox = b.preprocess(image, settings)
        raw_ref = torch_model(tensor)
        frame = {"id": image_id, "provenance": provenance, "shape": list(image.shape),
                 "input_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                 "letterbox": vars(letterbox), "results": {}}
        for name, model in onnx_models.items():
            frame["results"][name] = compare(raw_ref, model(tensor), letterbox, settings, tolerances)
        report["frames"].append(frame)
    report["passed"] = {name: all(f["results"][name]["passed"] for f in report["frames"])
                        for name in onnx_models}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("parity", help="export, then compare PyTorch and ONNX Runtime")
    run.add_argument("--out", type=Path, required=True, help="report directory (not in Git)")
    run.add_argument("--onnx", type=Path, default=AERIS_ONNX)
    run.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    config = b.load_config()
    settings = b.settings_from_config(config)
    try:
        model = TorchModel(config, settings)
        path = export_onnx(model, args.onnx, config["export"]["opset"])
        upstream = config["model"]["upstream_onnx_cross_check"]
        onnx_models = {
            "aeris_export": b.OnnxModel(path, settings, threads=args.threads),
            "upstream_export": b.OnnxModel(b.require_artifact(b.ROOT / upstream["path"],
                                                              upstream["sha256"]),
                                           settings, threads=args.threads)}
        images = parity_images(b.require_artifact(SAMPLE_DIR / "yolox-dog.jpg",
                                                  "5a9522051c3cec2bbd2f6323fccba32e8fbf3ddcc2b3e2fd46b04c720bc6f866"))
    except b.DetectorError as exc:
        print(exc, file=sys.stderr)
        return 2
    report = run_parity(config, settings, model, onnx_models, images)
    report["aeris_export"] = {"path": str(path.relative_to(b.ROOT)), "sha256": b.sha256_file(path),
                              "size_bytes": path.stat().st_size,
                              "onnxruntime": onnx_models["aeris_export"].version}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "parity.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    for frame in report["frames"]:
        for name, result in frame["results"].items():
            print(f"{frame['id']:15s} {name:16s} reg {result['raw_regression_max_abs']:.2e} "
                  f"prob {result['raw_probability_max_abs']:.2e} box {result['final_box_max_abs_px']:.2e} "
                  f"det {result['matched']}/{result['detections_reference']} "
                  f"{'PASS' if result['passed'] else 'FAIL'}")
    return 0 if report["passed"]["aeris_export"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
