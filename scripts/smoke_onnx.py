"""Run existing ONNX weights on a bounded photo-derived clip, without downloads."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import cv2

from aeris.detection import baseline as b
from aeris.video import pipeline as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.orchestrate import fingerprint, inventory  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Use a new directory to preserve previous runs')
    image_path = ROOT / 'data/raw/g1-bounded-sample/images/1143132.jpg'
    config = b.load_config()
    model = b.verified_model_path(config)
    photo = cv2.imread(str(image_path))
    if photo is None or photo.shape[0] < 500 or photo.shape[1] < 660:
        raise ValueError('Expected the existing FGVC 1143132 image, at least 660x500')
    output.mkdir(parents=True)
    clip, overlay = output / 'input.mp4', output / 'overlay.mp4'
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 400))
    if not writer.isOpened():
        raise RuntimeError('Cannot create smoke clip')
    try:
        for i in range(6):
            writer.write(photo[100:500, 4 * i:4 * i + 640].copy())
    finally:
        writer.release()
    os.environ['ORT_DISABLE_TELEMETRY'] = '1'
    runner = b.OnnxRuntimeRunner(model)
    result = p.process_video(clip, p.BaselineStages(config, runner),
                             sequence_id='fgvc-1143132-photo-pan', output_video=overlay)
    if result['processed_frames'] != 6 or not result['detections']:
        raise RuntimeError('Smoke failed: expected six processed frames and a detection')
    result['detections'] = [d.to_dict() for d in result['detections']]
    report = {
        'schema_version': 1,
        'run_at': datetime.now(timezone.utc).isoformat(),
        'command': ['PYTHONPATH=python', sys.executable, *sys.argv],
        'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                 text=True).strip(),
        'source_fingerprint': fingerprint(inventory(ROOT)),
        'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'],
                                                    cwd=ROOT, text=True).strip()),
        'input_kind': 'six-frame pan derived from a still photograph; not real video',
        'source': {'id': 'FGVC-Aircraft-2013b/1143132',
                   'sha256': b.sha256_file(image_path),
                   'rights': 'Non-commercial research only; no redistribution of image or clip',
                   'manifest': 'configs/data/source_manifest.json'},
        'config_sha256': b.sha256_file(b.CONFIG_PATH),
        'model_sha256': b.sha256_file(model),
        'execution_provider': runner.session.get_providers(),
        'precision': 'float32',
        'environment': {'os': platform.platform(), 'machine': platform.machine(),
                        'python': platform.python_version(),
                        'packages': {n: version(n) for n in
                                     ('numpy', 'opencv-python-headless', 'onnxruntime')}},
        'dependency_lock_sha256': b.sha256_file(
            ROOT / 'requirements/detection-macos-arm64-py314.txt'),
        'random_seed': None,
        'seed_reason': 'No random transformations; crop x=4*i, y=100, size=640x400, 10 fps',
        'artifact_sha256': {f.name: b.sha256_file(f) for f in (clip, overlay)},
        'accuracy': None,
        'accuracy_reason': 'No exhaustively labeled real video; smoke inference only',
        'result': result,
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(output / 'report.json'),
                      'frames': result['processed_frames'],
                      'detections': len(result['detections'])}))


if __name__ == '__main__':
    main()
