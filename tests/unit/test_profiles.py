"""Profile table is consistent, and the cascade gate runs the tiles only when needed."""
import json
import unittest

import numpy as np

from aeris.detection import baseline, profiles
from aeris.video.pipeline import BaselineStages


class FakeRunner:
    name = "fake"

    def __init__(self, size, confident_first_call):
        self.size, self.confident_first_call, self.calls = size, confident_first_call, 0

    def __call__(self, tensor):
        self.calls += 1
        cells = sum((self.size // s) ** 2 for s in baseline.STRIDES)
        raw = np.zeros((1, cells, 85), np.float32)
        raw[0, :, 4] = -9.0  # no objectness anywhere unless set below
        if self.calls == 1 and self.confident_first_call:   # crop 0 is the full frame
            raw[0, 0, 4], raw[0, 0, 5 + 4] = 1.0, 0.9      # COCO airplane
        return raw


class Profiles(unittest.TestCase):
    def test_every_profile_has_a_loadable_config(self):
        for name, p in profiles.PROFILES.items():
            config = json.loads(profiles.config_path(name).read_text())
            self.assertIn(config["input"]["size"], (416, 640), name)
            self.assertEqual(config["output"]["shape"][1],
                             sum((config["input"]["size"] // s) ** 2 for s in baseline.STRIDES), name)
            self.assertEqual(len(config["model"]["sha256"]), 64, name)
            self.assertTrue(p.tiles == 1 or p.gate is None or p.gate >= 0.3, name)  # gate >= demo min-score
        self.assertIn(profiles.DEFAULT_PROFILE, profiles.PROFILES)
        self.assertIn(profiles.WEB_DEFAULT_PROFILE, profiles.PROFILES)


class Cascade(unittest.TestCase):
    frame = np.zeros((720, 1280, 3), np.uint8)   # tiles exist: 450 px crops > 416 px input

    def passes(self, confident, gate):
        config = baseline.load_config()
        runner = FakeRunner(config["input"]["size"], confident)
        stages = BaselineStages(config, runner, tiles=2, gate=gate)
        prepared = stages.preprocess(self.frame)
        raw = stages.infer(prepared)
        stages.postprocess(raw, prepared, width=1280, height=720, sequence_id="t", frame_index=0, timestamp_s=0.0)
        return runner.calls

    def test_confident_full_frame_skips_tiles(self):
        self.assertEqual(self.passes(confident=True, gate=0.5), 1)

    def test_nothing_confident_runs_every_tile(self):
        self.assertEqual(self.passes(confident=False, gate=0.5), 5)

    def test_no_gate_always_tiles(self):
        self.assertEqual(self.passes(confident=True, gate=None), 5)


if __name__ == "__main__":
    unittest.main()
