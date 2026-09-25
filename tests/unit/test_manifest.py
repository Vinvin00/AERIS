"""Checks configs/data/source_manifest.json. Metadata only; no dataset is loaded."""
import copy
import json
import re
import unittest
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[2] / "configs/data/source_manifest.json"
BINS = ["[0,16)", "[16,32)", "[32,64)", "[64,inf)"]  # docs/CONTRACTS.md width bins
ROLES = {"still_image_smoke_baseline", "out_of_domain_reference", "primary_video_benchmark"}
STATUSES = {"not_acquired", "planned", "acquired"}
HEX = {"md5_published": 32, "md5_measured": 32, "sha256_measured": 64}
DUPLICATE_CHECKS = {"sha256 of encoded file bytes", "sha256 of decoded RGB frame pixels"}
PARTIAL_LABEL_UNAVAILABLE = {"AP", "precision", "false positives per frame"}

# Required keys per object; "*" means every list item. Values may still be null
# with an availability_reason, but the key itself must be present.
REQUIRED = {
    (): {"schema_version", "manifest_id", "audited_at", "scope_note", "class_mapping",
         "width_bins_px", "sources", "split_policy", "acquisition_plan"},
    ("class_mapping",): {"version", "classes"},
    ("sources", "*"): {"id", "name", "role", "acquisition_status", "source_uri", "provenance",
                       "rights", "modality", "classes", "annotation_capabilities",
                       "planned_evaluations", "eligible_evaluations", "counts", "discrepancies",
                       "width_bin_support", "retrieval", "grouping", "pretraining_overlap"},
    ("sources", "*", "rights"): {"terms_url", "terms_summary", "commercial_use",
                                 "redistribution_permitted", "repo_bundling_permitted"},
    ("sources", "*", "modality"): {"sensor", "media", "fixed_camera"},
    ("sources", "*", "classes"): {"source_taxonomy", "mapping_to_aeris"},
    ("sources", "*", "annotation_capabilities"): {
        "bbox", "objects_per_image", "exhaustive_labels", "default_annotation_status",
        "track_ids", "timestamps", "annotated_negative_frames", "occlusion_truncation_flags",
        "bbox_convention_stated", "bbox_convention_observed"},
    ("sources", "*", "retrieval"): {"artifacts", "instructions"},
    ("sources", "*", "retrieval", "artifacts", "*"): {"name", "url", "size_bytes", *HEX},
    ("sources", "*", "grouping"): {"group_key", "official_split", "leakage_risk"},
    ("split_policy",): {"unit", "procedure", "splits", "holdout_sealed", "holdout_rule",
                        "derivatives_inherit_source_split", "exact_duplicate_checks",
                        "near_duplicate_audit", "negatives", "tiny_objects",
                        "pilot_allocation", "interval_policy"},
    ("split_policy", "near_duplicate_audit"): {"method", "hamming_threshold"},
    ("split_policy", "negatives"): {"definition", "counts_only_complete_status", "requirement"},
    ("split_policy", "tiny_objects"): {"bins", "requirement"},
    ("split_policy", "pilot_allocation"): {"sessions_min", "development", "validation", "holdout"},
    ("acquisition_plan",): {"missing_labels", "caps", "effort_hours_estimate", "effort_basis",
                            "review_sample_fraction", "constraints", "decisions_needed"},
    ("acquisition_plan", "caps"): {"retained_media_gb", "annotation_and_review_hours", "clips_max"},
}


def load(text):
    def pairs(items):
        keys = [key for key, _ in items]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate key in {keys}")
        return dict(items)

    def constant(value):
        raise ValueError(f"nonfinite constant {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def nodes(node, pattern, path="$"):
    """Yield (path, value) for a REQUIRED pattern; missing parents are reported elsewhere."""
    if not pattern:
        yield path, node
    elif pattern[0] == "*":
        for index, item in enumerate(node if isinstance(node, list) else []):
            yield from nodes(item, pattern[1:], f"{path}[{index}]")
    elif isinstance(node, dict) and pattern[0] in node:
        yield from nodes(node[pattern[0]], pattern[1:], f"{path}.{pattern[0]}")


def key_problems(manifest):
    found = []
    for pattern, keys in REQUIRED.items():
        for path, node in nodes(manifest, pattern):
            if not isinstance(node, dict):
                found.append(f"{path}: expected object")
            else:
                found += [f"{path}: missing key {key}" for key in sorted(keys - set(node))]
    return found


def null_problems(node, path="$"):
    """Every null needs a sibling availability_reason entry; no stale reasons."""
    found = []
    if isinstance(node, dict):
        reasons = node.get("availability_reason", {})
        if not isinstance(reasons, dict):
            return [f"{path}.availability_reason: expected object"]
        nulls = {key for key, value in node.items() if value is None}
        for key in sorted(nulls):
            if not isinstance(reasons.get(key), str) or not reasons[key].strip():
                found.append(f"{path}.{key}: null without availability_reason")
        found += [f"{path}.availability_reason.{key}: stale reason" for key in set(reasons) - nulls]
        for key, value in node.items():
            found += null_problems(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found += null_problems(value, f"{path}[{index}]")
    return found


def texts(value, name, allow_empty=False):
    if (not isinstance(value, list) or (not value and not allow_empty)
            or any(not isinstance(item, str) or not item.strip() for item in value)):
        return [f"{name}: expected a {'' if allow_empty else 'nonempty '}list of nonempty strings"]
    return []


def count(value):
    return type(value) is int and value >= 0


def source_problems(source, class_ids):
    sid = source["id"]
    found = []
    caps, rights, media = source["annotation_capabilities"], source["rights"], source["modality"]["media"]
    if source["role"] not in ROLES:
        found.append(f"{sid}: unknown role")
    if source["acquisition_status"] not in STATUSES:
        found.append(f"{sid}: unknown acquisition_status")
    if rights["terms_url"] is not None and not rights["terms_url"].startswith("https://"):
        found.append(f"{sid}: terms_url must be https")
    if rights["repo_bundling_permitted"] is not False:
        found.append(f"{sid}: media must never be bundled in the repository")
    found += texts(source["planned_evaluations"], f"{sid}.planned_evaluations", allow_empty=True)
    found += texts(source["eligible_evaluations"], f"{sid}.eligible_evaluations", allow_empty=True)
    found += texts(source["discrepancies"], f"{sid}.discrepancies", allow_empty=True)
    artifacts = source["retrieval"]["artifacts"]
    if source["eligible_evaluations"] and source["acquisition_status"] != "acquired":
        found.append(f"{sid}: only acquired sources can have eligible_evaluations")
    if source["acquisition_status"] == "acquired" and (
            not artifacts or any(item["sha256_measured"] is None for item in artifacts)):
        found.append(f"{sid}: acquired sources need a measured SHA-256 for every artifact")
    evals = set(source["planned_evaluations"]) | set(source["eligible_evaluations"])
    if evals & {"tracking", "forecast"} and not (
            "video" in media and caps["track_ids"] is True and caps["timestamps"] is True):
        found.append(f"{sid}: tracking/forecast claims need video, track IDs and timestamps")
    if "detection" in evals and not (caps["exhaustive_labels"] is True
                                     and caps["annotated_negative_frames"] is True):
        found.append(f"{sid}: detection benchmark needs exhaustive labels and annotated negatives")
    if evals and caps["exhaustive_labels"] is not True:
        policy = source.get("metric_policy")
        if not isinstance(policy, dict) or not PARTIAL_LABEL_UNAVAILABLE <= set(
                policy.get("unavailable", [])) or not str(policy.get("unmatched_detections", "")).strip():
            found.append(f"{sid}: partial labels need metric_policy marking "
                         f"{sorted(PARTIAL_LABEL_UNAVAILABLE)} unavailable and scoring unmatched detections")
    mapping = source["classes"]["mapping_to_aeris"]
    if not isinstance(mapping, dict) or not set(mapping.values()) <= class_ids:
        found.append(f"{sid}: mapping_to_aeris uses an undefined AERIS class ID")
    support = source["width_bin_support"]
    if support is not None and (not isinstance(support, dict) or list(support) != BINS
                                or not all(count(value) for value in support.values())):
        found.append(f"{sid}: width_bin_support must give nonnegative integer counts for {BINS}")
    if not isinstance(source["retrieval"]["instructions"], str) or not source["retrieval"]["instructions"].strip():
        found.append(f"{sid}: retrieval.instructions must be nonempty")
    for artifact in artifacts:
        if artifact["size_bytes"] is not None and not count(artifact["size_bytes"]):
            found.append(f"{sid}: {artifact['name']} size_bytes must be a nonnegative integer")
        for key, length in HEX.items():
            value = artifact[key]
            if value is not None and not (isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value)):
                found.append(f"{sid}: {artifact['name']} {key} is not lowercase hex[{length}]")
    return found


def policy_problems(manifest):
    found = []
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        found.append("schema_version must be integer 1")
    if manifest["width_bins_px"] != BINS:
        found.append("width_bins_px must match CONTRACTS.md")
    class_ids = {int(key) for key in manifest["class_mapping"]["classes"]}
    ids = [source["id"] for source in manifest["sources"]]
    if len(ids) != len(set(ids)):
        found.append("duplicate source id")
    for source in manifest["sources"]:
        found += source_problems(source, class_ids)

    split = manifest["split_policy"]
    for key, expected in (("unit", "source_group"), ("holdout_sealed", True),
                          ("derivatives_inherit_source_split", True)):
        if type(split[key]) is not type(expected) or split[key] != expected:
            found.append(f"split_policy.{key} must be {expected!r}")
    if set(split["exact_duplicate_checks"]) != DUPLICATE_CHECKS:
        found.append("split_policy: need encoded-byte and decoded-pixel duplicate checks")
    negatives = split["negatives"]
    if negatives["counts_only_complete_status"] is not True or not (
            "complete" in negatives["definition"] and "zero objects" in negatives["definition"]):
        found.append("split_policy: only complete frames with zero objects can count as negatives")
    if split["tiny_objects"]["bins"] != BINS:
        found.append("split_policy.tiny_objects.bins must match CONTRACTS.md")
    found += texts(split["procedure"], "split_policy.procedure")
    for key in ("holdout_rule", "interval_policy"):
        if not isinstance(split[key], str) or not split[key].strip():
            found.append(f"split_policy.{key} must be nonempty")
    allocation = split["pilot_allocation"]
    shares = [allocation[name] for name in ("development", "validation", "holdout")]
    if (not all(count(value) for value in [*shares, allocation["sessions_min"]])
            or min(shares) < 1 or sum(shares) > allocation["sessions_min"]):
        found.append("split_policy.pilot_allocation: each split needs >=1 whole session (integers)")

    plan = manifest["acquisition_plan"]
    for key in ("missing_labels", "constraints", "decisions_needed"):
        found += texts(plan[key], f"acquisition_plan.{key}")
    labels = " ".join(item for item in plan["missing_labels"] if isinstance(item, str))
    for needed in ("track IDs", "negative", "tiny"):
        if needed not in labels:
            found.append(f"acquisition_plan.missing_labels must name {needed}")
    estimates = plan["effort_hours_estimate"]
    if not isinstance(estimates, dict) or not estimates:
        found.append("acquisition_plan.effort_hours_estimate must be nonempty")
    else:
        for name, value in estimates.items():
            if (not isinstance(value, list) or len(value) != 2
                    or any(type(item) not in (int, float) for item in value) or not 0 < value[0] <= value[1]):
                found.append(f"acquisition_plan.effort_hours_estimate.{name}: need [low, high] with 0 < low <= high")
    if not all(type(value) in (int, float) and value > 0 for value in plan["caps"].values()):
        found.append("acquisition_plan.caps must be positive numbers")
    return found


def problems(manifest):
    """Report problems instead of raising: structure first, then policy."""
    if not isinstance(manifest, dict):
        return ["$: expected object"]
    found = key_problems(manifest) + null_problems(manifest)
    if found:
        return found
    try:
        return policy_problems(manifest)
    except (TypeError, AttributeError, KeyError, ValueError) as error:
        return [f"malformed manifest: {error!r}"]


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load(MANIFEST.read_text())
        self.sources = {source["id"]: source for source in self.manifest["sources"]}

    def assertProblem(self, manifest, text):
        found = problems(manifest)
        self.assertTrue(any(text in problem for problem in found),
                        f"expected a problem containing {text!r}, got {found}")

    def test_committed_manifest_is_valid(self):
        self.assertEqual(problems(self.manifest), [])

    def test_strict_json(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":-Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                load(text)

    def test_every_required_key_is_enforced(self):
        checked = 0
        for pattern, keys in REQUIRED.items():
            for key in sorted(keys):
                bad = copy.deepcopy(self.manifest)
                path, node = next(nodes(bad, pattern))
                del node[key]
                with self.subTest(path=path, key=key):
                    self.assertProblem(bad, f"{path}: missing key {key}")
                checked += 1
        self.assertEqual(checked, sum(map(len, REQUIRED.values())))

    def test_malformed_values_are_reported_not_raised(self):
        for mutate, text in (
                (lambda m: m["sources"][0].__setitem__("availability_reason", None), "expected object"),
                (lambda m: m["acquisition_plan"]["missing_labels"].__setitem__(0, None), "missing_labels"),
                (lambda m: m["sources"].__setitem__(0, []), "expected object"),
                (lambda m: m["sources"][0]["classes"].__setitem__("mapping_to_aeris", None), "null without"),
                (lambda m: m["sources"][0]["retrieval"]["artifacts"][0].__setitem__("md5_measured", 5), "md5_measured")):
            bad = copy.deepcopy(self.manifest)
            mutate(bad)
            with self.subTest(text=text):
                self.assertProblem(bad, text)
        self.assertEqual(problems([]), ["$: expected object"])

    def test_null_needs_reason_and_reason_needs_null(self):
        bad = copy.deepcopy(self.manifest)
        bad["sources"][0]["provenance"] = None
        self.assertProblem(bad, "provenance: null without availability_reason")
        bad = copy.deepcopy(self.manifest)
        bad["sources"][0]["availability_reason"]["provenance"] = "stale"
        self.assertProblem(bad, "stale reason")

    def test_capability_claims_are_limited_by_labels_and_acquisition(self):
        for index, field, value, text in (
                (0, "planned_evaluations", "tracking", "tracking/forecast claims"),
                (0, "planned_evaluations", "detection", "exhaustive labels"),
                (2, "eligible_evaluations", "detection", "only acquired sources")):
            bad = copy.deepcopy(self.manifest)
            bad["sources"][index][field].append(value)
            with self.subTest(field=field, value=value):
                self.assertProblem(bad, text)
        bad = copy.deepcopy(self.manifest)
        bad["sources"][0]["classes"]["mapping_to_aeris"]["x"] = 7
        self.assertProblem(bad, "undefined AERIS class")
        bad = copy.deepcopy(self.manifest)
        bad["sources"][0]["acquisition_status"] = "acquired"
        self.assertProblem(bad, "measured SHA-256")

    def test_partial_labels_need_metric_policy(self):
        for mutate in (lambda s: s.pop("metric_policy"),
                       lambda s: s["metric_policy"]["unavailable"].remove("AP"),
                       lambda s: s["metric_policy"].__setitem__("unmatched_detections", " ")):
            bad = copy.deepcopy(self.manifest)
            mutate(bad["sources"][0])
            self.assertProblem(bad, "metric_policy")

    def test_rights_and_checksums(self):
        for mutate, text in (
                (lambda s: s["rights"].__setitem__("repo_bundling_permitted", True), "never be bundled"),
                (lambda s: s["rights"].__setitem__("terms_url", "http://example.org"), "https"),
                (lambda s: s["retrieval"]["artifacts"][0].__setitem__("md5_measured", "ABC"), "md5_measured"),
                (lambda s: s["retrieval"]["artifacts"][0].__setitem__("size_bytes", -1), "size_bytes"),
                (lambda s: s["retrieval"].__setitem__("instructions", ""), "instructions")):
            bad = copy.deepcopy(self.manifest)
            mutate(bad["sources"][0])
            with self.subTest(text=text):
                self.assertProblem(bad, text)

    def test_split_policy_mutations(self):
        encoded = "sha256 of encoded file bytes"
        for path, value, text in (
                (("unit",), "frame", "unit"),
                (("holdout_sealed",), False, "holdout_sealed"),
                (("holdout_sealed",), 1, "holdout_sealed"),
                (("derivatives_inherit_source_split",), False, "derivatives"),
                (("negatives", "counts_only_complete_status"), False, "zero objects"),
                (("negatives", "definition"), "any frame with no boxes", "zero objects"),
                (("tiny_objects", "bins"), BINS[:3], "tiny_objects"),
                (("pilot_allocation", "holdout"), 0, "pilot_allocation"),
                (("pilot_allocation", "holdout"), 1.5, "pilot_allocation"),
                (("exact_duplicate_checks",), [encoded], "decoded-pixel"),
                (("exact_duplicate_checks",), [encoded, encoded], "decoded-pixel"),
                (("interval_policy",), " ", "interval_policy"),
                (("procedure",), [""], "procedure")):
            bad = copy.deepcopy(self.manifest)
            node = bad["split_policy"]
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            with self.subTest(path=path, value=value):
                self.assertProblem(bad, text)
        bad = copy.deepcopy(self.manifest)
        bad["schema_version"] = True
        self.assertProblem(bad, "schema_version")

    def test_acquisition_plan_mutations(self):
        for key, value, text in (
                ("missing_labels", ["boxes"], "track IDs"),
                ("constraints", [""], "constraints"),
                ("decisions_needed", [], "decisions_needed"),
                ("effort_hours_estimate", {}, "effort_hours_estimate"),
                ("effort_hours_estimate", {"annotation": [14, 8]}, "low <= high"),
                ("caps", {"retained_media_gb": 0, "annotation_and_review_hours": 24, "clips_max": 12}, "caps")):
            bad = copy.deepcopy(self.manifest)
            bad["acquisition_plan"][key] = value
            with self.subTest(key=key, value=value):
                self.assertProblem(bad, text)

    def test_measured_fgvc_facts_and_discrepancies(self):
        fgvc = self.sources["fgvc-aircraft-2013b"]
        measured = fgvc["counts"]["measured_annotations_archive"]
        self.assertEqual(measured["images"], sum(measured["split_images"].values()))
        self.assertEqual(measured["images_in_more_than_one_split"], 0)
        self.assertNotEqual(measured["images"], fgvc["counts"]["published_landing_page"]["images"])
        artifact = fgvc["retrieval"]["artifacts"][0]
        self.assertNotEqual(artifact["md5_published"], artifact["md5_measured"])
        text = " ".join(fgvc["discrepancies"])
        self.assertIn(artifact["md5_measured"], text)
        self.assertIn("10,000", text)
        self.assertIn("Last-Modified", text)
        self.assertIn("sha256_measured", fgvc["retrieval"]["instructions"])
        self.assertEqual(sum(fgvc["width_bin_support"].values()), measured["images"])
        self.assertEqual(fgvc["width_bin_support"]["[32,64)"], 0)
        self.assertEqual(fgvc["planned_evaluations"], ["detection_smoke"])
        self.assertEqual(fgvc["annotation_capabilities"]["default_annotation_status"], "partial")

    def test_only_owned_pilot_can_back_video_claims(self):
        video = [s["id"] for s in self.manifest["sources"]
                 if set(s["planned_evaluations"]) & {"tracking", "forecast"}]
        self.assertEqual(video, ["aeris-owned-video-pilot"])
        pilot = self.sources["aeris-owned-video-pilot"]
        self.assertEqual(pilot["acquisition_status"], "planned")
        self.assertEqual(pilot["eligible_evaluations"], [])
        self.assertIsNone(pilot["pretraining_overlap"])


if __name__ == "__main__":
    unittest.main()
