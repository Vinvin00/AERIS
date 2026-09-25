"""Exercise the real acceptance path and failures in disposable repositories."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.orchestrate import DIMENSIONS, GateError, Workflow, file_hash, read, write


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ["AGENTS.md", "docs/PROJECT_STATE.md", "docs/CONTRACTS.md",
                     "orchestration/prompts/implementer.md", "orchestration/prompts/reviewer.md"]:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Test fixture context\n")
        self.tasks = [{"id": "T1", "title": "Fixture task", "objective": "Implement fixture",
                       "status": "planned", "depends_on": [], "read": [],
                       "write_globs": ["src/*"], "acceptance": {"AC1": "Works"},
                       "required_artifacts": ["src/result.txt"], "checks": ["check"],
                       "budget": "Offline fixture"},
                      {"id": "T2", "title": "Dependent task", "objective": "Consume fixture",
                       "status": "planned", "depends_on": ["T1"], "read": [],
                       "write_globs": ["next/*"], "acceptance": {"AC1": "Works"},
                       "required_artifacts": ["next/result.txt"], "checks": ["check"],
                       "budget": "Offline fixture"}]
        write(self.root / "orchestration/tasks.json", {"tasks": self.tasks})
        self.set_check(["{python}", "-c", "print('fixture gate executed')"])

    def set_check(self, argv, timeout=10):
        write(self.root / "orchestration/gates.json", {
            "checks": {"check": {"argv": argv, "timeout_s": timeout}}})
        self.flow = Workflow(self.root)

    def implement(self):
        self.flow.prepare("T1")
        (self.root / "src").mkdir(exist_ok=True)
        (self.root / "src/result.txt").write_text("implemented\n")
        folder = self.flow.attempt("T1")
        report = read(folder / "report.json")
        report.update(author_id="implementation-session", summary="Implemented fixture",
                      changed_files=["src/result.txt"],
                      criteria={"AC1": {"artifact": "src/result.txt", "explanation": "Fixture present"}},
                      handoff={"next_action": "Review output", "context": "No pending choices"})
        write(folder / "report.json", report)
        return folder

    def approve(self, folder):
        review = read(folder / "review.json")
        review.update(reviewer_id="separate-review-session", verdict="accept",
                      summary="Inspected the fixture and successful gate log",
                      scores={name: 3 for name in DIMENSIONS},
                      criteria={"AC1": {"passed": True, "reason": "Matches fixture behavior"}})
        write(folder / "review.json", review)

    def ready_for_accept(self):
        folder = self.implement()
        self.assertTrue(self.flow.verify("T1")["passed"])
        self.approve(folder)
        return folder

    def test_acceptance_unlocks_dependencies(self):
        folder = self.ready_for_accept()
        self.assertEqual(self.flow.status()[1]["status"], "waiting")
        self.flow.accept("T1")
        self.assertEqual(self.flow.status()[1]["status"], "ready")
        packet = Path(self.flow.prepare("T2")).read_text()
        self.assertIn("Implemented fixture", packet)
        self.assertTrue((folder / "source-before/AGENTS.md").exists())

    def test_dependencies_block_dispatch(self):
        with self.assertRaisesRegex(GateError, "Dependencies"):
            self.flow.prepare("T2")

    def test_missing_report_fields_block(self):
        self.flow.prepare("T1")
        with self.assertRaisesRegex(GateError, "author"):
            self.flow.verify("T1")

    def test_out_of_scope_changes_block(self):
        folder = self.implement()
        (self.root / "AGENTS.md").write_text("Weakened rules")
        report = read(folder / "report.json")
        report["changed_files"].append("AGENTS.md")
        write(folder / "report.json", report)
        with self.assertRaisesRegex(GateError, "Out-of-scope"):
            self.flow.verify("T1")

    def test_missing_criterion_evidence_blocks(self):
        folder = self.implement()
        report = read(folder / "report.json")
        report["criteria"]["AC1"]["artifact"] = "missing.txt"
        write(folder / "report.json", report)
        with self.assertRaisesRegex(GateError, "Missing evidence"):
            self.flow.verify("T1")

    def test_path_escape_blocks(self):
        folder = self.implement()
        report = read(folder / "report.json")
        report["criteria"]["AC1"]["artifact"] = "../outside.txt"
        write(folder / "report.json", report)
        with self.assertRaisesRegex(GateError, "escapes"):
            self.flow.verify("T1")

    def test_failed_command_cannot_pass_review(self):
        self.set_check(["{python}", "-c", "raise SystemExit(7)"])
        folder = self.implement()
        receipt = self.flow.verify("T1")
        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["checks"][0]["exit_code"], 7)
        self.approve(folder)
        decision = self.flow.evaluate("T1")
        self.assertFalse(decision["accepted"])
        self.assertIn("checks failed", (folder / "feedback.md").read_text())

    def test_source_changed_after_review_blocks(self):
        self.ready_for_accept()
        (self.root / "src/result.txt").write_text("different implementation\n")
        self.assertFalse(self.flow.evaluate("T1")["accepted"])

    def test_report_changed_after_verification_blocks(self):
        folder = self.ready_for_accept()
        report = read(folder / "report.json")
        report["summary"] = "Different claims"
        write(folder / "report.json", report)
        self.assertFalse(self.flow.evaluate("T1")["accepted"])

    def test_self_review_blocks(self):
        folder = self.ready_for_accept()
        review = read(folder / "review.json")
        review["reviewer_id"] = review["author_id"]
        write(folder / "review.json", review)
        self.assertFalse(self.flow.evaluate("T1")["accepted"])

    def test_tampered_log_blocks(self):
        folder = self.ready_for_accept()
        (folder / "check-check.log").write_text("fabricated pass")
        self.assertFalse(self.flow.evaluate("T1")["accepted"])

    def test_stale_review_blocks(self):
        folder = self.ready_for_accept()
        review = read(folder / "review.json")
        review["fingerprint"] = "old-source-hash"
        write(folder / "review.json", review)
        self.assertFalse(self.flow.evaluate("T1")["accepted"])

    def test_reverification_resets_approval(self):
        folder = self.ready_for_accept()
        self.flow.verify("T1")
        self.assertEqual(read(folder / "review.json")["verdict"], "revise")
        self.assertFalse(self.flow.evaluate("T1")["accepted"])

    def test_major_finding_blocks(self):
        folder = self.ready_for_accept()
        review = read(folder / "review.json")
        review["findings"] = [{"severity": "major", "resolved": False, "detail": "Wrong result"}]
        write(folder / "review.json", review)
        self.assertFalse(self.flow.evaluate("T1")["accepted"])

    def test_low_dimension_cannot_hide_in_average(self):
        folder = self.ready_for_accept()
        review = read(folder / "review.json")
        review["scores"] = {name: 4 for name in DIMENSIONS}
        review["scores"]["correctness"] = 2
        write(folder / "review.json", review)
        self.assertFalse(self.flow.evaluate("T1")["accepted"])

    def test_retry_preserves_ownership_baseline_and_feedback(self):
        self.implement()
        self.flow.abandon("T1", "Need a useful test")
        packet = Path(self.flow.prepare("T1", retry=True)).read_text()
        self.assertIn("Need a useful test", packet)
        base = read(self.flow.attempt("T1") / "base.json")
        self.assertNotIn("src/result.txt", base["files"])

    def test_three_attempt_limit(self):
        self.flow.prepare("T1")
        for index in range(3):
            self.flow.abandon("T1", "Repeated failure")
            if index < 2:
                self.flow.prepare("T1", retry=True)
        with self.assertRaisesRegex(GateError, "Three attempts"):
            self.flow.prepare("T1", retry=True)

    def test_closed_acceptance_cannot_be_overwritten(self):
        self.ready_for_accept()
        self.flow.accept("T1")
        before = file_hash(self.flow.attempt("T1") / "decision.json")
        with self.assertRaisesRegex(GateError, "closed"):
            self.flow.evaluate("T1")
        self.assertEqual(before, file_hash(self.flow.attempt("T1") / "decision.json"))

    def test_dependency_evidence_tampering_blocks_new_packet(self):
        folder = self.ready_for_accept()
        self.flow.accept("T1")
        (folder / "report.json").write_text("{}")
        with self.assertRaisesRegex(GateError, "evidence changed"):
            self.flow.prepare("T2")

    def test_dependency_cycle_is_rejected(self):
        self.tasks[0]["depends_on"] = ["T2"]
        write(self.root / "orchestration/tasks.json", {"tasks": self.tasks})
        with self.assertRaisesRegex(GateError, "cycle"):
            Workflow(self.root).lint()

    def test_only_one_active_writer(self):
        self.tasks[1]["depends_on"] = []
        write(self.root / "orchestration/tasks.json", {"tasks": self.tasks})
        flow = Workflow(self.root)
        flow.prepare("T1")
        with self.assertRaisesRegex(GateError, "owns this checkout"):
            flow.prepare("T2")

    def test_symlink_source_directory_rejected(self):
        (self.root / "linked").symlink_to(self.root / "docs", target_is_directory=True)
        with self.assertRaisesRegex(GateError, "Symlink"):
            self.flow.prepare("T1")

    def test_non_finite_json_rejected(self):
        path = self.root / "invalid.json"
        path.write_text('{"score": NaN}')
        with self.assertRaisesRegex(GateError, "Non-finite"):
            read(path)

    def test_check_modifying_source_is_rejected(self):
        self.set_check(["{python}", "-c",
                        "from pathlib import Path; Path('src/result.txt').write_text('changed')"])
        self.implement()
        with self.assertRaisesRegex(GateError, "Checks changed source"):
            self.flow.verify("T1")

    def test_missing_command_fails_with_receipt(self):
        self.set_check([str(self.root / "not-an-executable")])
        self.implement()
        receipt = self.flow.verify("T1")
        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["checks"][0]["exit_code"], 127)


class DiscoveryTests(unittest.TestCase):
    def test_empty_required_test_file_fails(self):
        runner = Path(__file__).resolve().parents[1] / "tools/check_tests.py"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "test_empty.py"
            target.write_text("# no test cases\n")
            result = subprocess.run([sys.executable, str(runner), str(target)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"No tests discovered", result.stderr)


if __name__ == "__main__":
    unittest.main()
