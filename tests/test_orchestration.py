"""Exercise the real acceptance path and failures in disposable repositories."""

import os
from pathlib import Path
import shutil
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



class GitRetryTests(unittest.TestCase):
    """TOOL-001: retry baselines over committed external changes, in disposable Git repositories."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = {name: "Test fixture context\n" for name in [
            "AGENTS.md", "docs/PROJECT_STATE.md", "docs/CONTRACTS.md",
            "orchestration/prompts/implementer.md", "orchestration/prompts/reviewer.md"]}
        self.files.update({"notes/f.txt": "v0\n", "notes/old.txt": "old\n", "notes/gone.txt": "gone\n",
                           "tools/helper.py": "print('x')\n"})
        self.tasks = [{"id": "T1", "title": "Fixture task", "objective": "Implement fixture",
                       "status": "planned", "depends_on": [], "read": [], "write_globs": ["src/*"],
                       "acceptance": {"AC1": "Works"}, "required_artifacts": ["src/result.txt"],
                       "checks": ["check"], "budget": "Offline fixture"}]
        for name, text in self.files.items():
            self.put(name, text)
        write(self.root / "orchestration/tasks.json", {"tasks": self.tasks})
        write(self.root / "orchestration/gates.json", {
            "checks": {"check": {"argv": ["{python}", "-c", "print('ok')"], "timeout_s": 10}}})
        self.git("init", "-q")
        self.git("config", "user.email", "t@example.invalid")
        self.git("config", "user.name", "tester")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "initial")
        self.branch = self.git("rev-parse", "--abbrev-ref", "HEAD").strip()
        self.base_head = self.git("rev-parse", "HEAD").strip()
        self.flow = Workflow(self.root)

    def git(self, *args):
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        return subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True,
                              text=True, env=env).stdout

    def put(self, name, text):
        path = self.root / name
        if text is None:
            path.unlink()
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def commit_files(self, changes, message="external"):
        for name, text in changes.items():
            self.put(name, text)
        self.git("add", "-A", "--", *changes)
        self.git("commit", "-q", "-m", message)

    def start(self):
        """First attempt with some uncommitted in-scope work, then closed."""
        self.flow.prepare("T1")
        self.put("src/result.txt", "implemented\n")
        self.flow.abandon("T1", "closed for test")

    def retry(self, **kw):
        self.flow = Workflow(self.root)
        return self.flow.prepare("T1", retry=True, **kw)

    def base(self):
        return read(self.flow.attempt("T1") / "base.json")

    def delta(self):
        from tools.orchestrate import changed, inventory
        return changed(self.base()["files"], inventory(self.root))

    def test_first_attempt_records_base_commit_and_dirty_flag(self):
        self.flow.prepare("T1")
        base = self.base()
        self.assertEqual(base["base_commit"], self.base_head)
        self.assertFalse(base["tree_dirty_at_base"])

    def test_clean_external_commit_is_rebased_audited_and_packeted(self):
        self.start()
        self.commit_files({"notes/ext.txt": "external\n"})
        packet = Path(self.retry()).read_text()
        base = self.base()
        self.assertEqual(self.delta(), ["src/result.txt"])
        entry = base["rebased"][0]
        self.assertEqual((entry["path"], entry["action"]), ("notes/ext.txt", "overlay"))
        self.assertEqual(len(entry["sha256"]), 64)
        self.assertEqual(self.flow.last_rebased, base["rebased"])
        self.assertIn("audit list", packet)
        self.assertIn("notes/ext.txt", packet)
        self.assertIn("notes/ext.txt", self.flow._rebased_note(self.flow.attempt("T1")))

    def test_baseline_keeps_attempt_one_work_visible(self):
        self.start()
        self.commit_files({"notes/ext.txt": "external\n"})
        self.retry()
        self.assertIn("src/result.txt", self.delta())

    def test_commit_touching_in_scope_and_out_of_scope_files_is_not_trusted(self):
        self.start()
        self.commit_files({"src/other.txt": "task\n", "notes/mixed.txt": "mixed\n"}, "mixed")
        self.retry()
        self.assertEqual(self.base()["rebased"], [])
        self.assertEqual(self.delta(), ["notes/mixed.txt", "src/other.txt", "src/result.txt"])

    def test_in_scope_committed_change_stays_in_the_delta(self):
        self.start()
        self.commit_files({"src/committed.txt": "task work\n"})
        self.retry()
        self.assertIn("src/committed.txt", self.delta())

    def test_file_tainted_by_a_mixed_commit_stays_untrusted_even_if_later_reverted(self):
        self.start()
        self.commit_files({"src/x.txt": "x\n", "notes/f.txt": "tainted\n"}, "mixed")
        self.commit_files({"notes/f.txt": "v1\n"}, "pure later edit")
        self.retry()
        self.assertEqual(self.base()["rebased"], [])
        self.assertIn("notes/f.txt", self.delta())

    def test_revert_to_base_content_leaves_no_delta_and_no_entry(self):
        self.start()
        self.commit_files({"notes/f.txt": "v1\n"})
        self.commit_files({"notes/f.txt": "v0\n"})
        self.retry()
        self.assertEqual(self.base()["rebased"], [])
        self.assertEqual(self.delta(), ["src/result.txt"])

    def test_rename_and_delete_by_external_commits(self):
        self.start()
        self.commit_files({"notes/old.txt": None, "notes/new.txt": "old\n"}, "rename")
        self.commit_files({"notes/gone.txt": None}, "delete")
        self.retry()
        actions = {(r["path"], r["action"]) for r in self.base()["rebased"]}
        self.assertEqual(actions, {("notes/old.txt", "drop"), ("notes/new.txt", "overlay"),
                                   ("notes/gone.txt", "drop")})
        self.assertEqual(self.delta(), ["src/result.txt"])

    def test_dirty_untracked_and_hidden_out_of_scope_files_stay_in_the_delta(self):
        self.start()
        self.commit_files({"notes/d.txt": "committed\n", "notes/s.txt": "committed\n"})
        self.put("notes/d.txt", "edited after commit\n")
        self.put("notes/s.txt", "hidden edit\n")
        self.git("update-index", "--assume-unchanged", "notes/s.txt")
        self.put("notes/untracked.txt", "new\n")
        self.retry()
        self.assertEqual(self.base()["rebased"], [])
        self.assertEqual(self.delta(), ["notes/d.txt", "notes/s.txt", "notes/untracked.txt", "src/result.txt"])

    def test_protected_paths_are_never_rebased(self):
        self.start()
        self.commit_files({"tools/helper.py": "print('weakened')\n"}, "tool")
        self.commit_files({"AGENTS.md": "changed\n"}, "agents")
        self.commit_files({"orchestration/prompts/reviewer.md": "lenient\n"}, "prompt")
        gates = read(self.root / "orchestration/gates.json")
        gates["checks"]["check"]["argv"] = ["{python}", "-c", "pass"]
        write(self.root / "orchestration/gates.json", gates)
        self.git("add", "-A", "--", "orchestration/gates.json")
        self.git("commit", "-q", "-m", "gate")
        self.retry()
        self.assertEqual(self.base()["rebased"], [])
        self.assertEqual(self.delta(), ["AGENTS.md", "orchestration/gates.json",
                                        "orchestration/prompts/reviewer.md", "src/result.txt",
                                        "tools/helper.py"])

    def test_tasks_json_is_rebased_only_when_the_tasks_own_entry_is_unchanged(self):
        self.start()
        other = {"id": "T9", "title": "Other", "objective": "o", "status": "planned", "depends_on": [],
                 "read": [], "write_globs": ["other/*"], "acceptance": {"AC1": "x"},
                 "required_artifacts": ["other/a.txt"], "checks": ["check"], "budget": "b"}
        write(self.root / "orchestration/tasks.json", {"tasks": self.tasks + [other]})
        self.git("add", "-A", "--", "orchestration/tasks.json")
        self.git("commit", "-q", "-m", "add unrelated task")
        self.retry()
        self.assertEqual([r["path"] for r in self.base()["rebased"]], ["orchestration/tasks.json"])
        self.assertEqual(self.delta(), ["src/result.txt"])

    def test_own_task_entry_edit_is_not_rebased(self):
        self.flow.prepare("T1")
        self.flow.abandon("T1", "closed for test")
        edited = [dict(self.tasks[0], budget="loosened")]
        write(self.root / "orchestration/tasks.json", {"tasks": edited})
        self.git("add", "-A", "--", "orchestration/tasks.json")
        self.git("commit", "-q", "-m", "edit own contract")
        self.retry()
        self.assertEqual(self.base()["rebased"], [])
        self.assertIn("orchestration/tasks.json", self.delta())

    def test_rewritten_history_fails_closed(self):
        self.start()
        self.git("checkout", "-q", "--orphan", "rewritten")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "rewritten history")
        with self.assertRaisesRegex(GateError, "not an ancestor"):
            self.retry()
        self.assertFalse((self.root / "orchestration/runs/T1/attempt-02").exists())   # no half-made attempt

    def test_merge_commit_in_range_fails_closed(self):
        self.start()
        self.git("switch", "-q", "-c", "side")
        self.commit_files({"notes/side.txt": "s\n"})
        self.git("switch", "-q", self.branch)
        self.commit_files({"notes/main.txt": "m\n"})
        self.git("merge", "-q", "--no-ff", "-m", "merge", "side")
        with self.assertRaisesRegex(GateError, "Merge"):
            self.retry()

    def test_git_errors_fail_closed(self):
        self.start()
        (self.root / ".git/HEAD").write_text("garbage\n")
        with self.assertRaisesRegex(GateError, "Git failed"):
            self.retry()

    def test_missing_git_binary_fails_closed_not_open(self):
        self.start()
        import unittest.mock as mock
        with mock.patch("tools.orchestrate.subprocess.run", side_effect=FileNotFoundError("git")):
            with self.assertRaisesRegex(GateError, "Git failed"):
                self.retry()

    def test_without_a_git_directory_behaviour_is_unchanged(self):
        self.start()
        shutil.rmtree(self.root / ".git")
        self.put("notes/ext.txt", "external\n")
        self.retry()
        self.assertNotIn("rebased", self.base())
        self.assertEqual(self.delta(), ["notes/ext.txt", "src/result.txt"])
        with self.assertRaisesRegex(GateError, "needs a Git checkout"):
            self.flow.abandon("T1", "again")
            self.flow.prepare("T1", retry=True, base_commit=self.base_head)

    def test_more_than_the_cap_of_rebased_files_is_refused(self):
        self.start()
        self.commit_files({f"notes/bulk{i}.txt": "x\n" for i in range(51)}, "bulk")
        with self.assertRaisesRegex(GateError, "limit"):
            self.retry()

    def test_legacy_attempt_needs_a_verified_explicit_base_commit(self):
        self.start()
        path = self.flow.attempt("T1") / "base.json"
        legacy = read(path)
        del legacy["base_commit"]
        write(path, legacy)
        self.commit_files({"notes/f.txt": "v1\n"})
        with self.assertRaisesRegex(GateError, "predates TOOL-001"):
            self.retry()
        with self.assertRaisesRegex(GateError, "does not match the recorded baseline"):
            self.retry(base_commit=self.git("rev-parse", "HEAD").strip())
        self.retry(base_commit=self.base_head)
        self.assertEqual(self.base()["base_commit"], self.base_head)
        self.assertEqual([r["path"] for r in self.base()["rebased"]], ["notes/f.txt"])

    def test_explicit_base_commit_cannot_override_a_recorded_one(self):
        self.start()
        self.commit_files({"notes/ext.txt": "external\n"})
        other = self.git("rev-parse", "HEAD").strip()
        with self.assertRaisesRegex(GateError, "already recorded"):
            self.retry(base_commit=other)

    def test_third_attempt_recomputes_from_the_first_baseline(self):
        self.start()
        self.commit_files({"notes/a.txt": "a\n"})
        self.retry()
        self.flow.abandon("T1", "again")
        self.commit_files({"notes/b.txt": "b\n"})
        self.retry()
        self.assertEqual(sorted(r["path"] for r in self.base()["rebased"]), ["notes/a.txt", "notes/b.txt"])
        self.assertEqual(self.delta(), ["src/result.txt"])


if __name__ == "__main__":
    unittest.main()
