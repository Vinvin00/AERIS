"""Offline, evidence-based task handoffs. This tool does not call an LLM or merge code."""

import argparse
import fnmatch
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unicodedata
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
                 "build", "dist", ".DS_Store"}
IGNORED_PREFIXES = ("orchestration/runs/", "data/raw/", "data/processed/",
                    "data/annotations/", "models/checkpoints/", "models/onnx/",
                    "models/tensorrt/", "experiments/runs/", "benchmarks/runs/")
DIMENSIONS = ("correctness", "tests", "reproducibility", "contracts", "scope")
# Never overlaid into a retry baseline: a weakened verifier or contract must not be committable-then-retried.
REBASE_DENYLIST = ("orchestration/gates.json", "tools/", "tests/test_orchestration.py",
                   "orchestration/prompts/", ".github/", "AGENTS.md", "docs/CONTRACTS.md")
# tools/ is on sys.path when the gates run, so a new tools/ file is overlaid only when the owner names it
# (prepare --retry --allow-new-tool) and it passes _check_new_tool. REBASE_NEVER is defensive: gate files
# exist at base, so the absent-at-base test already rejects them.
REBASE_NEVER = ("orchestrate.py", "check_tests.py")
NEVER_STEMS = ("sitecustomize", "usercustomize", "conftest", "__init__", "__main__")
REBASE_MAX_FILES = 50


class GateError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise GateError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    return digest(path.read_bytes())


def read(path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        GateError(f"Non-finite JSON value: {value}")))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temp = Path(handle.name)
    os.replace(temp, path)


def local(root, name):
    require(isinstance(name, str) and name.strip(), "Expected a repository-relative path")
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), f"Path escapes repository: {name}")
    return path


def inventory(root):
    result = {}
    for directory, dirs, files in os.walk(root):
        for name in dirs:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix() + "/"
            if name not in IGNORED_PARTS and not relative.startswith(IGNORED_PREFIXES):
                require(not path.is_symlink(), f"Symlink directory not permitted: {relative}")
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_PARTS
                         and not d.endswith(".egg-info") and not (Path(directory) / d).is_symlink()
                         and not ((Path(directory) / d).relative_to(root).as_posix() + "/")
                         .startswith(IGNORED_PREFIXES))
        for name in sorted(files):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if (name in IGNORED_PARTS or name.startswith(".env")
                    or relative.startswith(IGNORED_PREFIXES) or name.endswith((".pyc", ".pyo"))):
                continue
            require(not path.is_symlink(), f"Symlink not permitted in source inventory: {relative}")
            result[relative] = file_hash(path)
    return result


def fingerprint(files):
    return digest(json.dumps(files, sort_keys=True).encode())


def changed(before, after):
    return sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))


class Workflow:
    def __init__(self, root=ROOT):
        self.root = root.resolve()
        self.last_rebased = []
        self.runs = self.root / "orchestration/runs"
        self.tasks = read(self.root / "orchestration/tasks.json")["tasks"]
        self.gates = read(self.root / "orchestration/gates.json")["checks"]

    def state(self):
        path = self.runs / "state.json"
        return read(path) if path.exists() else {"accepted": {}}

    def task(self, task_id):
        match = [t for t in self.tasks if t["id"] == task_id]
        require(len(match) == 1, f"Unknown or duplicate task: {task_id}")
        return match[0]

    def lint(self):
        ids = [t["id"] for t in self.tasks]
        require(len(ids) == len(set(ids)), "Duplicate task IDs")
        for task in self.tasks:
            for key in ("title", "objective", "depends_on", "read", "write_globs",
                        "acceptance", "required_artifacts", "checks", "status", "budget"):
                require(key in task, f'{task["id"]}: missing {key}')
            require(task["status"] in ("planned", "deferred"), "Invalid task status")
            require(task["acceptance"] and task["checks"] and task["write_globs"],
                    "Task must have criteria, checks and ownership")
            require(len(task["acceptance"]) == len(set(task["acceptance"])), "Duplicate criteria")
            for dep in task["depends_on"]:
                require(dep in ids and dep != task["id"], f"Invalid dependency: {dep}")
            for name in task["read"]:
                require(local(self.root, name).is_file(), f"Missing context file: {name}")
            for name in task["required_artifacts"]:
                local(self.root, name)
                require(any(fnmatch.fnmatchcase(name, p) for p in task["write_globs"]),
                        f"Required artifact outside task scope: {name}")
            for check in task["checks"]:
                require(check in self.gates, f"Unknown check: {check}")
        visiting, visited = set(), set()

        def visit(task_id):
            require(task_id not in visiting, f"Dependency cycle: {task_id}")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dep in self.task(task_id)["depends_on"]:
                visit(dep)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in ids:
            visit(task_id)
        for name, gate in self.gates.items():
            require(isinstance(gate["argv"], list) and gate["argv"]
                    and all(isinstance(a, str) for a in gate["argv"]), f"Invalid argv: {name}")
            require(type(gate["timeout_s"]) is int and 0 < gate["timeout_s"] <= 3600,
                    f"Invalid timeout: {name}")
        return {"tasks": len(ids), "checks": len(self.gates), "valid": True}

    def status(self):
        accepted = self.state()["accepted"]
        return [{"id": t["id"], "title": t["title"], "status":
                 "accepted" if t["id"] in accepted else "deferred" if t["status"] == "deferred"
                 else "ready" if all(d in accepted for d in t["depends_on"]) else "waiting"}
                for t in self.tasks]

    def attempt(self, task_id):
        self.task(task_id)
        folders = sorted((self.runs / task_id).glob("attempt-*"))
        require(folders, f"Prepare {task_id} first")
        return folders[-1]

    # -- retry baselines over committed external changes (TOOL-001) ---------------------
    def _git(self, *args, text=True):
        """Read-only Git. Any failure raises: a retry baseline is never guessed."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        try:
            done = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True,
                                  text=text, timeout=60, env=env)
        except (OSError, subprocess.SubprocessError) as exc:
            raise GateError(f"Git failed ({' '.join(args[:2])}): {exc}") from exc
        require(done.returncode == 0, f"Git failed ({' '.join(args[:3])}): "
                f"{(done.stderr if text else done.stderr.decode(errors='replace')).strip()[:200]}")
        return done.stdout

    def _has_git(self):
        return (self.root / ".git").exists()

    def _head(self):
        out = self._git("rev-parse", "--verify", "HEAD").strip()
        require(len(out) == 40, "Unexpected HEAD value from Git")
        return out

    def _commit_files(self, commit, merge=False):
        """Files a commit changes; for a merge, those differing from its first parent. (Plain diff-tree
        prints nothing for a merge, `-m` diffs against every parent, and diff-tree has no --first-parent.)"""
        out = self._git("diff-tree", "--no-commit-id", "--name-only", "-r", "-z", "--no-renames",
                        *([f"{commit}^1", commit] if merge else ["--root", commit]), text=False)
        return [name.decode() for name in out.split(b"\0") if name]

    def _verify_explicit_base(self, commit, original):
        """A legacy attempt has no recorded base_commit; the given one must reproduce its snapshot."""
        for name, expected in original.items():
            probe = subprocess.run(["git", "-C", str(self.root), "cat-file", "-e", f"{commit}:{name}"],
                                   capture_output=True)
            if probe.returncode != 0:
                continue                      # untracked when the baseline was taken
            require(digest(self._git("show", f"{commit}:{name}", text=False)) == expected,
                    f"--base-commit does not match the recorded baseline ({name})")

    def rebase_candidates(self, task, original, base_commit, explicit, current, allow_new_tool=()):
        """Files to overlay onto the original baseline, as ``({path: sha256|None}, audit list)``."""
        head = self._head()
        commit = self._git("rev-parse", "--verify", f"{base_commit}^{{commit}}").strip()
        ancestor = subprocess.run(["git", "-C", str(self.root), "merge-base", "--is-ancestor", commit, head],
                                  capture_output=True)
        require(ancestor.returncode == 0, "base_commit is not an ancestor of HEAD "
                "(history was rewritten); refusing to rebase the retry baseline")
        if explicit:
            self._verify_explicit_base(commit, original)
        allowed = {self._check_new_tool(a, commit, head, original): a for a in allow_new_tool}
        commits = [c for c in self._git("rev-list", "--reverse", f"{commit}..{head}").split() if c]
        touched, tainted_files = {}, set()
        for c in commits:
            parents = self._git("rev-list", "--parents", "-n", "1", c).split()[1:]
            require(len(parents) in (1, 2),
                    f"Root commit or merge with {len(parents)} parents ({c[:10]}) in the retry range; "
                    "refusing to rebase")
            files = self._commit_files(c, len(parents) == 2)
            if any(fnmatch.fnmatchcase(f, pattern) for f in files for pattern in task["write_globs"]):
                tainted_files.update(files)   # task work (or a mixed commit): trust none of its files
            else:
                for f in files:
                    touched.setdefault(f, []).append(c[:10])
        overlay, audit = {}, []
        for path in sorted(touched):
            if path in tainted_files or (path.startswith(REBASE_DENYLIST) and path not in allowed.values()):
                continue
            if any(fnmatch.fnmatchcase(path, pattern) for pattern in task["write_globs"]):
                continue
            if path == "orchestration/tasks.json" and not self._own_task_unchanged(task, commit):
                continue
            probe = subprocess.run(["git", "-C", str(self.root), "cat-file", "-e", f"{head}:{path}"],
                                   capture_output=True)
            target = local(self.root, path)
            if probe.returncode != 0:                    # absent in HEAD: a trusted commit deleted it
                if target.exists() or path not in original:
                    continue
                overlay[path] = None
                audit.append({"path": path, "sha256": None, "commits": touched[path], "action": "drop"})
                continue
            if path in allowed.values() and not self._git("ls-tree", head, "--", path).startswith(("100644 blob", "100755 blob")):
                continue                                 # symlink or gitlink
            committed = self._git("show", f"{head}:{path}", text=False)
            if not target.is_file() or target.is_symlink() or target.read_bytes() != committed:
                continue                                 # dirty, untracked or hidden by a Git flag
            new = digest(committed)
            if path in current and original.get(path) != new:
                overlay[path] = new
                entry = {"path": path, "sha256": new, "commits": touched[path], "action": "overlay"}
                if path in allowed.values():
                    entry.update(size=len(committed), first_lines=[
                        line if len(line) <= 200 else line[:200] + "...[truncated]"
                        for line in committed.decode(errors="replace").splitlines()[:3]])
                audit.append(entry)
        for name in allow_new_tool:
            if name not in overlay:
                print(f"WARNING: --allow-new-tool {name} matched no overlay candidate (not committed, dirty, tainted, "
                      "not a regular file or already in the baseline); nothing was overlaid for it", file=sys.stderr)
        require(len(audit) <= REBASE_MAX_FILES,
                f"{len(audit)} files would be rebased (limit {REBASE_MAX_FILES}); refusing")
        return commit, overlay, audit

    def _check_new_tool(self, path, base, head, original):
        """Validate an owner-named tools/ path; returns it. Raises on anything that could shadow an import."""
        norm = lambda text: unicodedata.normalize("NFC", text).casefold()
        parts = path.split("/")
        require(len(parts) == 2 and parts[0] == "tools" and parts[1].endswith(".py"),
                f"--allow-new-tool needs a direct child tools/<name>.py, got {path!r}")
        name, stem = parts[1], parts[1][:-3]
        require(norm(name) not in REBASE_NEVER and norm(stem) not in {m.casefold() for m in sys.stdlib_module_names}
                and norm(stem) not in NEVER_STEMS and stem.isidentifier(),
                f"--allow-new-tool refuses {path}: gate, stdlib or special module name")
        search = [p for p in sys.path if Path(p or ".").resolve() not in (ROOT / "tools", self.root / "tools")]
        require(importlib.machinery.PathFinder.find_spec(stem, search) is None,
                f"--allow-new-tool refuses {path}: {stem} is an importable module")
        def tree(commit):
            out = self._git("ls-tree", "-r", "--name-only", "-z", commit, "--", "tools/", text=False)
            return {norm(n.decode()) for n in out.split(b"\0") if n}
        want = norm(path)
        at_base = {norm(k) for k in original} | tree(base)
        # Absent from base, and nothing with the same stem (file, package dir, extension module) in base or HEAD.
        # The base-tree entry for the path itself is also caught here (first == "<name>" starts with "<stem>.").
        for entry in at_base | (tree(head) - {want}):
            first = entry.split("/")[1] if entry.count("/") else ""
            require(first != norm(stem) and not first.startswith(norm(stem) + "."),
                    f"--allow-new-tool refuses {path}: collides with {entry}")
        return path

    def _own_task_unchanged(self, task, commit):
        try:
            before = json.loads(self._git("show", f"{commit}:orchestration/tasks.json"))["tasks"]
        except (GateError, ValueError, KeyError):
            return False
        return any(t.get("id") == task["id"] and t == task for t in before)

    def prepare(self, task_id, retry=False, base_commit=None, allow_new_tool=()):
        self.lint()
        task = self.task(task_id)
        accepted = self.state()["accepted"]
        require(task["status"] != "deferred", "Deferred task requires a separately reviewed activation")
        require(task_id not in accepted, "Task already accepted; create a follow-up task")
        require(all(dep in accepted for dep in task["depends_on"]), "Dependencies not accepted")
        # The shared checkout permits one active writer; no unattended concurrent mutation.
        for other in self.runs.glob("*/attempt-*/base.json"):
            meta = read(other)
            if meta["task_id"] != task_id and meta["task_id"] not in accepted:
                require((other.parent / "closed.json").exists(),
                        f'Another task owns this checkout: {meta["task_id"]}')
        attempts = sorted((self.runs / task_id).glob("attempt-*"))
        require(not allow_new_tool or (retry and attempts and self._has_git()),
                "--allow-new-tool is only valid with --retry on a Git checkout after a first attempt")
        if attempts and not retry:
            require(not (attempts[-1] / "closed.json").exists(), "Closed attempt; use --retry")
            require((attempts[-1] / "implementer.md").exists(),
                    f"{attempts[-1].name} has no implementer.md (an earlier prepare failed); remove the folder or abandon it")
            return str(attempts[-1] / "implementer.md")
        require(len(attempts) < 3, "Three attempts exhausted. Escalate with evidence; create a revised task.")
        if retry:
            require(attempts, "No previous attempt to retry")
            require((attempts[-1] / "closed.json").exists(), "Close attempt with evaluate or abandon first")
        for dep in task["depends_on"]:
            entry = accepted[dep]
            dep_folder = local(self.root, entry["evidence_dir"])
            require(file_hash(dep_folder / "report.json") == entry["report_sha256"]
                    and file_hash(dep_folder / "review.json") == entry["review_sha256"],
                    f"Accepted dependency evidence changed: {dep}")
        folder = self.runs / task_id / f"attempt-{len(attempts) + 1:02d}"
        files = inventory(self.root)
        # Preserve the original ownership baseline across retries.
        first = read(attempts[0] / "base.json") if attempts else None
        original = first["files"] if attempts else files
        base_files, rebased, recorded = dict(original), [], (first or {}).get("base_commit")
        if not attempts and self._has_git():
            recorded = self._head()
        if attempts and self._has_git():
            require(not (recorded and base_commit and base_commit != recorded),
                    "base_commit is already recorded for this task; do not pass --base-commit")
            recorded = recorded or base_commit
            require(recorded, "This attempt has no recorded base_commit (it predates TOOL-001); "
                    "retry with --base-commit <commit the first attempt was prepared on>")
            recorded, overlay, rebased = self.rebase_candidates(task, original, recorded,
                                                                not (first or {}).get("base_commit"), files,
                                                                allow_new_tool)
            for name, new_hash in overlay.items():
                if new_hash is None:
                    base_files.pop(name, None)
                else:
                    base_files[name] = new_hash
        else:
            require(not base_commit, "--base-commit needs a Git checkout and an earlier attempt")
        base = {"task_id": task_id, "attempt": len(attempts) + 1, "created_at": now(),
                "files": base_files, "packet_fingerprint": fingerprint(files)}
        if recorded:
            base["base_commit"] = recorded
        if not attempts and self._has_git():
            base["tree_dirty_at_base"] = bool(self._git("status", "--porcelain").strip())
        if attempts and self._has_git():
            base["rebased"] = rebased
            base["allowed_new_tools"] = sorted(allow_new_tool)
            base["head_at_retry"] = self._head()
        self.last_rebased = rebased
        sections = [f'# Implementation packet: {task_id}\n\n',
                    f'Packet fingerprint: `{base["packet_fingerprint"]}`. Attempt {base["attempt"]}/3.\n',
                    "Read the actual repository files before editing; this packet is a snapshot.\n",
                    "## Task contract\n```json\n" + json.dumps(task, indent=2) + "\n```\n"]
        context = list(dict.fromkeys(["AGENTS.md", "docs/PROJECT_STATE.md", "docs/CONTRACTS.md",
                                     "orchestration/prompts/implementer.md"] + task["read"]))
        omitted = []
        for path in sorted((self.root / "docs/decisions").glob("*.md")):
            rel = path.relative_to(self.root).as_posix()
            if rel in context:
                continue
            if (path.name == "0001-scope.md" or any(fnmatch.fnmatchcase(rel, g) for g in task["write_globs"])
                    or any(path.name.startswith(dep) for dep in task["depends_on"])):
                context.append(rel)
            else:
                omitted.append(rel)
        for name in context:
            content = local(self.root, name).read_text()
            sections.append(f"\n## Context: {name}\n\n{content}\n")
        if omitted:
            sections.append("\n## Decision records not included\n"
                            "Not named in `read`, not in this task's `write_globs`, not a direct dependency's and not "
                            "0001-scope.md; read them from the repository if needed:\n"
                            + "".join(f"- `{name}`\n" for name in omitted))
        for dep in task["depends_on"]:
            entry = accepted[dep]
            sections.append(f"\n## Accepted dependency {dep}\n" + json.dumps(entry, indent=2) + "\n")
            dep_folder = local(self.root, entry["evidence_dir"])
            sections.append((dep_folder / "report.json").read_text())
        if rebased:
            sections.append("\n## Baseline rebased over external commits (audit list)\n"
                            "These files are outside this task's write scope, clean in Git, and were changed only by "
                            "commits touching no write_globs file since the first attempt. They are NOT this task's "
                            "changes. The reviewer must check them:\n"
                            + "".join(self._audit_line(r, True) for r in rebased))
        if attempts:
            for name in ("feedback.md", "report.json"):
                path = attempts[-1] / name
                if path.exists():
                    sections.append(f"\n## Previous attempt: {name}\n" + path.read_text())
        sections.append(f"\nWrite your report to `{folder.relative_to(self.root)}/report.json`.\n")
        packet = "\n".join(sections)
        require(len(packet.encode()) <= 160_000, "Context exceeds 160 KB: split task/context before dispatch")
        folder.mkdir(parents=True)
        write(folder / "base.json", base)
        if not attempts:
            for name in base_files:
                target = folder / "source-before" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((self.root / name).read_bytes())
        report = {"task_id": task_id, "attempt": base["attempt"], "author_id": "",
                  "summary": "", "changed_files": [],
                  "criteria": {key: {"artifact": "", "explanation": ""} for key in task["acceptance"]},
                  "decisions": [], "limitations": [], "unresolved": [],
                  "handoff": {"next_action": "", "context": ""}}
        write(folder / "report.json", report)
        (folder / "implementer.md").write_text(packet)
        return str(folder / "implementer.md")

    def validate_report(self, task, folder, files):
        report = read(folder / "report.json")
        base = read(folder / "base.json")
        require(report["task_id"] == task["id"] and report["attempt"] == base["attempt"],
                "Report identity/attempt mismatch")
        require(isinstance(report["author_id"], str) and report["author_id"].strip(), "Missing author ID")
        require(isinstance(report["summary"], str) and report["summary"].strip(), "Missing summary")
        delta = changed(base["files"], files)
        require(delta, "No implementation changes")
        require(sorted(report["changed_files"]) == delta, "changed_files must exactly match source changes")
        for path in delta:
            require(any(fnmatch.fnmatchcase(path, pattern) for pattern in task["write_globs"]),
                    f"Out-of-scope change: {path}")
        for name in task["required_artifacts"]:
            require(name in files, f"Missing required artifact: {name}")
        require(set(report["criteria"]) == set(task["acceptance"]), "Criteria must match task exactly")
        for key, evidence in report["criteria"].items():
            path = local(self.root, evidence["artifact"])
            require(path.is_file() and evidence["artifact"] in files, f"Missing evidence for {key}")
            require(isinstance(evidence["explanation"], str) and evidence["explanation"].strip(),
                    f"No explanation for {key}")
        for key in ("decisions", "limitations", "unresolved"):
            require(isinstance(report[key], list) and all(isinstance(s, str) for s in report[key]),
                    f"Invalid {key}")
        require(not report["unresolved"], "Unresolved implementation issues must be addressed or escalated")
        require(all(isinstance(report["handoff"].get(k), str) and report["handoff"][k].strip()
                    for k in ("next_action", "context")), "Incomplete handoff")
        return report

    def verify(self, task_id):
        task, folder = self.task(task_id), self.attempt(task_id)
        require(not (folder / "closed.json").exists(), "Attempt is closed")
        # Invalidate previous receipts before doing any work.
        for name in ("verification.json", "decision.json"):
            (folder / name).unlink(missing_ok=True)
        files = inventory(self.root)
        self.validate_report(task, folder, files)
        report_hash = file_hash(folder / "report.json")
        receipts = []
        for check_id in task["checks"]:
            check = self.gates[check_id]
            argv = [sys.executable if arg == "{python}" else arg for arg in check["argv"]]
            try:
                process = subprocess.run(argv, cwd=self.root, capture_output=True,
                                         timeout=check["timeout_s"], check=False)
                output = process.stdout + process.stderr
                code = process.returncode
            except subprocess.TimeoutExpired as exc:
                output = (exc.stdout or b"") + (exc.stderr or b"") + b"\nCHECK TIMED OUT\n"
                code = 124
            except OSError as exc:
                output, code = str(exc).encode(), 127
            log = folder / f"check-{check_id}.log"
            log.write_bytes(output)
            receipts.append({"id": check_id, "argv": argv, "exit_code": code,
                             "log": log.relative_to(self.root).as_posix(), "sha256": file_hash(log)})
        require(files == inventory(self.root), "Checks changed source files; inspect changes and verify again")
        require(report_hash == file_hash(folder / "report.json"), "Report changed during verification")
        receipt = {"task_id": task_id, "created_at": now(), "fingerprint": fingerprint(files),
                   "report_sha256": report_hash, "checks": receipts,
                   "passed": all(item["exit_code"] == 0 for item in receipts)}
        write(folder / "verification.json", receipt)
        review = {"task_id": task_id, "attempt": read(folder / "base.json")["attempt"],
                  "reviewer_id": "", "author_id": read(folder / "report.json")["author_id"],
                  "fingerprint": receipt["fingerprint"], "report_sha256": report_hash,
                  "verification_sha256": file_hash(folder / "verification.json"),
                  "verdict": "revise", "scores": {key: 0 for key in DIMENSIONS},
                  "criteria": {key: {"passed": False, "reason": ""} for key in task["acceptance"]},
                  "findings": [], "summary": ""}
        # Re-verification invalidates the review. Never silently reuse a previous approval.
        write(folder / "review.json", review)
        source = (self.root / "orchestration/prompts/reviewer.md").read_text()
        (folder / "reviewer.md").write_text(source + "\n\n## Review inputs\n\n"
            + f"Task: {task_id}\nEvidence directory: {folder.relative_to(self.root)}\n"
            + "Read implementer.md, base.json, report.json, verification.json, every check log,\n"
            + "and the actual changed source files. Use Git diff where available. Fill review.json.\n"
            + f"Original source snapshot: orchestration/runs/{task_id}/attempt-01/source-before/\n"
            + "Compare that snapshot with current files, including deleted files. Git diff is also useful.\n"
            + self._rebased_note(folder))
        return receipt

    @staticmethod
    def _audit_line(r, with_hash):
        line = f"- {r['action']} `{r['path']}` " + (f"sha256 {r['sha256']} " if with_hash else "") \
            + f"commits {', '.join(r['commits'])}"
        if "size" in r:        # owner-allowed new tools/ file: show what it contains
            line += f" [ALLOWED NEW TOOL, {r['size']} bytes, first lines: {json.dumps(r['first_lines'])}]"
        return line + "\n"

    def _rebased_note(self, folder):
        rebased = read(folder / "base.json").get("rebased") or []
        if not rebased:
            return ""
        return ("\n## Baseline rebased over external commits (audit these)\n"
                + "".join(self._audit_line(r, False) for r in rebased)
                + "Confirm none of these is task work or a weakened verifier; they were excluded from the delta.\n")

    def evaluate(self, task_id):
        task, folder = self.task(task_id), self.attempt(task_id)
        require(not (folder / "closed.json").exists(), "Attempt is closed")
        problems = []
        try:
            files = inventory(self.root)
            report = self.validate_report(task, folder, files)
            receipt = read(folder / "verification.json")
            require(receipt["passed"] is True, "Required checks failed")
            require(receipt["fingerprint"] == fingerprint(files), "Source changed after verification")
            require(receipt["report_sha256"] == file_hash(folder / "report.json"), "Report changed after verification")
            require([item["id"] for item in receipt["checks"]] == task["checks"], "Incomplete check receipt")
            for item in receipt["checks"]:
                expected = [sys.executable if a == "{python}" else a for a in self.gates[item["id"]]["argv"]]
                require(item["argv"] == expected and item["exit_code"] == 0, "Invalid check execution")
                require(file_hash(local(self.root, item["log"])) == item["sha256"], "Check log changed")
            review = read(folder / "review.json")
            require(review["task_id"] == task_id and review["attempt"] == report["attempt"], "Review identity mismatch")
            require(isinstance(review["reviewer_id"], str) and review["reviewer_id"].strip()
                    and review["reviewer_id"].strip() != report["author_id"].strip(), "Independent reviewer required")
            require(review["author_id"] == report["author_id"], "Review author mismatch")
            require(review["fingerprint"] == receipt["fingerprint"]
                    and review["report_sha256"] == receipt["report_sha256"]
                    and review["verification_sha256"] == file_hash(folder / "verification.json"), "Stale review")
            require(review["verdict"] == "accept", "Reviewer requested revision or escalation")
            require(isinstance(review["summary"], str) and review["summary"].strip(), "Missing review reasoning")
            require(set(review["scores"]) == set(DIMENSIONS), "Incomplete rubric")
            for key, score in review["scores"].items():
                require(type(score) is int and 3 <= score <= 4, f"Rubric dimension below 3/4: {key}")
            require(set(review["criteria"]) == set(task["acceptance"]), "Incomplete criterion review")
            for key, value in review["criteria"].items():
                require(value["passed"] is True and isinstance(value["reason"], str)
                        and value["reason"].strip(), f"Criterion not supported: {key}")
            require(isinstance(review["findings"], list), "Invalid findings")
            for finding in review["findings"]:
                require(finding["severity"] in ("critical", "major", "minor")
                        and type(finding["resolved"]) is bool
                        and isinstance(finding["detail"], str) and finding["detail"].strip(), "Malformed finding")
                require(finding["resolved"] or finding["severity"] == "minor", "Unresolved blocking finding")
        except (GateError, KeyError, TypeError, OSError, ValueError) as exc:
            problems.append(str(exc))
        decision = {"task_id": task_id, "created_at": now(), "accepted": not problems,
                    "problems": problems, "fingerprint": fingerprint(inventory(self.root))}
        write(folder / "decision.json", decision)
        if problems:
            feedback = "# Revision required\n\n" + "\n".join(f"- {p}" for p in problems)
            if (folder / "review.json").exists():
                feedback += "\n\n## Reviewer evidence\n\n" + (folder / "review.json").read_text()
            feedback += "\n\nFix the cause without weakening gates. Prepare --retry for a new attempt.\n"
            (folder / "feedback.md").write_text(feedback)
            write(folder / "closed.json", {"reason": "revision_required", "created_at": now()})
        return decision

    def accept(self, task_id):
        require(task_id not in self.state()["accepted"], "Task already accepted")
        decision = self.evaluate(task_id)
        require(decision["accepted"], "Acceptance denied; read feedback.md")
        folder = self.attempt(task_id)
        state = self.state()
        state["accepted"][task_id] = {"accepted_at": now(), "fingerprint": decision["fingerprint"],
                                      "evidence_dir": folder.relative_to(self.root).as_posix(),
                                      "report_sha256": file_hash(folder / "report.json"),
                                      "review_sha256": file_hash(folder / "review.json")}
        write(self.runs / "state.json", state)
        write(folder / "closed.json", {"reason": "accepted", "created_at": now()})
        return state["accepted"][task_id]

    def abandon(self, task_id, reason):
        require(task_id not in self.state()["accepted"], "Cannot abandon accepted task")
        folder = self.attempt(task_id)
        require(reason.strip(), "Supply an escalation reason")
        write(folder / "closed.json", {"reason": reason, "created_at": now()})
        (folder / "feedback.md").write_text(f"# Task paused/escalated\n\n{reason}\n")
        return {"closed": str(folder), "reason": reason}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("lint", "status"):
        sub.add_parser(command)
    for command in ("prepare", "verify", "evaluate", "accept", "abandon", "changes"):
        child = sub.add_parser(command)
        child.add_argument("task")
        if command == "prepare":
            child.add_argument("--retry", action="store_true")
            child.add_argument("--allow-new-tool", action="append", default=[], metavar="tools/NAME.py",
                               help="retry only: also overlay this new direct-child tools/ file (repeatable)")
            child.add_argument("--base-commit", default=None,
                               help="commit the first attempt was prepared on (legacy attempts only)")
        if command == "abandon":
            child.add_argument("--reason", required=True)
    args = parser.parse_args()
    try:
        flow = Workflow()
        if args.command == "prepare":
            result = flow.prepare(args.task, args.retry, args.base_commit, args.allow_new_tool)
            for item in getattr(flow, "last_rebased", []):
                print(f"REBASED {item['action']} {item['path']} commits {','.join(item['commits'])}"
                      + (f" ALLOWED-NEW-TOOL size {item['size']} first lines {item['first_lines']}" if "size" in item else ""),
                      file=sys.stderr)
        elif args.command == "abandon":
            result = flow.abandon(args.task, args.reason)
        elif args.command == "changes":
            result = changed(read(flow.attempt(args.task) / "base.json")["files"], inventory(ROOT))
        elif args.command in ("lint", "status"):
            result = getattr(flow, args.command)()
        else:
            result = getattr(flow, args.command)(args.task)
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.command in ("verify", "evaluate"):
            return 0 if result.get("passed", result.get("accepted", False)) else 1
        return 0
    except (GateError, KeyError, TypeError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
