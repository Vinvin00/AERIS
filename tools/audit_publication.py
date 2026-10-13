#!/usr/bin/env python3
"""Read-only publication safety audit of this repository's reachable Git state.

Scans every blob reachable from any ref, every commit's author/committer metadata and
the currently tracked tree for secret-shaped strings, private-artifact filenames and
oversized or binary media. Findings carry the path, short blob id and rule name only:
matched text is never printed or stored. Stdlib and the ``git`` binary only; nothing is
written, downloaded, rewritten or pushed. Heuristic regexes find common token shapes, not
all secrets; see docs/publication_safety.md for limits.

    python3 tools/audit_publication.py [--output FILE] [--strict]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_SCAN_BYTES = 5 << 20
LARGE_BYTES = 1 << 20
CONTENT_RULES = {
    "private-key-block": rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY",
    "aws-access-key-id": rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "github-token": rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b",
    "slack-token": rb"\bxox[abprs]-[A-Za-z0-9-]{10,}\b",
    "google-api-key": rb"\bAIza[0-9A-Za-z_-]{35}\b",
    "llm-api-key": rb"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{32,}\b",
    "jwt": rb"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
    "credential-assignment": (rb"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)"
                              rb"\b\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"),
    "basic-auth-url": rb"[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s:@]{3,}@[^/\s]+",
}
CONTENT_REGEX = {name: re.compile(pattern) for name, pattern in CONTENT_RULES.items()}
NAME_RULES = {
    "env-file": re.compile(r"(?:^|/)\.env(?:\.(?!example$)[^/]*)?$"),
    "key-or-certificate-file": re.compile(r"\.(?:pem|key|p12|pfx|jks|keystore|crt|cer)$|(?:^|/)id_(?:rsa|dsa|ecdsa|ed25519)$"),
    "session-or-credential-file": re.compile(r"\.ses$|(?:^|/)(?:credentials|\.netrc|\.npmrc|\.pypirc)$|secrets?\.(?:json|ya?ml|toml)$"),
    "model-weights": re.compile(r"\.(?:onnx|pt|pth|ckpt|safetensors|engine|trt|tflite|h5|pb|npz)$"),
    "media": re.compile(r"\.(?:mp4|mov|avi|mkv|webm|jpg|jpeg|png|tif|tiff|bmp|gif)$", re.I),
    "archive": re.compile(r"\.(?:zip|tar|tgz|gz|bz2|xz|7z|rar)$"),
    "local-database-or-dump": re.compile(r"\.(?:sqlite3?|db|sql|dump|pkl|pickle)$"),
}
# Fixtures that are intentionally tracked; listed so an addition is a reviewed decision.
EXPECTED_TRACKED_MEDIA: set[str] = set()


def git(*args: str, text: bool = True) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)
    if result.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout.decode("utf-8", "replace") if text else result.stdout


def reachable_objects() -> dict[str, set[str]]:
    """blob id -> set of paths it was reachable under, from all refs (branches, tags, remotes)."""
    paths: dict[str, set[str]] = defaultdict(set)
    for line in git("rev-list", "--objects", "--all").splitlines():
        sha, _, path = line.partition(" ")
        if path:
            paths[sha].add(path)
    return paths


def blob_info(shas: list[str]) -> dict[str, tuple[str, int]]:
    proc = subprocess.run(["git", "cat-file", "--batch-check"], cwd=ROOT, input="\n".join(shas).encode(),
                          capture_output=True, check=True)
    info = {}
    for line in proc.stdout.decode().splitlines():
        sha, kind, size = line.split()
        info[sha] = (kind, int(size))
    return info


def read_blobs(shas: list[str]):
    proc = subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    assert proc.stdin and proc.stdout
    for sha in shas:
        proc.stdin.write(sha.encode() + b"\n")
        proc.stdin.flush()
        header = proc.stdout.readline().split()
        size = int(header[2])
        data = proc.stdout.read(size)
        proc.stdout.read(1)
        yield sha, data
    proc.stdin.close()
    proc.wait()


def scan_blob(data: bytes) -> list[str]:
    if b"\x00" in data[:8192]:
        return []
    return sorted(name for name, rx in CONTENT_REGEX.items() if rx.search(data))


def audit() -> dict:
    head = git("rev-parse", "HEAD").strip()
    paths = reachable_objects()
    info = blob_info(sorted(paths))
    blobs = {sha: paths[sha] for sha, (kind, _) in info.items() if kind == "blob"}
    tracked = set(git("ls-files", "-z").split("\0")) - {""}
    findings: list[dict] = []

    def add(rule: str, kind: str, sha: str, names: set[str], size: int, current: bool) -> None:
        findings.append({"rule": rule, "kind": kind, "blob": sha[:12], "paths": sorted(names)[:5],
                         "path_count": len(names), "bytes": size, "in_current_tree": current})

    scannable = [s for s in sorted(blobs) if info[s][1] <= MAX_SCAN_BYTES]
    for sha, data in read_blobs(scannable):
        hits = scan_blob(data)
        for rule in hits:
            add(rule, "content", sha, blobs[sha], info[sha][1], bool(blobs[sha] & tracked))
    for sha, names in blobs.items():
        size = info[sha][1]
        current = bool(names & tracked)
        for rule, rx in NAME_RULES.items():
            matched = {n for n in names if rx.search(n)}
            if matched:
                add(rule, "filename", sha, matched, size, current)
        if size > LARGE_BYTES:
            add("large-blob", "size", sha, names, size, current)
    for name in sorted(tracked):
        for rule, rx in NAME_RULES.items():
            if rx.search(name) and name not in EXPECTED_TRACKED_MEDIA and not any(
                    f["rule"] == rule and name in f["paths"] for f in findings):
                findings.append({"rule": rule, "kind": "filename", "blob": None, "paths": [name],
                                 "path_count": 1, "bytes": None, "in_current_tree": True})

    commits = git("log", "--all", "--format=%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%s").splitlines()
    identities: dict[str, int] = defaultdict(int)
    trailers: dict[str, int] = defaultdict(int)
    for line in commits:
        _, an, ae, cn, ce, _subject = line.split("\x1f")
        identities[f"{an} <{ae}>"] += 1
        if (cn, ce) != (an, ae):
            identities[f"{cn} <{ce}> (committer)"] += 1
    for message in git("log", "--all", "--format=%B%x1e").split("\x1e"):
        for m in re.finditer(r"(?mi)^(co-authored-by|signed-off-by):\s*(.+?)\s*<[^>]*>\s*$", message):
            trailers[f"{m.group(1).lower()}: {m.group(2)}"] += 1

    findings.sort(key=lambda f: (f["rule"], f["paths"], f["blob"] or ""))
    by_rule: dict[str, int] = defaultdict(int)
    for f in findings:
        by_rule[f["rule"]] += 1
    return {
        "head": head,
        "refs": sorted(git("for-each-ref", "--format=%(refname)").split()),
        "commits_scanned": len(commits),
        "blobs_reachable": len(blobs),
        "blobs_content_scanned": len(scannable),
        "blobs_skipped_over_limit": len(blobs) - len(scannable),
        "tracked_files": len(tracked),
        "scan_limit_bytes": MAX_SCAN_BYTES,
        "content_rules": sorted(CONTENT_RULES),
        "findings_by_rule": dict(sorted(by_rule.items())),
        "findings": findings,
        "identities": dict(sorted(identities.items())),
        "trailers": dict(sorted(trailers.items())),
        "not_covered": [
            "reflog, stashes, dangling or unreachable objects, and other clones or forks",
            "GitHub-side data: issues, PR comments, Actions logs, releases, settings",
            "entropy-based or provider-specific secrets the regexes do not describe",
            "binary or over-limit blobs are checked by filename and size only, not content",
            "licensing and rights of tracked content: a human judgment, see docs/publication_safety.md",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="write the JSON report here (default: stdout)")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 if any content-rule finding exists (secret-shaped strings)")
    args = parser.parse_args(argv)
    report = audit()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    print(f"{report['commits_scanned']} commits, {report['blobs_reachable']} blobs, "
          f"{len(report['findings'])} findings {report['findings_by_rule']}", file=sys.stderr)
    return 1 if args.strict and any(f["kind"] == "content" for f in report["findings"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
