# Operating AERIS with agents

The repository contains an assessed plan, an executable project foundation, and a
platform-independent orchestration kit. It does **not** contain a finished aircraft
detector. Agent sessions still need to be supplied by your chosen agent environment.

## Start one coordinator

Give an agent repository access and this prompt:

> Read AGENTS.md and orchestration/prompts/coordinator.md and act as the AERIS
> coordinator. You may delegate implementation and review to separate agents.
> Start with the earliest ready task, preserve one active writer per checkout,
> use the evidence gates, and continue through G1. Stop for concrete missing
> data/resource decisions or after three failed attempts. Do not activate deferred work.

Change `G1` to the milestone you want. The prompt authorizes delegation when actually
used; creating this scaffold did not start development agents or paid jobs.

## First local checks

Run from the repository root. Prefix with `rtk` when it is installed; the commands
below are the portable underlying commands.

```sh
python3 tools/orchestrate.py lint
python3 tools/check_tests.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
python3 tools/orchestrate.py status
```

The offline foundation requires Python >=3.11, CMake >=3.20 and a C++20 compiler.
For development, create a Python 3.11/3.12 environment and install
`pip install -c requirements/dev.txt -e '.[dev]'`, then run `ruff check .` and `pytest`.
The delivery's local `.venv` uses the available Python 3.14.6; production ML dependencies
should use the planned 3.11/3.12 environment and their compatibility checks.
ML/CV dependencies and locks are
introduced by their tasks. No GPU, model download or agent API key is needed here.

## Execute one task

```sh
python3 tools/orchestrate.py prepare AERIS-001
```

Give `orchestration/runs/AERIS-001/attempt-01/implementer.md` to an implementation
agent with repository access. It contains shared contracts, role instructions,
acceptance criteria, ownership and accepted dependency handoffs. The tool also
creates a structured report template and a source snapshot for review.

After the agent edits code and fills report.json:

```sh
python3 tools/orchestrate.py changes AERIS-001
python3 tools/orchestrate.py verify AERIS-001
```

Verification runs the approved checks and generates receipts/logs plus a reviewer
packet and review.json template. Give reviewer.md to a **separate** reviewer session.
The reviewer examines source and evidence and fills review.json. Then:

```sh
python3 tools/orchestrate.py evaluate AERIS-001
python3 tools/orchestrate.py accept AERIS-001
python3 tools/orchestrate.py status
```

Only acceptance unlocks dependent tasks. If checks or review fail, evaluate writes
feedback.md and closes the attempt. Retry with:

```sh
python3 tools/orchestrate.py prepare AERIS-001 --retry
```

An early error such as an incomplete report can be corrected before verification.
If handing it to a new attempt, first run `abandon AERIS-001 --reason 'specific failure'`.
Abandon closes evidence state; it does not revert files. Inspect/revert rejected code
before switching to a different task. Retries keep the original ownership baseline.

Do not run `evaluate` on an unfinished review: it deliberately counts as a rejected
attempt. Re-running `verify` invalidates any previous review and replaces its template.
Copy a completed review elsewhere if retaining it before re-verification matters.

## Context and evidence preservation

Commit accepted code, task contracts and ADRs. Keep `orchestration/runs/` in a backed-up
private artifact location: it is ignored by Git and contains source snapshots, task
reports, logs, reviews, feedback and state. If moving to another machine, transfer
that directory with the same source revision. Do not infer acceptance from Git alone.

Use `orchestration/prompts/recovery.md` after context loss. The full current contract
and each direct dependency's handoff are included in every new packet, up to a 160 KB
limit. Overlarge tasks must be split; content is never silently truncated.

For parallel development later, use separate worktrees, disjoint ownership and a
single coordinator-owned state ledger. This CLI enforces serial work in one checkout;
it is not a distributed scheduler or a merge service. After integration, re-run tests
and review on the combined source. Shared filesystem identity strings alone do not
prove independent review.

## What you still decide

Hardware/time budget, dataset permissions/access, project license, major scientific
scope changes, and optional paid resources. Routine implementation and correction
belong to the agents. No prompt can guarantee output quality; this system makes
missing evidence and stale results visible and blocks formal acceptance.
