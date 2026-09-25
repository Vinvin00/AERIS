# Recovery / context reset prompt

Resume AERIS from disk. Do not assume any previous chat result was accepted.

Read AGENTS.md, docs/PROJECT_STATE.md, docs/CONTRACTS.md, then run `status`. Inspect
orchestration/runs/state.json and the latest active attempt. Read its task contract,
implementer.md, report.json, verification.json, review.json and feedback.md where present.
Compare current source hashes with the receipt. If code/report/logs changed, reverify
and obtain a fresh review. Never carry approval across an unverified change.

Summarize: last accepted task, active attempt, completed artifacts, current blocker,
decisions, and next concrete command. Resume the active role, preserving write scope.
If a task exhausted three attempts, use the escalation template; do not reset history.
If the run archive is missing, stop claiming continuity, recover it from backup or
reconstruct the task from source and perform new verification and review.
