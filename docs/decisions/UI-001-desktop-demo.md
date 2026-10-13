# UI-001: local Qt demonstration

Status: proposed for review, 2026-10-13.

Use Qt Widgets for a native, offline-first interface. A landing page explains AERIS
and the three-step demo. The workspace offers the seven owner-selected recorded
excerpts, accepts a user video, and shows the result. The recorded sources are fetched only when selected and verified by
the Wikimedia-published SHA-1; existing ignored local excerpts and overlays are
used when present. No videos are committed. A QThread prepares missing samples,
while QProcess runs the existing CLI without freezing the UI and allows
cancellation. Each run gets a new local output directory. OpenCV plays the silent
overlay frames. This keeps the UI to the Qt Essentials wheel. Metric availability
comes from the CLI's `metrics.json`, never from UI inference. The source CLI awaits
separate task acceptance; this UI makes no measured quality or release claim.
