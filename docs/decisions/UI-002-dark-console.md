# UI-002: dark analysis console

Status: proposed for review, 2026-10-13.

The first UI was too light and its stacked preview, log and horizontal controls
were cut off in a compact window. Use a dark, high-contrast Qt palette with narrow
monospace labels, clear section numbers and a restrained green action color. The
visual language suggests an aerospace analysis instrument without claiming live
telemetry, measured accuracy or control capability.

Keep the landing page and workspace separate. Put source selection above a wide
video feed; place run actions in a vertical side panel so they remain visible at
900×600. Show run output when a run starts or a saved result loads. A scroll area
keeps text and output reachable at compact sizes. Preserve the existing CLI process,
sample provenance, own-video route, overlay playback and null-metric wording.

Visual captures at 1080×850 and 900×600 for both pages are stored in the ignored
`orchestration/runs/UI-002/attempt-01/visual/` evidence directory.
