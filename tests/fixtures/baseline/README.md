# AERIS-007 test fixtures (SYNTHETIC)

`part1/`, `part2/` and `part3/groundtruth.csv` are SYNTHETIC files in the AOT ground-truth
format, written by hand for `tests/unit/test_baseline.py`. They contain no AOT data. They
cover both column orders (part2 puts `id` second, as the real part2 does), camera 0 and
camera 1 copies of one flight (part1 and part3), every object category, nominal 6 px and
16 px marker boxes (including one cut by the frame edge, and one whose right/bottom were
derived as left + 16 in floating point, as the real CSVs do), measured extents in several
width bins, empty frames, bird-only frames and five capture days.
