"""Extra clips for the hosted, pre-recorded demo only (see tools/build_static_demo.py).

Kept apart from aeris.ui.samples so the local console and Qt app keep their seven-clip
catalog. Same rights rules apply: source is fetched from Wikimedia Commons and its
published SHA-1 must match before an excerpt is made.
"""
from __future__ import annotations

from aeris.ui.samples import BY_SLUG as CATALOG, Sample

DEMO_CLIPS = (
    Sample("tokyo-small", "Small airliner · Tokyo", "Small airliner crossing cloud, 4K source",
           "https://commons.wikimedia.org/wiki/File:Airplane_with_one_landing_light_-_filmed_in_tokyo_-_2025_10_5.webm",
           "https://upload.wikimedia.org/wikipedia/commons/6/63/Airplane_with_one_landing_light_-_filmed_in_tokyo_-_2025_10_5.webm",
           "Nesnad", "CC BY 4.0", "92c5d29b1aa10adafd6d9e9c2e9fbb67d2e9e964",
           "tokyo-small.webm", 0, 20, "source/tokyo-small-20s-1920x1080.mp4",
           "tokyo-small-run/overlay.mp4", "", 1920, 1080),
    Sample("b52-dubai", "B-52 · Dubai Airshow", "Bomber as a speck above a crowd",
           "https://commons.wikimedia.org/wiki/File:B-52_Stratofortress_flies_over_Dubai_Airshow_2025_(986813).webm",
           "https://upload.wikimedia.org/wikipedia/commons/7/70/B-52_Stratofortress_flies_over_Dubai_Airshow_2025_%28986813%29.webm",
           "U.S. Air Force video by Tech. Sgt. Justin Norton", "Public domain",
           "e3ffb0c431f3b556c836650a041365cf1cb20d60", "b52-dubai.webm", 0, 10,
           "source/b52-dubai-10s-1920x1080.mp4", "b52-dubai-run/overlay.mp4", "", 1920, 1080),
    Sample("f35-airshow", "F-35A · airshow", "Tiny, fast jet over cloud",
           "https://commons.wikimedia.org/wiki/File:F-35A_Airshow_Maneuvers_(47509546271).webm",
           "https://upload.wikimedia.org/wikipedia/commons/f/f5/F-35A_Airshow_Maneuvers_%2847509546271%29.webm",
           "See the Commons page (Flickr import)", "CC BY-SA 2.0",
           "8e34513d1f4bff43ec0a77e130cb6cfaa597d526", "f35-airshow.webm", 0, 20,
           "source/f35-airshow-20s-1920x1080.mp4", "f35-airshow-run/overlay.mp4", "", 1920, 1080),
    Sample("red-arrows", "Red Arrows · formation", "Nine-jet display team over a parked aircraft",
           "https://commons.wikimedia.org/wiki/File:Red_Arrows_whiz_over_the_Douglas_R4D_Sky_Train_at_RIAT_18_July_2025.webm",
           "https://upload.wikimedia.org/wikipedia/commons/c/c9/Red_Arrows_whiz_over_the_Douglas_R4D_Sky_Train_at_RIAT_18_July_2025.webm",
           "Jo Brodie", "CC0", "b25ce7ba2fff5486810894f059fea11aeb7fd10d",
           "red-arrows.webm", 0, 20, "source/red-arrows-20s-1920x1080.mp4",
           "red-arrows-run/overlay.mp4", "", 1920, 1080),
    Sample("air-force-one", "Air Force One · approach", "Large airliner from far to close",
           "https://commons.wikimedia.org/wiki/File:Air_Force_One_Landing_at_KPIT.webm",
           "https://upload.wikimedia.org/wikipedia/commons/e/e1/Air_Force_One_Landing_at_KPIT.webm",
           "Designism", "CC0", "93ff24731306f12cce508d000a978ce35ca772e9",
           "air-force-one.webm", 0, 20, "source/air-force-one-20s-1920x1080.mp4",
           "air-force-one-run/overlay.mp4", "", 1920, 1080),
)

BY_SLUG = {**CATALOG, **{sample.slug: sample for sample in DEMO_CLIPS}}
