"""Emit data/cv_family.json from the family map defined in 03_ground_truth.py.

Kept as a separate small script so the autoencoder and the error analysis can
import the same mapping without importing the ground-truth module (which would
pull in the ollama client).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "gt", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "03_ground_truth.py"))
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)

json.dump(gt.CV_FAMILY, open(os.path.join(DATA, "cv_family.json"), "w",
                            encoding="utf-8"), indent=1)
print(f"wrote {len(gt.CV_FAMILY)} CV family labels")
