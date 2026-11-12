import argparse
import os
from pathlib import Path
import yaml

def find_project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for p in [start] + list(start.parents):
        if (p / "CV").exists() and (p / "group assignment").exists():
            return p
        if (p / "DATA").exists():
            return p
    return start

def load_config(config_path=None):
    root = find_project_root()
    if config_path:
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = (root / cfg_path).resolve()
    else:
        cfg_path = root / "group assignment" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_project_root"] = str(root)
    cfg["_config_path"] = str(cfg_path)
    out_dir = root / cfg.get("output_dir", "group assignment")
    cfg["_out_dir"] = str(out_dir)
    return cfg

def add_common_args(parser):
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=None)
    return parser
