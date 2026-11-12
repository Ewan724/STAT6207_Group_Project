"""Configuration loading and shared path helpers.

The code never hard-codes absolute paths. Everything resolves from the project
root, which is the parent of the ``group assignment`` directory.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

GROUP_DIRNAME = "group assignment"


def find_project_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if parent.name == GROUP_DIRNAME:
            return parent.parent
        if (parent / GROUP_DIRNAME / "config.yaml").exists():
            return parent
    return here.parents[2]


PROJECT_ROOT = find_project_root()
GROUP_DIR = PROJECT_ROOT / GROUP_DIRNAME


def _resolve(base: Path, value: str | os.PathLike) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path
    root: Path = field(default=PROJECT_ROOT)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path_for(self, key: str) -> Path:
        value = self.get(f"paths.{key}")
        if value is None:
            raise KeyError(f"paths.{key} is not configured")
        return _resolve(self.root, value)

    @property
    def group_dir(self) -> Path:
        return _resolve(self.root, self.get("paths.group_dir", GROUP_DIRNAME))

    @property
    def data_root(self) -> Path:
        return _resolve(self.root, self.get("paths.data_root", "DATA"))

    @property
    def pool_cv_dir(self) -> Path:
        value = self.get("paths.pool_cv_dir")
        return _resolve(self.data_root, value) if value else self.data_root / "CV"

    @property
    def gt_pdf_dir(self) -> Path:
        value = self.get("paths.gt_pdf_dir")
        return _resolve(self.data_root, value) if value else self.data_root / "JD+HiredCV"

    @property
    def excel_path(self) -> Path | None:
        value = self.get("paths.excel_path")
        return _resolve(self.data_root, value) if value else None

    @property
    def processed(self) -> Path:
        return self.path_for("processed")

    @property
    def cache(self) -> Path:
        return self.path_for("cache")

    @property
    def models(self) -> Path:
        return self.path_for("models")

    @property
    def figures(self) -> Path:
        return self.path_for("figures")

    @property
    def tables(self) -> Path:
        return self.path_for("tables")

    @property
    def logs(self) -> Path:
        return self.path_for("logs")

    @property
    def seed(self) -> int:
        return int(self.get("seed", 42))

    def ensure_dirs(self) -> None:
        for p in [self.processed, self.cache, self.models, self.figures,
                  self.tables, self.logs, self.cache / "llm"]:
            p.mkdir(parents=True, exist_ok=True)

    def write_run_mode(self, mode: str, source: Path, reason: str) -> None:
        payload = {
            "run_mode": mode,
            "data_source": str(source),
            "data_source_name": source.name,
            "reason": reason,
            "seed": self.seed,
        }
        self.ensure_dirs()
        (self.processed / "run_mode.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def run_mode(self) -> dict[str, Any]:
        p = self.processed / "run_mode.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {"run_mode": "unknown"}


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else (GROUP_DIR / "config.yaml")
    if not cfg_path.is_absolute():
        if (PROJECT_ROOT / cfg_path).exists():
            cfg_path = PROJECT_ROOT / cfg_path
        else:
            cfg_path = cfg_path.resolve()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = Config(raw=raw, path=cfg_path)
    cfg.ensure_dirs()
    return cfg


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
