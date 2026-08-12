"""Configuration loading and path resolution."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path

    # -- section shortcuts --------------------------------------------------
    @property
    def http(self) -> dict[str, Any]:
        return self.raw["http"]

    @property
    def query(self) -> dict[str, Any]:
        return self.raw["query"]

    @property
    def frutas(self) -> dict[str, Any]:
        return self.raw["frutas"]

    @property
    def granos(self) -> dict[str, Any]:
        return self.raw["granos"]

    @property
    def update(self) -> dict[str, Any]:
        return self.raw["update"]

    @property
    def aggregate(self) -> dict[str, Any]:
        return self.raw["aggregate"]

    # -- resolved paths -----------------------------------------------------
    def path(self, key: str) -> Path:
        return (self.root / self.raw["paths"][key]).resolve()

    @property
    def data_dir(self) -> Path:
        return self.path("data")

    @property
    def logs_dir(self) -> Path:
        return self.path("logs")

    @property
    def duckdb_path(self) -> Path:
        return self.path("duckdb")

    @property
    def mapping_path(self) -> Path:
        return self.path("mapping")

    @property
    def crosswalk_path(self) -> Path:
        return self.root / "config" / "crosswalk_mercados.csv"

    @property
    def ciudades_path(self) -> Path:
        return self.root / "config" / "inpc_ciudades.csv"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def curated_dir(self) -> Path:
        return self.data_dir / "curated"

    @property
    def catalog_dir(self) -> Path:
        return self.data_dir / "catalog"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.csv"

    def ensure_dirs(self) -> None:
        for p in (
            self.data_dir,
            self.logs_dir,
            self.raw_dir,
            self.curated_dir,
            self.catalog_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


def load(path: str | os.PathLike | None = None) -> Config:
    root = PROJECT_ROOT
    cfg_path = Path(path) if path else root / "config" / "config.yml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(raw=raw, root=root)
    cfg.ensure_dirs()
    return cfg


def setup_logging(cfg: Config, name: str, verbose: bool = False) -> logging.Logger:
    """File + console logging. One log file per command, appended across runs."""
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sniim")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(cfg.logs_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(ch)
    return logger
