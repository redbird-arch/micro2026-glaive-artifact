#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = Path(os.environ.get("GLAIVE_SIM_EXP_OUT_ROOT", REPO_ROOT / "rebuttal" / "simulation_exp")).resolve()
RESULTS_DIR = OUT_ROOT / "results"
PLOTS_DIR = OUT_ROOT / "plots"

METHOD_COLORS = {
    "glaive": "#2a9d8f",
    "biring": "#7b2cbf",
    "halfringdr": "#7b2cbf",
    "mpibaseline": "#e76f51",
}
VARIANT_COLORS = {
    "real": "#111827",
    "balanced": "#00BFC4",
    "blend_0.25": "#009E73",
    "blend_0.50": "#CC79A7",
    "blend_0.75": "#E69F00",
    "eplb_x2": "#0072B2",
    "worst_top8": "#D55E00",
}
VARIANT_LABELS = {
    "real": "Real",
    "balanced": "Balanced",
    "blend_0.25": "Blend 25%",
    "blend_0.50": "Blend 50%",
    "blend_0.75": "Blend 75%",
    "eplb_x2": "EPLB x2",
    "worst_top8": "Worst-top8",
}
TOPOLOGY_MARKERS = {
    "mesh_nebula_8x4": "o",
    "clos_8x4": "s",
    "cm384_16x2": "^",
    "rail_optimized_8x4": "P",
    "torus_tpuv4_4x4x4": "D",
}
TOPOLOGY_JITTER = {
    "mesh_nebula_8x4": -0.012,
    "clos_8x4": -0.006,
    "cm384_16x2": 0.0,
    "rail_optimized_8x4": 0.012,
    "torus_tpuv4_4x4x4": 0.006,
}


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": True,
        }
    )
    return plt


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return float(value)


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def clean_label(value: str) -> str:
    return (
        value.replace("deepseekv3", "DeepSeek")
        .replace("olmoe", "OLMoE")
        .replace("qwen", "Qwen")
        .replace("evaluation_synthetic", "synthetic")
        .replace("_", " ")
    )


def variant_label(value: str) -> str:
    if value in VARIANT_LABELS:
        return VARIANT_LABELS[value]
    if value.startswith("blend_"):
        try:
            return f"Blend {float(value.split('_', 1)[1]) * 100:.0f}%"
        except (IndexError, ValueError):
            pass
    return clean_label(value)


def save_figure(
    plt,
    fig,
    figure_id: str,
    title: str,
    description: str,
    source_csvs: list[Path],
    bbox_inches: str | None = "tight",
) -> dict[str, object]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = PLOTS_DIR / f"{figure_id}.pdf"
    save_kwargs = {}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    fig.savefig(pdf_path, **save_kwargs)
    plt.close(fig)
    return {
        "figure_id": figure_id,
        "title": title,
        "description": description,
        "pdf_path": rel(pdf_path),
        "source_csvs": ";".join(rel(path) for path in source_csvs),
    }


def emit(entry: dict[str, object]) -> None:
    print(json.dumps(entry, ensure_ascii=False))


def skip(figure_id: str, reason: str) -> None:
    emit({"figure_id": figure_id, "skipped": True, "reason": reason})
