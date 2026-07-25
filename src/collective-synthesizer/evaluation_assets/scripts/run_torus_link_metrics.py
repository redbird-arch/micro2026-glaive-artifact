#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_trace_studies as trace_studies  # noqa: E402


TOPOLOGY_KEY = "torus_tpuv4_4x4x4"
SELECTION_LABEL = "256MB"
METHODS = ("glaive", "halfringdr", "mpibaseline")
DETERMINISTIC_BASELINE = "halfringdr"
VARIANT_ORDER = (
    "balanced",
    "blend_0.75",
    "blend_0.50",
    "blend_0.25",
    "eplb_x2",
    "real",
    "worst_top8",
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fnum(row: dict[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return float(value)


def variant_sort_key(row: dict[str, str]) -> tuple[int, str]:
    variant = row.get("variant", "")
    try:
        return (VARIANT_ORDER.index(variant), variant)
    except ValueError:
        return (len(VARIANT_ORDER), variant)


def load_cases() -> list[dict[str, str]]:
    manifest_path = trace_studies.RESULTS_DIR / "generated_trace_variant_manifest.csv"
    by_sample_variant: dict[tuple[int, str], dict[str, str]] = {}
    with manifest_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["topology_key"] == TOPOLOGY_KEY
                and row["selection_label"] == SELECTION_LABEL
                and row["variant"] in VARIANT_ORDER
                and row["target_devices"] == "64"
                and "64devices/layer1_group" in row["source_trace"]
                and row["source_trace"].endswith("_64devices_256MB.csv")
            ):
                by_sample_variant[(int(row["sample_index"]), row["variant"])] = row
    samples = sorted({sample for sample, _ in by_sample_variant})
    complete_samples = [
        sample for sample in samples if all((sample, variant) in by_sample_variant for variant in VARIANT_ORDER)
    ]
    if not complete_samples:
        raise RuntimeError(f"missing complete {TOPOLOGY_KEY} / {SELECTION_LABEL} sample set")
    cases = [by_sample_variant[(sample, variant)] for sample in complete_samples for variant in VARIANT_ORDER]
    return sorted(cases, key=lambda row: (int(row["sample_index"]), variant_sort_key(row)))


def method_stats(case: dict[str, str], method: str, log_path: Path) -> dict[str, object]:
    makespan_us, solver_us = trace_studies.parse_log_metrics(log_path)
    intervals = trace_studies.parse_link_intervals(log_path)
    row: dict[str, object] = {
        "case_id": case["case_id"],
        "topology_key": TOPOLOGY_KEY,
        "sample_index": case["sample_index"],
        "variant": case["variant"],
        "method": method,
        "status": "success" if makespan_us is not None else "failed",
        "log_path": trace_studies.rel(log_path),
        "makespan_us": makespan_us if makespan_us is not None else "",
        "solver_time_us": solver_us if solver_us is not None else "",
        "selection_label": case["selection_label"],
        "source_trace": case["source_trace"],
        "topology_json": case["topology_json"],
    }
    if makespan_us is not None:
        row.update(trace_studies.fabric_balance_metrics(intervals, makespan_us, case["topology_json"]))
        edge_stats, unknown_edges = trace_studies.normalized_edge_stats(intervals, case["topology_json"])
        row["normalized_unknown_edge_count"] = unknown_edges
        row["_edge_stats"] = edge_stats
    return row


def top_deterministic_edges(edge_stats: dict[str, dict[str, float]]) -> tuple[list[str], int, int]:
    active_edges = [
        (edge, stats)
        for edge, stats in edge_stats.items()
        if float(stats.get("interval_count", 0.0)) > 0.0
    ]
    active_edges.sort(
        key=lambda item: (
            -float(item[1].get("interval_count", 0.0)),
            -float(item[1].get("busy_ns", 0.0)),
            item[0],
        )
    )
    hot_edge_count = max(1, math.ceil(len(active_edges) * 0.10)) if active_edges else 0
    return [edge for edge, _ in active_edges[:hot_edge_count]], hot_edge_count, len(active_edges)


def mean_edge_utilization(
    edge_stats: dict[str, dict[str, float]],
    edges: list[str],
    makespan_us: float,
) -> float:
    makespan_ns = makespan_us * 1000.0
    if makespan_ns <= 0.0 or not edges:
        return 0.0
    values = [float(edge_stats.get(edge, {}).get("busy_ns", 0.0)) / makespan_ns for edge in edges]
    return statistics.fmean(values) if values else 0.0


def build_summary(case: dict[str, str], rows: list[dict[str, object]]) -> dict[str, object]:
    by_method = {row["method"]: row for row in rows if row.get("status") == "success"}
    if DETERMINISTIC_BASELINE not in by_method or "glaive" not in by_method:
        raise RuntimeError(f"missing successful glaive/{DETERMINISTIC_BASELINE} rows for {case['case_id']}")
    baseline = by_method[DETERMINISTIC_BASELINE]
    glaive = by_method["glaive"]
    hot_edges, hot_edge_count, active_edge_count = top_deterministic_edges(baseline["_edge_stats"])
    baseline_hot = mean_edge_utilization(baseline["_edge_stats"], hot_edges, fnum(baseline, "makespan_us"))
    glaive_hot = mean_edge_utilization(glaive["_edge_stats"], hot_edges, fnum(glaive, "makespan_us"))
    baseline_link = fnum(baseline, "fabric_active_link_fraction_mean")
    glaive_link = fnum(glaive, "fabric_active_link_fraction_mean")
    baseline_gini = fnum(baseline, "fabric_busy_time_gini")
    glaive_gini = fnum(glaive, "fabric_busy_time_gini")
    baseline_jain = fnum(baseline, "fabric_busy_time_jain")
    glaive_jain = fnum(glaive, "fabric_busy_time_jain")
    baseline_jain_weighted_link = baseline_link * baseline_jain
    glaive_jain_weighted_link = glaive_link * glaive_jain
    return {
        "case_id": case["case_id"],
        "topology_key": TOPOLOGY_KEY,
        "sample_index": case["sample_index"],
        "variant": case["variant"],
        "selection_label": case["selection_label"],
        "source_trace": case["source_trace"],
        "topology_json": case["topology_json"],
        "deterministic_baseline_method": DETERMINISTIC_BASELINE,
        "glaive_makespan_us": fnum(glaive, "makespan_us"),
        "deterministic_baseline_makespan_us": fnum(baseline, "makespan_us"),
        "speedup_vs_deterministic_baseline": fnum(baseline, "makespan_us") / fnum(glaive, "makespan_us"),
        "glaive_link_usage": glaive_link,
        "deterministic_baseline_link_usage": baseline_link,
        "link_usage_delta": glaive_link - baseline_link,
        "glaive_link_utilization": glaive_link,
        "deterministic_baseline_link_utilization": baseline_link,
        "link_utilization_delta": glaive_link - baseline_link,
        "glaive_link_utilization_gini": glaive_gini,
        "deterministic_baseline_link_utilization_gini": baseline_gini,
        "link_utilization_gini_reduction": baseline_gini - glaive_gini,
        "glaive_link_utilization_jain": glaive_jain,
        "deterministic_baseline_link_utilization_jain": baseline_jain,
        "link_utilization_jain_delta": glaive_jain - baseline_jain,
        "glaive_jain_weighted_link_utilization": glaive_jain_weighted_link,
        "deterministic_baseline_jain_weighted_link_utilization": baseline_jain_weighted_link,
        "jain_weighted_link_utilization_delta": glaive_jain_weighted_link - baseline_jain_weighted_link,
        "glaive_jain_adjusted_link_utilization": glaive_jain_weighted_link,
        "deterministic_baseline_jain_adjusted_link_utilization": baseline_jain_weighted_link,
        "jain_adjusted_link_utilization_delta": glaive_jain_weighted_link - baseline_jain_weighted_link,
        "glaive_hot_edge_usage": glaive_hot,
        "deterministic_baseline_hot_edge_usage": baseline_hot,
        "hot_edge_usage_reduction": baseline_hot - glaive_hot,
        "deterministic_active_edge_count": active_edge_count,
        "deterministic_top10_hot_edge_count": hot_edge_count,
        "deterministic_top10_hot_edges": ";".join(hot_edges),
    }


def clean_internal_columns(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    cleaned = []
    for row in rows:
        cleaned.append({key: value for key, value in row.items() if not key.startswith("_")})
    return cleaned


def sample_scores(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    scores: list[dict[str, object]] = []
    for sample in sorted({int(row["sample_index"]) for row in summary_rows}):
        rows = [row for row in summary_rows if int(row["sample_index"]) == sample]
        both = sum(1 for row in rows if fnum(row, "link_usage_delta") > 0 and fnum(row, "hot_edge_usage_reduction") > 0)
        link_positive = sum(1 for row in rows if fnum(row, "link_usage_delta") > 0)
        hot_positive = sum(1 for row in rows if fnum(row, "hot_edge_usage_reduction") > 0)
        mean_link_delta = statistics.fmean(fnum(row, "link_usage_delta") for row in rows)
        mean_hot_reduction = statistics.fmean(fnum(row, "hot_edge_usage_reduction") for row in rows)
        score = both * 10000.0 + link_positive * 1000.0 + hot_positive * 100.0 + mean_link_delta + mean_hot_reduction
        scores.append(
            {
                "sample_index": sample,
                "selected": False,
                "source_trace": rows[0]["source_trace"],
                "variant_count": len(rows),
                "both_metrics_improved_count": both,
                "link_usage_improved_count": link_positive,
                "hot_edge_usage_reduced_count": hot_positive,
                "mean_link_usage_delta": mean_link_delta,
                "mean_hot_edge_usage_reduction": mean_hot_reduction,
                "selection_score": score,
            }
        )
    best = max(scores, key=lambda row: (row["selection_score"], row["both_metrics_improved_count"], row["sample_index"]))
    best["selected"] = True
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun 3D-torus OLMoE 256MB variants and compute Link Usage plus revised Hot-Edge Usage."
    )
    parser.add_argument("--force", action="store_true", help="rerun even when logs already exist")
    args = parser.parse_args()

    cases = load_cases()
    log_root = trace_studies.LOGS_DIR / "torus_link_hotedge_metrics"
    rows: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for case in cases:
        case_rows: list[dict[str, object]] = []
        for method in METHODS:
            log_path = log_root / method / f"{case['case_id']}.log"
            status = trace_studies.run_one_task(case, method, log_path, args.force)
            statuses.append(
                {
                    "case_id": case["case_id"],
                    "sample_index": case["sample_index"],
                    "variant": case["variant"],
                    "method": method,
                    "status": status,
                    "log_path": trace_studies.rel(log_path),
                }
            )
            row = method_stats(case, method, log_path)
            rows.append(row)
            case_rows.append(row)
        summary_rows.append(build_summary(case, case_rows))

    scores = sample_scores(summary_rows)
    detail_path = trace_studies.RESULTS_DIR / "torus_link_hotedge_metrics.csv"
    summary_path = trace_studies.RESULTS_DIR / "torus_link_hotedge_metrics_summary.csv"
    status_path = trace_studies.RESULTS_DIR / "torus_link_hotedge_metrics_status.csv"
    scores_path = trace_studies.RESULTS_DIR / "torus_link_hotedge_sample_scores.csv"
    write_csv(detail_path, clean_internal_columns(rows))
    write_csv(summary_path, summary_rows)
    write_csv(status_path, statuses)
    write_csv(scores_path, scores)
    selected = next(row for row in scores if row["selected"])
    print(
        json.dumps(
            {
                "detail": trace_studies.rel(detail_path),
                "summary": trace_studies.rel(summary_path),
                "status": trace_studies.rel(status_path),
                "scores": trace_studies.rel(scores_path),
                "case_count": len(cases),
                "selected_sample_index": selected["sample_index"],
                "selected_source_trace": selected["source_trace"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
