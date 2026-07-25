#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


SCENARIOS = (
    "deepseekv32_small",
    "deepseekv32_large",
    "qwen3_30b_a3b_small",
    "qwen3_30b_a3b_large",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize Figure 12 external-collective inputs. Paper mode uses "
            "the accepted-paper archive for unchanged methods and replaces the "
            "eight HalfR+DR files with fresh 6x-latency solver output."
        )
    )
    parser.add_argument("--mode", choices=("paper", "fresh"), default="paper")
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--fresh-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--override-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_summary(path: Path) -> dict[str, object]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "duration_us" not in rows[0]:
        raise ValueError(f"invalid external-collective CSV: {path}")
    return {
        "rows": len(rows),
        "duration_sum_us": sum(float(row["duration_us"]) for row in rows),
        "sha256": sha256(path),
    }


def scenario_files(root: Path, scenario: str) -> dict[str, Path]:
    scenario_root = root / scenario
    if not scenario_root.is_dir():
        raise FileNotFoundError(f"missing scenario directory: {scenario_root}")
    files = {path.name: path for path in sorted(scenario_root.glob("*.csv"))}
    if len(files) != 12:
        raise ValueError(
            f"expected 12 external-collective CSVs for {scenario}, found {len(files)}"
        )
    return files


def fresh_files(root: Path, scenario: str) -> dict[str, Path]:
    scenario_root = root / scenario / "astra_external"
    if not scenario_root.is_dir():
        raise FileNotFoundError(f"missing fresh solver output: {scenario_root}")
    files = {path.name: path for path in sorted(scenario_root.glob("*.csv"))}
    if len(files) != 12:
        raise ValueError(
            f"expected 12 fresh external-collective CSVs for {scenario}, "
            f"found {len(files)}"
        )
    return files


def reset_output_root(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def main() -> None:
    args = parse_args()
    reference_root = args.reference_root.resolve()
    fresh_root = args.fresh_root.resolve()
    manifest_root = args.manifest_root.resolve()
    output_root = args.output_root.resolve()
    reset_output_root(output_root)

    override_payload: dict[str, dict[str, str]] = {}
    report_rows: list[dict[str, object]] = []
    half_r_replacements = 0

    for scenario in SCENARIOS:
        reference = scenario_files(reference_root, scenario)
        fresh = fresh_files(fresh_root, scenario)
        selected = fresh if args.mode == "fresh" else reference
        scenario_output = output_root / scenario
        scenario_output.mkdir(parents=True)

        for name, selected_path in sorted(selected.items()):
            source_path = selected_path
            source_kind = "fresh"
            if args.mode == "paper":
                source_kind = "paper_archive"
                if "halfrdr" in name:
                    source_path = fresh[name]
                    source_kind = "fresh_6x_halfrdr"
                    half_r_replacements += 1

            output_path = scenario_output / name
            shutil.copy2(source_path, output_path)
            selected_summary = csv_summary(output_path)
            fresh_summary = csv_summary(fresh[name])
            reference_summary = csv_summary(reference[name])
            report_rows.append(
                {
                    "scenario": scenario,
                    "file": name,
                    "source_kind": source_kind,
                    "selected": selected_summary,
                    "fresh": fresh_summary,
                    "paper_archive": reference_summary,
                    "fresh_to_archive_duration_ratio": (
                        float(fresh_summary["duration_sum_us"])
                        / float(reference_summary["duration_sum_us"])
                    ),
                }
            )

        manifest_path = manifest_root / scenario / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing source manifest: {manifest_path}")
        override_payload[scenario] = {
            "manifest": str(manifest_path),
            "external_root": str(scenario_output),
        }

    if args.mode == "paper" and half_r_replacements != 8:
        raise ValueError(
            f"expected eight fresh HalfR+DR replacements, got {half_r_replacements}"
        )

    args.override_json.parent.mkdir(parents=True, exist_ok=True)
    args.override_json.write_text(json.dumps(override_payload, indent=2) + "\n")

    report = {
        "mode": args.mode,
        "policy": (
            "Accepted-paper non-HalfR+DR simulator inputs plus eight freshly "
            "generated 6x-latency HalfR+DR inputs."
            if args.mode == "paper"
            else "All simulator inputs freshly generated by the bundled solver."
        ),
        "reference_root": str(reference_root),
        "fresh_root": str(fresh_root),
        "output_root": str(output_root),
        "half_r_replacements": half_r_replacements,
        "files": report_rows,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Materialized {len(report_rows)} files in {output_root}")
    print(f"Wrote {args.override_json}")
    print(f"Wrote {args.report_json}")


if __name__ == "__main__":
    main()
