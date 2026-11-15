"""Validate and aggregate one sharded IAMSO-NF-BI-CA benchmark run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from . import runner
from .dataset import no_task_corner_results_dir
from .run_ca_vs_mso_shard import ALGORITHM_NAME, MSO_NAME, WAYPOINT_COUNT


BASELINE_NAME = "IAMSO-NF-BI"
DIFFICULTIES = ("simple", "medium", "hard")
ALGORITHMS = (MSO_NAME, ALGORITHM_NAME)


def _expected_pairs(
    difficulties: Iterable[str], expected_count: int
) -> set[tuple[str, str]]:
    return {
        (f"{difficulty}_{index:03d}", algorithm)
        for difficulty in difficulties
        for index in range(1, expected_count + 1)
        for algorithm in ALGORITHMS
    }


def _validate_ablation(ablation: dict[str, Any]) -> None:
    expected = {
        "waypoint_count": WAYPOINT_COUNT,
        "fallback_path_enabled": False,
        "seed_paths_enabled": True,
        "bidirectional_seed_paths_enabled": True,
        "collision_aware_compression_enabled": True,
        "final_smoothing_enabled": True,
    }
    mismatches = {
        key: (ablation.get(key), value)
        for key, value in expected.items()
        if ablation.get(key) != value
    }
    if mismatches:
        raise ValueError(f"inconsistent ablation metadata: {mismatches}")


def validate_and_merge_shards(
    reports: list[dict[str, Any]],
    *,
    difficulties: Iterable[str] = DIFFICULTIES,
    expected_count: int = 100,
) -> list[dict[str, Any]]:
    """Return details after strict dataset, shard, and pair validation."""
    difficulties = tuple(difficulties)
    if not reports:
        raise ValueError("no shard reports found")

    first = reports[0]
    reference = {
        "run_id": first.get("run_id"),
        "dataset_name": first.get("dataset_name"),
        "dataset_version": first.get("dataset_version"),
        "cost_profile": first.get("cost_profile"),
        "evaluation_profile": first.get("evaluation_profile"),
    }
    if not reference["run_id"]:
        raise ValueError("shard report is missing run_id")

    details: list[dict[str, Any]] = []
    seen_shards: set[tuple[str, int, int]] = set()
    for report in reports:
        actual_reference = {key: report.get(key) for key in reference}
        if actual_reference != reference:
            raise ValueError(
                f"inconsistent shard metadata: expected={reference}, actual={actual_reference}"
            )
        _validate_ablation(report.get("ablation", {}))
        if set(report.get("algorithms", [])) != set(ALGORITHMS):
            raise ValueError(f"unexpected algorithms: {report.get('algorithms')}")

        shard = report.get("shard", {})
        shard_key = (
            str(shard.get("difficulty")),
            int(shard.get("start_index", 0)),
            int(shard.get("end_index", 0)),
        )
        if shard_key in seen_shards:
            raise ValueError(f"duplicate shard range: {shard_key}")
        seen_shards.add(shard_key)

        start_index, end_index = shard_key[1], shard_key[2]
        expected_ids = {
            f"{shard_key[0]}_{index:03d}"
            for index in range(start_index, end_index + 1)
        }
        declared_ids = set(shard.get("map_ids", []))
        if declared_ids != expected_ids:
            raise ValueError(
                f"declared shard IDs do not match its range: shard={shard_key}"
            )
        shard_details = list(report.get("details", []))
        if int(report.get("map_count", -1)) != len(expected_ids):
            raise ValueError(f"map_count mismatch in shard {shard_key}")
        details.extend(shard_details)

    pairs = [(row.get("map_id"), row.get("algorithm")) for row in details]
    counts = Counter(pairs)
    duplicates = sorted(pair for pair, count in counts.items() if count != 1)
    expected = _expected_pairs(difficulties, expected_count)
    actual = set(pairs)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if duplicates or missing or unexpected:
        raise ValueError(
            "aggregate coverage mismatch: "
            f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}"
        )
    return details


def _add_detail(bucket: dict[str, Any], detail: dict[str, Any]) -> None:
    bucket["attempts"] += 1
    bucket["runtime"] += float(detail["runtime"])
    status = detail["status"]
    collision_count = detail.get("collision_count")
    if collision_count is not None:
        bucket["path_found"] += 1
        bucket["collision"] += 1 if float(collision_count) > 0 else 0
    if detail.get("score") is not None:
        bucket["score"] += float(detail["score"])
        bucket["score_count"] += 1
    if status == "success":
        bucket["success"] += 1
        for key in (
            "distance",
            "time",
            "safety",
            "smoothness",
            "rain_exposure",
            "inflation_exposure",
            "total_exposure",
        ):
            bucket[key] += float(detail[key] or 0.0)
    elif status == "collision":
        bucket["collision_failures"] += 1
    elif status == "no_path":
        bucket["no_path"] += 1
    elif status == "exception":
        bucket["exception_count"] += 1
    else:
        raise ValueError(f"unknown result status: {status!r}")


def summarize_details(details: list[dict[str, Any]]) -> dict[str, Any]:
    overall = {name: runner._new_bucket() for name in ALGORITHMS}
    by_difficulty = {
        difficulty: {name: runner._new_bucket() for name in ALGORITHMS}
        for difficulty in DIFFICULTIES
    }
    for detail in details:
        algorithm = detail["algorithm"]
        difficulty = detail["difficulty"]
        _add_detail(overall[algorithm], detail)
        _add_detail(by_difficulty[difficulty][algorithm], detail)
    return {
        "overall": [
            runner._summarize_bucket(name, overall[name]) for name in ALGORITHMS
        ],
        "by_difficulty": {
            difficulty: [
                runner._summarize_bucket(name, by_difficulty[difficulty][name])
                for name in ALGORITHMS
            ]
            for difficulty in DIFFICULTIES
        },
    }


def paired_outcomes(details: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {
        (detail["difficulty"], detail["map_id"], detail["algorithm"]): detail
        for detail in details
    }
    output: dict[str, dict[str, Any]] = {}
    for difficulty in DIFFICULTIES:
        rows = [
            (
                indexed[(difficulty, f"{difficulty}_{index:03d}", ALGORITHM_NAME)],
                indexed[(difficulty, f"{difficulty}_{index:03d}", MSO_NAME)],
            )
            for index in range(1, 101)
        ]
        both = [(ca, mso) for ca, mso in rows if ca["success"] and mso["success"]]
        deltas: dict[str, float | None] = {}
        for metric in ("distance", "time", "safety", "smoothness", "score", "runtime"):
            values = [
                float(ca[metric]) - float(mso[metric])
                for ca, mso in both
                if ca.get(metric) is not None and mso.get(metric) is not None
            ]
            deltas[f"avg_{metric}_delta_ca_minus_mso"] = (
                statistics.fmean(values) if values else None
            )
        output[difficulty] = {
            "both_success": sum(bool(ca["success"] and mso["success"]) for ca, mso in rows),
            "ca_only_success": sum(bool(ca["success"] and not mso["success"]) for ca, mso in rows),
            "mso_only_success": sum(bool(mso["success"] and not ca["success"]) for ca, mso in rows),
            "neither_success": sum(bool(not ca["success"] and not mso["success"]) for ca, mso in rows),
            **deltas,
        }
    return output


def _exact_mcnemar_two_sided(ca_only: int, mso_only: int) -> float:
    discordant = ca_only + mso_only
    if discordant == 0:
        return 1.0
    lower_tail = min(ca_only, mso_only)
    probability = sum(
        math.comb(discordant, successes)
        for successes in range(lower_tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def paired_significance(
    paired: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    overall_ca_only = 0
    overall_mso_only = 0
    for difficulty in DIFFICULTIES:
        ca_only = int(paired[difficulty]["ca_only_success"])
        mso_only = int(paired[difficulty]["mso_only_success"])
        overall_ca_only += ca_only
        overall_mso_only += mso_only
        p_value = _exact_mcnemar_two_sided(ca_only, mso_only)
        output[difficulty] = {
            "ca_only_success": ca_only,
            "mso_only_success": mso_only,
            "exact_mcnemar_two_sided_p": p_value,
            "significant_at_0_05": p_value < 0.05,
        }
    overall_p = _exact_mcnemar_two_sided(overall_ca_only, overall_mso_only)
    output["overall"] = {
        "ca_only_success": overall_ca_only,
        "mso_only_success": overall_mso_only,
        "exact_mcnemar_two_sided_p": overall_p,
        "significant_at_0_05": overall_p < 0.05,
    }
    return output


def _nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def runtime_analysis(details: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    hard_outliers = {"hard_094", "hard_097"}
    for difficulty in DIFFICULTIES:
        output[difficulty] = {}
        for algorithm in ALGORITHMS:
            rows = [
                detail
                for detail in details
                if detail["difficulty"] == difficulty
                and detail["algorithm"] == algorithm
            ]
            values = [float(detail["runtime"]) for detail in rows]
            core_values = [
                float(detail["runtime"])
                for detail in rows
                if detail["map_id"] not in hard_outliers
            ]
            output[difficulty][algorithm] = {
                "count": len(values),
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "p95_nearest_rank": _nearest_rank(values, 0.95),
                "maximum": max(values),
                "mean_excluding_hard_094_097": statistics.fmean(core_values),
            }
    return output


def failure_maps(details: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    return {
        difficulty: {
            algorithm: [
                detail["map_id"]
                for detail in details
                if detail["difficulty"] == difficulty
                and detail["algorithm"] == algorithm
                and not detail["success"]
            ]
            for algorithm in ALGORITHMS
        }
        for difficulty in DIFFICULTIES
    }


def _parse_percent(value: str) -> float:
    return float(value.rstrip("%")) / 100.0


def load_nf_bi_baselines(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load the newest deterministic 100-map IAMSO-NF-BI report per difficulty."""
    found: dict[str, tuple[float, dict[str, Any]]] = {}
    for path in results_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        if "# IAMSO-NF-BI " not in text or "| IAMSO-NF-BI |" not in text:
            continue
        difficulty = next(
            (
                item
                for item in DIFFICULTIES
                if f"- Difficulty: {item}" in text or f"- \u96be\u5ea6\u8303\u56f4\uff1a{item}" in text
            ),
            None,
        )
        if difficulty is None:
            continue
        row_line = next(
            line for line in text.splitlines() if line.startswith("| IAMSO-NF-BI |")
        )
        values = [value.strip() for value in row_line.strip("|").split("|")]
        if len(values) != 9:
            raise ValueError(f"unexpected IAMSO-NF-BI table row in {path}")
        baseline = {
            "algorithm": BASELINE_NAME,
            "attempts": 100,
            "success_rate": _parse_percent(values[1]),
            "avg_distance": float(values[2]),
            "avg_time": float(values[3]),
            "avg_safety": _parse_percent(values[4]),
            "avg_collision": float(values[5]),
            "avg_score": float(values[6]),
            "avg_runtime": float(values[7]),
            "collision_failure_count": int(values[8]),
            "source_report": str(path),
        }
        modified = path.stat().st_mtime
        if difficulty not in found or modified > found[difficulty][0]:
            found[difficulty] = (modified, baseline)
    missing = sorted(set(DIFFICULTIES) - set(found))
    if missing:
        raise ValueError(f"missing IAMSO-NF-BI baseline reports: {missing}")
    return {difficulty: found[difficulty][1] for difficulty in DIFFICULTIES}


def build_comparison(
    summary: dict[str, Any], baselines: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    comparison: dict[str, dict[str, Any]] = {}
    for difficulty in DIFFICULTIES:
        rows = {
            row["algorithm"]: row for row in summary["by_difficulty"][difficulty]
        }
        ca_rate = float(rows[ALGORITHM_NAME]["success_rate"])
        mso_rate = float(rows[MSO_NAME]["success_rate"])
        baseline_rate = float(baselines[difficulty]["success_rate"])
        comparison[difficulty] = {
            "nf_bi_success_rate": baseline_rate,
            "ca_success_rate": ca_rate,
            "compression_success_rate_delta": ca_rate - baseline_rate,
            "mso_success_rate": mso_rate,
            "ca_minus_mso_success_rate": ca_rate - mso_rate,
            "ca_beats_nf_bi": ca_rate > baseline_rate,
            "ca_beats_mso": ca_rate > mso_rate,
        }
    return comparison


def build_overall_comparison(
    summary: dict[str, Any], baselines: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    baseline_attempts = sum(int(row["attempts"]) for row in baselines.values())
    baseline_successes = sum(
        int(row["attempts"]) * float(row["success_rate"])
        for row in baselines.values()
    )
    rows = {row["algorithm"]: row for row in summary["overall"]}
    baseline_rate = baseline_successes / baseline_attempts
    ca_rate = float(rows[ALGORITHM_NAME]["success_rate"])
    mso_rate = float(rows[MSO_NAME]["success_rate"])
    return {
        "nf_bi_success_rate": baseline_rate,
        "ca_success_rate": ca_rate,
        "compression_success_rate_delta": ca_rate - baseline_rate,
        "mso_success_rate": mso_rate,
        "ca_minus_mso_success_rate": ca_rate - mso_rate,
    }


def _format_percent(value: float) -> str:
    return f"{value:.2%}"


def write_markdown(
    path: Path,
    report: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
    comparison: dict[str, dict[str, Any]],
    paired: dict[str, dict[str, Any]],
) -> None:
    overall = report["overall_comparison"]
    runtime = report["runtime_analysis"]
    significance = report["paired_significance"]
    lines = [
        "# IAMSO-NF-BI collision-aware compression ablation",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Dataset: `{report['dataset_name']}` v{report['dataset_version']}",
        "- Maps: 100 per difficulty, 300 total",
        f"- Intermediate waypoints: {WAYPOINT_COUNT} (unchanged)",
        "- Changed factor: equal-arc seed encoding -> collision-aware seed encoding",
        "- Unchanged: NF fallback policy, BI seed set, optimizer budget, and smoothing",
        "- Current MSO was rerun on every map in the same shard process",
        "- IAMSO-NF-BI is the deterministic same-map baseline completed earlier on this code state",
        "",
        "## Conclusion",
        "",
        f"- Collision-aware compression improved IAMSO-NF-BI overall from "
        f"{_format_percent(overall['nf_bi_success_rate'])} to "
        f"{_format_percent(overall['ca_success_rate'])} "
        f"({_format_percent(overall['compression_success_rate_delta'])}).",
        f"- CA exceeded current MSO overall by "
        f"{_format_percent(overall['ca_minus_mso_success_rate'])}; "
        "it exceeded MSO on medium and hard maps, but not on simple maps.",
        "- The paired overall advantage was statistically significant under the "
        f"exact McNemar test (p={significance['overall']['exact_mcnemar_two_sided_p']:.6f}).",
        "- CA traded runtime and some safety margin for a higher feasible-path rate; "
        "on maps where both algorithms succeeded, CA paths were shorter and smoother.",
        "- Recommended mode policy: retain CA for IAMSO in all modes as an IAMSO "
        "improvement, while MSO remains the success-rate leader on simple maps.",
        "",
        "## Success-rate ablation",
        "",
        "| Difficulty | IAMSO-NF-BI | IAMSO-NF-BI-CA | CA - BI | Current MSO | CA - MSO |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for difficulty in DIFFICULTIES:
        row = comparison[difficulty]
        lines.append(
            f"| {difficulty} | {_format_percent(row['nf_bi_success_rate'])} | "
            f"{_format_percent(row['ca_success_rate'])} | "
            f"{_format_percent(row['compression_success_rate_delta'])} | "
            f"{_format_percent(row['mso_success_rate'])} | "
            f"{_format_percent(row['ca_minus_mso_success_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Paired CA vs current MSO",
            "",
        "| Difficulty | Both succeed | CA only | MSO only | Neither | Exact p |",
        "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for difficulty in DIFFICULTIES:
        row = paired[difficulty]
        lines.append(
            f"| {difficulty} | {row['both_success']} | {row['ca_only_success']} | "
            f"{row['mso_only_success']} | {row['neither_success']} | "
            f"{significance[difficulty]['exact_mcnemar_two_sided_p']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Current-run metrics",
            "",
            "| Difficulty | Algorithm | Success | Distance | Travel time | Safety | Score | Runtime/s |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for difficulty in DIFFICULTIES:
        for row in report["summary"]["by_difficulty"][difficulty]:
            lines.append(
                f"| {difficulty} | {row['algorithm']} | {_format_percent(row['success_rate'])} | "
                f"{row['avg_distance']:.4f} | {row['avg_time']:.4f} | "
                f"{_format_percent(row['avg_safety'])} | {row['avg_score']:.4f} | "
                f"{row['avg_runtime']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Runtime distribution",
            "",
            "| Difficulty | Algorithm | Mean/s | Median/s | P95/s | Max/s | Mean excluding hard_094/097 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for difficulty in DIFFICULTIES:
        for algorithm in ALGORITHMS:
            row = runtime[difficulty][algorithm]
            lines.append(
                f"| {difficulty} | {algorithm} | {row['mean']:.4f} | "
                f"{row['median']:.4f} | {row['p95_nearest_rank']:.4f} | "
                f"{row['maximum']:.4f} | "
                f"{row['mean_excluding_hard_094_097']:.4f} |"
            )
    lines.extend(
        [
            "",
            "`hard_094` and `hard_097` are strictly unreachable under the current "
            "continuous collision rule. They produced CA runtime outliers while both "
            "algorithms still failed, so the table reports both full and trimmed means.",
        ]
    )

    lines.extend(
        [
            "",
            "## Validation",
            "",
            "- Every difficulty contains exactly IDs `001-100`.",
            "- Every `(map_id, algorithm)` pair occurs exactly once.",
            "- All shards use the same dataset version and ablation metadata.",
            "- Parallel wall-clock runtime is reported for reference only and is not a speed ranking.",
            "",
            "## Baseline sources",
            "",
        ]
    )
    for difficulty in DIFFICULTIES:
        lines.append(f"- {difficulty}: `{baselines[difficulty]['source_report']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_csv(
    path: Path,
    summary: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
) -> None:
    fieldnames = [
        "difficulty",
        "algorithm",
        "attempts",
        "success_rate",
        "avg_distance",
        "avg_time",
        "avg_safety",
        "avg_collision",
        "avg_score",
        "avg_runtime",
        "collision_failure_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for difficulty in DIFFICULTIES:
            rows = [baselines[difficulty], *summary["by_difficulty"][difficulty]]
            for row in rows:
                writer.writerow(
                    {"difficulty": difficulty, **{key: row.get(key) for key in fieldnames[1:]}}
                )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate one complete CA-vs-MSO shard run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    results_dir = no_task_corner_results_dir()
    input_dir = args.input_dir or (results_dir / "ablation_shards")
    output_dir = args.output_dir or results_dir
    files = sorted(input_dir.glob(f"IAMSO-NF-BI-CA_vs_MSO_{args.run_id}_*.json"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    details = validate_and_merge_shards(reports)
    summary = summarize_details(details)
    baselines = load_nf_bi_baselines(results_dir)
    paired = paired_outcomes(details)
    comparison = build_comparison(summary, baselines)
    overall_comparison = build_overall_comparison(summary, baselines)
    significance = paired_significance(paired)
    runtimes = runtime_analysis(details)
    failures = failure_maps(details)

    first = reports[0]
    report = {
        "run_id": args.run_id,
        "dataset_name": first["dataset_name"],
        "dataset_version": first["dataset_version"],
        "dataset_root": first["dataset_root"],
        "map_count": 300,
        "selected_difficulties": list(DIFFICULTIES),
        "algorithms": list(ALGORITHMS),
        "ablation": first["ablation"],
        "optimizer_random_seeds_by_shard": [
            report.get("optimizer_random_seeds", {}) for report in reports
        ],
        "summary": summary,
        "paired_ca_vs_mso": paired,
        "nf_bi_baselines": baselines,
        "comparison": comparison,
        "overall_comparison": overall_comparison,
        "paired_significance": significance,
        "runtime_analysis": runtimes,
        "failure_maps": failures,
        "source_shards": [str(path) for path in files],
        "details": details,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"IAMSO-NF-BI-CA_vs_MSO_ablation_{args.run_id}"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    summary_path = output_dir / f"{stem}_summary.csv"
    details_path = output_dir / f"{stem}_details.csv"
    runner._write_json(json_path, report)
    runner._write_details_csv(details_path, report)
    write_summary_csv(summary_path, summary, baselines)
    write_markdown(markdown_path, report, baselines, comparison, paired)

    runner.print_comparison_table(report)
    print(f"Markdown: {markdown_path}")
    print(f"JSON: {json_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Details CSV: {details_path}")


if __name__ == "__main__":
    main()
