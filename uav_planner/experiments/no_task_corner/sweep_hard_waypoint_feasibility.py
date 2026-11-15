"""Sweep hard-map seed feasibility across waypoint counts."""

from __future__ import annotations

import csv
import json
import time

from uav_planner.costs import no_task_model_cost
from uav_planner.geometry import collision_cells_for_path, collision_segment_count_for_path
from uav_planner.metaheuristic.encoding import decode, encode_from_path, smooth_path

from .dataset import load_map_bundle, no_task_corner_maps_dir, no_task_corner_results_dir
from .diagnose_iamso_nf_bi_failures import FAILED_HARD_IDS
from .seed_ablation import build_bidirectional_seed_paths


WAYPOINT_COUNTS = (6, 8, 10, 12)
STRICTLY_UNREACHABLE = {"hard_094", "hard_097"}


def _is_safe(passable, path) -> bool:
    return (
        bool(path)
        and collision_segment_count_for_path(passable, path) == 0
        and not collision_cells_for_path(passable, path)
    )


def main() -> None:
    detail_rows: list[dict[str, object]] = []
    for index, map_id in enumerate(FAILED_HARD_IDS, start=1):
        print(f"[{index}/{len(FAILED_HARD_IDS)}] {map_id}", flush=True)
        layers, passable, meta = load_map_bundle(no_task_corner_maps_dir() / f"{map_id}.npz")
        start = tuple(meta["start"])
        goal = tuple(meta["goal"])

        if map_id in STRICTLY_UNREACHABLE:
            for waypoint_count in WAYPOINT_COUNTS:
                detail_rows.append(
                    {
                        "map_id": map_id,
                        "waypoint_count": waypoint_count,
                        "raw_seed_count": 0,
                        "decoded_safe_seed_count": 0,
                        "internal_safe_seed_count": 0,
                        "map_decoded_feasible": False,
                        "map_internal_feasible": False,
                        "note": "strictly_unreachable",
                    }
                )
            print("  skipped: strictly unreachable under continuous collision checks", flush=True)
            continue

        seed_paths, _ = build_bidirectional_seed_paths(
            passable, layers, no_task_model_cost, start, goal
        )
        for waypoint_count in WAYPOINT_COUNTS:
            decoded_safe = 0
            internal_safe = 0
            for path in seed_paths:
                encoded = encode_from_path(path, waypoint_count)
                decoded = decode(encoded, start, goal)
                internally_smoothed = smooth_path(decoded)
                decoded_safe += int(_is_safe(passable, decoded))
                internal_safe += int(_is_safe(passable, internally_smoothed))
            detail_rows.append(
                {
                    "map_id": map_id,
                    "waypoint_count": waypoint_count,
                    "raw_seed_count": len(seed_paths),
                    "decoded_safe_seed_count": decoded_safe,
                    "internal_safe_seed_count": internal_safe,
                    "map_decoded_feasible": decoded_safe > 0,
                    "map_internal_feasible": internal_safe > 0,
                    "note": "",
                }
            )
            print(
                f"  w={waypoint_count}: decoded={decoded_safe}/{len(seed_paths)} "
                f"internal={internal_safe}/{len(seed_paths)}",
                flush=True,
            )

    summary_rows = []
    reachable_count = len(FAILED_HARD_IDS) - len(STRICTLY_UNREACHABLE)
    for waypoint_count in WAYPOINT_COUNTS:
        rows = [
            row
            for row in detail_rows
            if row["waypoint_count"] == waypoint_count and not row["note"]
        ]
        raw_seeds = sum(int(row["raw_seed_count"]) for row in rows)
        decoded_safe_seeds = sum(int(row["decoded_safe_seed_count"]) for row in rows)
        internal_safe_seeds = sum(int(row["internal_safe_seed_count"]) for row in rows)
        decoded_maps = sum(bool(row["map_decoded_feasible"]) for row in rows)
        internal_maps = sum(bool(row["map_internal_feasible"]) for row in rows)
        summary_rows.append(
            {
                "waypoint_count": waypoint_count,
                "reachable_failed_maps": reachable_count,
                "maps_with_decoded_safe_seed": decoded_maps,
                "maps_with_internal_safe_seed": internal_maps,
                "decoded_map_coverage": decoded_maps / reachable_count,
                "internal_map_coverage": internal_maps / reachable_count,
                "raw_seed_count": raw_seeds,
                "decoded_safe_seed_count": decoded_safe_seeds,
                "internal_safe_seed_count": internal_safe_seeds,
                "decoded_seed_rate": decoded_safe_seeds / raw_seeds if raw_seeds else 0.0,
                "internal_seed_rate": internal_safe_seeds / raw_seeds if raw_seeds else 0.0,
            }
        )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = no_task_corner_results_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"IAMSO-NF-BI_困难失败地图航点数扫描_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    markdown_path = output_dir / f"{stem}.md"
    payload = {
        "waypoint_counts": list(WAYPOINT_COUNTS),
        "failed_map_count": len(FAILED_HARD_IDS),
        "reachable_failed_map_count": reachable_count,
        "strictly_unreachable": sorted(STRICTLY_UNREACHABLE),
        "summary": summary_rows,
        "details": detail_rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    lines = [
        "# IAMSO-NF-BI 困难失败地图航点数扫描",
        "",
        f"- 失败地图：{len(FAILED_HARD_IDS)} 张",
        f"- 严格碰撞口径可达：{reachable_count} 张",
        f"- 严格不可达：{', '.join(sorted(STRICTLY_UNREACHABLE))}",
        "- 本扫描仅评价种子压缩和 IAMSO 内部平滑，不运行完整优化。",
        "",
        "| 中间航点数 | 压缩后至少一条安全种子的地图 | 内部平滑后至少一条安全种子的地图 | 压缩种子安全率 | 内部平滑种子安全率 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['waypoint_count']} | {row['maps_with_decoded_safe_seed']}/{reachable_count} "
            f"| {row['maps_with_internal_safe_seed']}/{reachable_count} "
            f"| {row['decoded_seed_rate']:.2%} | {row['internal_seed_rate']:.2%} |"
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown: {markdown_path}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
