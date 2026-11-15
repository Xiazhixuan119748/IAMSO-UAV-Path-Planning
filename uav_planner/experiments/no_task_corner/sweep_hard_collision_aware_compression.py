"""Compare collision-aware seed compression on failed hard maps."""

from __future__ import annotations

import csv
import json
import time

from uav_planner.costs import no_task_model_cost
from uav_planner.geometry import collision_cells_for_path, collision_segment_count_for_path
from uav_planner.metaheuristic.encoding import decode, smooth_path

from .collision_aware_compression import collision_aware_encode, line_of_sight_simplify
from .dataset import load_map_bundle, no_task_corner_maps_dir, no_task_corner_results_dir
from .diagnose_iamso_nf_bi_failures import FAILED_HARD_IDS
from .seed_ablation import build_bidirectional_seed_paths


WAYPOINT_COUNTS = (6, 10, 12, 16)
STRICTLY_UNREACHABLE = {"hard_094", "hard_097"}


def _is_safe(passable, path) -> bool:
    return (
        bool(path)
        and collision_segment_count_for_path(passable, path) == 0
        and not collision_cells_for_path(passable, path)
    )


def main() -> None:
    details: list[dict[str, object]] = []
    for index, map_id in enumerate(FAILED_HARD_IDS, start=1):
        print(f"[{index}/{len(FAILED_HARD_IDS)}] {map_id}", flush=True)
        if map_id in STRICTLY_UNREACHABLE:
            print("  skipped: strictly unreachable", flush=True)
            continue

        layers, passable, meta = load_map_bundle(no_task_corner_maps_dir() / f"{map_id}.npz")
        start = tuple(meta["start"])
        goal = tuple(meta["goal"])
        seed_paths, seed_labels = build_bidirectional_seed_paths(
            passable, layers, no_task_model_cost, start, goal
        )
        for waypoint_count in WAYPOINT_COUNTS:
            encodable = 0
            decoded_safe = 0
            internal_safe = 0
            required_counts = []
            for label, path in zip(seed_labels, seed_paths):
                required_waypoints = max(0, len(line_of_sight_simplify(path, passable)) - 2)
                required_counts.append(required_waypoints)
                encoded = collision_aware_encode(path, passable, waypoint_count)
                if encoded is None:
                    continue
                encodable += 1
                decoded = decode(encoded, start, goal)
                internally_smoothed = smooth_path(decoded)
                decoded_safe += int(_is_safe(passable, decoded))
                internal_safe += int(_is_safe(passable, internally_smoothed))
            details.append(
                {
                    "map_id": map_id,
                    "waypoint_count": waypoint_count,
                    "raw_seed_count": len(seed_paths),
                    "encodable_seed_count": encodable,
                    "decoded_safe_seed_count": decoded_safe,
                    "internal_safe_seed_count": internal_safe,
                    "map_decoded_feasible": decoded_safe > 0,
                    "map_internal_feasible": internal_safe > 0,
                    "minimum_required_waypoints": min(required_counts) if required_counts else None,
                }
            )
            print(
                f"  w={waypoint_count}: encodable={encodable}/{len(seed_paths)} "
                f"decoded={decoded_safe}/{len(seed_paths)} internal={internal_safe}/{len(seed_paths)}",
                flush=True,
            )

    reachable_count = len(FAILED_HARD_IDS) - len(STRICTLY_UNREACHABLE)
    summary = []
    for waypoint_count in WAYPOINT_COUNTS:
        rows = [row for row in details if row["waypoint_count"] == waypoint_count]
        raw_count = sum(int(row["raw_seed_count"]) for row in rows)
        encodable_count = sum(int(row["encodable_seed_count"]) for row in rows)
        decoded_count = sum(int(row["decoded_safe_seed_count"]) for row in rows)
        internal_count = sum(int(row["internal_safe_seed_count"]) for row in rows)
        decoded_maps = sum(bool(row["map_decoded_feasible"]) for row in rows)
        internal_maps = sum(bool(row["map_internal_feasible"]) for row in rows)
        summary.append(
            {
                "waypoint_count": waypoint_count,
                "reachable_failed_maps": reachable_count,
                "maps_with_decoded_safe_seed": decoded_maps,
                "maps_with_internal_safe_seed": internal_maps,
                "decoded_map_coverage": decoded_maps / reachable_count,
                "internal_map_coverage": internal_maps / reachable_count,
                "raw_seed_count": raw_count,
                "encodable_seed_count": encodable_count,
                "decoded_safe_seed_count": decoded_count,
                "internal_safe_seed_count": internal_count,
                "encodable_seed_rate": encodable_count / raw_count if raw_count else 0.0,
                "internal_safe_seed_rate": internal_count / raw_count if raw_count else 0.0,
            }
        )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = no_task_corner_results_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"IAMSO-NF-BI_困难失败地图碰撞感知压缩消融_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(
            {
                "waypoint_counts": list(WAYPOINT_COUNTS),
                "strictly_unreachable": sorted(STRICTLY_UNREACHABLE),
                "summary": summary,
                "details": details,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)

    lines = [
        "# IAMSO-NF-BI 困难失败地图碰撞感知压缩消融",
        "",
        f"- 严格可达失败地图：{reachable_count} 张",
        f"- 严格不可达并跳过：{', '.join(sorted(STRICTLY_UNREACHABLE))}",
        "- 压缩策略：保留线段可见性所需拐点，再沿安全线段补足固定航点数。",
        "- 本实验不运行完整 IAMSO。",
        "",
        "| 中间航点数 | 压缩折线覆盖地图 | 内部平滑覆盖地图 | 可编码种子率 | 内部平滑安全种子率 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['waypoint_count']} | {row['maps_with_decoded_safe_seed']}/{reachable_count} "
            f"| {row['maps_with_internal_safe_seed']}/{reachable_count} "
            f"| {row['encodable_seed_rate']:.2%} | {row['internal_safe_seed_rate']:.2%} |"
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown: {markdown_path}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
