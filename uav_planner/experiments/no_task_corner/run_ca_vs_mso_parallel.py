"""Run the complete CA-vs-MSO benchmark with resumable process shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

from .dataset import no_task_corner_results_dir


SHARDS = (
    ("simple", 1, 1),
    ("simple", 2, 34),
    ("simple", 35, 67),
    ("simple", 68, 100),
    ("medium", 1, 34),
    ("medium", 35, 67),
    ("medium", 68, 100),
    ("hard", 1, 17),
    ("hard", 18, 34),
    ("hard", 35, 50),
    ("hard", 51, 67),
    ("hard", 68, 83),
    ("hard", 84, 100),
)


@dataclass(frozen=True)
class Shard:
    difficulty: str
    start_index: int
    end_index: int

    @property
    def label(self) -> str:
        return f"{self.difficulty}_{self.start_index:03d}-{self.end_index:03d}"

    @property
    def map_ids(self) -> set[str]:
        return {
            f"{self.difficulty}_{index:03d}"
            for index in range(self.start_index, self.end_index + 1)
        }


@dataclass
class Worker:
    shard: Shard
    process: subprocess.Popen[str]
    stdout_path: Path
    stderr_path: Path
    stdout_handle: object
    stderr_handle: object
    started_at: float


def benchmark_shards() -> tuple[Shard, ...]:
    shards = tuple(Shard(*values) for values in SHARDS)
    for difficulty in ("simple", "medium", "hard"):
        counts = Counter(
            map_id
            for shard in shards
            if shard.difficulty == difficulty
            for map_id in shard.map_ids
        )
        expected = {f"{difficulty}_{index:03d}" for index in range(1, 101)}
        if set(counts) != expected or any(count != 1 for count in counts.values()):
            raise RuntimeError(f"invalid shard partition for {difficulty}")
    return shards


def single_map_shards() -> tuple[Shard, ...]:
    """Return one resumable shard per map for interruption-safe serial runs."""
    return tuple(
        Shard(difficulty, index, index)
        for difficulty in ("simple", "medium", "hard")
        for index in range(1, 101)
    )


def result_path(output_dir: Path, run_id: str, shard: Shard) -> Path:
    return output_dir / (
        f"IAMSO-NF-BI-CA_vs_MSO_{run_id}_{shard.difficulty}_"
        f"{shard.start_index:03d}-{shard.end_index:03d}.json"
    )


def _validate_completed_result(path: Path, run_id: str, shard: Shard) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    actual_shard = report.get("shard", {})
    expected = {
        "difficulty": shard.difficulty,
        "start_index": shard.start_index,
        "end_index": shard.end_index,
        "map_ids": sorted(shard.map_ids),
    }
    actual = {
        "difficulty": actual_shard.get("difficulty"),
        "start_index": actual_shard.get("start_index"),
        "end_index": actual_shard.get("end_index"),
        "map_ids": sorted(actual_shard.get("map_ids", [])),
    }
    if report.get("run_id") != run_id or actual != expected:
        raise RuntimeError(f"existing result does not match requested shard: {path}")


def _write_manifest(
    path: Path,
    run_id: str,
    completed: list[str],
    running: dict[str, Worker],
    failed: dict[str, int],
) -> None:
    payload = {
        "run_id": run_id,
        "completed": sorted(completed),
        "running": {
            label: {
                "pid": worker.process.pid,
                "stdout": str(worker.stdout_path),
                "stderr": str(worker.stderr_path),
            }
            for label, worker in sorted(running.items())
        },
        "failed": failed,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _launch_worker(shard: Shard, run_id: str, log_dir: Path) -> Worker:
    stdout_path = log_dir / f"{shard.label}.stdout.log"
    stderr_path = log_dir / f"{shard.label}.stderr.log"
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    command = [
        sys.executable,
        "-u",
        "-m",
        "uav_planner.experiments.no_task_corner.run_ca_vs_mso_shard",
        "--difficulty",
        shard.difficulty,
        "--start-index",
        str(shard.start_index),
        "--end-index",
        str(shard.end_index),
        "--run-id",
        run_id,
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        creationflags=creationflags,
    )
    return Worker(
        shard=shard,
        process=process,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
        started_at=time.monotonic(),
    )


def run_parallel(
    run_id: str,
    max_workers: int,
    poll_seconds: float = 2.0,
    shards: tuple[Shard, ...] | None = None,
    manifest_name: str = "manifest.json",
) -> None:
    output_dir = no_task_corner_results_dir() / "ablation_shards"
    log_dir = output_dir / "logs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = log_dir / manifest_name

    selected_shards = shards or benchmark_shards()
    pending: deque[Shard] = deque()
    completed: list[str] = []
    for shard in selected_shards:
        path = result_path(output_dir, run_id, shard)
        if path.exists():
            _validate_completed_result(path, run_id, shard)
            completed.append(shard.label)
        else:
            pending.append(shard)

    running: dict[str, Worker] = {}
    failed: dict[str, int] = {}
    started_at = time.monotonic()
    last_progress = 0.0
    try:
        while pending or running:
            while pending and len(running) < max_workers:
                shard = pending.popleft()
                worker = _launch_worker(shard, run_id, log_dir)
                running[shard.label] = worker
                print(f"START {shard.label} pid={worker.process.pid}", flush=True)

            for label, worker in list(running.items()):
                return_code = worker.process.poll()
                if return_code is None:
                    continue
                worker.stdout_handle.close()
                worker.stderr_handle.close()
                del running[label]
                output_path = result_path(output_dir, run_id, worker.shard)
                if return_code == 0 and output_path.exists():
                    _validate_completed_result(output_path, run_id, worker.shard)
                    completed.append(label)
                    elapsed = time.monotonic() - worker.started_at
                    print(f"DONE  {label} elapsed={elapsed:.1f}s", flush=True)
                else:
                    failed[label] = int(return_code)
                    print(
                        f"FAIL  {label} code={return_code} stderr={worker.stderr_path}",
                        flush=True,
                    )

            now = time.monotonic()
            if now - last_progress >= 30.0:
                print(
                    f"PROGRESS completed={len(completed)}/{len(selected_shards)} "
                    f"running={len(running)} pending={len(pending)} "
                    f"elapsed={now - started_at:.1f}s",
                    flush=True,
                )
                last_progress = now
            _write_manifest(manifest_path, run_id, completed, running, failed)
            if pending or running:
                time.sleep(poll_seconds)
    finally:
        for worker in running.values():
            worker.stdout_handle.close()
            worker.stderr_handle.close()
        _write_manifest(manifest_path, run_id, completed, running, failed)

    if failed:
        raise RuntimeError(f"benchmark shards failed: {failed}")
    print(
        f"COMPLETE run_id={run_id} shards={len(completed)} "
        f"elapsed={time.monotonic() - started_at:.1f}s",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the sharded CA-vs-MSO benchmark")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(12, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--single-map-shards",
        action="store_true",
        help="checkpoint every map in a separate result file",
    )
    parser.add_argument(
        "--difficulty",
        choices=("simple", "medium", "hard"),
        help="run only one difficulty queue",
    )
    args = parser.parse_args(argv)
    if args.max_workers <= 0:
        parser.error("max-workers must be positive")
    shards = single_map_shards() if args.single_map_shards else benchmark_shards()
    if args.difficulty:
        shards = tuple(shard for shard in shards if shard.difficulty == args.difficulty)
    manifest_name = (
        f"manifest_{args.difficulty}.json" if args.difficulty else "manifest.json"
    )
    run_parallel(
        args.run_id,
        args.max_workers,
        shards=shards,
        manifest_name=manifest_name,
    )


if __name__ == "__main__":
    main()
