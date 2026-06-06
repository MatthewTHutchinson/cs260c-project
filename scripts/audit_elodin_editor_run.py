#!/usr/bin/env python3
"""Classify the latest Elodin editor run from local debug artifacts."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


IMAGE_EXTS = {".ppm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
INTERESTING_LOG_TOKENS = (
    "[FPV] First frame",
    "[GATE]",
    "[RACE]",
    "[FPV] render",
    "[FPV] collect",
    "[SOLVER] error",
    "Traceback",
    "panic",
    "error",
    "failed",
)


def count_frames(frame_dir: Path) -> int:
    if not frame_dir.exists() or not frame_dir.is_dir():
        return 0
    return sum(
        1
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def read_trace(path: Path) -> tuple[list[dict[str, str]], Counter[str], int]:
    if not path.exists():
        return [], Counter(), 0
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    modes = Counter(row.get("mode", "") for row in rows)
    fresh = sum(1 for row in rows if row.get("frame_fresh") == "1")
    return rows, modes, fresh


def interesting_log_lines(path: Path, limit: int = 25) -> list[str]:
    if not path.exists():
        return []
    matches: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        lowered = line.lower()
        if "bevy.org/learn/errors/b0004" in lowered:
            continue
        if any(token.lower() in lowered for token in INTERESTING_LOG_TOKENS):
            matches.append(line)
            if len(matches) >= limit:
                break
    return matches


def race_summary(path: Path) -> str | None:
    if not path.exists():
        return None
    summary = None
    for line in path.read_text(errors="replace").splitlines():
        if "[RACE]" in line:
            summary = line
    return summary


def gates_passed(summary: str | None) -> tuple[int, int] | None:
    if summary is None:
        return None
    marker = "gates_passed="
    if marker not in summary:
        return None
    tail = summary.split(marker, 1)[1].split(maxsplit=1)[0]
    try:
        passed, total = tail.split("/", 1)
        return int(passed), int(total)
    except ValueError:
        return None


def trace_progress(rows: list[dict[str, str]]) -> dict[str, int] | None:
    if not rows:
        return None

    progress: dict[str, int] = {}
    for key in ("last_gate_passed", "next_gate_index"):
        values = []
        for row in rows:
            raw = row.get(key)
            if raw in (None, ""):
                continue
            try:
                values.append(int(raw))
            except ValueError:
                continue
        if values:
            progress[f"latest_{key}"] = values[-1]
            progress[f"max_{key}"] = max(values)

    return progress or None


def command_source_counts(
    rows: list[dict[str, str]],
) -> tuple[Counter[str], dict[str, Counter[str]]]:
    sources: Counter[str] = Counter()
    by_mode: dict[str, Counter[str]] = {}
    for row in rows:
        source = row.get("command_source", "")
        if not source:
            continue
        mode = row.get("mode", "")
        sources[source] += 1
        by_mode.setdefault(mode, Counter())[source] += 1
    return sources, by_mode


def classify(
    rows: list[dict[str, str]],
    fresh_frames: int,
    saved_frames: int,
    modes: Counter[str],
    race: tuple[int, int] | None,
) -> str:
    if not rows:
        return (
            "NO_TRACE: the editor did not produce solver trace rows. It may have "
            "stopped before takeoff, failed before loading the CS260C solver, or "
            "written to a different TRACE_PATH."
        )

    active_modes = {mode for mode, count in modes.items() if mode and count > 0}
    non_search_modes = active_modes - {"search"}

    if fresh_frames == 0 and saved_frames == 0:
        return (
            "FPV_HANDOFF_FAILURE: the solver ran, but every row had frame_fresh=0 "
            "and no FPV frames were saved. The visible spin is the controller's "
            "search yaw fallback, not a detector or navigation decision yet."
        )

    if fresh_frames > 0 and not non_search_modes:
        return (
            "CV_DETECTION_FAILURE: FPV frames reached the solver, but the tracker "
            "never left search. Inspect saved frames and detector masks next."
        )

    if race is not None and race[0] == race[1] and race[1] > 0:
        return (
            "COURSE_COMPLETE: the run passed every gate reported by the harness."
        )

    if race is not None and race[0] > 0:
        return (
            "PARTIAL_COURSE_PROGRESS: the tracker and controller passed at least "
            "one gate. Remaining work is next-gate reacquisition, lateral control, "
            "or speed/altitude tuning."
        )

    if non_search_modes:
        return (
            "CONTROL_OR_NAVIGATION_FAILURE: the tracker produced gate estimates, "
            "so remaining spinning/flyaway behavior is likely command sign, gain, "
            "or FOV-aware navigation tuning."
        )

    return "INCONCLUSIVE: artifacts exist, but the failure mode is not obvious."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("logs/elodin_pilot_trace_editor.csv"),
        help="Pilot trace CSV written by scripts/run_elodin_editor.sh.",
    )
    parser.add_argument(
        "--frame-dir",
        type=Path,
        default=Path("logs/elodin_fpv_frames"),
        help="Directory where fresh FPV frames are saved.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("logs/elodin_editor_stdout.log"),
        help="Captured Elodin/editor stdout log.",
    )
    args = parser.parse_args()

    rows, modes, fresh_frames = read_trace(args.trace)
    saved_frames = count_frames(args.frame_dir)
    summary = race_summary(args.log)
    race = gates_passed(summary)
    verdict = classify(rows, fresh_frames, saved_frames, modes, race)

    print(f"trace={args.trace}")
    print(f"trace_rows={len(rows)}")
    print(f"frame_fresh_rows={fresh_frames}")
    print(f"saved_fpv_frames={saved_frames}")
    print(f"modes={dict(modes)}")
    if summary is not None:
        print(f"race_summary={summary}")
    if rows:
        print(f"first_tick={rows[0].get('tick', '')}")
        print(f"last_tick={rows[-1].get('tick', '')}")
        print(f"last_timestamp_s={rows[-1].get('timestamp_s', '')}")
        progress = trace_progress(rows)
        if progress is not None:
            for key, value in progress.items():
                print(f"{key}={value}")
        sources, by_mode = command_source_counts(rows)
        if sources:
            print(f"command_sources={dict(sources)}")
            print(
                "command_sources_by_mode="
                f"{ {mode: dict(counts) for mode, counts in by_mode.items()} }"
            )
    print(f"verdict={verdict}")

    log_lines = interesting_log_lines(args.log)
    if log_lines:
        print("interesting_log_lines:")
        for line in log_lines:
            print(f"  {line}")
    elif args.log.exists():
        print("interesting_log_lines=none")
    else:
        print(f"editor_log_missing={args.log}")


if __name__ == "__main__":
    main()
