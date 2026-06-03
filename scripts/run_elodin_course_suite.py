#!/usr/bin/env python3
"""Run selected Elodin course variants and summarize completion results."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_EDITOR = REPO_ROOT / "scripts" / "run_elodin_editor.sh"

RACE_RE = re.compile(
    r"\[RACE\]\s+course=(?P<course>\S+)\s+"
    r"gates_passed=(?P<passed>\d+)/(?P<total>\d+)\s+"
    r"lap_time=(?P<lap_time>[0-9.]+)s\s+"
    r"status=(?P<status>\S+)\s+"
    r"pass_times=\[(?P<pass_times>[^\]]*)\]"
)


def cleanup_stale_processes() -> None:
    patterns = [
        "elodin editor sim/main.py",
        "elodin run sim/main.py",
        "elodin render-server",
        "betaflight_SITL",
    ]
    for pattern in patterns:
        subprocess.run(
            ["pkill", "-f", pattern],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def parse_race_summary(log_path: Path) -> dict[str, str]:
    if not log_path.exists():
        return {}

    summary = None
    for line in log_path.read_text(errors="replace").splitlines():
        if "[RACE]" in line:
            summary = line
    if summary is None:
        return {}

    match = RACE_RE.search(summary)
    if match is None:
        return {"race_summary": summary}

    out = match.groupdict()
    out["race_summary"] = summary
    out["completed"] = str(out["status"] == "COMPLETE")
    return out


def run_course(
    *,
    camera_profile: str,
    course: str,
    out_root: Path,
    sim_time: float,
    timeout_s: float,
    idle_timeout_s: float,
    frame_stride: int,
) -> dict[str, str]:
    run_dir = (out_root / camera_profile / course).resolve()
    trace_path = run_dir / "trace.csv"
    frame_dir = run_dir / "frames"
    log_path = run_dir / "editor.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)
    trace_path.unlink(missing_ok=True)
    if frame_dir.exists():
        shutil.rmtree(frame_dir)

    env = os.environ.copy()
    env.update(
        {
            "ELODIN_COURSE": course,
            "ELODIN_CAMERA_PROFILE": camera_profile,
            "ELODIN_SIM_TIME": f"{sim_time:g}",
            "TRACE_PATH": str(trace_path),
            "FRAME_DIR": str(frame_dir),
            "FRAME_STRIDE": str(frame_stride),
            "LOG_PATH": str(log_path),
            "CLEAR_EDITOR_LOGS": "1",
            "CLEANUP_STALE_PROCESSES": "1",
        }
    )

    cleanup_stale_processes()
    start = time.monotonic()
    proc = subprocess.Popen(
        [str(RUN_EDITOR)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    saw_race_summary = False
    timed_out = False
    idle_timed_out = False
    log_pos = 0
    last_log_update = start
    try:
        while True:
            if log_path.exists():
                with log_path.open(errors="replace") as f:
                    f.seek(log_pos)
                    for line in f:
                        last_log_update = time.monotonic()
                        print(f"[{camera_profile}/{course}] {line}", end="", flush=True)
                        if "[RACE]" in line:
                            saw_race_summary = True
                    log_pos = f.tell()
            if saw_race_summary:
                break
            if proc.poll() is not None:
                break
            if time.monotonic() - start > timeout_s:
                timed_out = True
                break
            if time.monotonic() - last_log_update > idle_timeout_s:
                idle_timed_out = True
                break
            time.sleep(1.0)

        if saw_race_summary:
            try:
                proc.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                pass

        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=4.0)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=4.0)
    finally:
        cleanup_stale_processes()

    elapsed_s = time.monotonic() - start
    result = {
        "camera_profile": camera_profile,
        "course": course,
        "elapsed_wall_s": f"{elapsed_s:.1f}",
        "timed_out": str(timed_out),
        "idle_timed_out": str(idle_timed_out),
        "trace": str(trace_path),
        "log": str(log_path),
        "frames": str(frame_dir),
    }
    result.update(parse_race_summary(log_path))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--courses",
        default="easy,lateral_soft,low_high,four_gate_straight",
        help="Comma-separated ELODIN_COURSE names.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("logs/elodin_course_suite"))
    parser.add_argument(
        "--camera-profiles",
        default=os.environ.get("ELODIN_CAMERA_PROFILE", "vq1_pinhole"),
        help="Comma-separated camera profiles, e.g. vq1_pinhole,gatenet_fisheye.",
    )
    parser.add_argument("--sim-time", type=float, default=12.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--idle-timeout-s", type=float, default=45.0)
    parser.add_argument("--frame-stride", type=int, default=3)
    args = parser.parse_args()

    courses = [course.strip() for course in args.courses.split(",") if course.strip()]
    if not courses:
        raise ValueError("at least one course is required")
    camera_profiles = [
        profile.strip()
        for profile in args.camera_profiles.split(",")
        if profile.strip()
    ]
    if not camera_profiles:
        raise ValueError("at least one camera profile is required")

    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for camera_profile in camera_profiles:
        for course in courses:
            rows.append(
                run_course(
                    camera_profile=camera_profile,
                    course=course,
                    out_root=args.out_dir,
                    sim_time=args.sim_time,
                    timeout_s=args.timeout_s,
                    idle_timeout_s=args.idle_timeout_s,
                    frame_stride=max(1, args.frame_stride),
                )
            )

    summary_path = args.out_dir / "summary.csv"
    fieldnames = [
        "camera_profile",
        "course",
        "completed",
        "status",
        "passed",
        "total",
        "lap_time",
        "pass_times",
        "timed_out",
        "idle_timed_out",
        "elapsed_wall_s",
        "race_summary",
        "trace",
        "log",
        "frames",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"summary={summary_path}", flush=True)
    for row in rows:
        status = row.get("status", "NO_SUMMARY")
        passed = row.get("passed", "?")
        total = row.get("total", "?")
        lap = row.get("lap_time", "?")
        print(
            f"{row['camera_profile']}/{row['course']}: "
            f"{status} gates={passed}/{total} lap_time={lap}",
            flush=True,
        )

    failures = [row for row in rows if row.get("status") != "COMPLETE"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
