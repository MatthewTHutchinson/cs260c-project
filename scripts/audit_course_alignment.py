#!/usr/bin/env python3
"""Compare active debug course geometry with the Elodin harness course file.

This guards against a repeat of the legacy-track problem where benchmark names
and actual track geometry drifted apart. The courses checked here are validation
surfaces only; passing this audit does not mean they match the official VQ1
course.
"""

from __future__ import annotations

import argparse
import ast
import math
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ACTIVE_COURSE_FILE = Path(__file__).resolve().parents[1] / "algorithm" / "course_library.py"
DEFAULT_ELODIN_COURSE_FILE = Path("/Users/matthewhutchinson/dev/elodin-ai-grand-prix/sim/course.py")


@dataclass(frozen=True)
class ParsedGate:
    index: int
    center: tuple[float, float, float]
    yaw_deg: float


def literal(node: ast.AST) -> object:
    return ast.literal_eval(node)


def assignment_name(node: ast.stmt) -> tuple[str, ast.AST] | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id, node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id, node.value
    return None


def course_key(constant_name: str) -> str:
    return constant_name.removesuffix("_COURSE").lower()


def parse_gate_call(node: ast.AST, *, source: Path, course: str) -> ParsedGate:
    if not isinstance(node, ast.Call):
        raise ValueError(f"{source}: {course} contains a non-call gate entry")
    if not isinstance(node.func, ast.Name) or node.func.id not in {"Gate", "CourseGate"}:
        raise ValueError(f"{source}: {course} uses unsupported gate constructor")
    if len(node.args) < 2:
        raise ValueError(f"{source}: {course} gate is missing index or center")

    index = int(literal(node.args[0]))
    center_raw = literal(node.args[1])
    if not isinstance(center_raw, tuple) or len(center_raw) != 3:
        raise ValueError(f"{source}: {course} gate center is not a 3-tuple")
    center = tuple(float(value) for value in center_raw)
    yaw_deg = 0.0
    for keyword in node.keywords:
        if keyword.arg == "yaw_deg":
            yaw_deg = float(literal(keyword.value))
    return ParsedGate(index=index, center=center, yaw_deg=yaw_deg)


def parse_courses(path: Path) -> dict[str, tuple[ParsedGate, ...]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    courses: dict[str, tuple[ParsedGate, ...]] = {}
    for node in tree.body:
        assignment = assignment_name(node)
        if assignment is None:
            continue
        name, value = assignment
        if not name.endswith("_COURSE"):
            continue
        if not isinstance(value, (ast.Tuple, ast.List)):
            continue
        key = course_key(name)
        courses[key] = tuple(parse_gate_call(item, source=path, course=key) for item in value.elts)
    return dict(sorted(courses.items()))


def close(a: float, b: float, tol: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def compare_courses(
    active: dict[str, tuple[ParsedGate, ...]],
    harness: dict[str, tuple[ParsedGate, ...]],
    *,
    position_tol_m: float,
    yaw_tol_deg: float,
) -> list[str]:
    failures: list[str] = []
    active_keys = set(active)
    harness_keys = set(harness)
    for missing in sorted(active_keys - harness_keys):
        failures.append(f"missing_in_harness course={missing}")
    for extra in sorted(harness_keys - active_keys):
        failures.append(f"extra_in_harness course={extra}")

    for key in sorted(active_keys & harness_keys):
        active_gates = active[key]
        harness_gates = harness[key]
        if len(active_gates) != len(harness_gates):
            failures.append(f"gate_count_mismatch course={key} active={len(active_gates)} harness={len(harness_gates)}")
            continue
        for active_gate, harness_gate in zip(active_gates, harness_gates):
            prefix = f"course={key} gate={active_gate.index}"
            if active_gate.index != harness_gate.index:
                failures.append(f"index_mismatch {prefix} harness_index={harness_gate.index}")
            for axis, av, hv in zip(("x", "y", "z"), active_gate.center, harness_gate.center):
                if not close(av, hv, position_tol_m):
                    failures.append(f"center_mismatch {prefix} axis={axis} active={av:.6f} harness={hv:.6f}")
            if not close(active_gate.yaw_deg, harness_gate.yaw_deg, yaw_tol_deg):
                failures.append(
                    f"yaw_mismatch {prefix} active={active_gate.yaw_deg:.6f} harness={harness_gate.yaw_deg:.6f}"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-course-file", type=Path, default=DEFAULT_ACTIVE_COURSE_FILE)
    parser.add_argument("--elodin-course-file", type=Path, default=DEFAULT_ELODIN_COURSE_FILE)
    parser.add_argument("--position-tol-m", type=float, default=1e-6)
    parser.add_argument("--yaw-tol-deg", type=float, default=1e-6)
    args = parser.parse_args()

    active = parse_courses(args.active_course_file)
    harness = parse_courses(args.elodin_course_file)
    failures = compare_courses(
        active,
        harness,
        position_tol_m=args.position_tol_m,
        yaw_tol_deg=args.yaw_tol_deg,
    )

    print(f"active_course_file={args.active_course_file}")
    print(f"elodin_course_file={args.elodin_course_file}")
    print(f"active_courses={','.join(active)}")
    print(f"elodin_courses={','.join(harness)}")
    if failures:
        print("verdict=FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("verdict=PASS")
    print("note=active debug course geometry matches the Elodin harness course constants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
