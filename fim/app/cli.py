from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .pipeline_runner import run_step
from .steps_registry import get_step, list_steps, normalize_params


def _parse_kv(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise argparse.ArgumentTypeError("Expected key=value")
    k, v = s.split("=", 1)
    k = k.strip()
    v = v.strip()
    if v.lower() in {"true", "false"}:
        return k, (v.lower() == "true")
    # best-effort number parsing
    try:
        if "." in v or "e" in v.lower():
            return k, float(v)
        return k, int(v)
    except ValueError:
        return k, v


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="FIM pipeline CLI (schema-driven).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-steps", help="List available steps")

    ps = sub.add_parser("show-step", help="Show schema for a step")
    ps.add_argument("step_id", type=str)

    pr = sub.add_parser("run", help="Run a step with params")
    pr.add_argument("step_id", type=str)
    pr.add_argument(
        "--set",
        dest="kvs",
        action="append",
        default=[],
        help="Override param as key=value (repeatable)",
    )

    args = p.parse_args(argv)

    if args.cmd == "list-steps":
        for s in list_steps():
            print(f"{s.step_id}\t{s.title}")
        return 0

    if args.cmd == "show-step":
        step = get_step(args.step_id)
        payload = {
            "id": step.step_id,
            "title": step.title,
            "script": step.script,
            "essential": step.essential,
            "advanced": step.advanced,
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.cmd == "run":
        step = get_step(args.step_id)
        user_params: dict[str, Any] = {}
        for raw in args.kvs:
            k, v = _parse_kv(raw)
            user_params[k] = v
        params = normalize_params(step, user_params)
        res = run_step(step, params)
        print("ok:", res.ok)
        print("returncode:", res.returncode)
        print("command:", " ".join(res.command))
        print("\n--- stdout ---\n" + (res.stdout or ""))
        print("\n--- stderr ---\n" + (res.stderr or ""))
        return res.returncode

    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
