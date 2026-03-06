from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArgSpec:
    key: str
    type: str  # number|text|boolean|select
    default: Any = None
    options: list[Any] | None = None
    help: str | None = None


def _repo_root() -> Path:
    # fim/app/schema_generator.py -> fim/app -> fim -> repo root
    return Path(__file__).resolve().parents[2]


def _schema_path() -> Path:
    return _repo_root() / "fim" / "app" / "schemas" / "fim_params.schema.json"


def _literal(node: ast.AST) -> Any:
    """Best-effort literal evaluator for simple AST nodes."""
    # Handle common non-literal nodes we care about (e.g. type=float/int).
    if isinstance(node, ast.Name):
        return node.id
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _infer_ui_type(arg_type: Any, action: str | None, choices: list[Any] | None) -> str:
    if action in {"store_true", "store_false"}:
        return "boolean"
    if choices:
        return "select"
    if arg_type in (int, float) or arg_type in {"int", "float"}:
        return "number"
    return "text"


def scan_argparse_file(py_path: Path) -> dict[str, ArgSpec]:
    """Scan a Python file for argparse add_argument calls."""
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))

    found: dict[str, ArgSpec] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # match: <something>.add_argument(...)
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue

        # First positional arg should be '--foo' (we prefer long option)
        long_opt = None
        for a in node.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--"):
                long_opt = a.value
                break
        if not long_opt:
            continue

        # Extract kwargs
        kwargs: dict[str, Any] = {}
        for kw in node.keywords:
            if not kw.arg:
                continue
            kwargs[kw.arg] = _literal(kw.value)

        dest = kwargs.get("dest")
        key = dest if isinstance(dest, str) and dest else long_opt.lstrip("-").replace("-", "_")

        choices = kwargs.get("choices")
        if isinstance(choices, (tuple, list)):
            choices_list = list(choices)
        else:
            choices_list = None

        action = kwargs.get("action") if isinstance(kwargs.get("action"), str) else None

        arg_type = kwargs.get("type")
        if isinstance(arg_type, str):
            arg_type_val: Any = arg_type
        else:
            arg_type_val = arg_type  # may be None if not literal-evaluable

        ui_type = _infer_ui_type(arg_type_val, action, choices_list)

        default = kwargs.get("default", None)
        help_str = kwargs.get("help") if isinstance(kwargs.get("help"), str) else None

        found[key] = ArgSpec(
            key=key,
            type=ui_type,
            default=default,
            options=choices_list,
            help=help_str,
        )

    return found


def _index_params(param_list: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {p.get("key"): p for p in param_list if isinstance(p, dict) and isinstance(p.get("key"), str)}


def _update_param(existing: dict[str, Any], scanned: ArgSpec) -> None:
    # Keep existing label / grouping; update fields that come from code.
    # Don't downgrade UI types like file/number to plain text.
    existing_type = existing.get("type")
    if existing_type in {"file", "dir"}:
        pass
    elif scanned.type != "text" or existing_type not in {"number", "boolean", "select"}:
        existing["type"] = scanned.type

    # Sync select options with scanned choices.
    if scanned.options is not None and scanned.type == "select":
        existing_opts = existing.get("options")
        if isinstance(existing_opts, list) and any(isinstance(x, dict) for x in existing_opts):
            # Existing uses value/label dicts — preserve them, but add any new choices
            existing_values = {_option_value(o) for o in existing_opts}
            for choice in scanned.options:
                sc = str(choice)
                if sc not in existing_values:
                    existing_opts.append({"value": sc, "label": sc})
        else:
            existing["options"] = scanned.options

    # Only overwrite defaults when we can actually read a default from code.
    if scanned.default is not None:
        existing["default"] = scanned.default
    if scanned.help and "label" not in existing:
        existing["label"] = scanned.help


def merge_step_params(step_obj: dict[str, Any], scanned: dict[str, ArgSpec], *, add_new_to: str = "advanced") -> None:
    """Merge scanned args into a schema step object in-place.

    Strategy:
    - Update any existing params (by key) with scanned type/default/options.
    - Add newly discovered params not present in schema to the chosen list (default: advanced).
    """

    # Determine where params live in this step
    def _lists(container: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return list(container.get("essential", [])), list(container.get("advanced", []))

    # Collect all param containers we want to update
    containers: list[dict[str, Any]] = []
    if "essential" in step_obj or "advanced" in step_obj:
        containers.append(step_obj)
    if isinstance(step_obj.get("common"), dict):
        containers.append(step_obj["common"])
    if isinstance(step_obj.get("methods"), dict):
        for m in step_obj["methods"].values():
            if isinstance(m, dict):
                containers.append(m)

    # Build index of existing params
    existing_index: dict[str, dict[str, Any]] = {}
    for c in containers:
        ess, adv = _lists(c)
        existing_index.update(_index_params(ess))
        existing_index.update(_index_params(adv))

    # Update existing
    for k, spec in scanned.items():
        if k in existing_index:
            _update_param(existing_index[k], spec)

    # Add new
    for k, spec in scanned.items():
        if k in existing_index:
            continue
        target = step_obj
        if isinstance(step_obj.get("common"), dict):
            target = step_obj["common"]
        if add_new_to not in ("essential", "advanced"):
            add_new_to = "advanced"
        target.setdefault(add_new_to, []).append(
            {
                "key": spec.key,
                "label": spec.key,
                "type": spec.type,
                "default": spec.default,
                **({"options": spec.options} if spec.options and spec.type == "select" else {}),
            }
        )


def _option_value(opt: Any) -> str | None:
    """Extract the string value from a schema option (plain string, int, or {value, label} dict)."""
    if isinstance(opt, (str, int, float)):
        return str(opt)
    if isinstance(opt, dict):
        v = opt.get("value")
        return str(v) if v is not None else None
    return None


def _prune_select_options(param: dict[str, Any], valid_choices: list[Any]) -> None:
    """Remove select options that are no longer in the scanned argparse choices."""
    opts = param.get("options")
    if not isinstance(opts, list):
        return
    valid_set = {str(c) for c in valid_choices}
    param["options"] = [o for o in opts if _option_value(o) in valid_set]


def sync_step_params(step_obj: dict[str, Any], scanned: dict[str, ArgSpec]) -> list[str]:
    """Prune schema entries that no longer exist in the scanned argparse definitions.

    - Removes select options not in scanned choices.
    - Removes ``methods`` blocks whose key is not in any scanned select choices.

    Returns a list of human-readable messages describing what was pruned.
    """
    messages: list[str] = []

    # --- Collect all param containers ---
    def _all_params(container: dict[str, Any]) -> list[dict[str, Any]]:
        return list(container.get("essential", [])) + list(container.get("advanced", []))

    containers: list[dict[str, Any]] = []
    if "essential" in step_obj or "advanced" in step_obj:
        containers.append(step_obj)
    if isinstance(step_obj.get("common"), dict):
        containers.append(step_obj["common"])

    # --- 1. Prune select options ---
    all_select_choices: dict[str, set[str]] = {}  # key -> valid choice values
    for c in containers:
        for p in _all_params(c):
            key = p.get("key")
            if not key or p.get("type") != "select":
                continue
            spec = scanned.get(key)
            if spec and spec.options is not None:
                before = [_option_value(o) for o in (p.get("options") or [])]
                _prune_select_options(p, spec.options)
                after = [_option_value(o) for o in (p.get("options") or [])]
                removed = set(before) - set(after)
                if removed:
                    messages.append(f"  select '{key}': removed options {sorted(removed)}")
                all_select_choices[key] = {str(c) for c in spec.options}

    # --- 2. Prune methods blocks ---
    methods = step_obj.get("methods")
    if isinstance(methods, dict):
        # Find which select param's choices govern the methods keys
        # (match: a select param whose option values overlap with methods keys)
        valid_method_keys: set[str] | None = None
        for key, choices in all_select_choices.items():
            if choices & set(methods.keys()):
                valid_method_keys = choices
                break

        if valid_method_keys is not None:
            stale = [mk for mk in list(methods.keys()) if mk not in valid_method_keys]
            for mk in stale:
                del methods[mk]
                messages.append(f"  methods: removed block '{mk}'")

    return messages


def generate(write: bool = False, sync: bool = False) -> dict[str, Any]:
    schema_path = _schema_path()
    data = json.loads(schema_path.read_text(encoding="utf-8"))

    steps = data.get("steps", {})
    if not isinstance(steps, dict):
        raise ValueError("schema.steps must be an object")

    # Map steps -> python files to scan
    scan_map = {
        "tracking": _repo_root() / "fim" / "refactor" / "deformation_tracking.py",
        "inverse": _repo_root() / "fim" / "refactor" / "main_VFM.py",
        # distortion has no runnable script in this repo snapshot
    }

    for step_id, py_path in scan_map.items():
        if step_id not in steps:
            continue
        if not py_path.exists():
            continue
        scanned = scan_argparse_file(py_path)
        merge_step_params(steps[step_id], scanned, add_new_to="advanced")
        if sync:
            msgs = sync_step_params(steps[step_id], scanned)
            for m in msgs:
                print(f"[sync] {step_id}:\n{m}")

    if write:
        schema_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    return data


def main() -> None:
    p = argparse.ArgumentParser(description="Generate/refresh fim_params.schema.json from argparse definitions.")
    p.add_argument("--write", action="store_true", help="Write updates back to fim/app/schemas/fim_params.schema.json")
    p.add_argument(
        "--sync",
        action="store_true",
        help="Also prune stale options and method blocks that no longer exist in argparse choices",
    )
    args = p.parse_args()

    out = generate(write=args.write, sync=args.sync)
    if not args.write:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
