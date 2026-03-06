from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StepSpec:
    step_id: str
    title: str
    script: str | None
    essential: list[dict[str, Any]]
    advanced: list[dict[str, Any]]
    common: dict[str, Any] | None = None
    methods: dict[str, Any] | None = None


def _repo_root() -> Path:
    # fim/app/steps_registry.py -> fim/app -> fim -> repo root
    return Path(__file__).resolve().parents[2]


def load_schema_file() -> dict[str, Any]:
    # Prefer the app-owned schema location, but fall back to legacy path for backwards compatibility.
    schema_candidates = [
        _repo_root() / "fim" / "app" / "schemas" / "fim_params.schema.json",
        _repo_root() / "fim" / "legacy" / "fim_params.schema.json",
    ]
    schema_path = next((p for p in schema_candidates if p.exists()), None)
    if not schema_path:
        raise ValueError("Could not find fim_params.schema.json in app/schemas or legacy/")

    data = json.loads(schema_path.read_text(encoding="utf-8"))
    if "steps" not in data:
        raise ValueError(f"Invalid schema: missing 'steps' in {schema_path}")
    return data


def list_steps() -> list[StepSpec]:
    data = load_schema_file()
    steps = []
    for step_id, raw in data["steps"].items():
        steps.append(
            StepSpec(
                step_id=step_id,
                title=str(raw.get("title", step_id)),
                script=raw.get("script"),
                essential=list(raw.get("essential", [])),
                advanced=list(raw.get("advanced", [])),
                common=raw.get("common"),
                methods=raw.get("methods"),
            )
        )
    # stable ordering: distortion -> tracking -> inverse (if present)
    order = {"distortion": 0, "tracking": 1, "inverse": 2}
    steps.sort(key=lambda s: order.get(s.step_id, 999))
    return steps


def get_step(step_id: str) -> StepSpec:
    for s in list_steps():
        if s.step_id == step_id:
            return s
    raise KeyError(f"Unknown step_id: {step_id}")


def allowed_param_keys(step: StepSpec) -> set[str]:
    keys: set[str] = set()

    def _add(ps: list[dict[str, Any]]):
        for p in ps:
            k = p.get("key")
            if isinstance(k, str) and k:
                keys.add(k)

    _add(step.essential + step.advanced)
    if isinstance(step.common, dict):
        _add(list(step.common.get("essential", [])) + list(step.common.get("advanced", [])))
    if isinstance(step.methods, dict):
        for m in step.methods.values():
            if isinstance(m, dict):
                _add(list(m.get("essential", [])) + list(m.get("advanced", [])))
    return keys


def default_params(step: StepSpec) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def _defaults(ps: list[dict[str, Any]]):
        for p in ps:
            k = p.get("key")
            if isinstance(k, str) and k and "default" in p:
                out[k] = p.get("default")

    _defaults(step.essential + step.advanced)
    if isinstance(step.common, dict):
        _defaults(list(step.common.get("essential", [])) + list(step.common.get("advanced", [])))
    if isinstance(step.methods, dict):
        for m in step.methods.values():
            if isinstance(m, dict):
                _defaults(list(m.get("essential", [])) + list(m.get("advanced", [])))
    return out


def normalize_params(step: StepSpec, user_params: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults and drop unknown keys, so schema drives the interface.

    For steps that have method-specific schemas (e.g. Tracking), only include defaults/keys
    for the selected method to avoid passing unrelated placeholder params to the runnable script.
    """

    def _spec_lists_for_method(method: str | None) -> tuple[list[dict[str, Any]], set[str]]:
        ps: list[dict[str, Any]] = []
        allowed: set[str] = set()

        def _add(specs: list[dict[str, Any]]):
            for p in specs:
                ps.append(p)
                k = p.get("key")
                if isinstance(k, str) and k:
                    allowed.add(k)

        _add(step.essential + step.advanced)
        if isinstance(step.common, dict):
            _add(list(step.common.get("essential", [])) + list(step.common.get("advanced", [])))
        if method and isinstance(step.methods, dict):
            m = step.methods.get(method) or step.methods.get("physics") or {}
            if isinstance(m, dict):
                _add(list(m.get("essential", [])) + list(m.get("advanced", [])))
        return ps, allowed

    # Detect selected method from user params, otherwise from schema default.
    method: str | None = None
    if isinstance(step.methods, dict) and isinstance(step.common, dict):
        # Find the method selector key (e.g. "method" for tracking, "model" for inverse)
        method_key = "method"
        for p in list(step.common.get("essential", [])):
            if p.get("type") == "select" and p.get("key"):
                opts = p.get("options", [])
                opt_vals = {(o.get("value") if isinstance(o, dict) else str(o)) for o in opts}
                if opt_vals & set(step.methods.keys()):
                    method_key = p["key"]
                    break
        method = user_params.get(method_key)
        if not isinstance(method, str) or not method:
            for p in list(step.common.get("essential", [])):
                if p.get("key") == method_key and "default" in p:
                    method = p.get("default")
                    break

    specs, allowed = _spec_lists_for_method(method)

    # Defaults only from relevant specs
    out: dict[str, Any] = {}
    for p in specs:
        k = p.get("key")
        if isinstance(k, str) and k and "default" in p:
            out[k] = p.get("default")

    # Apply user overrides only for allowed keys
    for k, v in user_params.items():
        if k in allowed:
            out[k] = v
    return out
