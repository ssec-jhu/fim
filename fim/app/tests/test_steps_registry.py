from __future__ import annotations

import json
from pathlib import Path

import pytest

from fim.app.steps_registry import (
    StepSpec,
    allowed_param_keys,
    default_params,
    get_step,
    list_steps,
    load_schema_file,
    normalize_params,
)


class TestLoadSchemaFile:
    def test_returns_dict_with_steps(self):
        data = load_schema_file()
        assert isinstance(data, dict)
        assert "steps" in data

    def test_steps_not_empty(self):
        data = load_schema_file()
        assert len(data["steps"]) > 0


class TestListSteps:
    def test_returns_list_of_step_specs(self):
        steps = list_steps()
        assert isinstance(steps, list)
        assert all(isinstance(s, StepSpec) for s in steps)

    def test_ordering(self):
        steps = list_steps()
        ids = [s.step_id for s in steps]
        if "distortion" in ids and "tracking" in ids:
            assert ids.index("distortion") < ids.index("tracking")
        if "tracking" in ids and "inverse" in ids:
            assert ids.index("tracking") < ids.index("inverse")

    def test_step_fields_populated(self):
        steps = list_steps()
        for s in steps:
            assert isinstance(s.step_id, str) and s.step_id
            assert isinstance(s.title, str) and s.title


class TestGetStep:
    def test_valid_step_id(self):
        step = get_step("tracking")
        assert step.step_id == "tracking"

    def test_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="nonexistent"):
            get_step("nonexistent")


class TestAllowedParamKeys:
    def test_returns_set_of_strings(self):
        step = get_step("tracking")
        keys = allowed_param_keys(step)
        assert isinstance(keys, set)
        assert all(isinstance(k, str) for k in keys)
        assert len(keys) > 0

    def test_includes_common_and_method_keys(self):
        step = get_step("inverse")
        keys = allowed_param_keys(step)
        assert "model" in keys or "data_path" in keys

    def test_basic_step(self):
        step = StepSpec(
            step_id="test",
            title="Test",
            script=None,
            essential=[{"key": "a", "type": "text"}],
            advanced=[{"key": "b", "type": "number"}],
        )
        assert allowed_param_keys(step) == {"a", "b"}

    def test_with_common_and_methods(self):
        step = StepSpec(
            step_id="test",
            title="Test",
            script=None,
            essential=[],
            advanced=[],
            common={"essential": [{"key": "c"}], "advanced": [{"key": "d"}]},
            methods={"m1": {"essential": [{"key": "e"}], "advanced": [{"key": "f"}]}},
        )
        assert allowed_param_keys(step) == {"c", "d", "e", "f"}


class TestDefaultParams:
    def test_returns_dict(self):
        step = get_step("tracking")
        defaults = default_params(step)
        assert isinstance(defaults, dict)

    def test_includes_defaults_from_schema(self):
        step = StepSpec(
            step_id="test",
            title="Test",
            script=None,
            essential=[{"key": "x", "type": "number", "default": 42}],
            advanced=[{"key": "y", "type": "text", "default": "hello"}],
        )
        assert default_params(step) == {"x": 42, "y": "hello"}

    def test_skips_params_without_default(self):
        step = StepSpec(
            step_id="test",
            title="Test",
            script=None,
            essential=[{"key": "x", "type": "text"}],
            advanced=[],
        )
        assert default_params(step) == {}

    def test_with_common_and_methods(self):
        step = StepSpec(
            step_id="test",
            title="Test",
            script=None,
            essential=[],
            advanced=[],
            common={"essential": [{"key": "a", "default": 1}], "advanced": []},
            methods={"m1": {"essential": [{"key": "b", "default": 2}], "advanced": []}},
        )
        d = default_params(step)
        assert d["a"] == 1
        assert d["b"] == 2


class TestNormalizeParams:
    def test_fills_defaults(self):
        step = StepSpec(
            step_id="test",
            title="Test",
            script=None,
            essential=[{"key": "x", "type": "number", "default": 10}],
            advanced=[],
        )
        result = normalize_params(step, {})
        assert result["x"] == 10

    def test_user_override(self):
        step = StepSpec(
            step_id="test",
            title="Test",
            script=None,
            essential=[{"key": "x", "type": "number", "default": 10}],
            advanced=[],
        )
        result = normalize_params(step, {"x": 99})
        assert result["x"] == 99

    def test_drops_unknown_keys(self):
        step = StepSpec(
            step_id="test",
            title="Test",
            script=None,
            essential=[{"key": "x", "type": "number", "default": 10}],
            advanced=[],
        )
        result = normalize_params(step, {"x": 1, "unknown": 2})
        assert "unknown" not in result

    def test_method_aware_normalization(self):
        step = StepSpec(
            step_id="test",
            title="Test",
            script=None,
            essential=[],
            advanced=[],
            common={
                "essential": [
                    {
                        "key": "model",
                        "type": "select",
                        "default": "linear",
                        "options": [
                            {"value": "linear", "label": "Linear"},
                            {"value": "hgo", "label": "HGO"},
                        ],
                    }
                ],
                "advanced": [],
            },
            methods={
                "linear": {"essential": [{"key": "E1", "default": 100}], "advanced": []},
                "hgo": {"essential": [{"key": "C10", "default": 500}], "advanced": []},
            },
        )
        result = normalize_params(step, {"model": "hgo"})
        assert result.get("model") == "hgo"
        assert "C10" in result

    def test_default_method_when_not_specified(self):
        step = StepSpec(
            step_id="test",
            title="Test",
            script=None,
            essential=[],
            advanced=[],
            common={
                "essential": [
                    {
                        "key": "model",
                        "type": "select",
                        "default": "linear",
                        "options": [
                            {"value": "linear", "label": "Linear"},
                        ],
                    }
                ],
                "advanced": [],
            },
            methods={
                "linear": {"essential": [{"key": "E1", "default": 100}], "advanced": []},
            },
        )
        result = normalize_params(step, {})
        assert result.get("model") == "linear"
        assert result.get("E1") == 100
