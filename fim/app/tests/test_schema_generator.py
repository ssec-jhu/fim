from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from fim.app.schema_generator import (
    ArgSpec,
    _infer_ui_type,
    _literal,
    _option_value,
    _prune_select_options,
    merge_step_params,
    scan_argparse_file,
    sync_step_params,
)


class TestLiteral:
    def test_constant(self):
        import ast

        node = ast.Constant(value=42)
        assert _literal(node) == 42

    def test_name(self):
        import ast

        node = ast.Name(id="float")
        assert _literal(node) == "float"

    def test_unparseable(self):
        import ast

        node = ast.BinOp(left=ast.Constant(value=1), op=ast.Add(), right=ast.Constant(value=2))
        assert _literal(node) is None


class TestInferUIType:
    def test_store_true(self):
        assert _infer_ui_type(None, "store_true", None) == "boolean"

    def test_store_false(self):
        assert _infer_ui_type(None, "store_false", None) == "boolean"

    def test_choices(self):
        assert _infer_ui_type(None, None, ["a", "b"]) == "select"

    def test_int_type(self):
        assert _infer_ui_type("int", None, None) == "number"

    def test_float_type(self):
        assert _infer_ui_type("float", None, None) == "number"

    def test_default_text(self):
        assert _infer_ui_type("str", None, None) == "text"

    def test_none_type(self):
        assert _infer_ui_type(None, None, None) == "text"


class TestScanArgparseFile:
    def test_scan_simple_file(self, tmp_path):
        py = tmp_path / "script.py"
        py.write_text(
            textwrap.dedent("""\
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--alpha", type=float, default=1.5, help="Alpha value")
            parser.add_argument("--name", type=str, default="test", help="Name")
            parser.add_argument("--verbose", action="store_true", help="Verbose mode")
            parser.add_argument("--model", type=str, choices=["linear", "hgo"], default="linear")
            """)
        )
        result = scan_argparse_file(py)
        assert "alpha" in result
        assert result["alpha"].type == "number"
        assert result["alpha"].default == 1.5

        assert "name" in result
        assert result["name"].type == "text"

        assert "verbose" in result
        assert result["verbose"].type == "boolean"

        assert "model" in result
        assert result["model"].type == "select"
        assert result["model"].options == ["linear", "hgo"]

    def test_skips_positional_args(self, tmp_path):
        py = tmp_path / "script.py"
        py.write_text(
            textwrap.dedent("""\
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("input_file")
            parser.add_argument("--output", type=str)
            """)
        )
        result = scan_argparse_file(py)
        assert "input_file" not in result
        assert "output" in result

    def test_dest_override(self, tmp_path):
        py = tmp_path / "script.py"
        py.write_text(
            textwrap.dedent("""\
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--set", dest="kvs", action="append")
            """)
        )
        result = scan_argparse_file(py)
        assert "kvs" in result


class TestOptionValue:
    def test_string(self):
        assert _option_value("linear") == "linear"

    def test_int(self):
        assert _option_value(1) == "1"

    def test_float(self):
        assert _option_value(3.14) == "3.14"

    def test_dict_with_value(self):
        assert _option_value({"value": "hgo", "label": "HGO"}) == "hgo"

    def test_dict_without_value(self):
        assert _option_value({"label": "test"}) is None

    def test_none(self):
        assert _option_value(None) is None

    def test_other_type(self):
        assert _option_value([1, 2]) is None


class TestPruneSelectOptions:
    def test_prune_stale_options(self):
        param = {"type": "select", "options": ["a", "b", "c"]}
        _prune_select_options(param, ["a", "c"])
        assert param["options"] == ["a", "c"]

    def test_prune_dict_options(self):
        param = {
            "type": "select",
            "options": [
                {"value": "a", "label": "A"},
                {"value": "b", "label": "B"},
            ],
        }
        _prune_select_options(param, ["a"])
        assert len(param["options"]) == 1
        assert param["options"][0]["value"] == "a"

    def test_no_options(self):
        param = {"type": "select"}
        _prune_select_options(param, ["a"])  # should not raise

    def test_non_list_options(self):
        param = {"type": "select", "options": "invalid"}
        _prune_select_options(param, ["a"])  # should not raise


class TestMergeStepParams:
    def test_update_existing_param(self):
        step = {
            "essential": [{"key": "alpha", "label": "Alpha", "type": "number", "default": 1.0}],
            "advanced": [],
        }
        scanned = {"alpha": ArgSpec(key="alpha", type="number", default=2.0, help="New help")}
        merge_step_params(step, scanned)
        assert step["essential"][0]["default"] == 2.0

    def test_add_new_param(self):
        step = {
            "essential": [{"key": "alpha", "type": "number"}],
            "advanced": [],
        }
        scanned = {
            "alpha": ArgSpec(key="alpha", type="number"),
            "beta": ArgSpec(key="beta", type="text", default="val"),
        }
        merge_step_params(step, scanned, add_new_to="advanced")
        assert any(p["key"] == "beta" for p in step["advanced"])

    def test_add_new_to_common(self):
        step = {
            "common": {"essential": [], "advanced": []},
            "methods": {},
        }
        scanned = {"new_param": ArgSpec(key="new_param", type="number", default=5)}
        merge_step_params(step, scanned, add_new_to="advanced")
        assert any(p["key"] == "new_param" for p in step["common"]["advanced"])

    def test_preserves_file_type(self):
        step = {
            "essential": [{"key": "path", "type": "file", "default": "/a/b"}],
            "advanced": [],
        }
        scanned = {"path": ArgSpec(key="path", type="text", default="/c/d")}
        merge_step_params(step, scanned)
        assert step["essential"][0]["type"] == "file"
        assert step["essential"][0]["default"] == "/c/d"

    def test_preserves_dir_type(self):
        step = {
            "essential": [{"key": "out", "type": "dir"}],
            "advanced": [],
        }
        scanned = {"out": ArgSpec(key="out", type="text")}
        merge_step_params(step, scanned)
        assert step["essential"][0]["type"] == "dir"

    def test_updates_select_options_preserving_labels(self):
        step = {
            "essential": [
                {
                    "key": "model",
                    "type": "select",
                    "options": [{"value": "linear", "label": "Linear Model"}],
                }
            ],
            "advanced": [],
        }
        scanned = {"model": ArgSpec(key="model", type="select", options=["linear", "hgo"])}
        merge_step_params(step, scanned)
        opts = step["essential"][0]["options"]
        assert len(opts) == 2
        assert opts[0] == {"value": "linear", "label": "Linear Model"}

    def test_add_new_select_with_options(self):
        step = {"essential": [], "advanced": []}
        scanned = {"model": ArgSpec(key="model", type="select", options=["a", "b"])}
        merge_step_params(step, scanned, add_new_to="advanced")
        added = next(p for p in step["advanced"] if p["key"] == "model")
        assert added["options"] == ["a", "b"]

    def test_invalid_add_new_to_defaults_to_advanced(self):
        step = {"essential": [], "advanced": []}
        scanned = {"x": ArgSpec(key="x", type="text", default="val")}
        merge_step_params(step, scanned, add_new_to="invalid_section")
        assert any(p["key"] == "x" for p in step["advanced"])

    def test_with_methods_container(self):
        step = {
            "common": {"essential": [], "advanced": []},
            "methods": {
                "m1": {"essential": [{"key": "p1", "type": "number", "default": 10}], "advanced": []},
            },
        }
        scanned = {"p1": ArgSpec(key="p1", type="number", default=99)}
        merge_step_params(step, scanned)
        assert step["methods"]["m1"]["essential"][0]["default"] == 99


class TestSyncStepParams:
    def test_prunes_select_options(self):
        step = {
            "common": {
                "essential": [
                    {
                        "key": "model",
                        "type": "select",
                        "options": [
                            {"value": "linear", "label": "Linear"},
                            {"value": "old_model", "label": "Old Model"},
                        ],
                    }
                ],
                "advanced": [],
            },
            "methods": {
                "linear": {"essential": [], "advanced": []},
                "old_model": {"essential": [], "advanced": []},
            },
        }
        scanned = {"model": ArgSpec(key="model", type="select", options=["linear"])}
        msgs = sync_step_params(step, scanned)
        assert len(step["common"]["essential"][0]["options"]) == 1
        assert "old_model" not in step["methods"]
        assert len(msgs) > 0

    def test_no_changes_when_aligned(self):
        step = {
            "essential": [
                {"key": "model", "type": "select", "options": ["a", "b"]},
            ],
            "advanced": [],
        }
        scanned = {"model": ArgSpec(key="model", type="select", options=["a", "b"])}
        msgs = sync_step_params(step, scanned)
        assert len(msgs) == 0

    def test_no_methods_block(self):
        step = {
            "essential": [{"key": "x", "type": "select", "options": ["a", "b"]}],
            "advanced": [],
        }
        scanned = {"x": ArgSpec(key="x", type="select", options=["a"])}
        msgs = sync_step_params(step, scanned)
        assert len(step["essential"][0]["options"]) == 1

    def test_methods_without_matching_select(self):
        step = {
            "essential": [],
            "advanced": [],
            "methods": {"m1": {"essential": [], "advanced": []}},
        }
        scanned = {}
        msgs = sync_step_params(step, scanned)
        assert "m1" in step["methods"]


class TestGenerate:
    @patch("fim.app.schema_generator._schema_path")
    @patch("fim.app.schema_generator._repo_root")
    def test_generate_reads_schema(self, mock_root, mock_schema, tmp_path):
        from fim.app.schema_generator import generate

        schema = {
            "version": "1",
            "steps": {
                "inverse": {
                    "title": "Inverse",
                    "essential": [],
                    "advanced": [],
                }
            },
        }
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(schema))
        mock_schema.return_value = schema_file
        mock_root.return_value = tmp_path

        result = generate(write=False, sync=False)
        assert "steps" in result

    @patch("fim.app.schema_generator._schema_path")
    @patch("fim.app.schema_generator._repo_root")
    def test_generate_write(self, mock_root, mock_schema, tmp_path):
        from fim.app.schema_generator import generate

        schema = {"version": "1", "steps": {}}
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(schema))
        mock_schema.return_value = schema_file
        mock_root.return_value = tmp_path

        generate(write=True, sync=False)
        data = json.loads(schema_file.read_text())
        assert "steps" in data
