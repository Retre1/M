"""Tests for src/apexfx/config/loader.py."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from apexfx.config.loader import _deep_merge, _interpolate_env_vars, load_yaml


class TestInterpolateEnvVars:
    def test_with_set_env_var(self):
        with patch.dict(os.environ, {"MY_VAR": "hello"}):
            result = _interpolate_env_vars("${MY_VAR}")
            assert result == "hello"

    def test_with_default_value(self):
        # Ensure variable is NOT set
        env = {k: v for k, v in os.environ.items() if k != "UNSET_VAR_12345"}
        with patch.dict(os.environ, env, clear=True):
            result = _interpolate_env_vars("${UNSET_VAR_12345:fallback}")
            assert result == "fallback"

    def test_missing_var_no_default_keeps_placeholder(self):
        env = {k: v for k, v in os.environ.items() if k != "MISSING_VAR_XYZ"}
        with patch.dict(os.environ, env, clear=True):
            result = _interpolate_env_vars("${MISSING_VAR_XYZ}")
            assert result == "${MISSING_VAR_XYZ}"

    def test_nested_dict(self):
        with patch.dict(os.environ, {"DB_HOST": "localhost"}):
            data = {"connection": {"host": "${DB_HOST}"}}
            result = _interpolate_env_vars(data)
            assert result == {"connection": {"host": "localhost"}}

    def test_list(self):
        with patch.dict(os.environ, {"ITEM": "value"}):
            data = ["${ITEM}", "static"]
            result = _interpolate_env_vars(data)
            assert result == ["value", "static"]

    def test_non_string_passthrough(self):
        assert _interpolate_env_vars(42) == 42
        assert _interpolate_env_vars(True) is True


class TestDeepMerge:
    def test_flat_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"x": {"a": 1, "b": 2}, "y": 10}
        override = {"x": {"b": 99, "c": 3}}
        result = _deep_merge(base, override)
        assert result == {"x": {"a": 1, "b": 99, "c": 3}, "y": 10}

    def test_override_replaces_non_dict(self):
        base = {"x": {"a": 1}}
        override = {"x": "replaced"}
        result = _deep_merge(base, override)
        assert result == {"x": "replaced"}

    def test_base_not_mutated(self):
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"b": 1}}


class TestLoadYaml:
    def test_nonexistent_file_returns_empty_dict(self):
        result = load_yaml("/tmp/nonexistent_apexfx_test_file.yaml")
        assert result == {}

    def test_required_missing_raises(self):
        with pytest.raises(FileNotFoundError, match="Required config file missing"):
            load_yaml("/tmp/nonexistent_apexfx_test_file.yaml", required=True)

    def test_reads_actual_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\nnested:\n  x: 10\n")
            f.flush()
            path = f.name
        try:
            result = load_yaml(path)
            assert result == {"key": "value", "nested": {"x": 10}}
        finally:
            os.unlink(path)

    def test_env_var_interpolation_in_yaml(self):
        with patch.dict(os.environ, {"TEST_PORT": "8080"}):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write("port: ${TEST_PORT}\n")
                f.flush()
                path = f.name
            try:
                result = load_yaml(path)
                assert result == {"port": "8080"}
            finally:
                os.unlink(path)

    def test_empty_yaml_returns_empty_dict(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            path = f.name
        try:
            result = load_yaml(path)
            assert result == {}
        finally:
            os.unlink(path)
