"""Tests for Luau import parsing (TDD - write tests first).

Luau is a Roblox-specific dialect of Lua with additional syntax for:
- Type annotations
- GetService patterns (Roblox-specific)
- script.Parent paths
- @pkg/ style requires

These tests define expected behavior for parse_luau_imports().
"""

import tempfile
from pathlib import Path

import pytest


class TestLuauBasicRequire:
    """Tests for basic require statements in Luau."""

    def test_luau_imports_script_require(self):
        """Should parse require(script.Utils) pattern."""
        from tldr.cross_file_calls import parse_luau_imports

        with tempfile.NamedTemporaryFile(suffix=".luau", mode="w", delete=False) as f:
            f.write("""local Utils = require(script.Utils)
""")
            f.flush()

            imports = parse_luau_imports(f.name)

        assert len(imports) == 1
        assert imports[0]["module"] == "script.Utils"
        assert imports[0]["type"] == "require"

    def test_luau_imports_script_parent_require(self):
        """Should parse require(script.Parent.SharedModule) pattern."""
        from tldr.cross_file_calls import parse_luau_imports

        with tempfile.NamedTemporaryFile(suffix=".luau", mode="w", delete=False) as f:
            f.write("""local module = require(script.Parent.SharedModule)
""")
            f.flush()

            imports = parse_luau_imports(f.name)

        assert len(imports) == 1
        assert imports[0]["module"] == "script.Parent.SharedModule"
        assert imports[0]["type"] == "require"

    def test_luau_imports_string_literal_require(self):
        """Should parse require('@pkg/json') string literal pattern."""
        from tldr.cross_file_calls import parse_luau_imports

        with tempfile.NamedTemporaryFile(suffix=".luau", mode="w", delete=False) as f:
            f.write("""local json = require("@pkg/json")
""")
            f.flush()

            imports = parse_luau_imports(f.name)

        assert len(imports) == 1
        assert imports[0]["module"] == "@pkg/json"
        assert imports[0]["type"] == "require"


class TestLuauGetService:
    """Tests for Roblox GetService patterns."""

    def test_luau_imports_getservice_players(self):
        """Should parse game:GetService('Players') as import-like."""
        from tldr.cross_file_calls import parse_luau_imports

        with tempfile.NamedTemporaryFile(suffix=".luau", mode="w", delete=False) as f:
            f.write("""local Players = game:GetService("Players")
""")
            f.flush()

            imports = parse_luau_imports(f.name)

        assert len(imports) == 1
        assert imports[0]["module"] == "Players"
        assert imports[0]["type"] == "service"

    def test_luau_imports_multiple_getservice(self):
        """Should parse multiple GetService calls."""
        from tldr.cross_file_calls import parse_luau_imports

        with tempfile.NamedTemporaryFile(suffix=".luau", mode="w", delete=False) as f:
            f.write("""local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
""")
            f.flush()

            imports = parse_luau_imports(f.name)

        assert len(imports) == 3
        services = {imp["module"] for imp in imports}
        assert services == {"Players", "RunService", "ReplicatedStorage"}


class TestLuauMultipleImports:
    """Tests for files with multiple import types."""

    def test_luau_imports_mixed_requires_and_services(self):
        """Should parse both require and GetService in same file."""
        from tldr.cross_file_calls import parse_luau_imports

        with tempfile.NamedTemporaryFile(suffix=".luau", mode="w", delete=False) as f:
            f.write("""local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Utils = require(ReplicatedStorage.Utils)
local Config = require(script.Parent.Config)
""")
            f.flush()

            imports = parse_luau_imports(f.name)

        assert len(imports) == 3

        # Check service import
        service_imports = [i for i in imports if i["type"] == "service"]
        assert len(service_imports) == 1
        assert service_imports[0]["module"] == "ReplicatedStorage"

        # Check require imports
        require_imports = [i for i in imports if i["type"] == "require"]
        assert len(require_imports) == 2


class TestLuauEdgeCases:
    """Tests for edge cases and empty files."""

    def test_luau_imports_no_imports(self):
        """Should return empty list for file with no imports."""
        from tldr.cross_file_calls import parse_luau_imports

        with tempfile.NamedTemporaryFile(suffix=".luau", mode="w", delete=False) as f:
            f.write("""local x = 10
print(x)
""")
            f.flush()

            imports = parse_luau_imports(f.name)

        assert imports == []
        assert isinstance(imports, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
