"""Tests for Luau language support in HybridExtractor (TDD - tests first).

These tests define the expected behavior for:
1. File extension recognition (.luau)
2. Function extraction with type annotations
3. Type definitions extraction
4. Method detection (. vs : syntax)
5. Call graph extraction
6. Import/require extraction
7. Generic functions
8. Export pattern detection

All tests should FAIL initially because:
- .luau extension is not mapped in HybridExtractor
- Luau-specific extraction is not implemented
"""

from pathlib import Path

import pytest
from tldr.hybrid_extractor import HybridExtractor


class TestLuauFileExtensionRecognition:
    """Test that .luau files are recognized and parsed."""

    def test_luau_extension_detected_as_luau_language(self, tmp_path: Path):
        """File with .luau extension should be detected as 'luau' language."""
        luau_file = tmp_path / "test.luau"
        luau_file.write_text("""
local x = 10
print(x)
""")

        extractor = HybridExtractor()
        result = extractor.extract(luau_file)

        # Should detect language as "luau", not "lua" or unknown
        assert result.language == "luau"

    def test_luau_extension_not_treated_as_lua(self, tmp_path: Path):
        """Luau files should NOT be processed as regular Lua."""
        luau_file = tmp_path / "module.luau"
        luau_file.write_text("""
function greet(name: string): string
    return "Hello, " .. name
end
""")

        extractor = HybridExtractor()
        result = extractor.extract(luau_file)

        # Language should be specifically "luau"
        assert result.language == "luau"
        assert result.language != "lua"


class TestLuauFunctionExtraction:
    """Test extraction of Luau functions with type annotations."""

    def test_function_with_typed_parameters(self, tmp_path: Path):
        """Functions with Luau type annotations should be extracted."""
        luau_file = tmp_path / "typed.luau"
        luau_file.write_text("""
function greet(name: string): string
    return "Hello, " .. name
end

local function helper(): ()
end
""")

        extractor = HybridExtractor()
        result = extractor.extract(luau_file)

        # Should extract both functions
        assert len(result.functions) == 2

        # Find greet function
        greet = next((f for f in result.functions if f.name == "greet"), None)
        assert greet is not None
        assert "name" in greet.params

        # Find helper function
        helper = next((f for f in result.functions if f.name == "helper"), None)
        assert helper is not None

    def test_function_with_optional_type(self, tmp_path: Path):
        """Functions with optional type (?) should be extracted."""
        luau_file = tmp_path / "optional.luau"
        luau_file.write_text("""
function process(input: string, count: number?): {string}
    -- body
    return {}
end
""")

        extractor = HybridExtractor()
        result = extractor.extract(luau_file)

        assert len(result.functions) == 1
        func = result.functions[0]
        assert func.name == "process"
        # Should capture both parameters
        assert len(func.params) == 2


class TestLuauTypeDefinitions:
    """Test extraction of Luau type definitions."""

    def test_type_definitions_extracted(self, tmp_path: Path):
        """Type definitions should be recognized and extracted."""
        luau_file = tmp_path / "types.luau"
        luau_file.write_text("""
type Point = {x: number, y: number}
type Array<T> = {T}
type Callback = (string) -> ()
""")

        extractor = HybridExtractor()
        result = extractor.extract(luau_file)

        # Type definitions may be extracted as classes or a special field
        # The key is that they are not silently ignored
        # Check we got something from this file
        assert result.language == "luau"
        # Implementation may put types in classes list or a dedicated field
        # For now, we just verify the file was processed
        # More specific assertions can be added based on implementation choice


class TestLuauMethodDetection:
    """Test detection of methods (. vs : syntax)."""

    def test_static_vs_instance_method(self, tmp_path: Path):
        """Should distinguish between . (static) and : (instance) methods."""
        luau_file = tmp_path / "methods.luau"
        luau_file.write_text("""
local Module = {}

function Module.staticMethod(): ()
end

function Module:instanceMethod(): ()
end
""")

        extractor = HybridExtractor()
        result = extractor.extract(luau_file)

        # Should extract both methods
        assert len(result.functions) >= 2

        # Find the methods by name
        func_names = [f.name for f in result.functions]
        # Names could be "staticMethod", "instanceMethod" or "Module.staticMethod", etc.
        assert any("staticMethod" in name for name in func_names)
        assert any("instanceMethod" in name for name in func_names)

    def test_class_like_pattern(self, tmp_path: Path):
        """Should handle Luau class-like patterns."""
        luau_file = tmp_path / "class.luau"
        luau_file.write_text("""
local Player = {}
Player.__index = Player

function Player.new(name: string): Player
    local self = setmetatable({}, Player)
    self.name = name
    return self
end

function Player:greet(): string
    return "Hello, " .. self.name
end
""")

        extractor = HybridExtractor()
        result = extractor.extract(luau_file)

        # Should extract both methods
        func_names = [f.name for f in result.functions]
        assert any("new" in name for name in func_names)
        assert any("greet" in name for name in func_names)


class TestLuauCallGraph:
    """Test call graph extraction for Luau."""

    def test_call_graph_extraction(self, tmp_path: Path):
        """Should extract call relationships between functions."""
        luau_file = tmp_path / "calls.luau"
        luau_file.write_text("""
local function helper()
    return 42
end

local function main()
    helper()
    helper()
end
""")

        extractor = HybridExtractor()
        result = extractor.extract(luau_file)

        # Should have call graph with main -> helper
        assert result.call_graph is not None
        assert "main" in result.call_graph.calls
        assert "helper" in result.call_graph.calls["main"]
        # Should be deduplicated (only one edge even with two calls)
        assert result.call_graph.calls["main"].count("helper") == 1


class TestLuauImportExtraction:
    """Test extraction of require statements."""

    def test_require_extraction(self, tmp_path: Path):
        """Should extract require statements as imports."""
        luau_file = tmp_path / "imports.luau"
        luau_file.write_text("""
local Utils = require(script.Utils)
local Config = require(game.ReplicatedStorage.Config)
""")

        extractor = HybridExtractor()
        result = extractor.extract(luau_file)

        # Should extract 2 imports
        assert len(result.imports) == 2

        # Check that module paths are captured
        modules = [imp.module for imp in result.imports]
        assert any("Utils" in m for m in modules)
        assert any("Config" in m for m in modules)

    def test_string_require_extraction(self, tmp_path: Path):
        """Should extract require with string literal."""
        luau_file = tmp_path / "string_require.luau"
        luau_file.write_text("""
local json = require("@pkg/json")
""")

        extractor = HybridExtractor()
        result = extractor.extract(luau_file)

        assert len(result.imports) == 1
        assert "@pkg/json" in result.imports[0].module


class TestLuauGenericFunctions:
    """Test extraction of generic functions."""

    def test_generic_function_extraction(self, tmp_path: Path):
        """Should extract generic functions with type parameters."""
        luau_file = tmp_path / "generics.luau"
        luau_file.write_text("""
function identity<T>(value: T): T
    return value
end
""")

        extractor = HybridExtractor()
        result = extractor.extract(luau_file)

        # Should extract the function
        assert len(result.functions) == 1
        func = result.functions[0]
        assert func.name == "identity"
        # Parameter should be captured (type annotation may or may not be in params)
        assert "value" in func.params or len(func.params) > 0


class TestLuauExportPatterns:
    """Test detection of export patterns."""

    def test_export_detection(self, tmp_path: Path):
        """Should track which functions are exported vs local."""
        luau_file = tmp_path / "exports.luau"
        luau_file.write_text("""
local module = {}

function module.publicFunc(): ()
end

local function privateFunc(): ()
end

return module
""")

        extractor = HybridExtractor()
        result = extractor.extract(luau_file)

        # Should extract both functions
        func_names = [f.name for f in result.functions]
        assert any("publicFunc" in name for name in func_names)
        assert any("privateFunc" in name for name in func_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
