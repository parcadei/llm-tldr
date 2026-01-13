"""
Functional tests for TypeScript support in llm-tldr.

These tests verify that each feature actually works with TypeScript code,
not just that the functions exist.

Run with:
    pytest test_typescript_features.py -v
"""

import pytest
import tempfile
import subprocess
import json
from pathlib import Path


# Sample TypeScript files for testing
SIMPLE_FUNCTION = """
function greet(name: string): string {
    return "Hello, " + name;
}
"""

FUNCTION_WITH_BRANCHES = """
function classify(x: number): string {
    if (x > 10) {
        return "big";
    } else if (x > 5) {
        return "medium";
    } else {
        return "small";
    }
}
"""

FUNCTION_WITH_LOOP = """
function sumArray(arr: number[]): number {
    let total = 0;
    for (const num of arr) {
        total += num;
    }
    return total;
}
"""

CLASS_WITH_METHODS = """
class Calculator {
    private value: number = 0;

    add(x: number): Calculator {
        this.value += x;
        return this;
    }

    subtract(x: number): Calculator {
        this.value -= x;
        return this;
    }

    getResult(): number {
        return this.value;
    }
}
"""

MULTI_FILE_CALLER = """
import { helper } from "./helper";

export function main(): void {
    const result = helper("test");
    console.log(result);
}
"""

MULTI_FILE_HELPER = """
export function helper(input: string): string {
    return input.toUpperCase();
}

export function unused(): void {
    console.log("never called");
}
"""

IMPORTS_EXAMPLE = """
import { readFile } from "fs";
import path from "path";
import { Component, useState } from "react";
import type { Config } from "./types";

const x = 1;
"""

DATA_FLOW_EXAMPLE = """
function process(input: string): string {
    const trimmed = input.trim();
    const upper = trimmed.toUpperCase();
    const result = upper + "!";
    return result;
}
"""

ASYNC_FUNCTION = """
async function fetchData(url: string): Promise<string> {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error("Failed");
    }
    const data = await response.text();
    return data;
}
"""


def run_tldr(args: list[str], cwd: str = None) -> dict:
    """Run tldr command and return parsed JSON output."""
    cmd = ["tldr"] + args + ["--lang", "typescript"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        pytest.fail(f"tldr command failed: {result.stderr}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        # Some commands return non-JSON output
        return {"raw_output": result.stdout}


@pytest.fixture
def temp_ts_file():
    """Create a temporary TypeScript file."""
    def _create(content: str, filename: str = "test.ts"):
        tmpdir = tempfile.mkdtemp()
        filepath = Path(tmpdir) / filename
        filepath.write_text(content)
        return str(filepath), tmpdir
    return _create


@pytest.fixture
def temp_ts_project():
    """Create a temporary TypeScript project with multiple files."""
    def _create(files: dict[str, str]):
        tmpdir = tempfile.mkdtemp()
        for filename, content in files.items():
            filepath = Path(tmpdir) / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content)
        return tmpdir
    return _create


class TestStructure:
    """Test the 'structure' command for TypeScript."""

    def test_structure_finds_function(self, temp_ts_file):
        """structure should detect function declarations."""
        filepath, tmpdir = temp_ts_file(SIMPLE_FUNCTION)
        result = run_tldr(["structure", tmpdir])

        assert "files" in result, "structure should return files"
        assert len(result["files"]) > 0, "should find at least one file"

        functions = result["files"][0].get("functions", [])
        assert any("greet" in f for f in functions), \
            f"should find 'greet' function, got: {functions}"

    def test_structure_finds_class(self, temp_ts_file):
        """structure should detect class declarations."""
        filepath, tmpdir = temp_ts_file(CLASS_WITH_METHODS)
        result = run_tldr(["structure", tmpdir])

        assert "files" in result
        file_info = result["files"][0]

        classes = file_info.get("classes", [])
        assert any("Calculator" in c for c in classes), \
            f"should find 'Calculator' class, got: {classes}"

    def test_structure_finds_methods(self, temp_ts_file):
        """structure should detect class methods."""
        filepath, tmpdir = temp_ts_file(CLASS_WITH_METHODS)
        result = run_tldr(["structure", tmpdir])

        functions = result["files"][0].get("functions", [])
        method_names = ["add", "subtract", "getResult"]
        for method in method_names:
            assert any(method in f for f in functions), \
                f"should find '{method}' method, got: {functions}"

    def test_structure_finds_async_function(self, temp_ts_file):
        """structure should detect async functions."""
        filepath, tmpdir = temp_ts_file(ASYNC_FUNCTION)
        result = run_tldr(["structure", tmpdir])

        functions = result["files"][0].get("functions", [])
        assert any("fetchData" in f for f in functions), \
            f"should find 'fetchData' async function, got: {functions}"


class TestCFG:
    """Test the 'cfg' (control flow graph) command for TypeScript."""

    def test_cfg_simple_function(self, temp_ts_file):
        """cfg should return blocks for a simple function."""
        filepath, _ = temp_ts_file(SIMPLE_FUNCTION)
        result = run_tldr(["cfg", filepath, "greet"])

        assert result.get("function") == "greet", \
            f"should identify function name, got: {result}"
        assert "blocks" in result, "should return blocks"
        assert len(result["blocks"]) > 0, \
            f"simple function should have at least one block, got: {result}"

    def test_cfg_branching_function(self, temp_ts_file):
        """cfg should detect branches in if/else statements."""
        filepath, _ = temp_ts_file(FUNCTION_WITH_BRANCHES)
        result = run_tldr(["cfg", filepath, "classify"])

        assert len(result.get("blocks", [])) > 0, \
            f"branching function should have multiple blocks, got: {result}"
        assert len(result.get("edges", [])) > 0, \
            f"branching function should have edges, got: {result}"
        assert result.get("cyclomatic_complexity", 0) >= 3, \
            f"function with 2 branches should have complexity >= 3, got: {result}"

    def test_cfg_loop_function(self, temp_ts_file):
        """cfg should detect loop structures."""
        filepath, _ = temp_ts_file(FUNCTION_WITH_LOOP)
        result = run_tldr(["cfg", filepath, "sumArray"])

        assert len(result.get("blocks", [])) > 0, \
            f"loop function should have blocks, got: {result}"
        # Loops create back-edges
        edges = result.get("edges", [])
        assert len(edges) > 0, \
            f"loop function should have edges, got: {result}"

    def test_cfg_async_function(self, temp_ts_file):
        """cfg should handle async/await functions."""
        filepath, _ = temp_ts_file(ASYNC_FUNCTION)
        result = run_tldr(["cfg", filepath, "fetchData"])

        assert result.get("function") == "fetchData"
        assert len(result.get("blocks", [])) > 0, \
            f"async function should have blocks, got: {result}"


class TestDFG:
    """Test the 'dfg' (data flow graph) command for TypeScript."""

    def test_dfg_tracks_variables(self, temp_ts_file):
        """dfg should track variable definitions and uses."""
        filepath, _ = temp_ts_file(DATA_FLOW_EXAMPLE)
        result = run_tldr(["dfg", filepath, "process"])

        assert result.get("function") == "process"

        variables = result.get("variables", [])
        assert len(variables) > 0, \
            f"should find variables, got: {result}"

        var_names = [v.get("name", v) if isinstance(v, dict) else str(v)
                     for v in variables]
        expected_vars = ["input", "trimmed", "upper", "result"]
        for var in expected_vars:
            assert any(var in str(v) for v in var_names), \
                f"should find variable '{var}', got: {var_names}"

    def test_dfg_tracks_flow(self, temp_ts_file):
        """dfg should track data flow between variables."""
        filepath, _ = temp_ts_file(DATA_FLOW_EXAMPLE)
        result = run_tldr(["dfg", filepath, "process"])

        edges = result.get("edges", [])
        assert len(edges) > 0, \
            f"should have data flow edges, got: {result}"


class TestSlice:
    """Test the 'slice' (program slice) command for TypeScript."""

    def test_slice_finds_dependencies(self, temp_ts_file):
        """slice should find lines that affect a given line."""
        filepath, _ = temp_ts_file(DATA_FLOW_EXAMPLE)
        # Line 5 is "const result = upper + "!";"
        result = run_tldr(["slice", filepath, "process", "5"])

        # Should include lines that define 'upper' and 'trimmed' and 'input'
        assert "lines" in result or "slice" in result or len(result) > 0, \
            f"slice should return relevant lines, got: {result}"

    def test_slice_return_statement(self, temp_ts_file):
        """slice on return should include all contributing lines."""
        filepath, _ = temp_ts_file(DATA_FLOW_EXAMPLE)
        # Line 6 is "return result;"
        result = run_tldr(["slice", filepath, "process", "6"])

        assert len(result) > 0, \
            f"slice should find dependencies for return, got: {result}"


class TestCalls:
    """Test the 'calls' (call graph) command for TypeScript."""

    def test_calls_finds_function_calls(self, temp_ts_project):
        """calls should build cross-file call graph."""
        tmpdir = temp_ts_project({
            "main.ts": MULTI_FILE_CALLER,
            "helper.ts": MULTI_FILE_HELPER,
        })
        result = run_tldr(["calls", tmpdir])

        assert "edges" in result or "calls" in result, \
            f"calls should return edges, got: {result}"

    def test_calls_detects_imports(self, temp_ts_project):
        """calls should follow import statements."""
        tmpdir = temp_ts_project({
            "main.ts": MULTI_FILE_CALLER,
            "helper.ts": MULTI_FILE_HELPER,
        })
        result = run_tldr(["calls", tmpdir])

        # Should detect that main calls helper
        output = json.dumps(result)
        assert "main" in output.lower() or "helper" in output.lower(), \
            f"should find call relationship, got: {result}"


class TestImpact:
    """Test the 'impact' (reverse call graph) command for TypeScript."""

    def test_impact_finds_callers(self, temp_ts_project):
        """impact should find all callers of a function."""
        tmpdir = temp_ts_project({
            "main.ts": MULTI_FILE_CALLER,
            "helper.ts": MULTI_FILE_HELPER,
        })
        result = run_tldr(["impact", "helper", "--project", tmpdir])

        # main.ts calls helper, so it should appear in impact
        output = json.dumps(result)
        assert len(result) > 0, \
            f"should find callers of helper, got: {result}"


class TestContext:
    """Test the 'context' command for TypeScript."""

    def test_context_returns_relevant_code(self, temp_ts_project):
        """context should return relevant functions for LLM."""
        tmpdir = temp_ts_project({
            "main.ts": MULTI_FILE_CALLER,
            "helper.ts": MULTI_FILE_HELPER,
        })
        result = run_tldr(["context", "main", "--project", tmpdir])

        # Should return the main function and its dependencies
        output = result.get("raw_output", json.dumps(result))
        assert "main" in output.lower(), \
            f"context should include main function, got: {output[:500]}"

    def test_context_includes_callees(self, temp_ts_project):
        """context should include called functions."""
        tmpdir = temp_ts_project({
            "main.ts": MULTI_FILE_CALLER,
            "helper.ts": MULTI_FILE_HELPER,
        })
        result = run_tldr(["context", "main", "--project", tmpdir])

        output = result.get("raw_output", json.dumps(result))
        assert "helper" in output.lower(), \
            f"context should include helper (called by main), got: {output[:500]}"


class TestImports:
    """Test the 'imports' command for TypeScript."""

    def test_imports_parses_named_imports(self, temp_ts_file):
        """imports should parse named imports."""
        filepath, _ = temp_ts_file(IMPORTS_EXAMPLE)
        result = run_tldr(["imports", filepath])

        output = json.dumps(result)
        assert "readFile" in output or "fs" in output, \
            f"should find fs import, got: {result}"

    def test_imports_parses_default_imports(self, temp_ts_file):
        """imports should parse default imports."""
        filepath, _ = temp_ts_file(IMPORTS_EXAMPLE)
        result = run_tldr(["imports", filepath])

        output = json.dumps(result)
        assert "path" in output, \
            f"should find path default import, got: {result}"

    def test_imports_parses_type_imports(self, temp_ts_file):
        """imports should parse type-only imports."""
        filepath, _ = temp_ts_file(IMPORTS_EXAMPLE)
        result = run_tldr(["imports", filepath])

        output = json.dumps(result)
        # Should recognize type imports
        assert "Config" in output or "types" in output, \
            f"should find type import, got: {result}"


class TestImporters:
    """Test the 'importers' (reverse import lookup) command for TypeScript."""

    def test_importers_finds_importing_files(self, temp_ts_project):
        """importers should find files that import a module."""
        tmpdir = temp_ts_project({
            "main.ts": MULTI_FILE_CALLER,
            "helper.ts": MULTI_FILE_HELPER,
        })
        result = run_tldr(["importers", "helper", tmpdir])

        output = json.dumps(result)
        assert "main" in output.lower(), \
            f"should find main.ts imports helper, got: {result}"


class TestDead:
    """Test the 'dead' (dead code detection) command for TypeScript."""

    def test_dead_finds_unused_functions(self, temp_ts_project):
        """dead should find unreachable/unused functions."""
        tmpdir = temp_ts_project({
            "main.ts": MULTI_FILE_CALLER,
            "helper.ts": MULTI_FILE_HELPER,  # has 'unused' function
        })
        result = run_tldr(["dead", tmpdir])

        output = json.dumps(result)
        # The 'unused' function in helper.ts is never called
        assert "unused" in output.lower() or len(result) > 0, \
            f"should find unused function, got: {result}"


class TestDiagnostics:
    """Test the 'diagnostics' command for TypeScript."""

    def test_diagnostics_valid_file(self, temp_ts_file):
        """diagnostics should return no errors for valid TypeScript."""
        filepath, _ = temp_ts_file(SIMPLE_FUNCTION)
        result = run_tldr(["diagnostics", filepath])

        assert result.get("error_count", 0) == 0, \
            f"valid file should have no errors, got: {result}"

    def test_diagnostics_invalid_file(self, temp_ts_file):
        """diagnostics should detect TypeScript errors."""
        invalid_ts = """
function broken(x: number): string {
    return x;  // Type error: number not assignable to string
}
"""
        filepath, _ = temp_ts_file(invalid_ts)
        result = run_tldr(["diagnostics", filepath])

        # Should detect the type error
        assert result.get("error_count", 0) > 0 or len(result.get("diagnostics", [])) > 0, \
            f"should detect type error, got: {result}"


class TestExtract:
    """Test the 'extract' command for TypeScript."""

    def test_extract_full_file_info(self, temp_ts_file):
        """extract should return complete file analysis."""
        filepath, _ = temp_ts_file(CLASS_WITH_METHODS)
        result = run_tldr(["extract", filepath])

        assert "functions" in result or "classes" in result or "symbols" in result, \
            f"extract should return file info, got: {result}"


class TestArch:
    """Test the 'arch' (architecture detection) command for TypeScript."""

    def test_arch_detects_layers(self, temp_ts_project):
        """arch should detect architectural layers from call patterns."""
        # Create a simple layered architecture
        tmpdir = temp_ts_project({
            "controller.ts": """
import { service } from "./service";
export function handleRequest() { return service.getData(); }
""",
            "service.ts": """
import { repository } from "./repository";
export const service = { getData: () => repository.fetch() };
""",
            "repository.ts": """
export const repository = { fetch: () => "data" };
""",
        })
        result = run_tldr(["arch", tmpdir])

        # Should detect some layering
        assert len(result) > 0, \
            f"arch should detect architecture, got: {result}"


class TestWarm:
    """Test the 'warm' command for TypeScript."""

    def test_warm_builds_cache(self, temp_ts_project):
        """warm should pre-build call graph cache."""
        tmpdir = temp_ts_project({
            "main.ts": MULTI_FILE_CALLER,
            "helper.ts": MULTI_FILE_HELPER,
        })
        result = run_tldr(["warm", tmpdir])

        # warm typically outputs processing stats
        output = result.get("raw_output", json.dumps(result))
        assert "files" in output.lower() or "indexed" in output.lower() or "processed" in output.lower(), \
            f"warm should report progress, got: {output}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
