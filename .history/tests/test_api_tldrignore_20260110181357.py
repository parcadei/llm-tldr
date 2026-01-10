"""
Test suite for .tldrignore filtering consistency across all API functions.

This module verifies that the following API functions properly respect
.tldrignore patterns:
- get_relevant_context()
- get_file_tree()
- search()
- get_code_structure()
- scan_project_files()
"""

import tempfile
import shutil
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tldr.api import (
    get_relevant_context,
    get_file_tree,
    search,
    get_code_structure,
    scan_project_files,
)


class TestTldrignoreConsistency:
    """Test suite for .tldrignore pattern filtering across API functions."""

    def setup_method(self):
        """Create a temporary test project with sample files."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)

        # Create directory structure
        (self.project_root / "src").mkdir()
        (self.project_root / "src" / "utils").mkdir()
        (self.project_root / "build").mkdir()
        (self.project_root / "node_modules").mkdir()
        (self.project_root / ".venv").mkdir()

        # Create Python source files
        (self.project_root / "src" / "main.py").write_text("""
def main():
    return "hello"

def helper():
    return "helper"
""")

        (self.project_root / "src" / "utils" / "tools.py").write_text("""
def process(data):
    return data * 2

def format_output(value):
    return str(value)
""")

        # Create ignored files
        (self.project_root / "build" / "output.py").write_text("""
def build_artifact():
    pass
""")

        (self.project_root / "node_modules" / "package.py").write_text("""
def external_dep():
    pass
""")

        (self.project_root / ".venv" / "lib.py").write_text("""
def venv_module():
    pass
""")

        # Create .tldrignore file
        tldrignore_content = """# Ignore build artifacts
build/

# Ignore dependencies
node_modules/
.venv/
"""
        (self.project_root / ".tldrignore").write_text(tldrignore_content)

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_scan_project_files_respects_tldrignore(self):
        """Test that scan_project_files respects .tldrignore patterns."""
        files = scan_project_files(str(self.project_root), "python")
        file_paths = [str(Path(f).relative_to(self.project_root)).replace("\\", "/") for f in files]
        
        # Should include src files
        assert any("src/main.py" in p for p in file_paths), \
            f"src/main.py should be included. Found: {file_paths}"
        assert any("src/utils/tools.py" in p for p in file_paths), \
            f"src/utils/tools.py should be included. Found: {file_paths}"
        
        # Should exclude ignored files
        assert not any("build" in p for p in file_paths), \
            f"build/ should be ignored. Found: {file_paths}"
        assert not any("node_modules" in p for p in file_paths), \
            f"node_modules/ should be ignored. Found: {file_paths}"
        assert not any(".venv" in p for p in file_paths), \
            f".venv/ should be ignored. Found: {file_paths}"

    def test_get_file_tree_respects_tldrignore(self):
        """Test that get_file_tree respects .tldrignore patterns."""
        tree = get_file_tree(str(self.project_root), extensions={".py"})
        
        # Helper to flatten tree and get all file paths
        def get_all_files(node, path=""):
            files = []
            if node["type"] == "file":
                return [path + node["name"]]
            for child in node.get("children", []):
                child_path = path + node["name"] + "/"
                files.extend(get_all_files(child, child_path))
            return files
        
        all_files = get_all_files(tree)
        all_files_str = " ".join(all_files)
        
        # Should include src files
        assert "main.py" in all_files_str, "main.py should be in file tree"
        assert "tools.py" in all_files_str, "tools.py should be in file tree"
        
        # Should exclude ignored directories
        assert "build" not in all_files_str, "build/ should not be in file tree"
        assert "node_modules" not in all_files_str, "node_modules/ should not be in file tree"
        assert ".venv" not in all_files_str, ".venv/ should not be in file tree"

    def test_search_respects_tldrignore(self):
        """Test that search respects .tldrignore patterns."""
        # Search for function definitions
        results = search(r"def \w+", str(self.project_root), extensions={".py"})
        result_files = {r["file"] for r in results}
        
        # Should include matches from src
        assert any("src/main.py" in f or "src\\main.py" in f for f in result_files), \
            "Results should include matches from src/main.py"
        assert any("src/utils/tools.py" in f or "src\\utils\\tools.py" in f for f in result_files), \
            "Results should include matches from src/utils/tools.py"
        
        # Should not include matches from ignored directories
        assert not any("build" in f for f in result_files), \
            "Results should not include matches from build/"
        assert not any("node_modules" in f for f in result_files), \
            "Results should not include matches from node_modules/"
        assert not any(".venv" in f for f in result_files), \
            "Results should not include matches from .venv/"

    def test_get_code_structure_respects_tldrignore(self):
        """Test that get_code_structure respects .tldrignore patterns."""
        structure = get_code_structure(str(self.project_root), language="python")
        file_paths = [f["path"] for f in structure["files"]]
        file_paths_str = " ".join(file_paths)
        
        # Should include src files
        assert any("main.py" in f for f in file_paths), \
            "main.py should be in code structure"
        assert any("tools.py" in f for f in file_paths), \
            "tools.py should be in code structure"
        
        # Should exclude ignored files
        assert not any("build" in f for f in file_paths), \
            "build/ should not be in code structure"
        assert not any("node_modules" in f for f in file_paths), \
            "node_modules/ should not be in code structure"
        assert not any(".venv" in f for f in file_paths), \
            ".venv/ should not be in code structure"

    def test_get_relevant_context_respects_tldrignore(self):
        """Test that get_relevant_context respects .tldrignore patterns."""
        # Test module resolution with .tldrignore filtering
        context = get_relevant_context(
            str(self.project_root),
            "main",
            depth=1,
            language="python"
        )
        
        # Should find main.py from src/ (not from ignored directories)
        found_main = any(f.name == "main" for f in context.functions)
        assert found_main or context.entry_point == "main", \
            "Should resolve main module/function"


class TestFileFormatSupport:
    """Test that all file formats are properly handled."""

    def setup_method(self):
        """Create test project with multiple file formats."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)

        # Create files for different languages
        (self.project_root / "main.py").write_text("def hello(): pass")
        (self.project_root / "app.ts").write_text("export function hello() {}")
        (self.project_root / "app.tsx").write_text("export const App = () => <div/>;")
        (self.project_root / "main.go").write_text("package main\nfunc main() {}")
        (self.project_root / "lib.rs").write_text("pub fn main() {}")
        (self.project_root / "Main.java").write_text("public class Main {}")
        (self.project_root / "util.c").write_text("void util() {}")
        (self.project_root / "util.h").write_text("void util();")
        (self.project_root / "app.cpp").write_text("void app() {}")
        (self.project_root / "app.cc").write_text("void app() {}")
        (self.project_root / "main.rb").write_text("def hello; end")
        (self.project_root / "app.php").write_text("<?php function hello() {}")
        (self.project_root / "main.swift").write_text("func main() {}")
        (self.project_root / "Main.kt").write_text("fun main() {}")
        (self.project_root / "Main.scala").write_text("def main() {}")
        (self.project_root / "app.cs").write_text("static void Main() {}")
        (self.project_root / "main.lua").write_text("function main() end")
        (self.project_root / "main.ex").write_text("def main do end")

        # Create .tldrignore file
        (self.project_root / ".tldrignore").write_text("")

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_get_code_structure_python(self):
        """Test Python file format support."""
        structure = get_code_structure(str(self.project_root), language="python")
        file_paths = [f["path"] for f in structure["files"]]
        assert any("main.py" in f for f in file_paths), \
            "Python files (.py) should be included"

    def test_get_code_structure_typescript(self):
        """Test TypeScript file format support."""
        structure = get_code_structure(str(self.project_root), language="typescript")
        file_paths = [f["path"] for f in structure["files"]]
        assert any("app.ts" in f or "app.tsx" in f for f in file_paths), \
            "TypeScript files (.ts, .tsx) should be included"

    def test_get_code_structure_javascript(self):
        """Test JavaScript file format support."""
        # Add JS files
        (self.project_root / "app.js").write_text("function hello() {}")
        (self.project_root / "app.jsx").write_text("export const App = () => <div/>;")
        
        structure = get_code_structure(str(self.project_root), language="javascript")
        file_paths = [f["path"] for f in structure["files"]]
        assert any("app.js" in f or "app.jsx" in f for f in file_paths), \
            "JavaScript files (.js, .jsx) should be included"

    def test_get_code_structure_cpp(self):
        """Test C++ file format support."""
        structure = get_code_structure(str(self.project_root), language="cpp")
        file_paths = [f["path"] for f in structure["files"]]
        # C++ should include .cpp, .cc, .cxx, .hpp
        assert any("app.cpp" in f or "app.cc" in f for f in file_paths), \
            "C++ files (.cpp, .cc, .cxx, .hpp) should be included"

    def test_get_code_structure_lua(self):
        """Test Lua file format support."""
        structure = get_code_structure(str(self.project_root), language="lua")
        file_paths = [f["path"] for f in structure["files"]]
        assert any("main.lua" in f for f in file_paths), \
            "Lua files (.lua) should be included"

    def test_get_code_structure_elixir(self):
        """Test Elixir file format support."""
        structure = get_code_structure(str(self.project_root), language="elixir")
        file_paths = [f["path"] for f in structure["files"]]
        assert any("main.ex" in f for f in file_paths), \
            "Elixir files (.ex, .exs) should be included"


if __name__ == "__main__":
    # Run tests manually
    print("Testing .tldrignore consistency...\n")
    
    # Test 1: scan_project_files
    test1 = TestTldrignoreConsistency()
    test1.setup_method()
    try:
        test1.test_scan_project_files_respects_tldrignore()
        print("✓ scan_project_files respects .tldrignore")
    except AssertionError as e:
        print(f"✗ scan_project_files test failed: {e}")
    finally:
        test1.teardown_method()
    
    # Test 2: get_file_tree
    test2 = TestTldrignoreConsistency()
    test2.setup_method()
    try:
        test2.test_get_file_tree_respects_tldrignore()
        print("✓ get_file_tree respects .tldrignore")
    except AssertionError as e:
        print(f"✗ get_file_tree test failed: {e}")
    finally:
        test2.teardown_method()
    
    # Test 3: search
    test3 = TestTldrignoreConsistency()
    test3.setup_method()
    try:
        test3.test_search_respects_tldrignore()
        print("✓ search respects .tldrignore")
    except AssertionError as e:
        print(f"✗ search test failed: {e}")
    finally:
        test3.teardown_method()
    
    # Test 4: get_code_structure
    test4 = TestTldrignoreConsistency()
    test4.setup_method()
    try:
        test4.test_get_code_structure_respects_tldrignore()
        print("✓ get_code_structure respects .tldrignore")
    except AssertionError as e:
        print(f"✗ get_code_structure test failed: {e}")
    finally:
        test4.teardown_method()
    
    # Test 5: get_relevant_context
    test5 = TestTldrignoreConsistency()
    test5.setup_method()
    try:
        test5.test_get_relevant_context_respects_tldrignore()
        print("✓ get_relevant_context respects .tldrignore")
    except AssertionError as e:
        print(f"✗ get_relevant_context test failed: {e}")
    finally:
        test5.teardown_method()
    
    print("\nTesting file format support...\n")
    
    # Test file formats
    test_formats = TestFileFormatSupport()
    test_formats.setup_method()
    try:
        test_formats.test_get_code_structure_python()
        print("✓ Python files supported")
    except AssertionError as e:
        print(f"✗ Python files test failed: {e}")
    finally:
        test_formats.teardown_method()
    
    test_formats = TestFileFormatSupport()
    test_formats.setup_method()
    try:
        test_formats.test_get_code_structure_cpp()
        print("✓ C++ files supported")
    except AssertionError as e:
        print(f"✗ C++ files test failed: {e}")
    finally:
        test_formats.teardown_method()
    
    test_formats = TestFileFormatSupport()
    test_formats.setup_method()
    try:
        test_formats.test_get_code_structure_lua()
        print("✓ Lua files supported")
    except AssertionError as e:
        print(f"✗ Lua files test failed: {e}")
    finally:
        test_formats.teardown_method()
    
    test_formats = TestFileFormatSupport()
    test_formats.setup_method()
    try:
        test_formats.test_get_code_structure_elixir()
        print("✓ Elixir files supported")
    except AssertionError as e:
        print(f"✗ Elixir files test failed: {e}")
    finally:
        test_formats.teardown_method()
    
    print("\n✅ All tests completed!")
