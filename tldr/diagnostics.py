"""
Real-time diagnostics for TLDR.

Wraps type checkers (pyright, mypy) and linters (ruff) to provide
structured error output for LLM agents.

Supports:
- Python: pyright (type checker) + ruff (linter)
- TypeScript/JavaScript: tsc (type checker)
- Go: go vet (type checker) + golangci-lint (linter)
- Rust: cargo check (type checker) + clippy (linter)
- Java: javac (type checker) + checkstyle (linter)
- C/C++: clang/gcc (type checker) + cppcheck (linter)
- Ruby: rubocop (linter)
- PHP: phpstan (linter)
- Kotlin: kotlinc (type checker) + ktlint (linter)
- Swift: swiftc (type checker) + swiftlint (linter)
- C#: dotnet build (type checker)
- Scala: scalac (type checker)
- Elixir: mix compile (type checker) + credo (linter)
"""

import json
import re
import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree


# Mapping of language -> tools configuration
LANG_TOOLS: dict[str, dict] = {
    "python": {
        "type_checker": "pyright",
        "linter": "ruff",
    },
    "typescript": {
        "type_checker": "tsc",
        "linter": None,
    },
    "javascript": {
        "type_checker": None,
        "linter": None,  # Could add eslint
    },
    "go": {
        "type_checker": "go vet",
        "linter": "golangci-lint",
    },
    "rust": {
        "type_checker": "cargo check",
        "linter": "clippy",
    },
    "java": {
        "type_checker": "javac",
        "linter": "checkstyle",
    },
    "c": {
        "type_checker": "gcc",
        "linter": "cppcheck",
    },
    "cpp": {
        "type_checker": "g++",
        "linter": "cppcheck",
    },
    "ruby": {
        "type_checker": None,
        "linter": "rubocop",
    },
    "php": {
        "type_checker": None,
        "linter": "phpstan",
    },
    "kotlin": {
        "type_checker": "kotlinc",
        "linter": "ktlint",
    },
    "swift": {
        "type_checker": "swiftc",
        "linter": "swiftlint",
    },
    "csharp": {
        "type_checker": "dotnet build",
        "linter": None,
    },
    "scala": {
        "type_checker": "scalac",
        "linter": None,
    },
    "elixir": {
        "type_checker": "mix compile",
        "linter": "credo",
    },
}


def _detect_language(file_path: str) -> str:
    """Detect language from file extension."""
    ext = Path(file_path).suffix.lower()
    mapping = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".php": "php",
        ".kt": "kotlin",
        ".swift": "swift",
        ".cs": "csharp",
        ".scala": "scala",
        ".ex": "elixir",
        ".exs": "elixir",
    }
    return mapping.get(ext, "unknown")


def _parse_pyright_output(stdout: str) -> list[dict]:
    """Parse pyright JSON output into structured diagnostics."""
    try:
        data = json.loads(stdout)
        diagnostics = []
        for diag in data.get("generalDiagnostics", []):
            diagnostics.append({
                "file": diag.get("file", ""),
                "line": diag.get("range", {}).get("start", {}).get("line", 0) + 1,
                "column": diag.get("range", {}).get("start", {}).get("character", 0) + 1,
                "severity": diag.get("severity", "error"),
                "message": diag.get("message", ""),
                "rule": diag.get("rule", ""),
                "source": "pyright",
            })
        return diagnostics
    except json.JSONDecodeError:
        return []


def _parse_ruff_output(stdout: str) -> list[dict]:
    """Parse ruff JSON output into structured diagnostics."""
    try:
        data = json.loads(stdout)
        diagnostics = []
        for diag in data:
            diagnostics.append({
                "file": diag.get("filename", ""),
                "line": diag.get("location", {}).get("row", 0),
                "column": diag.get("location", {}).get("column", 0),
                "severity": "warning",  # ruff is mostly lint warnings
                "message": diag.get("message", ""),
                "rule": diag.get("code", ""),
                "source": "ruff",
            })
        return diagnostics
    except json.JSONDecodeError:
        return []


def _parse_tsc_output(stderr: str) -> list[dict]:
    """Parse tsc output into structured diagnostics."""
    diagnostics = []
    # tsc format: file(line,col): error TSxxxx: message
    pattern = r"(.+?)\((\d+),(\d+)\):\s*(error|warning)\s+(TS\d+):\s*(.+)"
    for line in stderr.strip().split("\n"):
        match = re.match(pattern, line)
        if match:
            diagnostics.append({
                "file": match.group(1),
                "line": int(match.group(2)),
                "column": int(match.group(3)),
                "severity": match.group(4),
                "message": match.group(6),
                "rule": match.group(5),
                "source": "tsc",
            })
    return diagnostics


def _parse_go_vet_output(stderr: str) -> list[dict]:
    """Parse go vet output into structured diagnostics."""
    diagnostics = []
    if not stderr.strip():
        return diagnostics
    # go vet format: file.go:line:col: message
    pattern = r"(.+?):(\d+):(\d+):\s*(.+)"
    for line in stderr.strip().split("\n"):
        match = re.match(pattern, line)
        if match:
            diagnostics.append({
                "file": match.group(1),
                "line": int(match.group(2)),
                "column": int(match.group(3)),
                "severity": "error",
                "message": match.group(4),
                "rule": "",
                "source": "go vet",
            })
    return diagnostics


def _parse_golangci_lint_output(stdout: str) -> list[dict]:
    """Parse golangci-lint JSON output into structured diagnostics."""
    try:
        data = json.loads(stdout)
        diagnostics = []
        for issue in data.get("Issues", []):
            pos = issue.get("Pos", {})
            diagnostics.append({
                "file": pos.get("Filename", ""),
                "line": pos.get("Line", 0),
                "column": pos.get("Column", 0),
                "severity": "warning",
                "message": issue.get("Text", ""),
                "rule": issue.get("FromLinter", ""),
                "source": "golangci-lint",
            })
        return diagnostics
    except json.JSONDecodeError:
        return []


def _parse_cargo_check_output(stdout: str) -> list[dict]:
    """Parse cargo check JSON output into structured diagnostics."""
    diagnostics = []
    if not stdout.strip():
        return diagnostics
    # cargo outputs one JSON object per line
    for line in stdout.strip().split("\n"):
        try:
            data = json.loads(line)
            if data.get("reason") != "compiler-message":
                continue
            msg = data.get("message", {})
            spans = msg.get("spans", [])
            if not spans:
                continue
            span = spans[0]
            code = msg.get("code", {})
            diagnostics.append({
                "file": span.get("file_name", ""),
                "line": span.get("line_start", 0),
                "column": span.get("column_start", 0),
                "severity": msg.get("level", "error"),
                "message": msg.get("message", ""),
                "rule": code.get("code", "") if code else "",
                "source": "cargo",
            })
        except json.JSONDecodeError:
            continue
    return diagnostics


def _parse_clippy_output(stdout: str) -> list[dict]:
    """Parse cargo clippy JSON output into structured diagnostics."""
    diagnostics = []
    if not stdout.strip():
        return diagnostics
    # clippy uses the same format as cargo check
    for line in stdout.strip().split("\n"):
        try:
            data = json.loads(line)
            if data.get("reason") != "compiler-message":
                continue
            msg = data.get("message", {})
            spans = msg.get("spans", [])
            if not spans:
                continue
            span = spans[0]
            code = msg.get("code", {})
            diagnostics.append({
                "file": span.get("file_name", ""),
                "line": span.get("line_start", 0),
                "column": span.get("column_start", 0),
                "severity": msg.get("level", "warning"),
                "message": msg.get("message", ""),
                "rule": code.get("code", "") if code else "",
                "source": "clippy",
            })
        except json.JSONDecodeError:
            continue
    return diagnostics


def _parse_rubocop_output(stdout: str) -> list[dict]:
    """Parse rubocop JSON output into structured diagnostics."""
    try:
        data = json.loads(stdout)
        diagnostics = []
        for file_info in data.get("files", []):
            file_path = file_info.get("path", "")
            for offense in file_info.get("offenses", []):
                loc = offense.get("location", {})
                diagnostics.append({
                    "file": file_path,
                    "line": loc.get("line", 0),
                    "column": loc.get("column", 0),
                    "severity": offense.get("severity", "warning"),
                    "message": offense.get("message", ""),
                    "rule": offense.get("cop_name", ""),
                    "source": "rubocop",
                })
        return diagnostics
    except json.JSONDecodeError:
        return []


def _parse_phpstan_output(stdout: str) -> list[dict]:
    """Parse phpstan JSON output into structured diagnostics."""
    try:
        data = json.loads(stdout)
        diagnostics = []
        for file_path, file_info in data.get("files", {}).items():
            for msg in file_info.get("messages", []):
                diagnostics.append({
                    "file": file_path,
                    "line": msg.get("line", 0),
                    "column": 0,  # phpstan doesn't provide column
                    "severity": "error",
                    "message": msg.get("message", ""),
                    "rule": "",
                    "source": "phpstan",
                })
        return diagnostics
    except json.JSONDecodeError:
        return []


def _parse_ktlint_output(stdout: str) -> list[dict]:
    """Parse ktlint JSON output into structured diagnostics."""
    try:
        data = json.loads(stdout)
        diagnostics = []
        for file_info in data:
            file_path = file_info.get("file", "")
            for error in file_info.get("errors", []):
                diagnostics.append({
                    "file": file_path,
                    "line": error.get("line", 0),
                    "column": error.get("column", 0),
                    "severity": "warning",
                    "message": error.get("message", ""),
                    "rule": error.get("rule", ""),
                    "source": "ktlint",
                })
        return diagnostics
    except json.JSONDecodeError:
        return []


def _parse_swiftlint_output(stdout: str) -> list[dict]:
    """Parse swiftlint JSON output into structured diagnostics."""
    try:
        data = json.loads(stdout)
        diagnostics = []
        for item in data:
            diagnostics.append({
                "file": item.get("file", ""),
                "line": item.get("line", 0),
                "column": item.get("column", 0),
                "severity": item.get("severity", "warning").lower(),
                "message": item.get("reason", ""),
                "rule": item.get("rule_id", ""),
                "source": "swiftlint",
            })
        return diagnostics
    except json.JSONDecodeError:
        return []


def _parse_cppcheck_output(stdout: str) -> list[dict]:
    """Parse cppcheck XML output into structured diagnostics."""
    diagnostics = []
    if not stdout.strip():
        return diagnostics
    try:
        root = ElementTree.fromstring(stdout)
        for error in root.findall(".//error"):
            location = error.find("location")
            if location is not None:
                diagnostics.append({
                    "file": location.get("file", ""),
                    "line": int(location.get("line", 0)),
                    "column": int(location.get("column", 0)),
                    "severity": error.get("severity", "error"),
                    "message": error.get("msg", ""),
                    "rule": error.get("id", ""),
                    "source": "cppcheck",
                })
        return diagnostics
    except ElementTree.ParseError:
        return []


def _parse_credo_output(stdout: str) -> list[dict]:
    """Parse credo JSON output into structured diagnostics."""
    try:
        data = json.loads(stdout)
        diagnostics = []
        for issue in data.get("issues", []):
            diagnostics.append({
                "file": issue.get("filename", ""),
                "line": issue.get("line_no", 0),
                "column": issue.get("column", 0),
                "severity": "warning",
                "message": issue.get("message", ""),
                "rule": issue.get("check", ""),
                "source": "credo",
            })
        return diagnostics
    except json.JSONDecodeError:
        return []


def _parse_javac_output(stderr: str) -> list[dict]:
    """Parse javac output into structured diagnostics."""
    diagnostics = []
    if not stderr.strip():
        return diagnostics
    # javac format: file.java:line: error: message
    pattern = r"(.+?):(\d+):\s*(error|warning):\s*(.+)"
    for line in stderr.strip().split("\n"):
        match = re.match(pattern, line)
        if match:
            diagnostics.append({
                "file": match.group(1),
                "line": int(match.group(2)),
                "column": 0,
                "severity": match.group(3),
                "message": match.group(4),
                "rule": "",
                "source": "javac",
            })
    return diagnostics


def _parse_checkstyle_output(stdout: str) -> list[dict]:
    """Parse checkstyle XML output into structured diagnostics."""
    diagnostics = []
    if not stdout.strip():
        return diagnostics
    try:
        root = ElementTree.fromstring(stdout)
        for file_elem in root.findall("file"):
            file_path = file_elem.get("name", "")
            for error in file_elem.findall("error"):
                diagnostics.append({
                    "file": file_path,
                    "line": int(error.get("line", 0)),
                    "column": int(error.get("column", 0)),
                    "severity": error.get("severity", "warning"),
                    "message": error.get("message", ""),
                    "rule": error.get("source", "").split(".")[-1],
                    "source": "checkstyle",
                })
        return diagnostics
    except ElementTree.ParseError:
        return []


def _parse_gcc_output(stderr: str) -> list[dict]:
    """Parse gcc/g++/clang output into structured diagnostics."""
    diagnostics = []
    if not stderr.strip():
        return diagnostics
    # gcc format: file.c:line:col: error: message
    pattern = r"(.+?):(\d+):(\d+):\s*(error|warning):\s*(.+)"
    for line in stderr.strip().split("\n"):
        match = re.match(pattern, line)
        if match:
            diagnostics.append({
                "file": match.group(1),
                "line": int(match.group(2)),
                "column": int(match.group(3)),
                "severity": match.group(4),
                "message": match.group(5),
                "rule": "",
                "source": "gcc",
            })
    return diagnostics


def _parse_kotlinc_output(stderr: str) -> list[dict]:
    """Parse kotlinc output into structured diagnostics."""
    diagnostics = []
    if not stderr.strip():
        return diagnostics
    # kotlinc format: file.kt:line:col: error: message
    pattern = r"(.+?):(\d+):(\d+):\s*(error|warning):\s*(.+)"
    for line in stderr.strip().split("\n"):
        match = re.match(pattern, line)
        if match:
            diagnostics.append({
                "file": match.group(1),
                "line": int(match.group(2)),
                "column": int(match.group(3)),
                "severity": match.group(4),
                "message": match.group(5),
                "rule": "",
                "source": "kotlinc",
            })
    return diagnostics


def _parse_swiftc_output(stderr: str) -> list[dict]:
    """Parse swiftc output into structured diagnostics."""
    diagnostics = []
    if not stderr.strip():
        return diagnostics
    # swiftc format: file.swift:line:col: error: message
    pattern = r"(.+?):(\d+):(\d+):\s*(error|warning):\s*(.+)"
    for line in stderr.strip().split("\n"):
        match = re.match(pattern, line)
        if match:
            diagnostics.append({
                "file": match.group(1),
                "line": int(match.group(2)),
                "column": int(match.group(3)),
                "severity": match.group(4),
                "message": match.group(5),
                "rule": "",
                "source": "swiftc",
            })
    return diagnostics


def _parse_dotnet_build_output(stderr: str) -> list[dict]:
    """Parse dotnet build output into structured diagnostics."""
    diagnostics = []
    if not stderr.strip():
        return diagnostics
    # dotnet format: file.cs(line,col): error CS0000: message
    pattern = r"(.+?)\((\d+),(\d+)\):\s*(error|warning)\s+(\w+):\s*(.+)"
    for line in stderr.strip().split("\n"):
        match = re.match(pattern, line)
        if match:
            diagnostics.append({
                "file": match.group(1),
                "line": int(match.group(2)),
                "column": int(match.group(3)),
                "severity": match.group(4),
                "message": match.group(6),
                "rule": match.group(5),
                "source": "dotnet",
            })
    return diagnostics


def _parse_scalac_output(stderr: str) -> list[dict]:
    """Parse scalac output into structured diagnostics."""
    diagnostics = []
    if not stderr.strip():
        return diagnostics
    # scalac format varies, common: file.scala:line: error: message
    pattern = r"(.+?):(\d+):\s*(error|warning):\s*(.+)"
    for line in stderr.strip().split("\n"):
        match = re.match(pattern, line)
        if match:
            diagnostics.append({
                "file": match.group(1),
                "line": int(match.group(2)),
                "column": 0,
                "severity": match.group(3),
                "message": match.group(4),
                "rule": "",
                "source": "scalac",
            })
    return diagnostics


def _parse_mix_compile_output(stderr: str) -> list[dict]:
    """Parse mix compile output into structured diagnostics."""
    diagnostics = []
    if not stderr.strip():
        return diagnostics
    # mix compile format: warning: message
    #   file.ex:line
    # Or: ** (CompileError) file.ex:line: message
    pattern = r"\*\*\s*\((\w+)\)\s*(.+?):(\d+):\s*(.+)"
    for line in stderr.strip().split("\n"):
        match = re.match(pattern, line)
        if match:
            diagnostics.append({
                "file": match.group(2),
                "line": int(match.group(3)),
                "column": 0,
                "severity": "error" if "Error" in match.group(1) else "warning",
                "message": match.group(4),
                "rule": "",
                "source": "mix",
            })
    return diagnostics


def get_diagnostics(
    file_path: str,
    language: str | None = None,
    include_lint: bool = True,
) -> dict:
    """
    Get type and lint diagnostics for a file.

    Args:
        file_path: Path to the source file
        language: Override language detection (python, typescript, go, rust, etc.)
        include_lint: Include linter diagnostics (default: True)

    Returns:
        Dict with 'diagnostics' list and metadata
    """
    path = Path(file_path).resolve()
    if not path.exists():
        return {"error": f"File not found: {file_path}", "diagnostics": []}

    lang = language or _detect_language(str(path))
    all_diagnostics = []
    tools_used = []

    if lang == "python":
        # Run pyright for type checking
        if shutil.which("pyright"):
            try:
                result = subprocess.run(
                    ["pyright", "--outputjson", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                all_diagnostics.extend(_parse_pyright_output(result.stdout))
                tools_used.append("pyright")
            except subprocess.TimeoutExpired:
                pass

        # Run ruff for linting
        if include_lint and shutil.which("ruff"):
            try:
                result = subprocess.run(
                    ["ruff", "check", "--output-format=json", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                all_diagnostics.extend(_parse_ruff_output(result.stdout))
                tools_used.append("ruff")
            except subprocess.TimeoutExpired:
                pass

    elif lang == "typescript":
        # Run tsc for type checking
        if shutil.which("tsc"):
            try:
                result = subprocess.run(
                    ["tsc", "--noEmit", "--pretty", "false", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                all_diagnostics.extend(_parse_tsc_output(result.stderr))
                tools_used.append("tsc")
            except subprocess.TimeoutExpired:
                pass

    elif lang == "go":
        # Run go vet for type checking
        if shutil.which("go"):
            try:
                result = subprocess.run(
                    ["go", "vet", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                all_diagnostics.extend(_parse_go_vet_output(result.stderr))
                tools_used.append("go vet")
            except subprocess.TimeoutExpired:
                pass

        # Run golangci-lint for linting
        if include_lint and shutil.which("golangci-lint"):
            try:
                result = subprocess.run(
                    ["golangci-lint", "run", "--out-format=json", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                all_diagnostics.extend(_parse_golangci_lint_output(result.stdout))
                tools_used.append("golangci-lint")
            except subprocess.TimeoutExpired:
                pass

    elif lang == "rust":
        # Run cargo check for type checking
        if shutil.which("cargo"):
            try:
                result = subprocess.run(
                    ["cargo", "check", "--message-format=json"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(path.parent),
                )
                all_diagnostics.extend(_parse_cargo_check_output(result.stdout))
                tools_used.append("cargo check")
            except subprocess.TimeoutExpired:
                pass

        # Run clippy for linting
        if include_lint and shutil.which("cargo"):
            try:
                result = subprocess.run(
                    ["cargo", "clippy", "--message-format=json"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(path.parent),
                )
                all_diagnostics.extend(_parse_clippy_output(result.stdout))
                tools_used.append("clippy")
            except subprocess.TimeoutExpired:
                pass

    elif lang == "java":
        # Run javac for type checking
        if shutil.which("javac"):
            try:
                result = subprocess.run(
                    ["javac", "-Xlint:all", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                all_diagnostics.extend(_parse_javac_output(result.stderr))
                tools_used.append("javac")
            except subprocess.TimeoutExpired:
                pass

        # Run checkstyle for linting
        if include_lint and shutil.which("checkstyle"):
            try:
                result = subprocess.run(
                    ["checkstyle", "-f", "xml", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                all_diagnostics.extend(_parse_checkstyle_output(result.stdout))
                tools_used.append("checkstyle")
            except subprocess.TimeoutExpired:
                pass

    elif lang in ("c", "cpp"):
        # Run gcc/g++ for type checking
        compiler = "g++" if lang == "cpp" else "gcc"
        if shutil.which(compiler):
            try:
                result = subprocess.run(
                    [compiler, "-fsyntax-only", "-Wall", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                all_diagnostics.extend(_parse_gcc_output(result.stderr))
                tools_used.append(compiler)
            except subprocess.TimeoutExpired:
                pass

        # Run cppcheck for linting
        if include_lint and shutil.which("cppcheck"):
            try:
                result = subprocess.run(
                    ["cppcheck", "--xml", "--enable=all", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                all_diagnostics.extend(_parse_cppcheck_output(result.stderr))
                tools_used.append("cppcheck")
            except subprocess.TimeoutExpired:
                pass

    elif lang == "ruby":
        # No type checker for Ruby by default
        # Run rubocop for linting
        if include_lint and shutil.which("rubocop"):
            try:
                result = subprocess.run(
                    ["rubocop", "--format", "json", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                all_diagnostics.extend(_parse_rubocop_output(result.stdout))
                tools_used.append("rubocop")
            except subprocess.TimeoutExpired:
                pass

    elif lang == "php":
        # No type checker for PHP by default
        # Run phpstan for linting
        if include_lint and shutil.which("phpstan"):
            try:
                result = subprocess.run(
                    ["phpstan", "analyse", "--error-format=json", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                all_diagnostics.extend(_parse_phpstan_output(result.stdout))
                tools_used.append("phpstan")
            except subprocess.TimeoutExpired:
                pass

    elif lang == "kotlin":
        # Run kotlinc for type checking
        if shutil.which("kotlinc"):
            try:
                result = subprocess.run(
                    ["kotlinc", "-d", "/dev/null", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                all_diagnostics.extend(_parse_kotlinc_output(result.stderr))
                tools_used.append("kotlinc")
            except subprocess.TimeoutExpired:
                pass

        # Run ktlint for linting
        if include_lint and shutil.which("ktlint"):
            try:
                result = subprocess.run(
                    ["ktlint", "--reporter=json", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                all_diagnostics.extend(_parse_ktlint_output(result.stdout))
                tools_used.append("ktlint")
            except subprocess.TimeoutExpired:
                pass

    elif lang == "swift":
        # Run swiftc for type checking
        if shutil.which("swiftc"):
            try:
                result = subprocess.run(
                    ["swiftc", "-typecheck", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                all_diagnostics.extend(_parse_swiftc_output(result.stderr))
                tools_used.append("swiftc")
            except subprocess.TimeoutExpired:
                pass

        # Run swiftlint for linting
        if include_lint and shutil.which("swiftlint"):
            try:
                result = subprocess.run(
                    ["swiftlint", "lint", "--reporter", "json", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                all_diagnostics.extend(_parse_swiftlint_output(result.stdout))
                tools_used.append("swiftlint")
            except subprocess.TimeoutExpired:
                pass

    elif lang == "csharp":
        # Run dotnet build for type checking
        if shutil.which("dotnet"):
            try:
                result = subprocess.run(
                    ["dotnet", "build", "--no-restore", str(path.parent)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                all_diagnostics.extend(_parse_dotnet_build_output(result.stderr))
                tools_used.append("dotnet build")
            except subprocess.TimeoutExpired:
                pass

    elif lang == "scala":
        # Run scalac for type checking
        if shutil.which("scalac"):
            try:
                result = subprocess.run(
                    ["scalac", "-d", "/tmp", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                all_diagnostics.extend(_parse_scalac_output(result.stderr))
                tools_used.append("scalac")
            except subprocess.TimeoutExpired:
                pass

    elif lang == "elixir":
        # Run mix compile for type checking
        if shutil.which("mix"):
            try:
                result = subprocess.run(
                    ["mix", "compile", "--warnings-as-errors"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(path.parent),
                )
                all_diagnostics.extend(_parse_mix_compile_output(result.stderr))
                tools_used.append("mix compile")
            except subprocess.TimeoutExpired:
                pass

        # Run credo for linting
        if include_lint and shutil.which("mix"):
            try:
                result = subprocess.run(
                    ["mix", "credo", "--format", "json", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(path.parent),
                )
                all_diagnostics.extend(_parse_credo_output(result.stdout))
                tools_used.append("credo")
            except subprocess.TimeoutExpired:
                pass

    # Sort by line number
    all_diagnostics.sort(key=lambda d: (d.get("file", ""), d.get("line", 0)))

    return {
        "file": str(path),
        "language": lang,
        "tools": tools_used,
        "diagnostics": all_diagnostics,
        "error_count": sum(1 for d in all_diagnostics if d.get("severity") == "error"),
        "warning_count": sum(1 for d in all_diagnostics if d.get("severity") == "warning"),
    }


def get_project_diagnostics(
    project_path: str,
    language: str = "python",
    include_lint: bool = True,
) -> dict:
    """
    Get diagnostics for entire project.

    Uses pyright/tsc on the whole project for faster checking.
    """
    path = Path(project_path).resolve()
    if not path.exists():
        return {"error": f"Path not found: {project_path}", "diagnostics": []}

    all_diagnostics = []
    tools_used = []

    if language == "python":
        # Run pyright on project
        if shutil.which("pyright"):
            try:
                result = subprocess.run(
                    ["pyright", "--outputjson", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(path),
                )
                all_diagnostics.extend(_parse_pyright_output(result.stdout))
                tools_used.append("pyright")
            except subprocess.TimeoutExpired:
                pass

        # Run ruff on project
        if include_lint and shutil.which("ruff"):
            try:
                result = subprocess.run(
                    ["ruff", "check", "--output-format=json", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(path),
                )
                all_diagnostics.extend(_parse_ruff_output(result.stdout))
                tools_used.append("ruff")
            except subprocess.TimeoutExpired:
                pass

    elif language == "typescript":
        # Run tsc on project
        if shutil.which("tsc"):
            try:
                result = subprocess.run(
                    ["tsc", "--noEmit", "--pretty", "false"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(path),
                )
                all_diagnostics.extend(_parse_tsc_output(result.stderr))
                tools_used.append("tsc")
            except subprocess.TimeoutExpired:
                pass

    elif language == "go":
        # Run go vet on project
        if shutil.which("go"):
            try:
                result = subprocess.run(
                    ["go", "vet", "./..."],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(path),
                )
                all_diagnostics.extend(_parse_go_vet_output(result.stderr))
                tools_used.append("go vet")
            except subprocess.TimeoutExpired:
                pass

        # Run golangci-lint on project
        if include_lint and shutil.which("golangci-lint"):
            try:
                result = subprocess.run(
                    ["golangci-lint", "run", "--out-format=json", "./..."],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(path),
                )
                all_diagnostics.extend(_parse_golangci_lint_output(result.stdout))
                tools_used.append("golangci-lint")
            except subprocess.TimeoutExpired:
                pass

    elif language == "rust":
        # Run cargo check on project
        if shutil.which("cargo"):
            try:
                result = subprocess.run(
                    ["cargo", "check", "--message-format=json"],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    cwd=str(path),
                )
                all_diagnostics.extend(_parse_cargo_check_output(result.stdout))
                tools_used.append("cargo check")
            except subprocess.TimeoutExpired:
                pass

        # Run clippy on project
        if include_lint and shutil.which("cargo"):
            try:
                result = subprocess.run(
                    ["cargo", "clippy", "--message-format=json"],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    cwd=str(path),
                )
                all_diagnostics.extend(_parse_clippy_output(result.stdout))
                tools_used.append("clippy")
            except subprocess.TimeoutExpired:
                pass

    elif language == "ruby":
        # Run rubocop on project
        if include_lint and shutil.which("rubocop"):
            try:
                result = subprocess.run(
                    ["rubocop", "--format", "json", "."],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(path),
                )
                all_diagnostics.extend(_parse_rubocop_output(result.stdout))
                tools_used.append("rubocop")
            except subprocess.TimeoutExpired:
                pass

    elif language == "elixir":
        # Run mix compile on project
        if shutil.which("mix"):
            try:
                result = subprocess.run(
                    ["mix", "compile", "--warnings-as-errors"],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    cwd=str(path),
                )
                all_diagnostics.extend(_parse_mix_compile_output(result.stderr))
                tools_used.append("mix compile")
            except subprocess.TimeoutExpired:
                pass

        # Run credo on project
        if include_lint and shutil.which("mix"):
            try:
                result = subprocess.run(
                    ["mix", "credo", "--format", "json"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(path),
                )
                all_diagnostics.extend(_parse_credo_output(result.stdout))
                tools_used.append("credo")
            except subprocess.TimeoutExpired:
                pass

    # Sort by file, then line
    all_diagnostics.sort(key=lambda d: (d.get("file", ""), d.get("line", 0)))

    return {
        "project": str(path),
        "language": language,
        "tools": tools_used,
        "diagnostics": all_diagnostics,
        "error_count": sum(1 for d in all_diagnostics if d.get("severity") == "error"),
        "warning_count": sum(1 for d in all_diagnostics if d.get("severity") == "warning"),
        "file_count": len(set(d.get("file", "") for d in all_diagnostics)),
    }


def format_diagnostics_for_llm(result: dict) -> str:
    """
    Format diagnostics as concise text for LLM context.
    """
    if result.get("error"):
        return f"Error: {result['error']}"

    diagnostics = result.get("diagnostics", [])
    if not diagnostics:
        return "No diagnostics found."

    lines = []
    errors = result.get("error_count", 0)
    warnings = result.get("warning_count", 0)
    lines.append(f"Found {errors} errors, {warnings} warnings")
    lines.append("")

    for d in diagnostics:
        severity = "E" if d.get("severity") == "error" else "W"
        rule = f" [{d['rule']}]" if d.get("rule") else ""
        lines.append(f"{severity} {d['file']}:{d['line']}:{d['column']}: {d['message']}{rule}")

    return "\n".join(lines)
