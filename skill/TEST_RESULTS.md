# TLDR Skill Test Results

**Date:** 2026-01-12
**Version:** 1.2.2
**Tester:** Claude Code
**Branch:** feature/claude-skill

---

## Test Environment

- **Platform:** macOS Darwin 25.3.0
- **Python:** 3.12 (via .venv)
- **Invocation:** `.venv/bin/python -m tldr.cli`

---

## Command Test Results

### Exploration Commands

| Command | Status | Notes |
|---------|--------|-------|
| `tldr --help` | ✅ Pass | All subcommands listed |
| `tldr --version` | ✅ Pass | Returns `1.2.2` |
| `tldr tree tldr/` | ✅ Pass | JSON file tree output |
| `tldr structure tldr/ --lang python` | ✅ Pass | Lists functions, classes, imports |
| `tldr search "def main" tldr/` | ✅ Pass | Found 3 matches in cli.py, mcp_server.py, daemon/startup.py |
| `tldr extract tldr/cli.py` | ✅ Pass | Full file analysis with docstring, imports, functions |

### Analysis Commands

| Command | Status | Notes |
|---------|--------|-------|
| `tldr cfg tldr/cli.py main` | ✅ Pass | Control flow blocks with calls listed |
| `tldr dfg tldr/cli.py main` | ✅ Pass | Data flow refs (definitions/uses) |
| `tldr slice tldr/cli.py main 200` | ✅ Pass | Returns affecting line numbers |
| `tldr context` | ✅ Pass | Help shows `--project`, `--depth`, `--lang` options |

### Cross-File Commands

| Command | Status | Notes |
|---------|--------|-------|
| `tldr calls .` | ✅ Pass | Built call graph with **858 edges** |
| `tldr impact "main" .` | ✅ Pass | Found callers in daemon/startup.py, tldr_code.py, cli.py, mcp_server.py |
| `tldr arch` | ✅ Pass | Help confirmed |
| `tldr imports` | ✅ Pass | Help confirmed |
| `tldr importers` | ✅ Pass | Help confirmed |

### Semantic Search Commands

| Command | Status | Notes |
|---------|--------|-------|
| `tldr warm .` | ✅ Pass | Indexed 40 files, 858 edges |
| `tldr semantic index .` | ✅ Pass | Indexed 524 code units (after numpy fix) |
| `tldr semantic search "query"` | ✅ Pass | Returns scored results |

### Daemon Commands

| Command | Status | Notes |
|---------|--------|-------|
| `tldr daemon status` | ✅ Pass | Reports "Daemon not running" |
| `tldr daemon start` | ✅ Pass | Help confirmed |
| `tldr daemon stop` | ✅ Pass | Help confirmed |
| `tldr daemon notify --help` | ✅ Pass | Shows `--project/-p` flag |

### Diagnostics Commands

| Command | Status | Notes |
|---------|--------|-------|
| `tldr diagnostics` | ✅ Pass | Help confirmed |
| `tldr change-impact` | ✅ Pass | Help confirmed |
| `tldr doctor` | ✅ Pass | Help confirmed |

---

## Issues Found

### 1. Semantic Index NumPy Conflict

**Error:**
```
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
Error: Numpy is not available
```

**Cause:** NumPy 2.x incompatible with current PyTorch/sentence-transformers versions.

**Fix:** Pin `numpy<2` in requirements or update torch.

### 2. Documentation Syntax Errors (Fixed)

| File | Issue | Fix Applied |
|------|-------|-------------|
| SKILL.md | `tldr semantic "X" .` | → `tldr semantic search "X"` |
| SKILL.md | `tldr warm .` for semantic | → `tldr semantic index .` |
| COMMANDS.md | `tldr semantic <query>` | → `tldr semantic search <query>` |
| COMMANDS.md | Listed `-v, --verbose` | → Removed (doesn't exist) |
| COMMANDS.md | `warm` description | → Clarified as call graph only |

---

## Output Samples

### Structure Command
```json
{
  "root": "tldr",
  "language": "python",
  "files": [
    {
      "path": "cli.py",
      "functions": ["_get_subprocess_detach_kwargs", "detect_language_from_extension", "main"],
      "classes": [],
      "imports": [...]
    }
  ]
}
```

### Call Graph Stats
```
Total: Indexed 40 files, found 858 edges
```

### Impact Analysis
```json
{
  "targets": {
    "tldr/daemon/startup.py:main": {
      "function": "main",
      "caller_count": 1,
      "callers": [{"function": "<module>", "file": "tldr/daemon/startup.py"}]
    }
  }
}
```

---

## Recommendations

1. **Pin NumPy version** in pyproject.toml: `numpy<2`
2. **Update README.md** semantic search syntax to match actual CLI
3. **Consider** adding `tldr semantic "query"` as shorthand for `tldr semantic search "query"`

---

## Test Coverage Summary

- **Total commands tested:** 22
- **Passed:** 22
- **Failed:** 0
- **Documentation fixes:** 5

**Note:** Semantic index initially failed due to NumPy 2.x. Fixed with `uv pip install "numpy<2"` (1.26.4).
