# TLDR: Code Analysis for AI Agents

**Give LLMs exactly the code they need. Nothing more.**

Your codebase is 100K lines. Claude's context window is 200K tokens. Raw code won't fit—and even if it did, the LLM would drown in irrelevant details.

TLDR extracts *structure* instead of dumping *text*. The result: **95% fewer tokens** while preserving everything needed to understand and edit code correctly.

```bash
pip install llm-tldr
tldr warm .                    # Index your project
tldr context main --project .  # Get LLM-ready summary
```

---

## How It Works

TLDR builds 5 analysis layers, each answering different questions:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: Program Dependence  → "What affects line 42?"      │
│ Layer 4: Data Flow           → "Where does this value go?"  │
│ Layer 3: Control Flow        → "How complex is this?"       │
│ Layer 2: Call Graph          → "Who calls this function?"   │
│ Layer 1: AST                 → "What functions exist?"      │
└─────────────────────────────────────────────────────────────┘
```

**Why layers?** Different tasks need different depth:
- Browsing code? Layer 1 (structure) is enough
- Refactoring? Layer 2 (call graph) shows what breaks
- Debugging null? Layer 5 (slice) shows only relevant lines

The daemon keeps indexes in memory for **100ms queries** instead of 30-second CLI spawns.

---

## The Workflow

### Before Reading Code
```bash
tldr tree src/                      # See file structure
tldr structure src/ --lang python   # See functions/classes
```

### Before Editing
```bash
tldr extract src/auth.py            # Full file analysis
tldr context login --project .      # LLM-ready summary (95% savings)
```

### Before Refactoring
```bash
tldr impact login .                 # Who calls this? (reverse call graph)
tldr change-impact                  # Which tests need to run?
```

### Debugging
```bash
tldr slice src/auth.py login 42     # What affects line 42?
tldr dfg src/auth.py login          # Trace data flow
```

### Finding Code by Behavior
```bash
tldr semantic "validate JWT tokens" .   # Natural language search
```

---

## Quick Setup

### 1. Install

```bash
pip install llm-tldr
```

### 2. Index Your Project

```bash
tldr warm /path/to/project
```

This builds all analysis layers and starts the daemon. Takes 30-60 seconds for a typical project, then queries are instant.

### 3. Start Using

```bash
tldr context main --project .   # Get context for a function
tldr impact helper_func .       # See who calls it
tldr semantic "error handling"  # Find by behavior
```

---

## Real Example: Why This Matters

**Scenario:** Debug why `user` is null on line 42.

**Without TLDR:**
1. Read the 150-line function
2. Trace every variable manually
3. Miss the bug because it's hidden in control flow

**With TLDR:**
```bash
tldr slice src/auth.py login 42
```

**Output:** Only 6 lines that affect line 42:
```python
3:   user = db.get_user(username)
7:   if user is None:
12:      raise NotFound
28:  token = create_token(user)  # ← BUG: skipped null check
35:  session.token = token
42:  return session
```

The bug is obvious. Line 28 uses `user` without going through the null check path.

---

## Command Reference

### Exploration
| Command | What It Does |
|---------|--------------|
| `tldr tree [path]` | File tree |
| `tldr structure [path] --lang <lang>` | Functions, classes, methods |
| `tldr search <pattern> [path]` | Text pattern search |
| `tldr extract <file>` | Full file analysis |

### Analysis
| Command | What It Does |
|---------|--------------|
| `tldr context <func> --project <path>` | LLM-ready summary (95% savings) |
| `tldr cfg <file> <function>` | Control flow graph |
| `tldr dfg <file> <function>` | Data flow graph |
| `tldr slice <file> <func> <line>` | Program slice |

### Cross-File
| Command | What It Does |
|---------|--------------|
| `tldr calls [path]` | Build call graph |
| `tldr impact <func> [path]` | Find all callers (reverse call graph) |
| `tldr dead [path]` | Find unreachable code |
| `tldr arch [path]` | Detect architecture layers |
| `tldr imports <file>` | Parse imports |
| `tldr importers <module> [path]` | Find files that import a module |

### Semantic
| Command | What It Does |
|---------|--------------|
| `tldr warm <path>` | Build all indexes (including embeddings) |
| `tldr semantic <query> [path]` | Natural language code search |

### Diagnostics
| Command | What It Does |
|---------|--------------|
| `tldr diagnostics <file>` | Type check + lint |
| `tldr change-impact [files]` | Find tests affected by changes |
| `tldr doctor` | Check/install diagnostic tools |

### Daemon
| Command | What It Does |
|---------|--------------|
| `tldr daemon start` | Start background daemon |
| `tldr daemon stop` | Stop daemon |
| `tldr daemon status` | Check status |

---

## Supported Languages

Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, Ruby, PHP, C#, Kotlin, Scala, Swift, Lua, Elixir

Language is auto-detected or specify with `--lang`.

---

## MCP Integration

For AI tools (Claude Desktop, Claude Code):

**Claude Desktop** - Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "tldr": {
      "command": "tldr-mcp",
      "args": ["--project", "/path/to/your/project"]
    }
  }
}
```

**Claude Code** - Add to `.claude/settings.json`:
```json
{
  "mcpServers": {
    "tldr": {
      "command": "tldr-mcp",
      "args": ["--project", "."]
    }
  }
}
```

---

## Performance

| Metric | Raw Code | TLDR | Improvement |
|--------|----------|------|-------------|
| Tokens for function context | 21,000 | 175 | **99% savings** |
| Tokens for codebase overview | 104,000 | 12,000 | **89% savings** |
| Query latency (daemon) | 30s | 100ms | **300x faster** |

---

## Deep Dive

For the full architecture explanation, benchmarks, and advanced workflows:

**[Full Documentation](./docs/TLDR.md)**

---

## License

Apache 2.0 - See LICENSE file.
