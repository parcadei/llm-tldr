---
name: llm-tldr
description: Code analysis for AI agents via 5-layer semantic indexing. Use when exploring codebases, finding functions by behavior, analyzing call graphs, tracing data flow, debugging with program slices, or preparing LLM-ready context. Triggers on "find code that does X", "what calls this", "trace this variable", "analyze this function", "code structure", "impact analysis".
license: Apache-2.0
compatibility: Requires Python 3.9+ and tree-sitter. Designed for Claude Code with MCP integration.
metadata:
  author: llm-tldr
  version: "1.2.2"
  source: https://github.com/jkneen/llm-tldr
---

# TLDR: Code Analysis for AI Agents

TLDR extracts *structure* instead of dumping *text*. Result: **95% fewer tokens** while preserving everything needed to understand and edit code.

## Quick Setup

```bash
pip install llm-tldr
tldr warm .  # Index the project (one-time, ~60s)
```

## When to Use This Skill

| User Intent | TLDR Command |
|-------------|--------------|
| "Find code that handles X" | `tldr semantic search "X"` |
| "What functions exist in this file?" | `tldr structure <file> --lang <lang>` |
| "What calls this function?" | `tldr impact <function> .` |
| "How does data flow through this?" | `tldr dfg <file> <function>` |
| "What affects line N?" (debugging) | `tldr slice <file> <function> <line>` |
| "Give me context for editing X" | `tldr context <function> --project .` |
| "Show the project structure" | `tldr tree .` |

## Core Commands

### Exploration
```bash
tldr tree <path>                     # File tree
tldr structure <path> --lang python  # Functions, classes, methods
tldr extract <file>                  # Full file analysis
```

### Semantic Search (Natural Language)
```bash
tldr semantic index .                      # Build embedding index first
tldr semantic search "validate JWT tokens" # Find by behavior, not text
```

### Impact Analysis
```bash
tldr calls .                         # Build call graph
tldr impact <function> .             # Who calls this? (reverse)
tldr dead .                          # Find unreachable code
```

### Debugging (Program Slicing)
```bash
tldr slice <file> <function> <line>  # What affects this line?
tldr dfg <file> <function>           # Data flow graph
tldr cfg <file> <function>           # Control flow graph
```

### LLM Context (95% Token Savings)
```bash
tldr context <function> --project .  # LLM-ready summary
```

## The 5 Analysis Layers

```
Layer 1: AST        → "What functions exist?"
Layer 2: Call Graph → "Who calls this function?"
Layer 3: CFG        → "How complex is this?"
Layer 4: DFG        → "Where does this value go?"
Layer 5: PDG        → "What affects line 42?"
```

## MCP Integration

For direct tool access in Claude Code, add to `.claude/settings.json`:

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

## Supported Languages

Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, Ruby, PHP, C#, Kotlin, Scala, Swift, Lua, Luau, Elixir

## Example: Debugging Workflow

**Problem:** Variable is null on line 42

Instead of reading 150 lines manually:
```bash
tldr slice src/auth.py login 42
```

**Output:** Only 6 lines that affect line 42:
```python
3:   user = db.get_user(username)
7:   if user is None:
12:      raise NotFound
28:  token = create_token(user)  # BUG: skipped null check
35:  session.token = token
42:  return session
```

## Daemon (100ms Queries)

```bash
tldr daemon start   # Start background daemon
tldr daemon status  # Check daemon status
tldr daemon stop    # Stop daemon
```

The daemon keeps indexes in memory for instant queries instead of 30-second CLI spawns.

See [Command Reference](references/COMMANDS.md) for full documentation.
