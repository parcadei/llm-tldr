# SKILL: Codebase Context Extraction (llm-tldr)

## Purpose

Use this tool to obtain minimal, high-signal context required for reasoning about an existing codebase without loading full source files.

## When to Use

Use this tool when:
- Context size is a limiting factor
- The codebase is unfamiliar
- You need callers, data flow, or change impact

Avoid this tool when:
- Working on small, self-contained files
- Writing new code without dependencies

## Installation and Setup

```bash
pip install llm-tldr
cd /path/to/project
tldr warm .                    # Index project (one-time setup)
```

## Essential Commands

### Structure Overview
```bash
tldr tree src/                       # File structure
tldr structure src/ --lang python    # Functions/classes overview
```

### Function Context
```bash
tldr context <function> --project .  # Function summary with dependencies
tldr extract <file>                  # Complete file analysis
```

### Impact Analysis
```bash
tldr impact <function> .             # Who calls this? (breaks if changed)
tldr calls .                         # Build full call graph
tldr arch .                          # Detect architecture layers
tldr dead .                          # Find unreachable code
```

### Finding Code by Intent
```bash
tldr semantic "validate auth tokens" .    # Natural language search
tldr semantic "error retry logic" .       # Finds behavior, not keywords
```

### Debugging
```bash
tldr slice <file> <func> <line>      # What affects this line?
tldr dfg <file> <function>           # Trace data flow
tldr cfg <file> <function>           # Control flow graph
tldr diagnostics <file>              # Type check + lint
```

## Default Agent Workflow

1. Get structure: `tldr tree` / `tldr structure`
2. Narrow context: `tldr context <function>`
3. Validate impact before changes: `tldr impact <function>`

## How It Works

llm-tldr builds 5 analysis layers:

1. AST (L1): Structure - functions/classes
2. Call Graph (L2): Dependencies - who calls what
3. Control Flow (L3): Logic paths - complexity metrics
4. Data Flow (L4): Value tracking - where data goes
5. Program Dependence (L5): Line-level impact - minimal slices

The semantic layer combines all 5 layers into searchable embeddings, enabling natural language search by what code does rather than what it says.

## Configuration

### Exclude Files (.tldrignore)

Create `.tldrignore` in project root (gitignore syntax):
```
node_modules/
.venv/
__pycache__/
dist/
build/
*.egg-info/
```

### Daemon Settings (.tldr/config.json)

```json
{
  "semantic": {
    "enabled": true,
    "auto_reindex_threshold": 20
  }
}
```

### Monorepo Support

For monorepos, create `.claude/workspace.json`:
```json
{
  "active_packages": ["packages/core", "packages/api"],
  "exclude_patterns": ["**/fixtures/**"]
}
```

## MCP Integration

**For Claude Code** (`.claude/settings.json`):
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

**For Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "tldr": {
      "command": "tldr-mcp",
      "args": ["--project", "/absolute/path/to/project"]
    }
  }
}
```

## Language Support

Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, Ruby, PHP, C#, Kotlin, Scala, Swift, Lua, Elixir (16 languages total). Language auto-detected or specify with `--lang`.

## Troubleshooting

**Daemon not responding?**
```bash
tldr daemon status
tldr daemon stop && tldr daemon start
```

**Index out of date?**
```bash
tldr warm .  # Full rebuild
```

**Semantic search not finding relevant code?**
```bash
ls .tldr/cache/semantic.faiss  # Check if index exists
tldr warm .                     # Rebuild if missing
```

## Principle

Rule of thumb: If raw code does not change your decision, do not include it in context.
