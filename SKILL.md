# llm-tldr: Efficient Code Context for LLM Agents

## Purpose

llm-tldr extracts code structure instead of dumping raw text, achieving **95% token savings** while preserving everything needed to understand and modify code. For coding agents working with large codebases, this means faster queries (100ms vs 30s) and fitting more context into limited token budgets.

## Quick Start

```bash
# Install and initialize project
pip install llm-tldr
cd /path/to/project
tldr warm .                    # Index project (30-60s one-time)
```

```bash
# Common operations (all <100ms with daemon running)
tldr context main --project .        # Get LLM-ready function summary
tldr semantic "JWT validation" .     # Search by behavior, not text
tldr impact login .                  # Find all callers (reverse call graph)
tldr extract src/auth.py             # Full file analysis
```

## Essential Commands

### Before Reading Code
```bash
tldr tree src/                       # File structure
tldr structure src/ --lang python    # Functions/classes overview
```

### Before Editing
```bash
tldr context <function> --project .  # 95% token savings vs raw code
tldr extract <file>                  # Complete file analysis
```

### Before Refactoring
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

## How It Works

llm-tldr builds 5 analysis layers from your code:

1. **AST (L1)**: Structure - what functions/classes exist
2. **Call Graph (L2)**: Dependencies - who calls what
3. **Control Flow (L3)**: Logic paths - complexity metrics
4. **Data Flow (L4)**: Value tracking - where data goes
5. **Program Dependence (L5)**: Line-level impact - minimal slices

**The semantic layer** combines all 5 layers into searchable embeddings using `bge-large-en-v1.5`, enabling natural language search by *what code does* rather than what it says.

**The daemon** keeps indexes in memory for instant queries instead of slow CLI spawns.

## Integration Patterns

### Auto-Update Daemon on File Changes

Add to your editor/Git hooks:
```bash
# Notify daemon when files change
tldr daemon notify "$FILE" --project .

# Git post-commit hook example
git diff --name-only HEAD~1 | xargs -I{} tldr daemon notify {} --project .
```

The daemon auto-rebuilds semantic index after 20 changed files.

### MCP Integration for Claude

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
      "args": ["/absolute/path/to/project"]
    }
  }
}
```

## Project Configuration

### Exclude Files (.tldrignore)

Create `.tldrignore` in project root (gitignore syntax):
```
node_modules/
.venv/
__pycache__/
dist/
build/
*.egg-info/
*.so
*.dll
.env
*.pem
large_fixtures/
vendor/
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

For monorepos, create `.claude/workspace.json` to scope indexing:
```json
{
  "active_packages": ["packages/core", "packages/api"],
  "exclude_patterns": ["**/fixtures/**"]
}
```

## Language Support

Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, Ruby, PHP, C#, Kotlin, Scala, Swift, Lua, Elixir (16 languages total). Language auto-detected or specify with `--lang`.

## Common Patterns

### Pattern 1: Function Context for LLM
```bash
# Get minimal context needed to understand/modify a function
tldr context handle_auth --project .
# Output: 95% fewer tokens than raw code, includes:
# - Function signature + docstring
# - What it calls + who calls it
# - Key data flows
# - First ~10 lines of code
```

### Pattern 2: Impact Analysis Before Refactoring
```bash
# Check what breaks if you change a function
tldr impact validate_token .
# Shows all callers across the codebase
# Use before renaming, changing signature, or removing functions
```

### Pattern 3: Finding Code by Behavior
```bash
# Traditional search finds keywords; semantic finds behavior
tldr semantic "handle database connection pooling" .
# Finds relevant code even without exact keyword matches
# Understands intent from call graphs and data flow
```

### Pattern 4: Debugging Data Flow
```bash
# Why is user null on line 42?
tldr slice src/auth.py login 42
# Shows ONLY the 6-8 lines that affect line 42
# Strips away irrelevant code automatically
```

### Pattern 5: Cross-Module Analysis
```bash
# Find all files that import a specific module
tldr importers auth.utils .

# Check which tests are affected by changes
tldr change-impact src/auth.py src/session.py
```

## Token Efficiency Examples

| Task | Raw Code | tldr | Savings |
|------|----------|------|---------|
| Function context | 21,000 tokens | 175 tokens | 99% |
| Codebase overview | 104,000 tokens | 12,000 tokens | 89% |
| File analysis | 8,500 tokens | 420 tokens | 95% |

## Performance

- **Query latency**: 100ms with daemon (vs 30s CLI cold start)
- **Indexing**: 30-60s for typical project
- **Memory**: Indexes stay in RAM for instant access
- **Auto-rebuild**: Triggers after 20 file changes

## Anti-Patterns to Avoid

❌ **Don't** paste raw code into LLM context when you have llm-tldr indexed
✅ **Do** use `tldr context` for 95% token savings

❌ **Don't** manually track which functions call what
✅ **Do** use `tldr impact` for instant reverse call graph

❌ **Don't** grep for code and hope you find what you need
✅ **Do** use `tldr semantic` to search by behavior

❌ **Don't** read entire files to understand one function
✅ **Do** use `tldr slice` to see only relevant lines

❌ **Don't** forget to warm the index after installing
✅ **Do** run `tldr warm .` once per project

## When to Use llm-tldr

**Use llm-tldr when:**
- Working with codebases >10K lines
- Need to understand unfamiliar code quickly
- Planning refactorings that affect multiple files
- Debugging complex data flows
- Preparing code context for LLM analysis
- Finding code by what it does, not what it's named

**Skip llm-tldr when:**
- Single-file scripts <500 lines
- You're writing new code from scratch
- Working with configuration files (YAML, JSON, etc.)
- Need the full implementation details immediately

## Further Reading

- **[README.md](./README.md)**: Project overview and quick start guide
- **[docs/TLDR.md](./docs/TLDR.md)**: Deep dive into architecture, benchmarks, and advanced workflows
- **[CONTRIBUTING.md](./CONTRIBUTING.md)**: How to contribute to the project

## Quick Troubleshooting

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
# Check if index exists
ls .tldr/cache/semantic.faiss
# Rebuild if missing
tldr warm .
```

**Queries too slow?**
```bash
# Ensure daemon is running
tldr daemon start
# Check status
tldr daemon status
```

---

**Key Principle**: llm-tldr optimizes for *understanding* code, not *viewing* it. Extract structure, trace dependencies, search by intent. Save tokens, work faster.
