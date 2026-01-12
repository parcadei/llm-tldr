# TLDR Command Reference

## Exploration Commands

| Command | Description |
|---------|-------------|
| `tldr tree [path]` | Display file tree |
| `tldr structure [path] --lang <lang>` | List functions, classes, methods |
| `tldr search <pattern> [path]` | Text pattern search |
| `tldr extract <file>` | Full file analysis with all layers |

## Analysis Commands

| Command | Description |
|---------|-------------|
| `tldr context <func> --project <path>` | LLM-ready summary (95% token savings) |
| `tldr cfg <file> <function>` | Control flow graph |
| `tldr dfg <file> <function>` | Data flow graph |
| `tldr slice <file> <func> <line>` | Program slice (what affects this line) |

## Cross-File Commands

| Command | Description |
|---------|-------------|
| `tldr calls [path]` | Build and display call graph |
| `tldr impact <func> [path]` | Reverse call graph (find all callers) |
| `tldr dead [path]` | Find unreachable/dead code |
| `tldr arch [path]` | Detect architecture layers |
| `tldr imports <file>` | Parse and list imports |
| `tldr importers <module> [path]` | Find files importing a module |

## Semantic Search Commands

| Command | Description |
|---------|-------------|
| `tldr warm <path>` | Build call graph index (for impact/dead code analysis) |
| `tldr semantic index [path]` | Build embedding index for semantic search |
| `tldr semantic search <query>` | Natural language code search |

## Diagnostics Commands

| Command | Description |
|---------|-------------|
| `tldr diagnostics <file>` | Type check + lint |
| `tldr change-impact [files]` | Find tests affected by changes |
| `tldr doctor` | Check/install diagnostic tools |

## Daemon Commands

| Command | Description |
|---------|-------------|
| `tldr daemon start` | Start background daemon |
| `tldr daemon stop` | Stop daemon |
| `tldr daemon status` | Check daemon status |
| `tldr daemon notify <file> --project .` | Notify daemon of file change |

## Common Flags

| Flag | Description |
|------|-------------|
| `--lang <language>` | Specify language (auto-detected by default) |
| `--project <path>` | Specify project root |
| `--no-ignore` | Bypass .tldrignore patterns |

## Language Support

Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, Ruby, PHP, C#, Kotlin, Scala, Swift, Lua, Luau, Elixir

Language is auto-detected from file extension or specify with `--lang`.
