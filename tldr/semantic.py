"""
Semantic search for code using 5-layer embeddings.

Embeds functions/methods using all 5 TLDR analysis layers:
- L1: Signature + docstring
- L2: Top callers + callees (from call graph)
- L3: Control flow summary
- L4: Data flow summary
- L5: Dependencies

Uses BAAI/bge-large-en-v1.5 for embeddings (1024 dimensions)
and FAISS for fast vector similarity search.
"""

import json
import logging
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger("tldr.semantic")

ALL_LANGUAGES = ["python", "typescript", "javascript", "go", "rust", "java", "c", "cpp", "ruby", "php", "kotlin", "swift", "csharp", "scala", "lua", "elixir"]

# Lazy imports for heavy dependencies
_model = None
_model_name = None  # Track which model is loaded
_model_lock = threading.Lock()  # Thread safety for model singleton

# Supported models with approximate download sizes
SUPPORTED_MODELS = {
    "bge-large-en-v1.5": {
        "hf_name": "BAAI/bge-large-en-v1.5",
        "size": "1.3GB",
        "dimension": 1024,
        "description": "High quality, recommended for production",
    },
    "all-MiniLM-L6-v2": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "size": "80MB",
        "dimension": 384,
        "description": "Lightweight, good for testing",
    },
}

DEFAULT_MODEL = "bge-large-en-v1.5"


@dataclass(slots=True)
class EmbeddingUnit:
    """A code unit (function/method/class) for embedding.

    Contains information from all 5 TLDR layers:
    - L1: signature, docstring
    - L2: calls, called_by
    - L3: cfg_summary
    - L4: dfg_summary
    - L5: dependencies
    """

    name: str
    qualified_name: str
    file: str
    line: int
    language: str
    unit_type: str  # "function" | "method" | "class"
    signature: str
    docstring: str
    calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    cfg_summary: str = ""
    dfg_summary: str = ""
    dependencies: str = ""
    code_preview: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "file": self.file,
            "line": self.line,
            "language": self.language,
            "unit_type": self.unit_type,
            "signature": self.signature,
            "docstring": self.docstring,
            "calls": self.calls,
            "called_by": self.called_by,
            "cfg_summary": self.cfg_summary,
            "dfg_summary": self.dfg_summary,
            "dependencies": self.dependencies,
            "code_preview": self.code_preview,
        }


MODEL_NAME = "BAAI/bge-large-en-v1.5"  # Legacy, use SUPPORTED_MODELS


def _model_exists_locally(hf_name: str) -> bool:
    """Check if a model is already downloaded locally."""
    try:
        from huggingface_hub import try_to_load_from_cache

        # Check if model config exists in cache
        result = try_to_load_from_cache(hf_name, "config.json")
        return result is not None
    except Exception:
        return False


def _confirm_download(model_key: str) -> bool:
    """Prompt user to confirm model download. Returns True if confirmed."""
    model_info = SUPPORTED_MODELS.get(model_key, {})
    size = model_info.get("size", "unknown size")
    hf_name = model_info.get("hf_name", model_key)

    # Skip prompt if TLDR_AUTO_DOWNLOAD is set or not a TTY
    if os.environ.get("TLDR_AUTO_DOWNLOAD") == "1":
        return True
    if not sys.stdin.isatty():
        # Non-interactive: warn but proceed
        print(f"⚠️  Downloading {hf_name} ({size})...", file=sys.stderr)
        return True

    print(f"\n⚠️  Semantic search requires embedding model: {hf_name}", file=sys.stderr)
    print(f"   Download size: {size}", file=sys.stderr)
    print("   (Set TLDR_AUTO_DOWNLOAD=1 to skip this prompt)\n", file=sys.stderr)

    try:
        response = input("Continue with download? [Y/n] ").strip().lower()
        return response in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def get_model(model_name: Optional[str] = None):
    """Lazy-load the embedding model (cached, thread-safe).

    Args:
        model_name: Model key from SUPPORTED_MODELS, or None for default.
                   Can also be a full HuggingFace model name.

    Returns:
        SentenceTransformer model instance.

    Raises:
        ValueError: If model not found or user declines download.
    """
    global _model, _model_name

    # Resolve model name
    if model_name is None:
        model_name = DEFAULT_MODEL

    # Get HuggingFace name
    if model_name in SUPPORTED_MODELS:
        hf_name = SUPPORTED_MODELS[model_name]["hf_name"]
    else:
        # Allow arbitrary HuggingFace model names
        hf_name = model_name

    # Thread-safe model loading with double-checked locking
    # First check without lock for fast path (model already loaded)
    if _model is not None and _model_name == hf_name:
        return _model

    with _model_lock:
        # Re-check under lock to prevent race condition
        if _model is not None and _model_name == hf_name:
            return _model

        # Check if model needs downloading (outside lock would cause UI issues)
        if not _model_exists_locally(hf_name):
            model_key = model_name if model_name in SUPPORTED_MODELS else None
            if model_key and not _confirm_download(model_key):
                raise ValueError(
                    "Model download declined. Use --model to choose a smaller model."
                )

        from sentence_transformers import SentenceTransformer
        import torch

        # Auto-select best device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        _model = SentenceTransformer(hf_name, device=device)

        # Best-effort dtype optimization (don't fail model load)
        try:
            _model = _model.to(torch.bfloat16)
        except Exception:
            pass  # Fall back to fp32 if bf16 unsupported

        # Apply torch.compile() for optimized execution (PyTorch 2.0+)
        try:
            if hasattr(torch, "compile"):
                _model = torch.compile(_model)
        except Exception:
            pass  # Silently fall back if compile unavailable

        _model_name = hf_name
        return _model


def _parse_identifier_to_words(name: str) -> str:
    """Parse camelCase/snake_case/PascalCase identifier to space-separated words.

    Converts code identifiers into natural language for better semantic search.

    Examples:
        getUserData -> get user data
        get_user_data -> get user data
        XMLParser -> xml parser
        _private_method -> private method
        HTMLElement -> html element

    Args:
        name: The identifier to parse.

    Returns:
        Space-separated lowercase words.
    """
    import re

    # Remove leading/trailing underscores
    name = name.strip("_")
    if not name:
        return ""

    # Handle snake_case: replace underscores with spaces
    name = name.replace("_", " ")

    # Use regex for camelCase/PascalCase splitting
    # This handles acronyms correctly: XMLParser -> XML Parser -> xml parser
    # Pattern: split before uppercase that follows lowercase, or before uppercase followed by lowercase
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)  # camelCase
    words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", words)  # ACRONYMWord

    # Clean up multiple spaces and lowercase
    words = " ".join(words.split()).lower()

    return words


def _extract_inline_comments(code: str) -> List[str]:
    """Extract inline comments from code preview.

    Parses # comments and extracts their text for semantic embedding.
    Comments often contain valuable natural language describing intent.

    Args:
        code: The code string to parse.

    Returns:
        List of comment strings (without # prefix).
    """
    import re

    comments = []
    for line in code.split("\n"):
        # Match # comments (but not in strings - simple heuristic)
        # Skip shebang lines
        match = re.search(r'(?<!["\'#])\s*#\s*(.+?)$', line)
        if match:
            comment = match.group(1).strip()
            # Filter out noise comments
            if len(comment) > 3 and not comment.startswith("!"):
                comments.append(comment)

    return comments


def _generate_semantic_description(unit: "EmbeddingUnit") -> str:
    """Generate natural language description when docstring is missing.

    Creates a semantic description from code structure for functions
    that lack docstrings, improving embedding quality.

    Args:
        unit: The EmbeddingUnit to describe.

    Returns:
        Generated natural language description.
    """
    parts = []

    # Parse function name into natural language
    name_words = _parse_identifier_to_words(unit.name)
    if name_words:
        # Create a sentence from the name
        if unit.unit_type == "method":
            parts.append(f"Method that {name_words}")
        elif unit.unit_type == "class":
            parts.append(f"Class for {name_words}")
        else:
            parts.append(f"Function that {name_words}")

    # Extract parameter semantics from signature
    if unit.signature:
        import re

        # Extract parameter names from signature
        param_match = re.search(r"\((.*?)\)", unit.signature)
        if param_match:
            params_str = param_match.group(1)
            if params_str and params_str not in ("self", "cls"):
                # Parse parameter names
                param_names = []
                for param in params_str.split(","):
                    param = param.strip()
                    if not param or param in ("self", "cls"):
                        continue
                    # Extract name before : or =
                    name = param.split(":")[0].split("=")[0].strip()
                    if name and name not in ("self", "cls"):
                        param_names.append(_parse_identifier_to_words(name))

                if param_names:
                    parts.append(f"Takes {', '.join(param_names[:5])} as input")

    # Describe complexity in natural language
    if unit.cfg_summary:
        import re

        complexity_match = re.search(r"complexity:(\d+)", unit.cfg_summary)
        if complexity_match:
            complexity = int(complexity_match.group(1))
            if complexity == 1:
                parts.append("Simple linear logic")
            elif complexity <= 3:
                parts.append("Contains conditional logic")
            elif complexity <= 7:
                parts.append("Moderate complexity with multiple branches")
            else:
                parts.append("Complex control flow with many decision points")

    # Describe data handling
    if unit.dfg_summary:
        import re

        vars_match = re.search(r"vars:(\d+)", unit.dfg_summary)
        if vars_match:
            var_count = int(vars_match.group(1))
            if var_count > 10:
                parts.append("Processes multiple data variables")

    # Extract inline comments for additional context
    if unit.code_preview:
        comments = _extract_inline_comments(unit.code_preview)
        if comments:
            parts.append("Notes: " + "; ".join(comments[:3]))

    return ". ".join(parts) if parts else ""


def build_embedding_text(unit: EmbeddingUnit) -> str:
    """Build rich text for embedding from all 5 layers.

    Creates a single text string containing information from all
    analysis layers, suitable for embedding with a language model.
    Prioritizes natural language over code syntax for better semantic search.

    Args:
        unit: The EmbeddingUnit containing code analysis.

    Returns:
        A text string combining all layer information.
    """
    parts = []

    # Primary: Natural language description (most important for semantic search)
    # Use docstring if available, otherwise generate description
    if unit.docstring:
        parts.append(f"Description: {unit.docstring}")
    else:
        # Generate semantic description from code structure
        generated = _generate_semantic_description(unit)
        if generated:
            parts.append(f"Description: {generated}")

    # Parse function name as natural language (helps match semantic queries)
    name_words = _parse_identifier_to_words(unit.name)
    if name_words and name_words != unit.name.lower():
        parts.append(f"Purpose: {name_words}")

    # L1: Signature (contains type hints which have semantic value)
    if unit.signature:
        parts.append(f"Signature: {unit.signature}")

    # L2: Call graph with natural language framing
    if unit.calls:
        calls_words = [_parse_identifier_to_words(c) for c in unit.calls[:5]]
        calls_str = ", ".join(filter(None, calls_words))
        if calls_str:
            parts.append(f"Uses: {calls_str}")

    if unit.called_by:
        callers_words = [_parse_identifier_to_words(c) for c in unit.called_by[:5]]
        callers_str = ", ".join(filter(None, callers_words))
        if callers_str:
            parts.append(f"Used by: {callers_str}")

    # L5: Dependencies (module names can have semantic meaning)
    if unit.dependencies:
        parts.append(f"Dependencies: {unit.dependencies}")

    # Code preview (last - code syntax is less useful for semantic matching)
    if unit.code_preview:
        # Include comments from code which are natural language
        comments = _extract_inline_comments(unit.code_preview)
        if comments:
            parts.append(f"Code comments: {'; '.join(comments[:5])}")

        # Include code but truncated (syntax is less useful than semantics)
        code_lines = unit.code_preview.split("\n")[:8]  # First 8 lines
        parts.append(f"Code:\n{chr(10).join(code_lines)}")

    # Add name and type for context (at start for clarity)
    type_str = unit.unit_type if unit.unit_type else "function"
    parts.insert(0, f"{type_str.capitalize()}: {unit.name}")

    return "\n".join(parts)


def compute_embedding(text: str, model_name: Optional[str] = None):
    """Compute embedding vector for text.

    Args:
        text: The text to embed.
        model_name: Model to use (from SUPPORTED_MODELS or HF name).

    Returns:
        numpy array with L2-normalized embedding.
    """
    import numpy as np

    model = get_model(model_name)

    # BGE models work best with instruction prefix for queries
    # For document embedding, we use text directly
    embedding = model.encode(text, normalize_embeddings=True)

    return np.array(embedding, dtype=np.float32)


# Threshold for switching to parallel processing
MIN_FILES_FOR_PARALLEL = 15


def extract_units_from_project(
    project_path: str, lang: str = "python", respect_ignore: bool = True
) -> List[EmbeddingUnit]:
    """Extract all functions/methods/classes from a project.

    Uses existing TLDR APIs:
    - tldr.api.get_code_structure() for L1 (signatures)
    - tldr.cross_file_calls for L2 (call graph)
    - CFG/DFG extractors for L3/L4 summaries
    - tldr.api.get_imports for L5 (dependencies)

    Args:
        project_path: Path to project root.
        lang: Programming language ("python", "typescript", "go", "rust").
        respect_ignore: If True, respect .tldrignore patterns (default True).

    Returns:
        List of EmbeddingUnit objects with enriched metadata.
    """
    from tldr.api import get_code_structure, build_project_call_graph
    from tldr.tldrignore import load_ignore_patterns, should_ignore

    project = Path(project_path).resolve()
    units = []

    # Get code structure (L1)
    # Use max_results=0 for unlimited files - the default of 100 would truncate large projects
    structure = get_code_structure(str(project), language=lang, max_results=0)

    # Filter ignored files
    if respect_ignore:
        spec = load_ignore_patterns(project)
        structure["files"] = [
            f
            for f in structure.get("files", [])
            if not should_ignore(project / f.get("path", ""), project, spec)
        ]

    # Build call graph (L2)
    try:
        call_graph = build_project_call_graph(str(project), language=lang)

        # Build call/called_by maps
        calls_map = {}  # func -> [called functions]
        called_by_map = {}  # func -> [calling functions]

        for edge in call_graph.edges:
            src_file, src_func, dst_file, dst_func = edge

            # Forward: src calls dst
            if src_func not in calls_map:
                calls_map[src_func] = []
            calls_map[src_func].append(dst_func)

            # Backward: dst is called by src
            if dst_func not in called_by_map:
                called_by_map[dst_func] = []
            called_by_map[dst_func].append(src_func)
    except Exception:
        # Call graph may not be available for all projects
        calls_map = {}
        called_by_map = {}

    # Process files in parallel for better performance
    files = structure.get("files", [])
    max_workers = int(os.environ.get("TLDR_MAX_WORKERS", os.cpu_count() or 4))

    # Use parallel processing if we have enough files to justify overhead
    if len(files) >= MIN_FILES_FOR_PARALLEL and max_workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _process_file_for_extraction,
                        file_info,
                        str(project),
                        lang,
                        calls_map,
                        called_by_map,
                    ): file_info
                    for file_info in files
                }

                for future in as_completed(futures):
                    file_info = futures[future]
                    try:
                        file_units = future.result(timeout=60)  # 60s per file timeout
                        units.extend(file_units)
                    except Exception as e:
                        logger.warning(f"Failed to process {file_info.get('path', 'unknown')}: {e}")
                        # Continue with other files

        except Exception as e:
            # Fallback to sequential if parallel fails
            logger.warning(f"Parallel extraction failed: {e}, falling back to sequential")
            for file_info in files:
                try:
                    file_units = _process_file_for_extraction(
                        file_info, str(project), lang, calls_map, called_by_map
                    )
                    units.extend(file_units)
                except Exception as fe:
                    logger.warning(f"Failed to process {file_info.get('path', 'unknown')}: {fe}")
    else:
        # Sequential processing for single file or when parallel is disabled
        for file_info in files:
            try:
                file_units = _process_file_for_extraction(
                    file_info, str(project), lang, calls_map, called_by_map
                )
                units.extend(file_units)
            except Exception as e:
                logger.warning(f"Failed to process {file_info.get('path', 'unknown')}: {e}")

    return units


def _build_parent_map(tree) -> dict:
    """Build a mapping from each AST node to its parent in O(N) time.

    Args:
        tree: The root AST node.

    Returns:
        Dict mapping child nodes to their parent nodes.
    """
    import ast

    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _get_parent_class(node, parents: dict) -> str | None:
    """Find the parent ClassDef name for a node, if any.

    Walks up the parent chain to find if this node is directly
    inside a class body (i.e., is a method).

    Args:
        node: The AST node to check.
        parents: Parent map from _build_parent_map.

    Returns:
        Class name if node is a direct method of a class, None otherwise.
    """
    import ast

    current = parents.get(node)
    while current:
        if isinstance(current, ast.ClassDef):
            # Check if node is a direct child in the class body
            if node in current.body:
                return current.name
            # Node is nested deeper (e.g., inside a nested function), not a direct method
            return None
        current = parents.get(current)
    return None


def _build_signature_from_node(node, lang: str) -> str:
    """Build function signature string from AST node.

    Extracts signature during single AST walk to avoid redundant parsing.

    Args:
        node: AST FunctionDef or AsyncFunctionDef node.
        lang: Programming language.

    Returns:
        Signature string like "def func(arg1: int, arg2) -> str".
    """
    if lang != "python":
        return f"function {node.name}(...)"

    import ast

    try:
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        returns = ""
        if node.returns:
            returns = f" -> {ast.unparse(node.returns)}"

        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(args)}){returns}"
    except Exception:
        return f"def {node.name}(...)"


def _parse_file_ast(file_path: Path, lang: str, content: Optional[str] = None) -> tuple:
    """Parse file AST to extract line numbers, code previews, signatures, and docstrings.

    Reads file content once and returns it along with parsed AST for reuse by callers,
    eliminating redundant file I/O operations (reduces 4N+1 reads to 1 read per file).

    Args:
        file_path: Path to the source file.
        lang: Programming language.
        content: Optional pre-read file content to avoid redundant I/O.

    Returns:
        Tuple of (result_dict, content, ast_tree) where:
        - result_dict: {
            "functions": {func_name: {"line": int, "code_preview": str, "signature": str, "docstring": str|None}},
            "classes": {class_name: {"line": int, "docstring": str|None}},
            "methods": {"ClassName.method": {"line": int, "code_preview": str, "signature": str, "docstring": str|None}}
          }
        - content: File content (read once, reusable by callers)
        - ast_tree: Parsed AST (for Python) or None (reusable by callers)

    Performance: O(N) where N is number of AST nodes.
    Uses parent map for efficient parent class lookup instead of nested ast.walk().
    Extracts signature and docstring in the same pass to eliminate 2N redundant ast.parse() calls.
    """
    result = {"functions": {}, "classes": {}, "methods": {}}
    ast_tree = None

    # Read content only if not provided (single I/O per file)
    if content is None:
        if not file_path.exists():
            return result, "", None
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return result, "", None

    lines = content.split("\n")

    try:
        if lang == "python":
            import ast

            ast_tree = ast.parse(content)

            # Build parent map in O(N) - replaces O(N^3) nested ast.walk() calls
            parents = _build_parent_map(ast_tree)

            for node in ast.walk(ast_tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # O(depth) parent lookup instead of O(N^2) nested walk
                    parent_class = _get_parent_class(node, parents)

                    # Extract code preview (first 10 lines of body)
                    # AST uses 1-indexed line numbers, Python lists are 0-indexed
                    start_line = node.lineno - 1  # Convert to 0-indexed for list access
                    end_line = getattr(
                        node, "end_lineno", node.lineno + 10
                    )  # end_lineno is 1-indexed, works as exclusive slice bound
                    body_lines = lines[start_line : min(end_line, start_line + 10)]
                    code_preview = "\n".join(body_lines[:10])

                    # Extract signature and docstring in the same AST walk
                    # This eliminates 2N redundant ast.parse() calls per file
                    signature = _build_signature_from_node(node, lang)
                    docstring = ast.get_docstring(node)

                    func_data = {
                        "line": node.lineno,
                        "code_preview": code_preview,
                        "signature": signature,
                        "docstring": docstring,
                    }

                    if parent_class:
                        result["methods"][f"{parent_class}.{node.name}"] = func_data
                    else:
                        result["functions"][node.name] = func_data

                elif isinstance(node, ast.ClassDef):
                    result["classes"][node.name] = {
                        "line": node.lineno,
                        "docstring": ast.get_docstring(node),
                    }

    except Exception:
        # Return empty result on any parsing error, but preserve content
        pass

    return result, content, ast_tree


def _get_file_dependencies(file_path: Path, lang: str) -> str:
    """Get file-level import dependencies as a string."""
    if not file_path.exists():
        return ""

    try:
        from tldr.api import get_imports

        imports = get_imports(str(file_path), language=lang)

        # Extract module names (limit to first 5 for brevity)
        modules = []
        for imp in imports[:5]:
            module = imp.get("module", "")
            if module:
                modules.append(module)

        return ", ".join(modules) if modules else ""
    except Exception:
        return ""


def _get_cfg_summary(
    file_path: Path, func_name: str, lang: str, content: Optional[str] = None
) -> str:
    """Get CFG summary (complexity, block count) for a function.

    Args:
        file_path: Path to the source file.
        func_name: Name of the function to analyze.
        lang: Programming language.
        content: Optional pre-read file content to avoid redundant I/O.

    Returns:
        CFG summary string (e.g., "complexity:3, blocks:5").
    """
    if content is None:
        if not file_path.exists():
            return ""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    try:
        if lang == "python":
            from tldr.cfg_extractor import extract_python_cfg

            cfg = extract_python_cfg(content, func_name)
            return f"complexity:{cfg.cyclomatic_complexity}, blocks:{len(cfg.blocks)}"
        # Add other languages as needed
    except Exception:
        pass

    return ""


def _get_dfg_summary(
    file_path: Path, func_name: str, lang: str, content: Optional[str] = None
) -> str:
    """Get DFG summary (variable count, def-use chains) for a function.

    Args:
        file_path: Path to the source file.
        func_name: Name of the function to analyze.
        lang: Programming language.
        content: Optional pre-read file content to avoid redundant I/O.

    Returns:
        DFG summary string (e.g., "vars:5, def-use chains:8").
    """
    if content is None:
        if not file_path.exists():
            return ""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    try:
        if lang == "python":
            from tldr.dfg_extractor import extract_python_dfg

            dfg = extract_python_dfg(content, func_name)

            # Count unique variables and def-use chains
            var_names = set()
            for ref in dfg.var_refs:
                var_names.add(ref.name)

            return f"vars:{len(var_names)}, def-use chains:{len(dfg.dataflow_edges)}"
        # Add other languages as needed
    except Exception:
        pass

    return ""


# Cached wrappers for CFG/DFG extraction to avoid repeated file I/O and parsing.
# Uses string paths (hashable) as cache keys.
# Cache size of 1000 handles typical project sizes; entries evicted LRU when exceeded.
# NOTE: mtime_ns is included in cache key to auto-invalidate on file changes.


def _get_file_mtime_ns(file_path: Path) -> int:
    """Get file modification time in nanoseconds, or 0 if unavailable."""
    try:
        return file_path.stat().st_mtime_ns
    except (OSError, IOError):
        return 0


def _process_file_for_extraction(
    file_info: Dict[str, Any],
    project_path: str,
    lang: str,
    calls_map: Dict[str, List[str]],
    called_by_map: Dict[str, List[str]],
) -> List[EmbeddingUnit]:
    """Process a single file and extract all units. Top-level for pickling.

    This function reads the file ONCE and extracts all information in a single pass,
    avoiding the O(n*m) file read issue where n=files and m=functions.

    Args:
        file_info: Dict with 'path', 'functions', 'classes' from get_code_structure.
        project_path: Absolute path to project root.
        lang: Programming language.
        calls_map: Map of function name -> list of called functions.
        called_by_map: Map of function name -> list of calling functions.

    Returns:
        List of EmbeddingUnit objects for this file.
    """
    units = []
    project = Path(project_path)
    file_path = file_info.get("path", "")
    full_path = project / file_path

    if not full_path.exists():
        return units

    try:
        # Read file content ONCE
        content = full_path.read_text(encoding="utf-8", errors="replace")
        lines = content.split('\n')
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return units

    # Use tree-sitter based extraction for ALL languages (not just Python)
    # This provides consistent line numbers, signatures, and docstrings
    ast_info = {"functions": {}, "classes": {}, "methods": {}}
    all_signatures = {}
    all_docstrings = {}

    try:
        from tldr.ast_extractor import extract_file
        module_info = extract_file(str(full_path))

        # Build lookup dicts from extracted info
        for func in module_info.functions:
            # Extract code preview (first 10 lines from function start)
            start_idx = max(0, func.line_number - 1)
            end_idx = min(start_idx + 10, len(lines))
            code_preview = '\n'.join(lines[start_idx:end_idx])

            ast_info["functions"][func.name] = {
                "line": func.line_number,
                "code_preview": code_preview,
            }
            all_signatures[func.name] = func.signature()
            all_docstrings[func.name] = func.docstring or ""

        for cls in module_info.classes:
            ast_info["classes"][cls.name] = {"line": cls.line_number}

            # Process methods within each class
            for method in cls.methods:
                method_key = f"{cls.name}.{method.name}"
                start_idx = max(0, method.line_number - 1)
                end_idx = min(start_idx + 10, len(lines))
                code_preview = '\n'.join(lines[start_idx:end_idx])

                ast_info["methods"][method_key] = {
                    "line": method.line_number,
                    "code_preview": code_preview,
                }
                all_signatures[method_key] = method.signature()
                all_docstrings[method_key] = method.docstring or ""

    except Exception as e:
        logger.debug(f"AST extraction failed for {file_path}: {e}")

    # Get dependencies (imports) - single call
    dependencies = ""
    try:
        from tldr.api import get_imports
        imports = get_imports(str(full_path), language=lang)
        modules = [imp.get("module", "") for imp in imports[:5] if imp.get("module")]
        dependencies = ", ".join(modules)
    except Exception:
        pass

    # Pre-compute CFG/DFG for all functions at once
    cfg_cache = {}
    dfg_cache = {}

    if lang == "python":
        # Get all function names we need to process
        all_func_names = list(file_info.get("functions", []))
        for class_info in file_info.get("classes", []):
            if isinstance(class_info, dict):
                all_func_names.extend(class_info.get("methods", []))

        for func_name in all_func_names:
            try:
                from tldr.cfg_extractor import extract_python_cfg
                cfg = extract_python_cfg(content, func_name)
                cfg_cache[func_name] = f"complexity:{cfg.cyclomatic_complexity}, blocks:{len(cfg.blocks)}"
            except Exception:
                cfg_cache[func_name] = ""

            try:
                from tldr.dfg_extractor import extract_python_dfg
                dfg = extract_python_dfg(content, func_name)
                var_names = set(ref.name for ref in dfg.var_refs)
                dfg_cache[func_name] = f"vars:{len(var_names)}, def-use chains:{len(dfg.dataflow_edges)}"
            except Exception:
                dfg_cache[func_name] = ""

    # Process functions
    for func_name in file_info.get("functions", []):
        func_info = ast_info.get("functions", {}).get(func_name, {})
        unit = EmbeddingUnit(
            name=func_name,
            qualified_name=f"{file_path.replace('/', '.')}.{func_name}",
            file=file_path,
            line=func_info.get("line", 1),
            language=lang,
            unit_type="function",
            signature=all_signatures.get(func_name, f"def {func_name}(...)"),
            docstring=all_docstrings.get(func_name, ""),
            calls=calls_map.get(func_name, [])[:5],
            called_by=called_by_map.get(func_name, [])[:5],
            cfg_summary=cfg_cache.get(func_name, ""),
            dfg_summary=dfg_cache.get(func_name, ""),
            dependencies=dependencies,
            code_preview=func_info.get("code_preview", ""),
        )
        units.append(unit)

    # Process classes
    for class_info in file_info.get("classes", []):
        if isinstance(class_info, dict):
            class_name = class_info.get("name", "")
            methods = class_info.get("methods", [])
        else:
            class_name = class_info
            methods = []

        class_line = ast_info.get("classes", {}).get(class_name, {}).get("line", 1)

        # Add class itself
        unit = EmbeddingUnit(
            name=class_name,
            qualified_name=f"{file_path.replace('/', '.')}.{class_name}",
            file=file_path,
            line=class_line,
            language=lang,
            unit_type="class",
            signature=f"class {class_name}",
            docstring="",
            calls=[],
            called_by=[],
            cfg_summary="",
            dfg_summary="",
            dependencies=dependencies,
            code_preview="",
        )
        units.append(unit)

        # Add methods
        for method in methods:
            method_key = f"{class_name}.{method}"
            method_info = ast_info.get("methods", {}).get(method_key, {})

            unit = EmbeddingUnit(
                name=method,
                qualified_name=f"{file_path.replace('/', '.')}.{method_key}",
                file=file_path,
                line=method_info.get("line", 1),
                language=lang,
                unit_type="method",
                signature=all_signatures.get(method_key, f"def {method}(self, ...)"),
                docstring=all_docstrings.get(method_key, ""),
                calls=calls_map.get(method, [])[:5],
                called_by=called_by_map.get(method, [])[:5],
                cfg_summary=cfg_cache.get(method, ""),
                dfg_summary=dfg_cache.get(method, ""),
                dependencies=dependencies,
                code_preview=method_info.get("code_preview", ""),
            )
            units.append(unit)

    return units


def _get_progress_console():
    """Get rich Console if available and TTY, else None."""
    if not sys.stdout.isatty():
        return None
    if os.environ.get("NO_PROGRESS") or os.environ.get("CI"):
        return None
    try:
        from rich.console import Console

        return Console()
    except ImportError:
        return None


def _detect_project_languages(project_path: Path, respect_ignore: bool = True) -> List[str]:
    """Scan project files to detect present languages."""
    from tldr.tldrignore import load_ignore_patterns, should_ignore
    
    # Extension map (copied from cli.py to avoid circular import)
    EXTENSION_TO_LANGUAGE = {
        '.java': 'java',
        '.py': 'python',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.go': 'go',
        '.rs': 'rust',
        '.c': 'c',
        '.h': 'c',
        '.cpp': 'cpp',
        '.hpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.hh': 'cpp',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.cs': 'csharp',
        '.kt': 'kotlin',
        '.kts': 'kotlin',
        '.scala': 'scala',
        '.sc': 'scala',
        '.lua': 'lua',
        '.ex': 'elixir',
        '.exs': 'elixir',
    }
    
    found_languages = set()
    spec = load_ignore_patterns(project_path) if respect_ignore else None
    
    for root, dirs, files in os.walk(project_path):
        # Prune common heavy dirs immediately for speed
        dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', '.tldr', 'venv', '__pycache__', '.idea', '.vscode'}]
        
        for file in files:
             file_path = Path(root) / file
             
             # Check ignore patterns
             if respect_ignore and should_ignore(file_path, project_path, spec):
                 continue
                 
             ext = file_path.suffix.lower()
             if ext in EXTENSION_TO_LANGUAGE:
                 found_languages.add(EXTENSION_TO_LANGUAGE[ext])

    # Return sorted list intersect with ALL_LANGUAGES to ensure validity
    return sorted(list(found_languages & set(ALL_LANGUAGES)))


def build_semantic_index(
    project_path: str,
    lang: str = "all",
    model: Optional[str] = None,
    show_progress: bool = True,
    respect_ignore: bool = True,
) -> int:
    """Build and save FAISS index + metadata for a project.

    Creates a unified index at .tldr/cache/semantic/:
    - index.faiss - Vector index
    - metadata.json - Unit metadata with language info

    Args:
        project_path: Path to project root.
        lang: Language to index - "all" (default) auto-detects and indexes all.
        model: Model name from SUPPORTED_MODELS or HuggingFace name.
        show_progress: Show progress spinner (default: True).
        respect_ignore: If True, respect .tldrignore patterns (default True).

    Returns:
        Number of indexed units.
    """
    import faiss
    from tldr.tldrignore import ensure_tldrignore

    console = _get_progress_console() if show_progress else None

    # Ensure .tldrignore exists (create with defaults if not)
    project = Path(project_path).resolve()
    created, message = ensure_tldrignore(project)
    if created and console:
        console.print(f"[yellow]{message}[/yellow]")

    # Resolve model name early to get HF name for metadata
    model_key = model if model else DEFAULT_MODEL
    if model_key in SUPPORTED_MODELS:
        hf_name = SUPPORTED_MODELS[model_key]["hf_name"]
    else:
        hf_name = model_key

    # Unified cache directory - no language suffix
    # All languages stored together in single index
    cache_dir = project / ".tldr" / "cache" / "semantic"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Extract all units (respecting .tldrignore)
    indexed_languages: List[str] = []
    if console:
        with console.status("[bold green]Extracting code units...") as status:
            if lang == "all":
                # Auto-detect which languages are present
                status.update("[bold green]Scanning project languages...")
                indexed_languages = _detect_project_languages(project, respect_ignore=respect_ignore)
                if not indexed_languages:
                    console.print("[yellow]No supported languages detected in project[/yellow]")
                    return 0
                console.print(f"[dim]Detected languages: {', '.join(indexed_languages)}[/dim]")

                units = []
                for lang_name in indexed_languages:
                    status.update(f"[bold green]Extracting {lang_name} code units...")
                    units.extend(extract_units_from_project(str(project), lang=lang_name, respect_ignore=respect_ignore))
            else:
                indexed_languages = [lang]
                units = extract_units_from_project(str(project), lang=lang, respect_ignore=respect_ignore)
            status.update(f"[bold green]Extracted {len(units)} code units")
    else:
        if lang == "all":
            indexed_languages = _detect_project_languages(project, respect_ignore=respect_ignore)
            if not indexed_languages:
                return 0
            units = []
            for lang_name in indexed_languages:
                units.extend(extract_units_from_project(str(project), lang=lang_name, respect_ignore=respect_ignore))
        else:
            indexed_languages = [lang]
            units = extract_units_from_project(str(project), lang=lang, respect_ignore=respect_ignore)

    if not units:
        logger.warning(
            "No indexable code units found. Check project contains supported source files."
        )
        return 0

    # Build all texts first for batch encoding
    texts = [build_embedding_text(unit) for unit in units]

    # Batch encode embeddings (10-50x faster than sequential)
    # SentenceTransformers handles batching internally with optimal GPU utilization
    model_obj = get_model(model)
    if console:
        console.print(f"[bold green]Computing embeddings for {len(texts)} units...")
    embeddings_matrix = model_obj.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=show_progress and console is not None,
    )

    # Build FAISS index (inner product for normalized vectors = cosine similarity)
    if console:
        with console.status("[bold green]Building FAISS index..."):
            dimension = embeddings_matrix.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings_matrix)
    else:
        dimension = embeddings_matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_matrix)

    # Save index and metadata atomically using temp files + rename
    # This prevents corruption if crash occurs between writes
    index_file = cache_dir / "index.faiss"
    metadata_file = cache_dir / "metadata.json"
    temp_index = cache_dir / "index.faiss.tmp"
    temp_metadata = cache_dir / "metadata.json.tmp"

    # Prepare metadata with version for future migrations
    metadata = {
        "version": 2,  # v2 = unified index format
        "units": [u.to_dict() for u in units],
        "model": hf_name,
        "dimension": dimension,
        "count": len(units),
        "languages": indexed_languages,
    }

    # Write to temp files first
    faiss.write_index(index, str(temp_index))
    temp_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Atomic replace (os.replace works cross-platform, unlike Path.rename on Windows)
    os.replace(temp_index, index_file)
    os.replace(temp_metadata, metadata_file)

    if console:
        console.print(f"[bold green]✓[/] Indexed {len(units)} code units")

    return len(units)


def semantic_search(
    project_path: str,
    query: str,
    k: int = 5,
    expand_graph: bool = False,
    model: Optional[str] = None,
    lang: Optional[str] = None,  # Ignored - searches unified index
) -> List[dict]:
    """Search for code units semantically.

    Searches the unified index containing all languages.

    Args:
        project_path: Path to project root.
        query: Natural language query.
        k: Number of results to return.
        expand_graph: If True, include callers/callees in results.
        model: Model to use for query embedding. If None, uses
               the model from the index metadata.
        lang: Deprecated - ignored. Searches unified index with all languages.

    Returns:
        List of result dictionaries with name, file, line, score, language, etc.
    """
    import faiss
    import warnings

    # Validate inputs
    if query is None:
        raise TypeError("query cannot be None")
    if not query.strip():
        raise ValueError("query cannot be empty")
    if k <= 0:
        raise ValueError("k must be positive")

    # Deprecation warning for lang parameter
    if lang is not None:
        warnings.warn(
            "lang parameter is deprecated. Unified index searches all languages.",
            DeprecationWarning,
            stacklevel=2
        )

    project = Path(project_path).resolve()
    # Unified cache directory - same as build_semantic_index
    cache_dir = project / ".tldr" / "cache" / "semantic"

    index_file = cache_dir / "index.faiss"
    metadata_file = cache_dir / "metadata.json"

    # Check index exists - with migration hint for old lang-specific indexes
    if not index_file.exists():
        # Check for old language-specific indexes
        for old_lang in ALL_LANGUAGES:
            old_index = cache_dir / old_lang / "index.faiss"
            if old_index.exists():
                raise FileNotFoundError(
                    f"Found old index at {old_index}. "
                    f"Run 'tldr semantic index .' to migrate to unified index."
                )
        raise FileNotFoundError(
            f"Semantic index not found. Run: tldr semantic index {project}"
        )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata not found at {metadata_file}. Run: tldr semantic index ."
        )

    # Load index and metadata
    try:
        index = faiss.read_index(str(index_file))
    except RuntimeError as e:
        if "not recognized" in str(e):
            raise ValueError(
                f"Corrupted index at {index_file}. Rebuild: tldr semantic index ."
            ) from e
        raise
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    units = metadata["units"]

    # Validate dimension matches
    expected_dim = metadata.get("dimension")
    if expected_dim and index.d != expected_dim:
        raise ValueError(
            f"Index dimension ({index.d}) != metadata ({expected_dim}). "
            "Rebuild: tldr semantic index ."
        )

    # Use model from metadata if not specified (ensures matching embeddings)
    index_model = metadata.get("model")
    if model is None and index_model:
        model = index_model

    # Embed query (with instruction prefix for BGE)
    query_text = f"Represent this code search query: {query}"
    query_embedding = compute_embedding(query_text, model_name=model)
    query_embedding = query_embedding.reshape(1, -1)

    # Search
    k = min(k, len(units))
    scores, indices = index.search(query_embedding, k)

    # Build results
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0 or idx >= len(units):
            continue

        unit = units[idx]
        result = {
            "name": unit["name"],
            "qualified_name": unit["qualified_name"],
            "file": unit["file"],
            "line": unit["line"],
            "language": unit.get("language", "unknown"),
            "unit_type": unit["unit_type"],
            "signature": unit["signature"],
            "docstring": unit.get("docstring", ""),
            "code_preview": unit.get("code_preview", ""),
            "score": float(score),
        }

        # Include graph expansion if requested
        if expand_graph:
            result["calls"] = unit.get("calls", [])
            result["called_by"] = unit.get("called_by", [])
            result["related"] = list(
                set(unit.get("calls", []) + unit.get("called_by", []))
            )

        results.append(result)

    return results
