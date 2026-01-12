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
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

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

        _model = SentenceTransformer(hf_name)
        _model_name = hf_name
        return _model


def build_embedding_text(unit: EmbeddingUnit) -> str:
    """Build rich text for embedding from all 5 layers.

    Creates a single text string containing information from all
    analysis layers, suitable for embedding with a language model.

    Args:
        unit: The EmbeddingUnit containing code analysis.

    Returns:
        A text string combining all layer information.
    """
    parts = []

    # L1: Signature + docstring
    if unit.signature:
        parts.append(f"Signature: {unit.signature}")
    if unit.docstring:
        parts.append(f"Description: {unit.docstring}")

    # L2: Call graph (forward - callees)
    if unit.calls:
        calls_str = ", ".join(unit.calls[:5])  # Top 5
        parts.append(f"Calls: {calls_str}")

    # L2: Call graph (backward - callers)
    if unit.called_by:
        callers_str = ", ".join(unit.called_by[:5])  # Top 5
        parts.append(f"Called by: {callers_str}")

    # L3: Control flow summary
    if unit.cfg_summary:
        parts.append(f"Control flow: {unit.cfg_summary}")

    # L4: Data flow summary
    if unit.dfg_summary:
        parts.append(f"Data flow: {unit.dfg_summary}")

    # L5: Dependencies
    if unit.dependencies:
        parts.append(f"Dependencies: {unit.dependencies}")

    # Code preview (first 10 lines of function body)
    if unit.code_preview:
        parts.append(f"Code:\n{unit.code_preview}")

    # Add name and type for context
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


def _extract_units_from_file(
    args: Tuple[Dict[str, Any], str, str, Dict[str, List[str]], Dict[str, List[str]]],
) -> List[EmbeddingUnit]:
    """Worker function for parallel unit extraction from a single file.

    Processes one file and returns all EmbeddingUnit objects found.
    Designed to be called via ProcessPoolExecutor.map().

    Args:
        args: Tuple of (file_info, project_path, lang, calls_map, called_by_map)
            - file_info: Dict with 'path', 'functions', 'classes' keys
            - project_path: String path to project root
            - lang: Programming language
            - calls_map: Dict mapping function names to their callees
            - called_by_map: Dict mapping function names to their callers

    Returns:
        List of EmbeddingUnit objects extracted from the file.
    """
    file_info, project_path_str, lang, calls_map, called_by_map = args
    project = Path(project_path_str)
    units: List[EmbeddingUnit] = []

    file_path = file_info.get("path", "")
    full_path = project / file_path

    # Parse AST once per file - extracts line numbers, code preview, signature, and docstring
    ast_info, file_content, _ = _parse_file_ast(full_path, lang)

    # Get imports for dependencies (L5)
    dependencies = _get_file_dependencies(full_path, lang)

    # Process functions
    for func_name in file_info.get("functions", []):
        func_data = ast_info.get("functions", {}).get(func_name, {})
        signature = func_data.get("signature")
        docstring = func_data.get("docstring")
        line = func_data.get("line", 1)

        # Get CFG summary (L3) - pass pre-read content to avoid file I/O
        cfg_summary = _get_cfg_summary(full_path, func_name, lang, file_content)

        # Get DFG summary (L4) - pass pre-read content to avoid file I/O
        dfg_summary = _get_dfg_summary(full_path, func_name, lang, file_content)

        code_preview = func_data.get("code_preview", "")

        unit = EmbeddingUnit(
            name=func_name,
            qualified_name=f"{file_path.replace('/', '.')}.{func_name}",
            file=file_path,
            line=line,
            language=lang,
            unit_type="function",
            signature=signature or f"def {func_name}(...)",
            docstring=docstring or "",
            calls=calls_map.get(func_name, [])[:5],
            called_by=called_by_map.get(func_name, [])[:5],
            cfg_summary=cfg_summary,
            dfg_summary=dfg_summary,
            dependencies=dependencies,
            code_preview=code_preview,
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

        class_data = ast_info.get("classes", {}).get(class_name, {})
        class_line = class_data.get("line", 1)
        class_docstring = class_data.get("docstring") or ""

        # Add class itself
        unit = EmbeddingUnit(
            name=class_name,
            qualified_name=f"{file_path.replace('/', '.')}.{class_name}",
            file=file_path,
            line=class_line,
            language=lang,
            unit_type="class",
            signature=f"class {class_name}",
            docstring=class_docstring,
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
            method_qualified = f"{class_name}.{method}"
            method_data = ast_info.get("methods", {}).get(f"{class_name}.{method}", {})
            method_line = method_data.get("line", 1)
            method_preview = method_data.get("code_preview", "")
            method_signature = (
                method_data.get("signature") or f"def {method}(self, ...)"
            )
            method_docstring = method_data.get("docstring") or ""

            cfg_summary = _get_cfg_summary(full_path, method, lang, file_content)
            dfg_summary = _get_dfg_summary(full_path, method, lang, file_content)

            unit = EmbeddingUnit(
                name=method,
                qualified_name=f"{file_path.replace('/', '.')}.{method_qualified}",
                file=file_path,
                line=method_line,
                language=lang,
                unit_type="method",
                signature=method_signature,
                docstring=method_docstring,
                calls=calls_map.get(method, [])[:5],
                called_by=called_by_map.get(method, [])[:5],
                cfg_summary=cfg_summary,
                dfg_summary=dfg_summary,
                dependencies=dependencies,
                code_preview=method_preview,
            )
            units.append(unit)

    return units


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
    structure = get_code_structure(str(project), language=lang)

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

    # Process each file in structure
    files = structure.get("files", [])
    project_str = str(project)

    if len(files) >= MIN_FILES_FOR_PARALLEL:
        # Parallel processing for large projects
        # Each worker processes one file independently
        max_workers = min(os.cpu_count() or 4, 8)
        args_list = [(f, project_str, lang, calls_map, called_by_map) for f in files]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_extract_units_from_file, args_list))

        # Flatten results from all workers
        for batch in results:
            units.extend(batch)
    else:
        # Sequential processing for small projects (avoids process spawn overhead)
        for file_info in files:
            args = (file_info, project_str, lang, calls_map, called_by_map)
            file_units = _extract_units_from_file(args)
            units.extend(file_units)

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


@lru_cache(maxsize=1000)
def _get_cfg_summary_cached(
    file_path_str: str, func_name: str, lang: str, mtime_ns: int
) -> str:
    """Cached version of CFG summary extraction.

    Args:
        file_path_str: String path to the file (must be string for hashability).
        func_name: Name of the function to analyze.
        lang: Programming language.
        mtime_ns: File modification time in nanoseconds (for cache invalidation).

    Returns:
        CFG summary string (complexity and block count).
    """
    # mtime_ns is only used as cache key - not passed to inner function
    _ = mtime_ns
    return _get_cfg_summary(Path(file_path_str), func_name, lang)


@lru_cache(maxsize=1000)
def _get_dfg_summary_cached(
    file_path_str: str, func_name: str, lang: str, mtime_ns: int
) -> str:
    """Cached version of DFG summary extraction.

    Args:
        file_path_str: String path to the file (must be string for hashability).
        func_name: Name of the function to analyze.
        lang: Programming language.
        mtime_ns: File modification time in nanoseconds (for cache invalidation).

    Returns:
        DFG summary string (variable count and def-use chains).
    """
    # mtime_ns is only used as cache key - not passed to inner function
    _ = mtime_ns
    return _get_dfg_summary(Path(file_path_str), func_name, lang)


def _get_function_signature(
    file_path: Path,
    func_name: str,
    lang: str,
    content: Optional[str] = None,
    ast_tree=None,
) -> Optional[str]:
    """Extract function signature from file.

    Args:
        file_path: Path to the source file.
        func_name: Name of the function to extract signature for.
        lang: Programming language.
        content: Optional pre-read file content to avoid redundant I/O.
        ast_tree: Optional pre-parsed AST tree (for Python) to avoid redundant parsing.

    Returns:
        Function signature string or None if not found.
    """
    import ast as ast_module

    if content is None:
        if not file_path.exists():
            return None
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

    try:
        if lang == "python":
            if ast_tree is None:
                ast_tree = ast_module.parse(content)
            for node in ast_module.walk(ast_tree):
                if isinstance(node, ast_module.FunctionDef) and node.name == func_name:
                    # Build signature from args
                    args = []
                    for arg in node.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            arg_str += f": {ast_module.unparse(arg.annotation)}"
                        args.append(arg_str)

                    returns = ""
                    if node.returns:
                        returns = f" -> {ast_module.unparse(node.returns)}"

                    return f"def {func_name}({', '.join(args)}){returns}"

        # For other languages, return simple signature
        return f"function {func_name}(...)"

    except Exception:
        return None


def _get_function_docstring(
    file_path: Path,
    func_name: str,
    lang: str,
    content: Optional[str] = None,
    ast_tree=None,
) -> Optional[str]:
    """Extract function docstring from file.

    Args:
        file_path: Path to the source file.
        func_name: Name of the function to extract docstring for.
        lang: Programming language.
        content: Optional pre-read file content to avoid redundant I/O.
        ast_tree: Optional pre-parsed AST tree (for Python) to avoid redundant parsing.

    Returns:
        Function docstring or None if not found.
    """
    import ast as ast_module

    if content is None:
        if not file_path.exists():
            return None
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

    try:
        if lang == "python":
            if ast_tree is None:
                ast_tree = ast_module.parse(content)
            for node in ast_module.walk(ast_tree):
                if isinstance(node, ast_module.FunctionDef) and node.name == func_name:
                    return ast_module.get_docstring(node)

        return None

    except Exception:
        return None


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


def build_semantic_index(
    project_path: str,
    lang: str = "python",
    model: Optional[str] = None,
    show_progress: bool = True,
    respect_ignore: bool = True,
) -> int:
    """Build and save FAISS index + metadata for a project.

    Creates:
    - .tldr/cache/semantic/index.faiss - Vector index
    - .tldr/cache/semantic/metadata.json - Unit metadata

    Args:
        project_path: Path to project root.
        lang: Programming language.
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

    cache_dir = project / ".tldr" / "cache" / "semantic"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Extract all units (respecting .tldrignore)
    if console:
        with console.status("[bold green]Extracting code units...") as status:
            units = extract_units_from_project(
                str(project), lang=lang, respect_ignore=respect_ignore
            )
            status.update(f"[bold green]Extracted {len(units)} code units")
    else:
        units = extract_units_from_project(
            str(project), lang=lang, respect_ignore=respect_ignore
        )

    if not units:
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

    # Prepare metadata
    metadata = {
        "units": [u.to_dict() for u in units],
        "model": hf_name,
        "dimension": dimension,
        "count": len(units),
    }

    # Write to temp files first
    faiss.write_index(index, str(temp_index))
    temp_metadata.write_text(json.dumps(metadata, indent=2))

    # Atomic rename (POSIX guarantees rename atomicity on same filesystem)
    temp_index.rename(index_file)
    temp_metadata.rename(metadata_file)

    if console:
        console.print(f"[bold green]✓[/] Indexed {len(units)} code units")

    return len(units)


def semantic_search(
    project_path: str,
    query: str,
    k: int = 5,
    expand_graph: bool = False,
    model: Optional[str] = None,
) -> List[dict]:
    """Search for code units semantically.

    Args:
        project_path: Path to project root.
        query: Natural language query.
        k: Number of results to return.
        expand_graph: If True, include callers/callees in results.
        model: Model to use for query embedding. If None, uses
               the model from the index metadata.

    Returns:
        List of result dictionaries with name, file, line, score, etc.
    """
    import faiss

    # Handle empty query
    if not query or not query.strip():
        return []

    project = Path(project_path).resolve()
    cache_dir = project / ".tldr" / "cache" / "semantic"

    index_file = cache_dir / "index.faiss"
    metadata_file = cache_dir / "metadata.json"

    # Check index exists
    if not index_file.exists():
        raise FileNotFoundError(
            f"Semantic index not found at {index_file}. Run build_semantic_index first."
        )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata not found at {metadata_file}. Run build_semantic_index first."
        )

    # Load index and metadata
    index = faiss.read_index(str(index_file))
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    units = metadata["units"]

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
            "unit_type": unit["unit_type"],
            "signature": unit["signature"],
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
