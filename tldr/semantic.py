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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

logger = logging.getLogger("tldr.semantic")

ALL_LANGUAGES = ["python", "typescript", "javascript", "go", "rust", "java", "c", "cpp", "ruby", "php", "kotlin", "swift", "csharp", "scala", "lua", "luau", "elixir", "zig"]

# Lazy imports for heavy dependencies
_model = None
_model_name = None  # Track which model is loaded

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

# Project root markers - files that indicate a project root
PROJECT_ROOT_MARKERS = [".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod", ".tldr"]


def _find_project_root(start_path: Path) -> Path:
    """Find project root by walking up from start_path.

    Looks for common project markers (.git, pyproject.toml, etc.).
    Also respects CLAUDE_PROJECT_DIR environment variable.

    Args:
        start_path: Path to start searching from.

    Returns:
        Project root path, or start_path if no markers found.
    """
    # Check environment variable first
    env_root = os.environ.get("CLAUDE_PROJECT_DIR")
    if env_root:
        env_path = Path(env_root).resolve()
        if env_path.exists():
            return env_path

    # Walk up looking for project markers
    current = start_path.resolve()
    while current != current.parent:
        for marker in PROJECT_ROOT_MARKERS:
            if (current / marker).exists():
                return current
        current = current.parent

    # No markers found - use start_path
    return start_path.resolve()


@dataclass
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
    print(f"   (Set TLDR_AUTO_DOWNLOAD=1 to skip this prompt)\n", file=sys.stderr)

    try:
        response = input("Continue with download? [Y/n] ").strip().lower()
        return response in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def get_model(model_name: Optional[str] = None):
    """Lazy-load the embedding model (cached).

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

    # Return cached model if same
    if _model is not None and _model_name == hf_name:
        return _model

    # Check if model needs downloading
    if not _model_exists_locally(hf_name):
        model_key = model_name if model_name in SUPPORTED_MODELS else None
        if model_key and not _confirm_download(model_key):
            raise ValueError(f"Model download declined. Use --model to choose a smaller model.")

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


def extract_units_from_project(project_path: str, lang: str = "python", respect_ignore: bool = True, progress_callback=None) -> List[EmbeddingUnit]:
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
    from tldr.api import get_code_structure, build_project_call_graph, get_imports
    from tldr.tldrignore import load_ignore_patterns, should_ignore

    project = Path(project_path).resolve()
    units = []

    # Load ignore spec before getting structure
    ignore_spec = load_ignore_patterns(project) if respect_ignore else None

    # Get code structure (L1) - use high limit for semantic index
    structure = get_code_structure(str(project), language=lang, max_results=100000, ignore_spec=ignore_spec)

    # Filter ignored files
    if respect_ignore:
        spec = load_ignore_patterns(project)
        structure["files"] = [
            f for f in structure.get("files", [])
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

    # Use parallel processing if we have multiple files
    if len(files) > 1 and max_workers > 1:
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
                        file_units = future.result(timeout=60)
                        units.extend(file_units)
                        if progress_callback:
                            progress_callback(file_info.get('path', 'unknown'), len(units), len(files))
                    except Exception as e:
                        logger.warning(f"Failed to process {file_info.get('path', 'unknown')}: {e}")

        except Exception as e:
            logger.warning(f"Parallel extraction failed: {e}, falling back to sequential")
            for file_info in files:
                try:
                    file_units = _process_file_for_extraction(
                        file_info, str(project), lang, calls_map, called_by_map
                    )
                    units.extend(file_units)
                    if progress_callback:
                        progress_callback(file_info.get('path', 'unknown'), len(units), len(files))
                except Exception as fe:
                    logger.warning(f"Failed to process {file_info.get('path', 'unknown')}: {fe}")
    else:
        for file_info in files:
            try:
                file_units = _process_file_for_extraction(
                    file_info, str(project), lang, calls_map, called_by_map
                )
                units.extend(file_units)
                if progress_callback:
                    progress_callback(file_info.get('path', 'unknown'), len(units), len(files))
            except Exception as e:
                logger.warning(f"Failed to process {file_info.get('path', 'unknown')}: {e}")

    return units


def _parse_file_ast(file_path: Path, lang: str) -> dict:
    """Parse file AST to extract line numbers and code previews.

    Returns:
        Dict with structure:
        {
            "functions": {func_name: {"line": int, "code_preview": str}},
            "classes": {class_name: {"line": int}},
            "methods": {"ClassName.method": {"line": int, "code_preview": str}}
        }
    """
    result = {"functions": {}, "classes": {}, "methods": {}}

    if not file_path.exists():
        return result

    try:
        content = file_path.read_text()
        lines = content.split('\n')

        if lang == "python":
            import ast
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # Check if this is a method (inside a class)
                    parent_class = None
                    for potential_parent in ast.walk(tree):
                        if isinstance(potential_parent, ast.ClassDef):
                            if node in ast.walk(potential_parent) and node.name != potential_parent.name:
                                # Check if node is a direct child method
                                for item in potential_parent.body:
                                    if item is node:
                                        parent_class = potential_parent.name
                                        break

                    # Extract code preview (first 10 lines of body)
                    start_line = node.lineno
                    end_line = getattr(node, 'end_lineno', start_line + 10)
                    body_lines = lines[start_line - 1:min(end_line, start_line + 10) - 1]
                    code_preview = '\n'.join(body_lines[:10])

                    if parent_class:
                        result["methods"][f"{parent_class}.{node.name}"] = {
                            "line": node.lineno,
                            "code_preview": code_preview
                        }
                    else:
                        result["functions"][node.name] = {
                            "line": node.lineno,
                            "code_preview": code_preview
                        }

                elif isinstance(node, ast.ClassDef):
                    result["classes"][node.name] = {"line": node.lineno}

    except Exception:
        # Return empty result on any parsing error
        pass

    return result


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


def _get_cfg_summary(file_path: Path, func_name: str, lang: str) -> str:
    """Get CFG summary (complexity, block count) for a function."""
    if not file_path.exists():
        return ""

    try:
        content = file_path.read_text()

        # Import the appropriate CFG extractor based on language
        from tldr import cfg_extractor

        extractor_map = {
            "python": cfg_extractor.extract_python_cfg,
            "typescript": cfg_extractor.extract_typescript_cfg,
            "javascript": cfg_extractor.extract_typescript_cfg,  # JS uses TS extractor
            "go": cfg_extractor.extract_go_cfg,
            "rust": cfg_extractor.extract_rust_cfg,
            "java": cfg_extractor.extract_java_cfg,
            "c": cfg_extractor.extract_c_cfg,
            "cpp": cfg_extractor.extract_cpp_cfg,
            "php": cfg_extractor.extract_php_cfg,
            "ruby": cfg_extractor.extract_ruby_cfg,
            "swift": cfg_extractor.extract_swift_cfg,
            "csharp": cfg_extractor.extract_csharp_cfg,
            "kotlin": cfg_extractor.extract_kotlin_cfg,
            "scala": cfg_extractor.extract_scala_cfg,
            "lua": cfg_extractor.extract_lua_cfg,
            "luau": cfg_extractor.extract_luau_cfg,
            "elixir": cfg_extractor.extract_elixir_cfg,
            "zig": cfg_extractor.extract_zig_cfg,
        }

        extractor = extractor_map.get(lang)
        if extractor:
            cfg = extractor(content, func_name)
            return f"complexity:{cfg.cyclomatic_complexity}, blocks:{len(cfg.blocks)}"
    except Exception:
        pass

    return ""


def _get_dfg_summary(file_path: Path, func_name: str, lang: str) -> str:
    """Get DFG summary (variable count, def-use chains) for a function."""
    if not file_path.exists():
        return ""

    try:
        content = file_path.read_text()

        # Import the appropriate DFG extractor based on language
        from tldr import dfg_extractor

        extractor_map = {
            "python": dfg_extractor.extract_python_dfg,
            "typescript": dfg_extractor.extract_typescript_dfg,
            "javascript": dfg_extractor.extract_typescript_dfg,  # JS uses TS extractor
            "go": dfg_extractor.extract_go_dfg,
            "rust": dfg_extractor.extract_rust_dfg,
            "java": dfg_extractor.extract_java_dfg,
            "c": dfg_extractor.extract_c_dfg,
            "cpp": dfg_extractor.extract_cpp_dfg,
            "php": dfg_extractor.extract_php_dfg,
            "ruby": dfg_extractor.extract_ruby_dfg,
            "swift": dfg_extractor.extract_swift_dfg,
            "csharp": dfg_extractor.extract_csharp_dfg,
            "kotlin": dfg_extractor.extract_kotlin_dfg,
            "scala": dfg_extractor.extract_scala_dfg,
            "lua": dfg_extractor.extract_lua_dfg,
            "luau": dfg_extractor.extract_luau_dfg,
            "elixir": dfg_extractor.extract_elixir_dfg,
            "zig": dfg_extractor.extract_zig_dfg,
        }

        extractor = extractor_map.get(lang)
        if extractor:
            dfg = extractor(content, func_name)

            # Count unique variables and def-use chains
            var_names = set()
            for ref in dfg.var_refs:
                var_names.add(ref.name)

            return f"vars:{len(var_names)}, def-use chains:{len(dfg.dataflow_edges)}"
    except Exception:
        pass

    return ""


def _get_function_signature(file_path: Path, func_name: str, lang: str) -> Optional[str]:
    """Extract function signature from file."""
    if not file_path.exists():
        return None

    try:
        content = file_path.read_text()

        if lang == "python":
            import ast
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    # Build signature from args
                    args = []
                    for arg in node.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        args.append(arg_str)

                    returns = ""
                    if node.returns:
                        returns = f" -> {ast.unparse(node.returns)}"

                    return f"def {func_name}({', '.join(args)}){returns}"


        # For other languages, return simple signature
        return f"function {func_name}(...)"

    except Exception:
        return None


def _get_function_docstring(file_path: Path, func_name: str, lang: str) -> Optional[str]:
    """Extract function docstring from file."""
    if not file_path.exists():
        return None

    try:
        content = file_path.read_text()

        if lang == "python":
            import ast
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    return ast.get_docstring(node)

        return None

    except Exception:
        return None


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
        content = full_path.read_text()
        lines = content.split('\n')
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return units

    # Parse AST once for all function info
    ast_info = {"functions": {}, "classes": {}, "methods": {}}
    all_signatures = {}
    all_docstrings = {}

    if lang == "python":
        try:
            import ast
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Check if this is a method (inside a class)
                    parent_class = None
                    for potential_parent in ast.walk(tree):
                        if isinstance(potential_parent, ast.ClassDef):
                            for item in potential_parent.body:
                                if item is node:
                                    parent_class = potential_parent.name
                                    break

                    # Extract code preview (first 10 lines of body)
                    start_line = node.lineno
                    end_line = getattr(node, 'end_lineno', start_line + 10)
                    body_lines = lines[start_line - 1:min(end_line, start_line + 10) - 1]
                    code_preview = '\n'.join(body_lines[:10])

                    # Build signature
                    args = []
                    for arg in node.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        args.append(arg_str)
                    returns = ""
                    if node.returns:
                        returns = f" -> {ast.unparse(node.returns)}"
                    signature = f"def {node.name}({', '.join(args)}){returns}"

                    # Get docstring
                    docstring = ast.get_docstring(node) or ""

                    if parent_class:
                        key = f"{parent_class}.{node.name}"
                        ast_info["methods"][key] = {
                            "line": node.lineno,
                            "code_preview": code_preview
                        }
                        all_signatures[key] = signature
                        all_docstrings[key] = docstring
                    else:
                        ast_info["functions"][node.name] = {
                            "line": node.lineno,
                            "code_preview": code_preview
                        }
                        all_signatures[node.name] = signature
                        all_docstrings[node.name] = docstring

                elif isinstance(node, ast.ClassDef):
                    ast_info["classes"][node.name] = {"line": node.lineno}

        except Exception as e:
            logger.debug(f"AST parse failed for {file_path}: {e}")

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

    # Language-to-extractor mapping for CFG/DFG analysis
    def _get_extractors(language: str):
        """Return (cfg_extractor, dfg_extractor) for the given language."""
        if language == "python":
            from tldr.cfg_extractor import extract_python_cfg
            from tldr.dfg_extractor import extract_python_dfg
            return extract_python_cfg, extract_python_dfg
        elif language in ("typescript", "javascript"):
            from tldr.cfg_extractor import extract_typescript_cfg
            from tldr.dfg_extractor import extract_typescript_dfg
            return extract_typescript_cfg, extract_typescript_dfg
        return None, None

    cfg_extractor, dfg_extractor = _get_extractors(lang)

    if cfg_extractor and dfg_extractor:
        # Get all function names we need to process
        all_func_names = list(file_info.get("functions", []))
        for class_info in file_info.get("classes", []):
            if isinstance(class_info, dict):
                all_func_names.extend(class_info.get("methods", []))

        for func_name in all_func_names:
            try:
                cfg = cfg_extractor(content, func_name)
                cfg_cache[func_name] = f"complexity:{cfg.cyclomatic_complexity}, blocks:{len(cfg.blocks)}"
            except Exception:
                cfg_cache[func_name] = ""

            try:
                dfg = dfg_extractor(content, func_name)
                var_names = {ref.name for ref in dfg.var_refs}
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
        '.luau': 'luau',
        '.ex': 'elixir',
        '.exs': 'elixir',
        '.zig': 'zig',
    }

    found_languages = set()
    spec = load_ignore_patterns(project_path) if respect_ignore else None

    for root, dirs, files in os.walk(project_path):
        # Prune common heavy dirs immediately for speed
        dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', '.tldr', 'venv', '.venv', '__pycache__', '.idea', '.vscode', 'env', '.env', 'vendor', 'deps', '_build', 'cover'}]

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
    import numpy as np
    from tldr.tldrignore import ensure_tldrignore

    console = _get_progress_console() if show_progress else None

    # Resolve paths: scan_path is where to look for code, project_root is where to store cache
    scan_path = Path(project_path).resolve()
    project_root = _find_project_root(scan_path)

    # Ensure .tldrignore exists at project root (create with defaults if not)
    created, message = ensure_tldrignore(project_root)
    if created and console:
        console.print(f"[yellow]{message}[/yellow]")

    # Resolve model name early to get HF name for metadata
    model_key = model if model else DEFAULT_MODEL
    if model_key in SUPPORTED_MODELS:
        hf_name = SUPPORTED_MODELS[model_key]["hf_name"]
    else:
        hf_name = model_key

    # Always store cache at project root, not scan path
    cache_dir = project_root / ".tldr" / "cache" / "semantic"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Extract all units (respecting .tldrignore) - scan from scan_path, not project_root
    if console:
        with console.status("[bold green]Extracting code units...") as status:
            def update_progress(file_path, units_count, total_files):
                short_path = file_path if len(file_path) < 50 else "..." + file_path[-47:]
                status.update(f"[bold green]Processing {short_path}... ({units_count} units)")

            if lang == "all":
                status.update("[bold green]Scanning project languages...")
                target_languages = _detect_project_languages(scan_path, respect_ignore=respect_ignore)
                if not target_languages:
                    console.print("[yellow]No supported languages detected in project[/yellow]")
                    return 0
                if console:
                    console.print(f"[dim]Detected languages: {', '.join(target_languages)}[/dim]")

                units = []
                for lang_name in target_languages:
                    status.update(f"[bold green]Extracting {lang_name} code units...")
                    units.extend(extract_units_from_project(str(scan_path), lang=lang_name, respect_ignore=respect_ignore, progress_callback=update_progress))
            else:
                units = extract_units_from_project(str(scan_path), lang=lang, respect_ignore=respect_ignore, progress_callback=update_progress)
            status.update(f"[bold green]Extracted {len(units)} code units")
    else:
        if lang == "all":
            target_languages = _detect_project_languages(scan_path, respect_ignore=respect_ignore)
            if not target_languages:
                return 0
            units = []
            for lang_name in target_languages:
                units.extend(extract_units_from_project(str(scan_path), lang=lang_name, respect_ignore=respect_ignore))
        else:
            units = extract_units_from_project(str(scan_path), lang=lang, respect_ignore=respect_ignore)

    if not units:
        return 0

    import numpy as np

    BATCH_SIZE = 64
    num_units = len(units)
    texts = [build_embedding_text(unit) for unit in units]

    if console:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Computing embeddings...", total=num_units)

            model_obj = get_model(model)
            all_embeddings = []

            for i in range(0, num_units, BATCH_SIZE):
                chunk_end = min(i + BATCH_SIZE, num_units)
                chunk_texts = texts[i:chunk_end]

                current_unit = units[i]
                short_path = current_unit.file if len(current_unit.file) < 40 else "..." + current_unit.file[-37:]
                progress.update(task, description=f"[bold green]Embedding {short_path}::{current_unit.name}")

                result = model_obj.encode(
                    chunk_texts,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                all_embeddings.extend(np.array(result, dtype=np.float32))

                progress.update(task, completed=chunk_end)

            embeddings_matrix = np.vstack(all_embeddings)
    else:
        model_obj = get_model(model)
        result = model_obj.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True
        )
        embeddings_matrix = np.array(result, dtype=np.float32)

    dimension = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_matrix)

    # Save index
    index_file = cache_dir / "index.faiss"
    faiss.write_index(index, str(index_file))

    # Save metadata with actual model used
    metadata = {
        "units": [u.to_dict() for u in units],
        "model": hf_name,
        "dimension": dimension,
        "count": len(units),
    }
    metadata_file = cache_dir / "metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))

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
    import numpy as np

    # Handle empty query
    if not query or not query.strip():
        return []

    # Find project root for cache location (matches build_semantic_index behavior)
    scan_path = Path(project_path).resolve()
    project_root = _find_project_root(scan_path)
    cache_dir = project_root / ".tldr" / "cache" / "semantic"

    index_file = cache_dir / "index.faiss"
    metadata_file = cache_dir / "metadata.json"

    # Check index exists
    if not index_file.exists():
        raise FileNotFoundError(f"Semantic index not found at {index_file}. Run build_semantic_index first.")

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_file}. Run build_semantic_index first.")

    # Load index and metadata
    index = faiss.read_index(str(index_file))
    metadata = json.loads(metadata_file.read_text())
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
            result["related"] = list(set(unit.get("calls", []) + unit.get("called_by", [])))

        results.append(result)

    return results
