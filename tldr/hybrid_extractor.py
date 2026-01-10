#!/usr/bin/env python3
"""
Hybrid code structure extractor - best approach per language.

Strategy:
- Python: ast module (fastest, most accurate)
- TypeScript/JavaScript: tree-sitter (if available) else Pygments
- Other languages: Pygments fallback (broad support, less metadata)

Output is unified across all extractors.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ast_extractor import extract_python, ModuleInfo, FunctionInfo, ClassInfo, ImportInfo, CallGraphInfo
from .signature_extractor_pygments import SignatureExtractor

logger = logging.getLogger(__name__)

# File size limit - tree-sitter memory usage is ~10-200x file size
# 5MB matches mcp-server-tree-sitter default; override with TLDR_MAX_FILE_SIZE
DEFAULT_MAX_FILE_SIZE = 5_000_000  # 5MB
MAX_FILE_SIZE = int(os.environ.get("TLDR_MAX_FILE_SIZE", DEFAULT_MAX_FILE_SIZE))


class FileTooLargeError(Exception):
    """Raised when a file exceeds MAX_FILE_SIZE."""
    def __init__(self, file_path: Path, size: int, limit: int):
        self.file_path = file_path
        self.size = size
        self.limit = limit
        super().__init__(
            f"File {file_path} is {size:,} bytes, exceeds limit of {limit:,} bytes. "
            f"Set TLDR_MAX_FILE_SIZE environment variable to increase limit."
        )


class ParseError(Exception):
    """Raised when tree-sitter parsing fails."""
    def __init__(self, file_path: Path, language: str, error: Exception):
        self.file_path = file_path
        self.language = language
        self.original_error = error
        super().__init__(f"Failed to parse {file_path} as {language}: {error}")

# Check tree-sitter availability
TREE_SITTER_AVAILABLE = False
TREE_SITTER_GO_AVAILABLE = False
TREE_SITTER_RUST_AVAILABLE = False
try:
    from tree_sitter import Language, Parser
    import tree_sitter_typescript
    import tree_sitter_javascript
    TREE_SITTER_AVAILABLE = True
except ImportError:
    pass

try:
    import tree_sitter_go
    TREE_SITTER_GO_AVAILABLE = True
except ImportError:
    pass

try:
    import tree_sitter_rust
    TREE_SITTER_RUST_AVAILABLE = True
except ImportError:
    pass

try:
    import tree_sitter_java
    TREE_SITTER_JAVA_AVAILABLE = True
except ImportError:
    TREE_SITTER_JAVA_AVAILABLE = False

TREE_SITTER_C_AVAILABLE = False
try:
    from tree_sitter import Language, Parser
    import tree_sitter_c
    TREE_SITTER_C_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_CPP_AVAILABLE = False
try:
    from tree_sitter import Language, Parser
    import tree_sitter_cpp
    TREE_SITTER_CPP_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_RUBY_AVAILABLE = False
try:
    from tree_sitter import Language, Parser
    import tree_sitter_ruby
    TREE_SITTER_RUBY_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_KOTLIN_AVAILABLE = False
try:
    from tree_sitter import Language, Parser
    import tree_sitter_kotlin
    TREE_SITTER_KOTLIN_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_SWIFT_AVAILABLE = False
try:
    from tree_sitter import Language, Parser
    import tree_sitter_swift
    TREE_SITTER_SWIFT_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_CSHARP_AVAILABLE = False
try:
    from tree_sitter import Language, Parser
    import tree_sitter_c_sharp
    TREE_SITTER_CSHARP_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_SCALA_AVAILABLE = False
try:
    from tree_sitter import Language, Parser
    import tree_sitter_scala
    TREE_SITTER_SCALA_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_LUA_AVAILABLE = False
try:
    from tree_sitter import Language, Parser
    import tree_sitter_lua
    TREE_SITTER_LUA_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_ELIXIR_AVAILABLE = False
try:
    from tree_sitter import Language, Parser
    import tree_sitter_elixir
    TREE_SITTER_ELIXIR_AVAILABLE = True
except ImportError:
    pass


class HybridExtractor:
    """
    Extract code structure using best available method per language.

    Priority:
    1. Native AST (Python) - fastest, richest
    2. Tree-sitter (JS/TS) - fast, good metadata
    3. Pygments (fallback) - broad support, signatures only
    """

    # File extension to extractor method mapping
    PYTHON_EXTENSIONS = {".py", ".pyx", ".pyi"}
    TREE_SITTER_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}
    GO_EXTENSIONS = {".go"}
    RUST_EXTENSIONS = {".rs"}
    JAVA_EXTENSIONS = {".java"}
    C_EXTENSIONS = {".c", ".h"}
    CPP_EXTENSIONS = {".cpp", ".hpp", ".cc", ".cxx", ".hh"}
    RUBY_EXTENSIONS = {".rb"}
    KOTLIN_EXTENSIONS = {".kt", ".kts"}
    SWIFT_EXTENSIONS = {".swift"}
    CSHARP_EXTENSIONS = {".cs"}
    SCALA_EXTENSIONS = {".scala", ".sc"}
    LUA_EXTENSIONS = {".lua"}
    ELIXIR_EXTENSIONS = {".ex", ".exs"}

    def __init__(self):
        self._pygments_extractor = SignatureExtractor()
        self._ts_parsers: dict[str, Any] = {}

    def _safe_decode(self, data: bytes) -> str:
        """Safely decode bytes to string, replacing invalid UTF-8.

        This prevents UnicodeDecodeError crashes on malformed input.
        Invalid bytes are replaced with the Unicode replacement character.
        """
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")

    def extract(self, file_path: str | Path, base_path: str | None = None) -> ModuleInfo:
        """Extract from any supported file.

        Args:
            file_path: Path to the file to extract
            base_path: Optional base directory for path containment validation

        Raises:
            PathTraversalError: If path contains directory traversal patterns
        """
        # Security: Import and validate path containment
        from .api import _validate_path_containment
        _validate_path_containment(str(file_path), base_path)
        file_path = Path(file_path)

        # File size check - prevent memory exhaustion on large files
        try:
            file_size = file_path.stat().st_size
            if file_size > MAX_FILE_SIZE:
                raise FileTooLargeError(file_path, file_size, MAX_FILE_SIZE)
        except OSError as e:
            logger.warning(f"Could not stat file {file_path}: {e}")
            # Continue anyway - let the actual read fail if there's a problem

        suffix = file_path.suffix.lower()

        # Python - use native AST
        if suffix in self.PYTHON_EXTENSIONS:
            return extract_python(file_path)

        # JS/TS - use tree-sitter if available
        if suffix in self.TREE_SITTER_EXTENSIONS:
            if TREE_SITTER_AVAILABLE:
                result = self._try_tree_sitter(lambda fp: self._extract_tree_sitter(fp, suffix), file_path, "typescript")
                if result:
                    return result
            logger.debug(f"Tree-sitter not available or failed, using Pygments for {suffix}")

        # Go - use tree-sitter-go if available
        if suffix in self.GO_EXTENSIONS:
            if TREE_SITTER_GO_AVAILABLE and TREE_SITTER_AVAILABLE:
                result = self._try_tree_sitter(self._extract_go, file_path, "go")
                if result:
                    return result
            logger.debug(f"Tree-sitter-go not available or failed, using Pygments for {suffix}")

        # Rust - use tree-sitter-rust if available
        if suffix in self.RUST_EXTENSIONS:
            if TREE_SITTER_RUST_AVAILABLE and TREE_SITTER_AVAILABLE:
                result = self._try_tree_sitter(self._extract_rust, file_path, "rust")
                if result:
                    return result
            logger.debug(f"Tree-sitter-rust not available or failed, using Pygments for {suffix}")

        # Java - use tree-sitter-java if available
        if suffix in self.JAVA_EXTENSIONS:
            if TREE_SITTER_JAVA_AVAILABLE and TREE_SITTER_AVAILABLE:
                result = self._try_tree_sitter(self._extract_java, file_path, "java")
                if result:
                    return result
            logger.debug(f"Tree-sitter-java not available or failed, using Pygments for {suffix}")

        # C - use tree-sitter-c if available
        if suffix in self.C_EXTENSIONS:
            if TREE_SITTER_C_AVAILABLE:
                result = self._try_tree_sitter(self._extract_c, file_path, "c")
                if result:
                    return result
            logger.debug(f"Tree-sitter-c not available or failed, using Pygments for {suffix}")

        # C++ - use tree-sitter-cpp if available
        if suffix in self.CPP_EXTENSIONS:
            if TREE_SITTER_CPP_AVAILABLE:
                result = self._try_tree_sitter(self._extract_cpp, file_path, "cpp")
                if result:
                    return result
            logger.debug(f"Tree-sitter-cpp not available or failed, using Pygments for {suffix}")

        # Ruby - use tree-sitter-ruby if available
        if suffix in self.RUBY_EXTENSIONS:
            if TREE_SITTER_RUBY_AVAILABLE:
                result = self._try_tree_sitter(self._extract_ruby, file_path, "ruby")
                if result:
                    return result
            logger.debug(f"Tree-sitter-ruby not available or failed, using Pygments for {suffix}")

        # Kotlin - use tree-sitter-kotlin if available
        if suffix in self.KOTLIN_EXTENSIONS:
            if TREE_SITTER_KOTLIN_AVAILABLE:
                result = self._try_tree_sitter(self._extract_kotlin, file_path, "kotlin")
                if result:
                    return result
            logger.debug(f"Tree-sitter-kotlin not available or failed, using Pygments for {suffix}")

        # Swift - use tree-sitter-swift if available
        if suffix in self.SWIFT_EXTENSIONS:
            if TREE_SITTER_SWIFT_AVAILABLE:
                result = self._try_tree_sitter(self._extract_swift, file_path, "swift")
                if result:
                    return result
            logger.debug(f"Tree-sitter-swift not available or failed, using Pygments for {suffix}")

        # C# - use tree-sitter-c-sharp if available
        if suffix in self.CSHARP_EXTENSIONS:
            if TREE_SITTER_CSHARP_AVAILABLE:
                result = self._try_tree_sitter(self._extract_csharp, file_path, "csharp")
                if result:
                    return result
            logger.debug(f"Tree-sitter-c-sharp not available or failed, using Pygments for {suffix}")

        # Scala - use tree-sitter-scala if available
        if suffix in self.SCALA_EXTENSIONS:
            if TREE_SITTER_SCALA_AVAILABLE:
                result = self._try_tree_sitter(self._extract_scala, file_path, "scala")
                if result:
                    return result
            logger.debug(f"Tree-sitter-scala not available or failed, using Pygments for {suffix}")

        # Lua - use tree-sitter-lua if available
        if suffix in self.LUA_EXTENSIONS:
            if TREE_SITTER_LUA_AVAILABLE:
                result = self._try_tree_sitter(self._extract_lua, file_path, "lua")
                if result:
                    return result
            logger.debug(f"Tree-sitter-lua not available or failed, using Pygments for {suffix}")

        # Elixir - use tree-sitter-elixir if available
        if suffix in self.ELIXIR_EXTENSIONS:
            if TREE_SITTER_ELIXIR_AVAILABLE:
                result = self._try_tree_sitter(self._extract_elixir, file_path, "elixir")
                if result:
                    return result
            logger.debug(f"Tree-sitter-elixir not available or failed, using Pygments for {suffix}")

        # Fallback to Pygments
        return self._extract_pygments(file_path)

    def _extract_pygments(self, file_path: Path) -> ModuleInfo:
        """Extract using Pygments (signatures only)."""
        try:
            signatures_text = self._pygments_extractor.get_signatures(str(file_path))
            signatures = self._parse_signatures(signatures_text)
        except Exception as e:
            logger.warning(f"Pygments extraction failed for {file_path}: {e}")
            signatures = []

        # Convert to ModuleInfo format
        # Pygments doesn't give us enough info to distinguish classes from functions
        # so we put everything in functions
        functions = [
            FunctionInfo(
                name=sig.split("(")[0].strip() if "(" in sig else sig,
                params=self._extract_params_from_sig(sig),
                return_type=None,
                docstring=None,
            )
            for sig in signatures
        ]

        return ModuleInfo(
            file_path=str(file_path),
            language=self._detect_language(file_path),
            docstring=None,
            functions=functions,
        )

    def _extract_tree_sitter(self, file_path: Path, suffix: str) -> ModuleInfo:
        """Extract using tree-sitter for JS/TS."""
        lang_map = {
            ".ts": "typescript",
            ".tsx": "tsx",
            ".js": "javascript",
            ".jsx": "javascript",
            ".mjs": "javascript",
            ".cjs": "javascript",
        }
        language = lang_map.get(suffix, "javascript")

        with open(file_path, "rb") as f:
            source = f.read()

        parser = self._get_ts_parser(language)
        tree = self._safe_parse(parser, source, file_path, language)

        module_info = ModuleInfo(
            file_path=str(file_path),
            language=language,
            docstring=None,
        )

        # First pass: collect defined function names
        defined_names = self._collect_ts_definitions(tree.root_node, source)

        self._extract_ts_nodes(tree.root_node, source, module_info, defined_names)
        return module_info

    def _collect_ts_definitions(self, node, source: bytes) -> set[str]:
        """Collect all defined function/method names."""
        names: set[str] = set()
        for child in node.children:
            if child.type == "function_declaration":
                for c in child.children:
                    if c.type == "identifier":
                        names.add(self._safe_decode(source[c.start_byte:c.end_byte]))
                        break
            elif child.type == "class_declaration":
                class_name = ""
                for c in child.children:
                    if c.type in ("identifier", "type_identifier"):
                        class_name = self._safe_decode(source[c.start_byte:c.end_byte])
                    elif c.type == "class_body":
                        for method in c.children:
                            if method.type == "method_definition":
                                for m in method.children:
                                    if m.type == "property_identifier":
                                        names.add(self._safe_decode(source[m.start_byte:m.end_byte]))
                                        break
            # Recurse
            if child.type in ("program", "export_statement"):
                names.update(self._collect_ts_definitions(child, source))
        return names

    def _get_ts_parser(self, language: str) -> Any:
        """Get or create tree-sitter parser."""
        if language not in self._ts_parsers:
            parser = Parser()
            if language in ("typescript", "tsx"):
                if language == "tsx":
                    parser.language = Language(tree_sitter_typescript.language_tsx())
                else:
                    parser.language = Language(tree_sitter_typescript.language_typescript())
            else:
                parser.language = Language(tree_sitter_javascript.language())
            self._ts_parsers[language] = parser
        return self._ts_parsers[language]

    def _safe_parse(self, parser: Any, source: bytes, file_path: Path, language: str) -> Any:
        """Safely parse source code, catching tree-sitter errors."""
        try:
            return parser.parse(source)
        except Exception as e:
            logger.error(f"Tree-sitter parse failed for {file_path} ({language}): {e}")
            raise ParseError(file_path, language, e)

    def _try_tree_sitter(self, extractor_method, file_path: Path, language: str) -> ModuleInfo | None:
        """Try tree-sitter extraction, return None on failure to allow Pygments fallback."""
        try:
            return extractor_method(file_path)
        except (ParseError, OSError, ValueError, RuntimeError) as e:
            logger.warning(f"Tree-sitter extraction failed for {file_path} ({language}), falling back to Pygments: {e}")
            return None
        except Exception as e:
            # Catch-all for unexpected errors (corrupted grammars, etc.)
            logger.warning(f"Unexpected error during tree-sitter extraction for {file_path} ({language}): {e}")
            return None

    def _extract_ts_nodes(self, node, source: bytes, module_info: ModuleInfo, defined_names: set[str] | None = None):
        """Recursively extract from tree-sitter nodes."""
        prev_comment = None  # Track JSDoc comments

        for child in node.children:
            node_type = child.type

            # Track JSDoc comments (/** ... */)
            if node_type == "comment":
                comment_text = self._safe_decode(source[child.start_byte:child.end_byte])
                if comment_text.startswith("/**"):
                    prev_comment = self._parse_jsdoc(comment_text)
                continue

            # Imports
            if node_type == "import_statement":
                text = self._safe_decode(source[child.start_byte:child.end_byte]).rstrip(";")
                # Strip "import " prefix since statement() will add it back
                if text.startswith("import "):
                    text = text[7:]
                module_info.imports.append(ImportInfo(
                    module=text,
                    names=[],
                    is_from=False,
                    line_number=child.start_point[0] + 1,
                ))
                prev_comment = None

            # Functions
            elif node_type in ("function_declaration", "arrow_function", "method_definition"):
                func = self._extract_ts_function(child, source)
                if func:
                    if prev_comment:
                        func.docstring = prev_comment
                    module_info.functions.append(func)
                    # Extract calls
                    if defined_names:
                        self._extract_ts_calls(child, func.name, source, module_info.call_graph, defined_names)
                prev_comment = None

            # Classes
            elif node_type == "class_declaration":
                cls = self._extract_ts_class(child, source, prev_comment, defined_names, module_info.call_graph)
                if cls:
                    module_info.classes.append(cls)
                prev_comment = None

            # Recurse into containers
            elif node_type in ("export_statement", "lexical_declaration", "program",
                            "variable_declaration", "variable_declarator", "statement_block",
                            "export_clause"):
                self._extract_ts_nodes(child, source, module_info, defined_names)
                prev_comment = None
            else:
                prev_comment = None

    def _extract_ts_calls(self, node, caller_name: str, source: bytes, call_graph: CallGraphInfo, defined_names: set[str]):
        """Extract function calls from a TS/JS function body."""
        for child in node.children:
            if child.type == "call_expression":
                callee = self._get_ts_call_name(child, source)
                if callee and callee in defined_names:
                    call_graph.add_call(caller_name, callee)
            # Recurse into all children
            self._extract_ts_calls(child, caller_name, source, call_graph, defined_names)

    def _get_ts_call_name(self, node, source: bytes) -> str | None:
        """Get the name of a called function from a call_expression."""
        for child in node.children:
            if child.type == "identifier":
                return self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "member_expression":
                # Get the method name (rightmost part)
                for c in child.children:
                    if c.type == "property_identifier":
                        return self._safe_decode(source[c.start_byte:c.end_byte])
        return None

    def _parse_jsdoc(self, comment: str) -> str:
        """Parse JSDoc comment into clean docstring."""
        # Remove /** and */ and leading asterisks
        lines = comment.split("\n")
        cleaned = []
        for line in lines:
            line = line.strip()
            if line.startswith("/**"):
                line = line[3:].strip()
            if line.endswith("*/"):
                line = line[:-2].strip()
            if line.startswith("*"):
                line = line[1:].strip()
            if line:
                cleaned.append(line)
        return " ".join(cleaned)

    def _extract_ts_function(self, node, source: bytes) -> FunctionInfo | None:
        """Extract TypeScript/JavaScript function."""
        name = ""
        params = []
        return_type = None
        is_async = False

        text = self._safe_decode(source[node.start_byte:node.end_byte])
        is_async = text.strip().startswith("async")

        for child in node.children:
            # Handle different identifier types
            if child.type in ("identifier", "property_identifier") and not name:
                name = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "formal_parameters":
                for p in child.children:
                    if p.type not in ("(", ")", ","):
                        params.append(self._safe_decode(source[p.start_byte:p.end_byte]))
            elif child.type == "type_annotation":
                return_type = self._safe_decode(source[child.start_byte:child.end_byte]).lstrip(": ")

        if not name:
            return None

        return FunctionInfo(
            name=name,
            params=params,
            return_type=return_type,
            docstring=None,
            is_async=is_async,
            line_number=node.start_point[0] + 1,
        )

    def _extract_ts_class(
        self,
        node,
        source: bytes,
        class_docstring: str | None = None,
        defined_names: set[str] | None = None,
        call_graph: CallGraphInfo | None = None,
    ) -> ClassInfo | None:
        """Extract TypeScript/JavaScript class."""
        name = ""
        bases = []
        methods = []

        for child in node.children:
            # TypeScript uses type_identifier, JS uses identifier
            if child.type in ("identifier", "type_identifier") and not name:
                name = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "extends_clause":
                for c in child.children:
                    if c.type == "identifier":
                        bases.append(self._safe_decode(source[c.start_byte:c.end_byte]))
            elif child.type == "implements_clause":
                for c in child.children:
                    if c.type == "identifier":
                        bases.append(f"implements {self._safe_decode(source[c.start_byte:c.end_byte])}")
            elif child.type == "class_body":
                # Extract methods with JSDoc support
                prev_comment = None
                for body_child in child.children:
                    if body_child.type == "comment":
                        comment_text = self._safe_decode(source[body_child.start_byte:body_child.end_byte])
                        if comment_text.startswith("/**"):
                            prev_comment = self._parse_jsdoc(comment_text)
                        continue

                    if body_child.type in ("method_definition", "public_field_definition"):
                        method = self._extract_ts_function(body_child, source)
                        if method:
                            method.is_method = True
                            if prev_comment:
                                method.docstring = prev_comment
                            methods.append(method)
                            # Extract calls from this method
                            if defined_names and call_graph and name:
                                caller_name = f"{name}.{method.name}"
                                self._extract_ts_calls(body_child, caller_name, source, call_graph, defined_names)
                        prev_comment = None
                    else:
                        prev_comment = None

        if not name:
            return None

        return ClassInfo(
            name=name,
            bases=bases,
            docstring=class_docstring,
            methods=methods,
            line_number=node.start_point[0] + 1,
        )

    # === Go Extraction ===

    def _extract_go(self, file_path: Path) -> ModuleInfo:
        """Extract using tree-sitter for Go."""
        with open(file_path, "rb") as f:
            source = f.read()

        parser = self._get_go_parser()
        tree = self._safe_parse(parser, source, file_path, "go")

        module_info = ModuleInfo(
            file_path=str(file_path),
            language="go",
            docstring=None,
        )

        # First pass: collect defined function/method names
        defined_names = self._collect_go_definitions(tree.root_node, source)

        self._extract_go_nodes(tree.root_node, source, module_info, defined_names)
        return module_info

    def _collect_go_definitions(self, node, source: bytes) -> set[str]:
        """Collect all defined function/method names."""
        names: set[str] = set()
        for child in node.children:
            if child.type == "function_declaration":
                for c in child.children:
                    if c.type == "identifier":
                        names.add(self._safe_decode(source[c.start_byte:c.end_byte]))
                        break
            elif child.type == "method_declaration":
                for c in child.children:
                    if c.type == "field_identifier":
                        names.add(self._safe_decode(source[c.start_byte:c.end_byte]))
                        break
            # Recurse into source_file
            if child.type in ("source_file",):
                names.update(self._collect_go_definitions(child, source))
        return names

    def _get_go_parser(self) -> Any:
        """Get or create Go tree-sitter parser."""
        if "go" not in self._ts_parsers:
            parser = Parser()
            parser.language = Language(tree_sitter_go.language())
            self._ts_parsers["go"] = parser
        return self._ts_parsers["go"]

    def _extract_go_nodes(self, node, source: bytes, module_info: ModuleInfo, defined_names: set[str] | None = None):
        """Recursively extract from Go tree-sitter nodes."""
        for child in node.children:
            node_type = child.type

            # Imports
            if node_type == "import_declaration":
                self._extract_go_imports(child, source, module_info)

            # Functions
            elif node_type == "function_declaration":
                func = self._extract_go_function(child, source)
                if func:
                    module_info.functions.append(func)
                    # Extract calls from the function body
                    if defined_names:
                        self._extract_go_calls(child, func.name, source, module_info.call_graph, defined_names)

            # Methods (functions with receiver)
            elif node_type == "method_declaration":
                method = self._extract_go_method(child, source)
                if method:
                    # For Go, we collect methods separately but mark them
                    method.is_method = True
                    module_info.functions.append(method)
                    # Extract calls from the method body
                    if defined_names:
                        self._extract_go_calls(child, method.name, source, module_info.call_graph, defined_names)

            # Type declarations (struct, interface)
            elif node_type == "type_declaration":
                cls = self._extract_go_type(child, source)
                if cls:
                    module_info.classes.append(cls)

            # Recurse into source_file and other containers
            if node_type in ("source_file",):
                self._extract_go_nodes(child, source, module_info, defined_names)

    def _extract_go_calls(self, node, caller_name: str, source: bytes, call_graph: CallGraphInfo, defined_names: set[str]):
        """Extract function calls from a Go function body."""
        for child in node.children:
            if child.type == "call_expression":
                callee = self._get_go_call_name(child, source)
                if callee and callee in defined_names:
                    call_graph.add_call(caller_name, callee)
            # Recurse into all children
            self._extract_go_calls(child, caller_name, source, call_graph, defined_names)

    def _get_go_call_name(self, node, source: bytes) -> str | None:
        """Get the name of a called function from a call_expression."""
        for child in node.children:
            if child.type == "identifier":
                return self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "selector_expression":
                # Method call like s.prepare() - get the method name (rightmost part)
                for c in child.children:
                    if c.type == "field_identifier":
                        return self._safe_decode(source[c.start_byte:c.end_byte])
        return None

    def _extract_go_imports(self, node, source: bytes, module_info: ModuleInfo):
        """Extract Go import declarations."""
        for child in node.children:
            if child.type == "import_spec":
                text = self._safe_decode(source[child.start_byte:child.end_byte]).strip('"')
                module_info.imports.append(ImportInfo(
                    module=text,
                    names=[],
                    is_from=False,
                    line_number=child.start_point[0] + 1,
                ))
            elif child.type == "import_spec_list":
                for spec in child.children:
                    if spec.type == "import_spec":
                        text = self._safe_decode(source[spec.start_byte:spec.end_byte]).strip('"')
                        # Handle alias: `alias "path"`
                        parts = text.split()
                        if len(parts) == 2:
                            module_info.imports.append(ImportInfo(
                                module=parts[1].strip('"'),
                                names=[parts[0]],
                                is_from=False,
                                line_number=spec.start_point[0] + 1,
                            ))
                        else:
                            module_info.imports.append(ImportInfo(
                                module=text.strip('"'),
                                names=[],
                                is_from=False,
                                line_number=spec.start_point[0] + 1,
                            ))

    def _extract_go_function(self, node, source: bytes) -> FunctionInfo | None:
        """Extract Go function declaration."""
        name = ""
        params = []
        return_type = None

        for child in node.children:
            if child.type == "identifier" and not name:
                name = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "parameter_list":
                params = self._extract_go_params(child, source)
            elif child.type == "result":
                return_type = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "type_identifier" and not return_type:
                return_type = self._safe_decode(source[child.start_byte:child.end_byte])

        if not name:
            return None

        return FunctionInfo(
            name=name,
            params=params,
            return_type=return_type,
            docstring=None,
            line_number=node.start_point[0] + 1,
        )

    def _extract_go_method(self, node, source: bytes) -> FunctionInfo | None:
        """Extract Go method declaration (function with receiver)."""
        name = ""
        params = []
        return_type = None
        receiver = ""

        for child in node.children:
            if child.type == "parameter_list":
                # First param list is receiver, second is params
                if not receiver:
                    recv_params = self._extract_go_params(child, source)
                    if recv_params:
                        receiver = recv_params[0]
                else:
                    params = self._extract_go_params(child, source)
            elif child.type == "field_identifier":
                name = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "result":
                return_type = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "type_identifier" and not return_type:
                return_type = self._safe_decode(source[child.start_byte:child.end_byte])

        if not name:
            return None

        # Prepend receiver to indicate this is a method
        if receiver:
            name = f"({receiver}) {name}"

        return FunctionInfo(
            name=name,
            params=params,
            return_type=return_type,
            docstring=None,
            is_method=True,
            line_number=node.start_point[0] + 1,
        )

    def _extract_go_params(self, node, source: bytes) -> list[str]:
        """Extract Go parameter list."""
        params = []
        for child in node.children:
            if child.type == "parameter_declaration":
                param = self._safe_decode(source[child.start_byte:child.end_byte])
                params.append(param)
        return params

    def _extract_go_type(self, node, source: bytes) -> ClassInfo | None:
        """Extract Go type declaration (struct or interface)."""
        name = ""
        methods = []
        bases = []

        for child in node.children:
            if child.type == "type_spec":
                for spec_child in child.children:
                    if spec_child.type == "type_identifier":
                        name = self._safe_decode(source[spec_child.start_byte:spec_child.end_byte])
                    elif spec_child.type == "struct_type":
                        # Extract embedded types as "bases"
                        for field in spec_child.children:
                            if field.type == "field_declaration_list":
                                for decl in field.children:
                                    if decl.type == "field_declaration":
                                        # Check for embedded type (no field name, just type)
                                        field_text = self._safe_decode(source[decl.start_byte:decl.end_byte]).strip()
                                        if " " not in field_text and not field_text.startswith("//"):
                                            bases.append(field_text)
                    elif spec_child.type == "interface_type":
                        # Extract interface methods
                        for iface_child in spec_child.children:
                            if iface_child.type == "method_elem":
                                method_name = ""
                                method_params = []
                                method_return = None
                                for m_child in iface_child.children:
                                    if m_child.type == "field_identifier":
                                        method_name = self._safe_decode(source[m_child.start_byte:m_child.end_byte])
                                    elif m_child.type == "parameter_list":
                                        method_params = self._extract_go_params(m_child, source)
                                    elif m_child.type == "type_identifier":
                                        method_return = self._safe_decode(source[m_child.start_byte:m_child.end_byte])
                                if method_name:
                                    methods.append(FunctionInfo(
                                        name=method_name,
                                        params=method_params,
                                        return_type=method_return,
                                        docstring=None,
                                        is_method=True,
                                        line_number=iface_child.start_point[0] + 1,
                                    ))

        if not name:
            return None

        return ClassInfo(
            name=name,
            bases=bases,
            docstring=None,
            methods=methods,
            line_number=node.start_point[0] + 1,
        )

    # === Rust Extraction ===

    def _extract_rust(self, file_path: Path) -> ModuleInfo:
        """Extract using tree-sitter for Rust."""
        with open(file_path, "rb") as f:
            source = f.read()

        parser = self._get_rust_parser()
        tree = self._safe_parse(parser, source, file_path, "rust")

        module_info = ModuleInfo(
            file_path=str(file_path),
            language="rust",
            docstring=None,
        )

        # Collect all defined function/method names for call graph filtering
        defined_names = self._collect_rust_definitions(tree.root_node, source)

        self._extract_rust_nodes(tree.root_node, source, module_info, defined_names)
        return module_info

    def _collect_rust_definitions(self, node, source: bytes) -> set[str]:
        """Collect all defined function/method names in Rust code."""
        names: set[str] = set()
        for child in node.children:
            if child.type == "function_item":
                # Top-level function
                for c in child.children:
                    if c.type == "identifier":
                        names.add(self._safe_decode(source[c.start_byte:c.end_byte]))
                        break
            elif child.type == "impl_item":
                # Impl block methods
                for c in child.children:
                    if c.type == "declaration_list":
                        for item in c.children:
                            if item.type == "function_item":
                                for fc in item.children:
                                    if fc.type == "identifier":
                                        names.add(self._safe_decode(source[fc.start_byte:fc.end_byte]))
                                        break
            # Recurse into modules
            if child.type in ("source_file", "mod_item", "declaration_list"):
                names.update(self._collect_rust_definitions(child, source))
        return names

    def _get_rust_parser(self) -> Any:
        """Get or create Rust tree-sitter parser."""
        if "rust" not in self._ts_parsers:
            parser = Parser()
            parser.language = Language(tree_sitter_rust.language())
            self._ts_parsers["rust"] = parser
        return self._ts_parsers["rust"]

    def _extract_rust_nodes(self, node, source: bytes, module_info: ModuleInfo, defined_names: set[str] | None = None):
        """Recursively extract from Rust tree-sitter nodes."""
        if defined_names is None:
            defined_names = set()

        for child in node.children:
            node_type = child.type

            # Use declarations
            if node_type == "use_declaration":
                text = self._safe_decode(source[child.start_byte:child.end_byte])
                # Strip "use " prefix and trailing semicolon for clean display
                module = text.replace("use ", "").rstrip(";").strip()
                module_info.imports.append(ImportInfo(
                    module=module,
                    names=[],
                    is_from=False,
                    line_number=child.start_point[0] + 1,
                ))

            # Functions
            elif node_type == "function_item":
                func = self._extract_rust_function(child, source)
                if func:
                    module_info.functions.append(func)
                    # Extract call graph from function body
                    self._extract_rust_calls(child, func.name, source, module_info.call_graph, defined_names)

            # Struct definitions
            elif node_type == "struct_item":
                cls = self._extract_rust_struct(child, source)
                if cls:
                    module_info.classes.append(cls)

            # Trait definitions
            elif node_type == "trait_item":
                cls = self._extract_rust_trait(child, source)
                if cls:
                    module_info.classes.append(cls)

            # Impl blocks
            elif node_type == "impl_item":
                self._extract_rust_impl(child, source, module_info, defined_names)

            # Recurse into module items
            if node_type in ("source_file", "mod_item", "declaration_list"):
                self._extract_rust_nodes(child, source, module_info, defined_names)

    def _extract_rust_function(self, node, source: bytes) -> FunctionInfo | None:
        """Extract Rust function item."""
        name = ""
        params = []
        return_type = None
        is_async = False

        text = self._safe_decode(source[node.start_byte:node.end_byte])
        is_async = "async fn" in text or "async " in text.split("fn")[0] if "fn" in text else False

        for child in node.children:
            if child.type == "identifier":
                name = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "parameters":
                params = self._extract_rust_params(child, source)
            elif child.type == "type_identifier" or child.type.endswith("_type"):
                # Return type comes after ->
                return_type = self._safe_decode(source[child.start_byte:child.end_byte])

        # Look for return type after ->
        if "->" in text and not return_type:
            ret_part = text.split("->")[1].split("{")[0].strip()
            return_type = ret_part

        if not name:
            return None

        return FunctionInfo(
            name=name,
            params=params,
            return_type=return_type,
            docstring=None,
            is_async=is_async,
            line_number=node.start_point[0] + 1,
        )

    def _extract_rust_params(self, node, source: bytes) -> list[str]:
        """Extract Rust parameter list."""
        params = []
        for child in node.children:
            if child.type == "parameter":
                param = self._safe_decode(source[child.start_byte:child.end_byte])
                params.append(param)
            elif child.type == "self_parameter":
                param = self._safe_decode(source[child.start_byte:child.end_byte])
                params.append(param)
        return params

    def _extract_rust_struct(self, node, source: bytes) -> ClassInfo | None:
        """Extract Rust struct definition."""
        name = ""

        for child in node.children:
            if child.type == "type_identifier":
                name = self._safe_decode(source[child.start_byte:child.end_byte])
                break

        if not name:
            return None

        return ClassInfo(
            name=name,
            bases=[],
            docstring=None,
            methods=[],
            line_number=node.start_point[0] + 1,
        )

    def _extract_rust_trait(self, node, source: bytes) -> ClassInfo | None:
        """Extract Rust trait definition."""
        name = ""
        methods = []

        for child in node.children:
            if child.type == "type_identifier":
                name = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "declaration_list":
                for item in child.children:
                    if item.type == "function_signature_item":
                        sig = self._safe_decode(source[item.start_byte:item.end_byte])
                        fn_name = sig.split("(")[0].replace("fn ", "").strip() if "(" in sig else sig
                        methods.append(FunctionInfo(
                            name=fn_name,
                            params=[],
                            return_type=None,
                            docstring=None,
                            is_method=True,
                            line_number=item.start_point[0] + 1,
                        ))

        if not name:
            return None

        return ClassInfo(
            name=f"trait {name}",
            bases=[],
            docstring=None,
            methods=methods,
            line_number=node.start_point[0] + 1,
        )

    def _extract_rust_impl(self, node, source: bytes, module_info: ModuleInfo, defined_names: set[str] | None = None):
        """Extract Rust impl block methods and associate with struct/trait."""
        if defined_names is None:
            defined_names = set()
        impl_type = ""
        trait_name = ""

        for child in node.children:
            if child.type == "type_identifier":
                impl_type = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "generic_type":
                impl_type = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "declaration_list":
                for item in child.children:
                    if item.type == "function_item":
                        func = self._extract_rust_function(item, source)
                        if func:
                            func.is_method = True
                            # Tag with impl type
                            caller_name = f"({impl_type}) {func.name}"
                            func.name = caller_name
                            module_info.functions.append(func)
                            # Extract call graph from method body
                            self._extract_rust_calls(item, caller_name, source, module_info.call_graph, defined_names)

    def _extract_rust_calls(self, node, caller_name: str, source: bytes, call_graph: CallGraphInfo, defined_names: set[str]):
        """Extract function calls from a Rust function body."""
        for child in node.children:
            # Skip macro invocations (like println!)
            if child.type == "macro_invocation":
                continue
            if child.type == "call_expression":
                callee = self._get_rust_call_name(child, source)
                if callee and callee in defined_names:
                    call_graph.add_call(caller_name, callee)
            # Recurse into all children
            self._extract_rust_calls(child, caller_name, source, call_graph, defined_names)

    def _get_rust_call_name(self, node, source: bytes) -> str | None:
        """Get the name of a called function from a call_expression."""
        for child in node.children:
            if child.type == "identifier":
                return self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "field_expression":
                # Method call like self.foo() - get the field name (rightmost part)
                for c in child.children:
                    if c.type == "field_identifier":
                        return self._safe_decode(source[c.start_byte:c.end_byte])
            elif child.type == "scoped_identifier":
                # Qualified call like Foo::bar() - get the last identifier
                for c in child.children:
                    if c.type == "identifier":
                        return self._safe_decode(source[c.start_byte:c.end_byte])
        return None

    # === Java Extraction ===

    def _extract_java(self, file_path: Path) -> ModuleInfo:
        """Extract using tree-sitter for Java."""
        with open(file_path, "rb") as f:
            source = f.read()

        parser = self._get_java_parser()
        tree = self._safe_parse(parser, source, file_path, "java")

        module_info = ModuleInfo(
            file_path=str(file_path),
            language="java",
            docstring=None,
        )

        # Collect all defined method names for call graph filtering
        defined_names = self._collect_java_definitions(tree.root_node, source)

        self._extract_java_nodes(tree.root_node, source, module_info, defined_names)
        return module_info

    def _collect_java_definitions(self, node, source: bytes) -> set[str]:
        """Collect all defined method names in Java code."""
        names: set[str] = set()
        for child in node.children:
            if child.type == "method_declaration":
                # Find method name
                for c in child.children:
                    if c.type == "identifier":
                        names.add(self._safe_decode(source[c.start_byte:c.end_byte]))
                        break
            elif child.type == "class_declaration":
                # Look inside class body
                for c in child.children:
                    if c.type == "class_body":
                        names.update(self._collect_java_definitions(c, source))
            # Recurse into program and class_body
            if child.type in ("program", "class_body"):
                names.update(self._collect_java_definitions(child, source))
        return names

    def _get_java_parser(self) -> Any:
        """Get or create Java tree-sitter parser."""
        if "java" not in self._ts_parsers:
            parser = Parser()
            parser.language = Language(tree_sitter_java.language())
            self._ts_parsers["java"] = parser
        return self._ts_parsers["java"]

    def _extract_java_nodes(self, node, source: bytes, module_info: ModuleInfo, defined_names: set[str] | None = None):
        """Recursively extract from Java tree-sitter nodes."""
        if defined_names is None:
            defined_names = set()

        for child in node.children:
            node_type = child.type

            # Import declarations
            if node_type == "import_declaration":
                self._extract_java_import(child, source, module_info)

            # Class declarations
            elif node_type == "class_declaration":
                cls = self._extract_java_class(child, source, module_info, defined_names)
                if cls:
                    module_info.classes.append(cls)

            # Interface declarations (treat like class)
            elif node_type == "interface_declaration":
                cls = self._extract_java_interface(child, source)
                if cls:
                    module_info.classes.append(cls)

            # Recurse into program node
            if node_type == "program":
                self._extract_java_nodes(child, source, module_info, defined_names)

    def _extract_java_import(self, node, source: bytes, module_info: ModuleInfo):
        """Extract Java import declaration."""
        # Get the full import text and parse it
        import_text = self._safe_decode(source[node.start_byte:node.end_byte])
        # Strip "import " prefix and trailing semicolon
        module = import_text.replace("import ", "").replace("static ", "").rstrip(";").strip()
        module_info.imports.append(ImportInfo(
            module=module,
            names=[],
            is_from=False,
            line_number=node.start_point[0] + 1,
        ))

    def _extract_java_class(self, node, source: bytes, module_info: ModuleInfo, defined_names: set[str]) -> ClassInfo | None:
        """Extract Java class declaration."""
        name = ""
        methods = []
        bases = []

        for child in node.children:
            if child.type == "identifier":
                name = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "superclass":
                # Extract extends clause
                for c in child.children:
                    if c.type == "type_identifier":
                        bases.append(self._safe_decode(source[c.start_byte:c.end_byte]))
            elif child.type == "super_interfaces":
                # Extract implements clause
                for c in child.children:
                    if c.type == "type_list":
                        for t in c.children:
                            if t.type == "type_identifier":
                                bases.append(self._safe_decode(source[t.start_byte:t.end_byte]))
            elif child.type == "class_body":
                # Extract methods from class body
                for body_child in child.children:
                    if body_child.type == "method_declaration":
                        method = self._extract_java_method(body_child, source)
                        if method:
                            methods.append(method)
                            module_info.functions.append(method)
                            # Extract call graph from method body
                            self._extract_java_calls(body_child, method.name, source, module_info.call_graph, defined_names)
                    elif body_child.type == "constructor_declaration":
                        method = self._extract_java_constructor(body_child, source, name)
                        if method:
                            methods.append(method)
                            module_info.functions.append(method)

        if not name:
            return None

        return ClassInfo(
            name=name,
            bases=bases,
            docstring=None,
            methods=methods,
            line_number=node.start_point[0] + 1,
        )

    def _extract_java_interface(self, node, source: bytes) -> ClassInfo | None:
        """Extract Java interface declaration."""
        name = ""
        methods = []
        bases = []

        for child in node.children:
            if child.type == "identifier":
                name = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "extends_interfaces":
                for c in child.children:
                    if c.type == "type_list":
                        for t in c.children:
                            if t.type == "type_identifier":
                                bases.append(self._safe_decode(source[t.start_byte:t.end_byte]))
            elif child.type == "interface_body":
                for body_child in child.children:
                    if body_child.type == "method_declaration":
                        method = self._extract_java_method(body_child, source)
                        if method:
                            methods.append(method)

        if not name:
            return None

        return ClassInfo(
            name=name,
            bases=bases,
            docstring=None,
            methods=methods,
            line_number=node.start_point[0] + 1,
        )

    def _extract_java_method(self, node, source: bytes) -> FunctionInfo | None:
        """Extract Java method declaration."""
        name = ""
        params = []
        return_type = None

        for child in node.children:
            if child.type == "identifier":
                name = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "formal_parameters":
                params = self._extract_java_params(child, source)
            elif child.type in ("type_identifier", "void_type", "integral_type", "floating_point_type", "boolean_type", "generic_type", "array_type"):
                return_type = self._safe_decode(source[child.start_byte:child.end_byte])

        if not name:
            return None

        return FunctionInfo(
            name=name,
            params=params,
            return_type=return_type,
            docstring=None,
            is_method=True,
            line_number=node.start_point[0] + 1,
        )

    def _extract_java_constructor(self, node, source: bytes, class_name: str) -> FunctionInfo | None:
        """Extract Java constructor declaration."""
        name = class_name  # Constructor has same name as class
        params = []

        for child in node.children:
            if child.type == "identifier":
                name = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "formal_parameters":
                params = self._extract_java_params(child, source)

        return FunctionInfo(
            name=name,
            params=params,
            return_type=None,  # Constructors have no return type
            docstring=None,
            is_method=True,
            line_number=node.start_point[0] + 1,
        )

    def _extract_java_params(self, node, source: bytes) -> list[str]:
        """Extract Java formal parameters."""
        params = []
        for child in node.children:
            if child.type == "formal_parameter":
                param = self._safe_decode(source[child.start_byte:child.end_byte])
                params.append(param)
            elif child.type == "spread_parameter":
                param = self._safe_decode(source[child.start_byte:child.end_byte])
                params.append(param)
        return params

    def _extract_java_calls(self, node, caller_name: str, source: bytes, call_graph: CallGraphInfo, defined_names: set[str]):
        """Extract method calls from a Java method body."""
        for child in node.children:
            if child.type == "method_invocation":
                callee = self._get_java_call_name(child, source)
                if callee and callee in defined_names:
                    call_graph.add_call(caller_name, callee)
            # Recurse into all children
            self._extract_java_calls(child, caller_name, source, call_graph, defined_names)

    def _get_java_call_name(self, node, source: bytes) -> str | None:
        """Get the name of a called method from a method_invocation node."""
        for child in node.children:
            if child.type == "identifier":
                return self._safe_decode(source[child.start_byte:child.end_byte])
        return None

    # === C Extraction ===

    def _extract_c(self, file_path: Path) -> ModuleInfo:
        """Extract using tree-sitter for C."""
        with open(file_path, "rb") as f:
            source = f.read()

        parser = self._get_c_parser()
        tree = self._safe_parse(parser, source, file_path, "c")

        module_info = ModuleInfo(
            file_path=str(file_path),
            language="c",
            docstring=None,
        )

        # Collect all defined function names for call graph filtering
        defined_names = self._collect_c_definitions(tree.root_node, source)

        self._extract_c_nodes(tree.root_node, source, module_info, defined_names)
        return module_info

    def _collect_c_definitions(self, node, source: bytes) -> set[str]:
        """Collect all defined function names in C code."""
        names: set[str] = set()
        for child in node.children:
            if child.type == "function_definition":
                # Find function name in declarator
                for c in child.children:
                    if c.type == "function_declarator":
                        for dc in c.children:
                            if dc.type == "identifier":
                                names.add(self._safe_decode(source[dc.start_byte:dc.end_byte]))
                                break
            # Recurse into translation_unit
            if child.type == "translation_unit":
                names.update(self._collect_c_definitions(child, source))
        return names

    def _get_c_parser(self) -> Any:
        """Get or create C tree-sitter parser."""
        if "c" not in self._ts_parsers:
            parser = Parser()
            parser.language = Language(tree_sitter_c.language())
            self._ts_parsers["c"] = parser
        return self._ts_parsers["c"]

    def _extract_c_nodes(self, node, source: bytes, module_info: ModuleInfo, defined_names: set[str] | None = None):
        """Recursively extract from C tree-sitter nodes."""
        if defined_names is None:
            defined_names = set()

        for child in node.children:
            node_type = child.type

            # Include directives (imports)
            if node_type == "preproc_include":
                self._extract_c_include(child, source, module_info)

            # Function definitions
            elif node_type == "function_definition":
                func = self._extract_c_function(child, source)
                if func:
                    module_info.functions.append(func)
                    # Extract call graph from function body
                    self._extract_c_calls(child, func.name, source, module_info.call_graph, defined_names)

            # Recurse into container nodes (FIXED: was only translation_unit)
            if node_type in ("translation_unit", "struct_specifier", "union_specifier"):
                self._extract_c_nodes(child, source, module_info, defined_names)

    def _extract_c_include(self, node, source: bytes, module_info: ModuleInfo):
        """Extract C #include directive."""
        # Get the included file path
        for child in node.children:
            if child.type == "string_literal":
                # Local include "file.h"
                module = self._safe_decode(source[child.start_byte:child.end_byte]).strip('"')
                module_info.imports.append(ImportInfo(
                    module=module,
                    names=[],
                    is_from=False,
                    line_number=node.start_point[0] + 1,
                ))
                return
            elif child.type == "system_lib_string":
                # System include <file.h>
                module = self._safe_decode(source[child.start_byte:child.end_byte]).strip('<>')
                module_info.imports.append(ImportInfo(
                    module=module,
                    names=[],
                    is_from=False,
                    line_number=node.start_point[0] + 1,
                ))
                return

    def _extract_c_function(self, node, source: bytes) -> FunctionInfo | None:
        """Extract C function definition."""
        name = ""
        params = []
        return_type = None

        for child in node.children:
            # Return type comes before the declarator
            if child.type in ("primitive_type", "type_identifier", "sized_type_specifier"):
                return_type = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "pointer_declarator":
                # Pointer return type like int* or char*
                for pc in child.children:
                    if pc.type == "function_declarator":
                        for dc in pc.children:
                            if dc.type == "identifier":
                                name = self._safe_decode(source[dc.start_byte:dc.end_byte])
                            elif dc.type == "parameter_list":
                                params = self._extract_c_params(dc, source)
                if return_type:
                    return_type = return_type + "*"
            elif child.type == "function_declarator":
                for dc in child.children:
                    if dc.type == "identifier":
                        name = self._safe_decode(source[dc.start_byte:dc.end_byte])
                    elif dc.type == "parameter_list":
                        params = self._extract_c_params(dc, source)

        if not name:
            return None

        return FunctionInfo(
            name=name,
            params=params,
            return_type=return_type,
            docstring=None,
            is_method=False,
            line_number=node.start_point[0] + 1,
        )

    def _extract_c_params(self, node, source: bytes) -> list[str]:
        """Extract C function parameters."""
        params = []
        for child in node.children:
            if child.type == "parameter_declaration":
                param = self._safe_decode(source[child.start_byte:child.end_byte])
                params.append(param)
        return params

    def _extract_c_calls(self, node, caller_name: str, source: bytes, call_graph: CallGraphInfo, defined_names: set[str]):
        """Extract function calls from a C function body."""
        for child in node.children:
            if child.type == "call_expression":
                callee = self._get_c_call_name(child, source)
                if callee and callee in defined_names:
                    call_graph.add_call(caller_name, callee)
            # Recurse into all children
            self._extract_c_calls(child, caller_name, source, call_graph, defined_names)

    def _get_c_call_name(self, node, source: bytes) -> str | None:
        """Get the name of a called function from a call_expression node."""
        for child in node.children:
            if child.type == "identifier":
                return self._safe_decode(source[child.start_byte:child.end_byte])
        return None

    # === C++ Extraction ===

    def _extract_cpp(self, file_path: Path) -> ModuleInfo:
        """Extract using tree-sitter for C++."""
        with open(file_path, "rb") as f:
            source = f.read()

        parser = self._get_cpp_parser()
        tree = self._safe_parse(parser, source, file_path, "cpp")

        module_info = ModuleInfo(
            file_path=str(file_path),
            language="cpp",
            docstring=None,
        )

        # Collect all defined function names for call graph filtering
        defined_names = self._collect_cpp_definitions(tree.root_node, source)

        self._extract_cpp_nodes(tree.root_node, source, module_info, defined_names)
        return module_info

    def _collect_cpp_definitions(self, node, source: bytes) -> set[str]:
        """Collect all defined function names in C++ code."""
        names: set[str] = set()
        for child in node.children:
            if child.type == "function_definition":
                # Find function name in declarator
                for c in child.children:
                    if c.type == "function_declarator":
                        for dc in c.children:
                            if dc.type == "identifier":
                                names.add(self._safe_decode(source[dc.start_byte:dc.end_byte]))
                                break
            # Recurse into translation_unit
            if child.type == "translation_unit":
                names.update(self._collect_cpp_definitions(child, source))
        return names

    def _get_cpp_parser(self) -> Any:
        """Get or create C++ tree-sitter parser."""
        if "cpp" not in self._ts_parsers:
            parser = Parser()
            parser.language = Language(tree_sitter_cpp.language())
            self._ts_parsers["cpp"] = parser
        return self._ts_parsers["cpp"]

    def _extract_cpp_nodes(self, node, source: bytes, module_info: ModuleInfo, defined_names: set[str] | None = None):
        """Recursively extract from C++ tree-sitter nodes."""
        if defined_names is None:
            defined_names = set()

        for child in node.children:
            node_type = child.type

            # Include directives (imports)
            if node_type == "preproc_include":
                self._extract_cpp_include(child, source, module_info)

            # Function definitions
            elif node_type == "function_definition":
                func = self._extract_cpp_function(child, source)
                if func:
                    module_info.functions.append(func)
                    # Extract call graph from function body
                    self._extract_cpp_calls(child, func.name, source, module_info.call_graph, defined_names)

            # Recurse into container nodes (FIXED: was only translation_unit, missed namespaces/classes)
            if node_type in ("translation_unit", "namespace_definition",
                             "class_specifier", "struct_specifier",
                             "declaration", "declaration_list"):
                self._extract_cpp_nodes(child, source, module_info, defined_names)

    def _extract_cpp_include(self, node, source: bytes, module_info: ModuleInfo):
        """Extract C++ #include directive."""
        # Get the included file path
        for child in node.children:
            if child.type == "string_literal":
                # Local include "file.hpp"
                module = self._safe_decode(source[child.start_byte:child.end_byte]).strip('"')
                module_info.imports.append(ImportInfo(
                    module=module,
                    names=[],
                    is_from=False,
                    line_number=node.start_point[0] + 1,
                ))
                return
            elif child.type == "system_lib_string":
                # System include <file.h>
                module = self._safe_decode(source[child.start_byte:child.end_byte]).strip('<>')
                module_info.imports.append(ImportInfo(
                    module=module,
                    names=[],
                    is_from=False,
                    line_number=node.start_point[0] + 1,
                ))
                return

    def _extract_cpp_function(self, node, source: bytes) -> FunctionInfo | None:
        """Extract C++ function definition."""
        name = ""
        params = []
        return_type = None

        for child in node.children:
            # Return type comes before the declarator
            if child.type in ("primitive_type", "type_identifier", "sized_type_specifier"):
                return_type = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "pointer_declarator":
                # Pointer return type like int* or char*
                for pc in child.children:
                    if pc.type == "function_declarator":
                        for dc in pc.children:
                            if dc.type == "identifier":
                                name = self._safe_decode(source[dc.start_byte:dc.end_byte])
                            elif dc.type == "parameter_list":
                                params = self._extract_cpp_params(dc, source)
                if return_type:
                    return_type = return_type + "*"
            elif child.type == "function_declarator":
                for dc in child.children:
                    if dc.type == "identifier":
                        name = self._safe_decode(source[dc.start_byte:dc.end_byte])
                    elif dc.type == "parameter_list":
                        params = self._extract_cpp_params(dc, source)

        if not name:
            return None

        return FunctionInfo(
            name=name,
            params=params,
            return_type=return_type,
            docstring=None,
            is_method=False,
            line_number=node.start_point[0] + 1,
        )

    def _extract_cpp_params(self, node, source: bytes) -> list[str]:
        """Extract C++ function parameters."""
        params = []
        for child in node.children:
            if child.type == "parameter_declaration":
                param = self._safe_decode(source[child.start_byte:child.end_byte])
                params.append(param)
        return params

    def _extract_cpp_calls(self, node, caller_name: str, source: bytes, call_graph: CallGraphInfo, defined_names: set[str]):
        """Extract function calls from a C++ function body."""
        for child in node.children:
            if child.type == "call_expression":
                callee = self._get_cpp_call_name(child, source)
                if callee and callee in defined_names:
                    call_graph.add_call(caller_name, callee)
            # Recurse into all children
            self._extract_cpp_calls(child, caller_name, source, call_graph, defined_names)

    def _get_cpp_call_name(self, node, source: bytes) -> str | None:
        """Get the name of a called function from a call_expression node."""
        for child in node.children:
            if child.type == "identifier":
                return self._safe_decode(source[child.start_byte:child.end_byte])
        return None

    # === Ruby Extraction ===

    def _extract_ruby(self, file_path: Path) -> ModuleInfo:
        """Extract using tree-sitter for Ruby."""
        with open(file_path, "rb") as f:
            source = f.read()

        parser = self._get_ruby_parser()
        tree = self._safe_parse(parser, source, file_path, "ruby")

        module_info = ModuleInfo(
            file_path=str(file_path),
            language="ruby",
            docstring=None,
        )

        # Collect all defined method names for call graph filtering
        defined_names = self._collect_ruby_definitions(tree.root_node, source)

        self._extract_ruby_nodes(tree.root_node, source, module_info, defined_names)
        return module_info

    def _collect_ruby_definitions(self, node, source: bytes) -> set[str]:
        """Collect all defined method names in Ruby code."""
        names: set[str] = set()
        for child in node.children:
            if child.type == "method":
                name_node = child.child_by_field_name("name")
                if name_node:
                    names.add(self._safe_decode(source[name_node.start_byte:name_node.end_byte]))
            # Recurse into class bodies, module bodies etc
            names.update(self._collect_ruby_definitions(child, source))
        return names

    def _get_ruby_parser(self) -> Any:
        """Get or create Ruby tree-sitter parser."""
        if "ruby" not in self._ts_parsers:
            parser = Parser()
            parser.language = Language(tree_sitter_ruby.language())
            self._ts_parsers["ruby"] = parser
        return self._ts_parsers["ruby"]

    def _extract_ruby_nodes(self, node, source: bytes, module_info: ModuleInfo, defined_names: set[str] | None = None):
        """Recursively extract from Ruby tree-sitter nodes."""
        if defined_names is None:
            defined_names = set()

        for child in node.children:
            node_type = child.type

            # Method definitions
            if node_type == "method":
                func = self._extract_ruby_method(child, source)
                if func:
                    module_info.functions.append(func)
                    # Extract call graph from method body
                    self._extract_ruby_calls(child, func.name, source, module_info.call_graph, defined_names)

            # Class definitions
            elif node_type == "class":
                cls = self._extract_ruby_class(child, source, defined_names, module_info.call_graph)
                if cls:
                    module_info.classes.append(cls)

            # Module definitions (treat like class for structure)
            elif node_type == "module":
                # Recurse into module body
                body_node = child.child_by_field_name("body")
                if body_node:
                    self._extract_ruby_nodes(body_node, source, module_info, defined_names)

            # Require statements (imports)
            elif node_type == "call":
                import_info = self._extract_ruby_require(child, source)
                if import_info:
                    module_info.imports.append(import_info)

            # Recurse into other nodes
            else:
                self._extract_ruby_nodes(child, source, module_info, defined_names)

    def _extract_ruby_method(self, node, source: bytes) -> FunctionInfo | None:
        """Extract a Ruby method definition."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        name = self._safe_decode(source[name_node.start_byte:name_node.end_byte])

        # Extract parameters
        params = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            params = self._extract_ruby_params(params_node, source)

        return FunctionInfo(
            name=name,
            params=params,
            return_type=None,  # Ruby doesn't have static return types
            docstring=None,
            is_method=True,
            line_number=node.start_point[0] + 1,
        )

    def _extract_ruby_params(self, node, source: bytes) -> list[str]:
        """Extract Ruby method parameters."""
        params = []
        for child in node.children:
            if child.type in ("identifier", "optional_parameter", "splat_parameter", "keyword_parameter", "block_parameter"):
                param_text = self._safe_decode(source[child.start_byte:child.end_byte])
                params.append(param_text)
        return params

    def _extract_ruby_class(self, node, source: bytes, defined_names: set[str], call_graph: CallGraphInfo) -> ClassInfo | None:
        """Extract a Ruby class definition."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        name = self._safe_decode(source[name_node.start_byte:name_node.end_byte])

        # Extract superclass if present
        superclass = None
        superclass_node = node.child_by_field_name("superclass")
        if superclass_node:
            superclass = self._safe_decode(source[superclass_node.start_byte:superclass_node.end_byte])

        methods = []
        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                if child.type == "method":
                    method = self._extract_ruby_method(child, source)
                    if method:
                        methods.append(method)
                        self._extract_ruby_calls(child, f"{name}.{method.name}", source, call_graph, defined_names)

        return ClassInfo(
            name=name,
            methods=methods,
            bases=[superclass] if superclass else [],
            docstring=None,
            line_number=node.start_point[0] + 1,
        )

    def _extract_ruby_require(self, node, source: bytes) -> ImportInfo | None:
        """Extract a Ruby require/require_relative statement."""
        method_node = node.child_by_field_name("method")
        if not method_node:
            return None

        method_name = self._safe_decode(source[method_node.start_byte:method_node.end_byte])
        if method_name not in ("require", "require_relative"):
            return None

        args_node = node.child_by_field_name("arguments")
        if not args_node:
            return None

        # Find the string argument
        for child in args_node.children:
            if child.type == "string":
                string_content = child.child_by_field_name("content")
                if string_content:
                    module = self._safe_decode(source[string_content.start_byte:string_content.end_byte])
                else:
                    text = self._safe_decode(source[child.start_byte:child.end_byte])
                    module = text.strip("'\"")

                return ImportInfo(
                    module=module,
                    names=[],
                    is_from=method_name == "require_relative",
                )

        return None

    def _extract_ruby_calls(self, node, caller_name: str, source: bytes, call_graph: CallGraphInfo, defined_names: set[str]):
        """Extract method calls from a Ruby method body."""
        for child in node.children:
            if child.type == "call":
                callee = self._get_ruby_call_name(child, source)
                if callee and callee in defined_names:
                    call_graph.add_call(caller_name, callee)
            # Recurse into all children
            self._extract_ruby_calls(child, caller_name, source, call_graph, defined_names)

    def _get_ruby_call_name(self, node, source: bytes) -> str | None:
        """Get the name of a called method from a call node."""
        method_node = node.child_by_field_name("method")
        if method_node:
            return self._safe_decode(source[method_node.start_byte:method_node.end_byte])
        return None

    def _extract_kotlin(self, file_path: Path) -> ModuleInfo:
        """Extract using tree-sitter-kotlin."""
        with open(file_path, "rb") as f:
            source = f.read()

        parser = self._get_kotlin_parser()
        tree = self._safe_parse(parser, source, file_path, "kotlin")

        module_info = ModuleInfo(
            file_path=str(file_path),
            language="kotlin",
            docstring=None,
        )

        # First pass: collect defined function names
        defined_names = self._collect_kotlin_definitions(tree.root_node, source)

        self._extract_kotlin_nodes(tree.root_node, source, module_info, defined_names)
        return module_info

    def _get_kotlin_parser(self) -> Any:
        """Get or create tree-sitter Kotlin parser."""
        if "kotlin" not in self._ts_parsers:
            kotlin_lang = Language(tree_sitter_kotlin.language())
            parser = Parser(kotlin_lang)
            self._ts_parsers["kotlin"] = parser
        return self._ts_parsers["kotlin"]

    def _collect_kotlin_definitions(self, node, source: bytes) -> set[str]:
        """Collect all defined function/method names in Kotlin."""
        names: set[str] = set()
        for child in node.children:
            if child.type == "function_declaration":
                for c in child.children:
                    if c.type == "identifier":
                        names.add(self._safe_decode(source[c.start_byte:c.end_byte]))
                        break
            elif child.type == "class_declaration":
                for c in child.children:
                    if c.type == "class_body":
                        for member in c.children:
                            if member.type == "function_declaration":
                                for m in member.children:
                                    if m.type == "identifier":
                                        names.add(self._safe_decode(source[m.start_byte:m.end_byte]))
                                        break
            # Recurse
            if child.type in ("source_file", "package_header", "import_list"):
                names.update(self._collect_kotlin_definitions(child, source))
        return names

    def _extract_kotlin_nodes(self, node, source: bytes, module_info: ModuleInfo, defined_names: set[str] | None = None):
        """Extract Kotlin nodes into module info."""
        call_graph = module_info.call_graph or CallGraphInfo()
        module_info.call_graph = call_graph

        for child in node.children:
            if child.type == "function_declaration":
                func_info = self._extract_kotlin_function(child, source)
                if func_info:
                    module_info.functions.append(func_info)
                    # Extract calls from function body
                    if defined_names:
                        self._extract_kotlin_calls(child, func_info.name, source, call_graph, defined_names)

            elif child.type == "class_declaration":
                class_info = self._extract_kotlin_class(child, source, defined_names or set(), call_graph)
                if class_info:
                    module_info.classes.append(class_info)

            elif child.type == "import":
                import_info = self._extract_kotlin_import(child, source)
                if import_info:
                    module_info.imports.append(import_info)

            # Recurse for nested structures (FIXED: was only source_file, missed objects/companions)
            if child.type in ("source_file", "class_declaration", "class_body",
                              "object_declaration", "companion_object"):
                self._extract_kotlin_nodes(child, source, module_info, defined_names)

    def _extract_kotlin_function(self, node, source: bytes) -> FunctionInfo | None:
        """Extract function info from Kotlin function_declaration."""
        name = None
        params = []
        return_type = None

        for child in node.children:
            if child.type == "identifier":
                name = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "function_value_parameters":
                params = self._extract_kotlin_params(child, source)
            elif child.type == "type_identifier" or child.type == "user_type":
                return_type = self._safe_decode(source[child.start_byte:child.end_byte])

        if not name:
            return None

        return FunctionInfo(
            name=name,
            params=params,
            return_type=return_type,
            docstring=None,
            line_number=node.start_point[0] + 1,
        )

    def _extract_kotlin_params(self, node, source: bytes) -> list[str]:
        """Extract parameter list from Kotlin function_value_parameters."""
        params = []
        for child in node.children:
            if child.type == "parameter":
                for c in child.children:
                    if c.type == "identifier":
                        params.append(self._safe_decode(source[c.start_byte:c.end_byte]))
                        break
        return params

    def _extract_kotlin_class(self, node, source: bytes, defined_names: set[str], call_graph: CallGraphInfo) -> ClassInfo | None:
        """Extract class info from Kotlin class_declaration."""
        name = None
        methods = []

        for child in node.children:
            if child.type == "identifier":
                name = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "class_body":
                for member in child.children:
                    if member.type == "function_declaration":
                        method = self._extract_kotlin_function(member, source)
                        if method:
                            methods.append(method)
                            # Extract calls from method body
                            self._extract_kotlin_calls(member, method.name, source, call_graph, defined_names)

        if not name:
            return None

        return ClassInfo(
            name=name,
            methods=methods,
            bases=[],
            docstring=None,
            line_number=node.start_point[0] + 1,
        )

    def _extract_kotlin_import(self, node, source: bytes) -> ImportInfo | None:
        """Extract import info from Kotlin import_header."""
        text = self._safe_decode(source[node.start_byte:node.end_byte]).strip()
        if text.startswith("import "):
            module = text[7:].strip()
            # Handle alias: import foo.bar as baz
            alias = None
            if " as " in module:
                parts = module.split(" as ")
                module = parts[0].strip()
                alias = parts[1].strip()
            return ImportInfo(
                module=module,
                names=[],
            )
        return None

    def _extract_kotlin_calls(self, node, caller_name: str, source: bytes, call_graph: CallGraphInfo, defined_names: set[str]):
        """Extract function calls from a Kotlin function body."""
        for child in node.children:
            if child.type == "call_expression":
                callee = self._get_kotlin_call_name(child, source)
                if callee and callee in defined_names:
                    call_graph.add_call(caller_name, callee)
            # Recurse into all children
            self._extract_kotlin_calls(child, caller_name, source, call_graph, defined_names)

    def _get_kotlin_call_name(self, node, source: bytes) -> str | None:
        """Get the name of a called function from a call_expression node."""
        for child in node.children:
            if child.type == "identifier":
                return self._safe_decode(source[child.start_byte:child.end_byte])
        return None

    # === Swift Extraction ===

    def _extract_swift(self, file_path: Path) -> ModuleInfo:
        """Extract using tree-sitter-swift."""
        with open(file_path, "rb") as f:
            source = f.read()

        parser = self._get_swift_parser()
        tree = self._safe_parse(parser, source, file_path, "swift")

        module_info = ModuleInfo(
            file_path=str(file_path),
            language="swift",
            docstring=None,
        )

        # First pass: collect defined function names
        defined_names = self._collect_swift_definitions(tree.root_node, source)

        self._extract_swift_nodes(tree.root_node, source, module_info, defined_names)
        return module_info

    def _get_swift_parser(self) -> Any:
        """Get or create tree-sitter Swift parser."""
        if "swift" not in self._ts_parsers:
            swift_lang = Language(tree_sitter_swift.language())
            parser = Parser(swift_lang)
            self._ts_parsers["swift"] = parser
        return self._ts_parsers["swift"]

    def _collect_swift_definitions(self, node, source: bytes) -> set[str]:
        """Collect all defined function/method names in Swift."""
        names: set[str] = set()
        for child in node.children:
            if child.type == "function_declaration":
                name_node = child.child_by_field_name("name")
                if name_node:
                    names.add(self._safe_decode(source[name_node.start_byte:name_node.end_byte]))
            elif child.type == "class_declaration":
                # Find methods in class body
                for c in child.children:
                    if c.type == "class_body":
                        for member in c.children:
                            if member.type == "function_declaration":
                                name_node = member.child_by_field_name("name")
                                if name_node:
                                    names.add(self._safe_decode(source[name_node.start_byte:name_node.end_byte]))
            # Recurse
            if child.type == "source_file":
                names.update(self._collect_swift_definitions(child, source))
        return names

    def _extract_swift_nodes(self, node, source: bytes, module_info: ModuleInfo, defined_names: set[str] | None = None):
        """Extract Swift nodes into module info."""
        call_graph = module_info.call_graph or CallGraphInfo()
        module_info.call_graph = call_graph

        for child in node.children:
            if child.type == "function_declaration":
                func_info = self._extract_swift_function(child, source)
                if func_info:
                    module_info.functions.append(func_info)
                    # Extract calls from function body
                    if defined_names:
                        self._extract_swift_calls(child, func_info.name, source, call_graph, defined_names)

            elif child.type == "class_declaration":
                class_info = self._extract_swift_class(child, source, defined_names or set(), call_graph)
                if class_info:
                    module_info.classes.append(class_info)

            elif child.type == "import_declaration":
                import_info = self._extract_swift_import(child, source)
                if import_info:
                    module_info.imports.append(import_info)

            # Recurse for nested structures (FIXED: was only source_file, missed extensions/structs/protocols)
            if child.type in ("source_file", "class_declaration", "class_body",
                              "struct_declaration", "extension_declaration",
                              "protocol_declaration", "protocol_body",
                              "enum_declaration"):
                self._extract_swift_nodes(child, source, module_info, defined_names)

    def _extract_swift_function(self, node, source: bytes) -> FunctionInfo | None:
        """Extract function info from Swift function_declaration."""
        name = None
        params = []
        return_type = None

        name_node = node.child_by_field_name("name")
        if name_node:
            name = self._safe_decode(source[name_node.start_byte:name_node.end_byte])

        for child in node.children:
            if child.type == "parameter":
                for c in child.children:
                    if c.type == "simple_identifier":
                        params.append(self._safe_decode(source[c.start_byte:c.end_byte]))
                        break
            elif child.type == "type_annotation":
                # Return type after ->
                for c in child.children:
                    if c.type in ("type_identifier", "user_type", "simple_identifier"):
                        return_type = self._safe_decode(source[c.start_byte:c.end_byte])
                        break

        if not name:
            return None

        return FunctionInfo(
            name=name,
            params=params,
            return_type=return_type,
            docstring=None,
            line_number=node.start_point[0] + 1,
        )

    def _extract_swift_class(self, node, source: bytes, defined_names: set[str], call_graph: CallGraphInfo) -> ClassInfo | None:
        """Extract class info from Swift class_declaration."""
        name = None
        methods = []

        for child in node.children:
            if child.type == "type_identifier":
                name = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "class_body":
                for member in child.children:
                    if member.type == "function_declaration":
                        method = self._extract_swift_function(member, source)
                        if method:
                            methods.append(method)
                            # Extract calls from method body
                            self._extract_swift_calls(member, method.name, source, call_graph, defined_names)

        if not name:
            return None

        return ClassInfo(
            name=name,
            methods=methods,
            bases=[],
            docstring=None,
            line_number=node.start_point[0] + 1,
        )

    def _extract_swift_import(self, node, source: bytes) -> ImportInfo | None:
        """Extract import info from Swift import_declaration."""
        text = self._safe_decode(source[node.start_byte:node.end_byte]).strip()
        if text.startswith("import "):
            module = text[7:].strip()
            return ImportInfo(
                module=module,
                names=[],
            )
        return None

    def _extract_swift_calls(self, node, caller_name: str, source: bytes, call_graph: CallGraphInfo, defined_names: set[str]):
        """Extract function calls from a Swift function body."""
        for child in node.children:
            if child.type == "call_expression":
                callee = self._get_swift_call_name(child, source)
                if callee and callee in defined_names:
                    call_graph.add_call(caller_name, callee)
            # Recurse into all children
            self._extract_swift_calls(child, caller_name, source, call_graph, defined_names)

    def _get_swift_call_name(self, node, source: bytes) -> str | None:
        """Get the name of a called function from a call_expression node."""
        for child in node.children:
            if child.type == "simple_identifier":
                return self._safe_decode(source[child.start_byte:child.end_byte])
        return None

    # === C# Extraction ===

    def _extract_csharp(self, file_path: Path) -> ModuleInfo:
        """Extract using tree-sitter-c-sharp."""
        with open(file_path, "rb") as f:
            source = f.read()

        parser = self._get_csharp_parser()
        tree = self._safe_parse(parser, source, file_path, "csharp")

        module_info = ModuleInfo(
            file_path=str(file_path),
            language="csharp",
            docstring=None,
        )

        self._extract_csharp_nodes(tree.root_node, source, module_info)
        return module_info

    def _get_csharp_parser(self) -> Any:
        """Get or create tree-sitter C# parser."""
        if "csharp" not in self._ts_parsers:
            csharp_lang = Language(tree_sitter_c_sharp.language())
            parser = Parser(csharp_lang)
            self._ts_parsers["csharp"] = parser
        return self._ts_parsers["csharp"]

    def _extract_csharp_nodes(self, node, source: bytes, module_info: ModuleInfo):
        """Extract C# nodes into module info."""
        for child in node.children:
            if child.type == "method_declaration":
                func_info = self._extract_csharp_method(child, source)
                if func_info:
                    module_info.functions.append(func_info)
            elif child.type == "class_declaration":
                class_info = self._extract_csharp_class(child, source)
                if class_info:
                    module_info.classes.append(class_info)
            elif child.type == "using_directive":
                import_info = self._extract_csharp_import(child, source)
                if import_info:
                    module_info.imports.append(import_info)
            # Recurse for namespaces, classes, and other containers
            if child.type in ("compilation_unit", "namespace_declaration", "file_scoped_namespace_declaration",
                              "class_declaration", "struct_declaration", "interface_declaration", "declaration_list"):
                self._extract_csharp_nodes(child, source, module_info)

    def _extract_csharp_method(self, node, source: bytes) -> FunctionInfo | None:
        """Extract method info from C# method_declaration."""
        name = None
        params = []
        return_type = None

        for child in node.children:
            if child.type == "identifier":
                name = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "parameter_list":
                params = self._extract_csharp_params(child, source)
            elif child.type in ("predefined_type", "generic_name", "identifier", "nullable_type"):
                if return_type is None:  # First type is return type
                    return_type = self._safe_decode(source[child.start_byte:child.end_byte])

        if not name:
            return None

        return FunctionInfo(
            name=name,
            params=params,
            return_type=return_type,
            docstring=None,
            line_number=node.start_point[0] + 1,
        )

    def _extract_csharp_params(self, node, source: bytes) -> list[str]:
        """Extract parameter list from C# parameter_list."""
        params = []
        for child in node.children:
            if child.type == "parameter":
                for c in child.children:
                    if c.type == "identifier":
                        params.append(self._safe_decode(source[c.start_byte:c.end_byte]))
                        break
        return params

    def _extract_csharp_class(self, node, source: bytes) -> ClassInfo | None:
        """Extract class info from C# class_declaration."""
        name = None
        methods = []

        for child in node.children:
            if child.type == "identifier":
                name = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "declaration_list":
                for member in child.children:
                    if member.type == "method_declaration":
                        method = self._extract_csharp_method(member, source)
                        if method:
                            methods.append(method)

        if not name:
            return None

        return ClassInfo(
            name=name,
            methods=methods,
            bases=[],
            docstring=None,
            line_number=node.start_point[0] + 1,
        )

    def _extract_csharp_import(self, node, source: bytes) -> ImportInfo | None:
        """Extract import info from C# using_directive."""
        for child in node.children:
            if child.type in ("identifier", "qualified_name"):
                module = self._safe_decode(source[child.start_byte:child.end_byte])
                return ImportInfo(
                    module=module,
                    names=[],
                )
        return None

    # === Scala Extraction ===

    def _extract_scala(self, file_path: Path) -> ModuleInfo:
        """Extract using tree-sitter-scala."""
        with open(file_path, "rb") as f:
            source = f.read()

        parser = self._get_scala_parser()
        tree = self._safe_parse(parser, source, file_path, "scala")

        module_info = ModuleInfo(
            file_path=str(file_path),
            language="scala",
            docstring=None,
        )

        # First pass: collect defined function names
        defined_names = self._collect_scala_definitions(tree.root_node, source)

        self._extract_scala_nodes(tree.root_node, source, module_info, defined_names)
        return module_info

    def _get_scala_parser(self) -> Any:
        """Get or create tree-sitter Scala parser."""
        if "scala" not in self._ts_parsers:
            scala_lang = Language(tree_sitter_scala.language())
            parser = Parser(scala_lang)
            self._ts_parsers["scala"] = parser
        return self._ts_parsers["scala"]

    def _collect_scala_definitions(self, node, source: bytes) -> set[str]:
        """Collect all defined function/method names in Scala."""
        names: set[str] = set()
        for child in node.children:
            if child.type == "function_definition":
                name_node = child.child_by_field_name("name")
                if name_node:
                    names.add(self._safe_decode(source[name_node.start_byte:name_node.end_byte]))
            elif child.type in ("class_definition", "object_definition"):
                for c in child.children:
                    if c.type == "template_body":
                        for member in c.children:
                            if member.type == "function_definition":
                                name_node = member.child_by_field_name("name")
                                if name_node:
                                    names.add(self._safe_decode(source[name_node.start_byte:name_node.end_byte]))
            # Recurse
            names.update(self._collect_scala_definitions(child, source))
        return names

    def _extract_scala_nodes(self, node, source: bytes, module_info: ModuleInfo, defined_names: set[str] | None = None):
        """Extract Scala nodes into module info."""
        call_graph = module_info.call_graph or CallGraphInfo()
        module_info.call_graph = call_graph

        for child in node.children:
            if child.type == "function_definition":
                func_info = self._extract_scala_function(child, source)
                if func_info:
                    module_info.functions.append(func_info)
                    # Extract calls from function body
                    if defined_names:
                        self._extract_scala_calls(child, func_info.name, source, call_graph, defined_names)

            elif child.type in ("class_definition", "object_definition"):
                class_info = self._extract_scala_class(child, source, defined_names or set(), call_graph)
                if class_info:
                    module_info.classes.append(class_info)

            elif child.type == "import_declaration":
                import_info = self._extract_scala_import(child, source)
                if import_info:
                    module_info.imports.append(import_info)

            # Recurse for nested structures
            if child.type in ("compilation_unit", "package_clause", "template_body",
                              "object_definition", "class_definition", "trait_definition"):
                self._extract_scala_nodes(child, source, module_info, defined_names)

    def _extract_scala_function(self, node, source: bytes) -> FunctionInfo | None:
        """Extract function info from Scala function_definition."""
        name = None
        params = []
        return_type = None

        name_node = node.child_by_field_name("name")
        if name_node:
            name = self._safe_decode(source[name_node.start_byte:name_node.end_byte])

        for child in node.children:
            if child.type == "parameters":
                params = self._extract_scala_params(child, source)
            elif child.type in ("type_identifier", "generic_type", "simple_type"):
                return_type = self._safe_decode(source[child.start_byte:child.end_byte])

        if not name:
            return None

        return FunctionInfo(
            name=name,
            params=params,
            return_type=return_type,
            docstring=None,
            line_number=node.start_point[0] + 1,
        )

    def _extract_scala_params(self, node, source: bytes) -> list[str]:
        """Extract parameter list from Scala parameters."""
        params = []
        for child in node.children:
            if child.type == "parameter":
                name_node = child.child_by_field_name("name")
                if name_node:
                    params.append(self._safe_decode(source[name_node.start_byte:name_node.end_byte]))
        return params

    def _extract_scala_class(self, node, source: bytes, defined_names: set[str], call_graph: CallGraphInfo) -> ClassInfo | None:
        """Extract class info from Scala class_definition or object_definition."""
        name = None
        methods = []

        name_node = node.child_by_field_name("name")
        if name_node:
            name = self._safe_decode(source[name_node.start_byte:name_node.end_byte])

        for child in node.children:
            if child.type == "template_body":
                for member in child.children:
                    if member.type == "function_definition":
                        method = self._extract_scala_function(member, source)
                        if method:
                            methods.append(method)
                            # Extract calls from method body
                            self._extract_scala_calls(member, method.name, source, call_graph, defined_names)

        if not name:
            return None

        return ClassInfo(
            name=name,
            methods=methods,
            bases=[],
            docstring=None,
            line_number=node.start_point[0] + 1,
        )

    def _extract_scala_import(self, node, source: bytes) -> ImportInfo | None:
        """Extract import info from Scala import_declaration."""
        text = self._safe_decode(source[node.start_byte:node.end_byte]).strip()
        if text.startswith("import "):
            module = text[7:].strip()
            # Handle selective imports: import foo.{A, B}
            names = []
            if "{" in module:
                base = module.split("{")[0].rstrip(".")
                selectors = module.split("{")[1].rstrip("}").split(",")
                names = [s.strip() for s in selectors if s.strip()]
                module = base
            return ImportInfo(
                module=module,
                names=names,
            )
        return None

    def _extract_scala_calls(self, node, caller_name: str, source: bytes, call_graph: CallGraphInfo, defined_names: set[str]):
        """Extract function calls from a Scala function body."""
        for child in node.children:
            if child.type == "call_expression":
                callee = self._get_scala_call_name(child, source)
                if callee and callee in defined_names:
                    call_graph.add_call(caller_name, callee)
            # Recurse into all children
            self._extract_scala_calls(child, caller_name, source, call_graph, defined_names)

    def _get_scala_call_name(self, node, source: bytes) -> str | None:
        """Get the name of a called function from a call_expression node."""
        for child in node.children:
            if child.type == "identifier":
                return self._safe_decode(source[child.start_byte:child.end_byte])
        return None

    # === Lua Extraction ===

    def _extract_lua(self, file_path: Path) -> ModuleInfo:
        """Extract using tree-sitter-lua."""
        with open(file_path, "rb") as f:
            source = f.read()

        parser = self._get_lua_parser()
        tree = self._safe_parse(parser, source, file_path, "lua")

        module_info = ModuleInfo(
            file_path=str(file_path),
            language="lua",
            docstring=None,
        )

        # First pass: collect defined function names
        defined_names = self._collect_lua_definitions(tree.root_node, source)

        self._extract_lua_nodes(tree.root_node, source, module_info, defined_names)
        return module_info

    def _get_lua_parser(self) -> Any:
        """Get or create tree-sitter Lua parser."""
        if "lua" not in self._ts_parsers:
            lua_lang = Language(tree_sitter_lua.language())
            parser = Parser(lua_lang)
            self._ts_parsers["lua"] = parser
        return self._ts_parsers["lua"]

    def _collect_lua_definitions(self, node, source: bytes) -> set[str]:
        """Collect all defined function names in Lua."""
        names: set[str] = set()

        for child in node.children:
            # function name() ... end or local function name() ... end
            if child.type == "function_declaration":
                # Find the identifier child (the function name)
                for grandchild in child.children:
                    if grandchild.type == "identifier":
                        names.add(self._safe_decode(source[grandchild.start_byte:grandchild.end_byte]))
                        break
                    elif grandchild.type in ("dot_index_expression", "method_index_expression"):
                        # Table.method or Table:method
                        field = grandchild.child_by_field_name("field")
                        if field:
                            names.add(self._safe_decode(source[field.start_byte:field.end_byte]))
                        break

            # Recurse
            names.update(self._collect_lua_definitions(child, source))

        return names

    def _extract_lua_nodes(self, node, source: bytes, module_info: ModuleInfo, defined_names: set[str] | None = None):
        """Extract Lua nodes into module info."""
        call_graph = module_info.call_graph or CallGraphInfo()
        module_info.call_graph = call_graph

        for child in node.children:
            # function name() ... end or local function name() ... end
            if child.type == "function_declaration":
                func_info = self._extract_lua_function(child, source)
                if func_info:
                    module_info.functions.append(func_info)
                    # Extract calls from function body
                    if defined_names:
                        self._extract_lua_calls(child, func_info.name, source, call_graph, defined_names)

            # require statements
            elif child.type == "function_call":
                import_info = self._extract_lua_require(child, source)
                if import_info:
                    module_info.imports.append(import_info)

            # Recurse for other nodes
            else:
                self._extract_lua_nodes(child, source, module_info, defined_names)

    def _extract_lua_function(self, node, source: bytes) -> FunctionInfo | None:
        """Extract function info from function_declaration."""
        # Find the identifier child (the function name)
        name = None
        for child in node.children:
            if child.type == "identifier":
                name = self._safe_decode(source[child.start_byte:child.end_byte])
                break
            elif child.type in ("dot_index_expression", "method_index_expression"):
                # Table.method or Table:method
                field = child.child_by_field_name("field")
                if field:
                    name = self._safe_decode(source[field.start_byte:field.end_byte])
                break

        if not name:
            return None

        params = self._extract_lua_params(node, source)

        return FunctionInfo(
            name=name,
            params=params,
            return_type=None,  # Lua is dynamically typed
            docstring=None,
            line_number=node.start_point[0] + 1,
        )

    def _extract_lua_params(self, node, source: bytes) -> list[str]:
        """Extract parameters from a Lua function node."""
        params = []
        for child in node.children:
            if child.type == "parameters":
                for param in child.children:
                    if param.type == "identifier":
                        params.append(self._safe_decode(source[param.start_byte:param.end_byte]))
                    elif param.type == "spread":
                        params.append("...")
                break
        return params

    def _extract_lua_require(self, node, source: bytes) -> ImportInfo | None:
        """Extract require statement from a function_call node."""
        func_name = None
        module = None

        for child in node.children:
            if child.type == "identifier":
                func_name = self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type == "arguments":
                # Get the first string argument
                for arg in child.children:
                    if arg.type == "string":
                        text = self._safe_decode(source[arg.start_byte:arg.end_byte])
                        # Strip quotes
                        if text.startswith('"') and text.endswith('"'):
                            module = text[1:-1]
                        elif text.startswith("'") and text.endswith("'"):
                            module = text[1:-1]
                        break
            elif child.type == "string":
                # require "module" syntax
                text = self._safe_decode(source[child.start_byte:child.end_byte])
                if text.startswith('"') and text.endswith('"'):
                    module = text[1:-1]
                elif text.startswith("'") and text.endswith("'"):
                    module = text[1:-1]

        if func_name == "require" and module:
            return ImportInfo(
                module=module,
                names=[],
            )
        return None

    def _extract_lua_calls(self, node, caller_name: str, source: bytes, call_graph: CallGraphInfo, defined_names: set[str]):
        """Extract function calls from a Lua function body."""
        for child in node.children:
            if child.type == "function_call":
                callee = self._get_lua_call_name(child, source)
                if callee and callee in defined_names:
                    call_graph.add_call(caller_name, callee)
            # Recurse into all children
            self._extract_lua_calls(child, caller_name, source, call_graph, defined_names)

    def _get_lua_call_name(self, node, source: bytes) -> str | None:
        """Get the name of a called function from a function_call node."""
        for child in node.children:
            if child.type == "identifier":
                return self._safe_decode(source[child.start_byte:child.end_byte])
            elif child.type in ("dot_index_expression", "method_index_expression"):
                # Table.method() or Table:method()
                field = child.child_by_field_name("field")
                if field:
                    return self._safe_decode(source[field.start_byte:field.end_byte])
        return None

    # === Elixir Extraction ===

    def _extract_elixir(self, file_path: Path) -> ModuleInfo:
        """Extract using tree-sitter-elixir."""
        with open(file_path, "rb") as f:
            source = f.read()

        parser = self._get_elixir_parser()
        tree = self._safe_parse(parser, source, file_path, "elixir")

        module_info = ModuleInfo(
            file_path=str(file_path),
            language="elixir",
            docstring=None,
        )

        self._extract_elixir_nodes(tree.root_node, source, module_info)
        return module_info

    def _get_elixir_parser(self) -> Any:
        """Get or create tree-sitter Elixir parser."""
        if "elixir" not in self._ts_parsers:
            elixir_lang = Language(tree_sitter_elixir.language())
            parser = Parser(elixir_lang)
            self._ts_parsers["elixir"] = parser
        return self._ts_parsers["elixir"]

    def _extract_elixir_nodes(self, node, source: bytes, module_info: ModuleInfo):
        """Extract Elixir nodes into module info."""
        for child in node.children:
            if child.type == "call":
                # Check if this is a def/defp/defmodule call
                call_name = self._get_elixir_call_identifier(child, source)
                if call_name in ("def", "defp"):
                    func_info = self._extract_elixir_function(child, source, call_name == "defp")
                    if func_info:
                        module_info.functions.append(func_info)
                elif call_name == "defmodule":
                    # Extract module name
                    args = child.child_by_field_name("arguments")
                    if args:
                        for arg in args.children:
                            if arg.type == "alias":
                                module_info.docstring = f"Module: {self._safe_decode(source[arg.start_byte:arg.end_byte])}"
                                break
            # Recurse
            self._extract_elixir_nodes(child, source, module_info)

    def _get_elixir_call_identifier(self, node, source: bytes) -> str | None:
        """Get the identifier from an Elixir call node."""
        for child in node.children:
            if child.type == "identifier":
                return self._safe_decode(source[child.start_byte:child.end_byte])
        return None

    def _extract_elixir_function(self, node, source: bytes, is_private: bool) -> FunctionInfo | None:
        """Extract an Elixir function definition (def/defp)."""
        # Note: tree-sitter-elixir uses child types, not field names
        args = None
        for child in node.children:
            if child.type == "arguments":
                args = child
                break
        if not args:
            return None

        name = None
        params = []

        for arg_child in args.children:
            if arg_child.type == "call":
                # Function with parameters: def func_name(args)
                for c in arg_child.children:
                    if c.type == "identifier":
                        name = self._safe_decode(source[c.start_byte:c.end_byte])
                        break
                # Extract parameters - use child type iteration for elixir
                inner_args = None
                for c in arg_child.children:
                    if c.type == "arguments":
                        inner_args = c
                        break
                if inner_args:
                    for param in inner_args.children:
                        if param.type == "identifier":
                            params.append(self._safe_decode(source[param.start_byte:param.end_byte]))
            elif arg_child.type == "identifier":
                # Function without parameters: def func_name do
                name = self._safe_decode(source[arg_child.start_byte:arg_child.end_byte])

        if not name:
            return None

        return FunctionInfo(
            name=name,
            params=params,
            return_type=None,
            docstring=None,
            line_number=node.start_point[0] + 1,
        )

    def _parse_signatures(self, text: str) -> list[str]:
        """Parse Pygments signature output."""
        if not text or not text.strip():
            return []
        lines = text.strip().split('\n')
        return [
            line.strip().lstrip("- ").lstrip("* ")
            for line in lines
            if line.strip() and not line.startswith("#") and not line.startswith("```")
        ]

    def _extract_params_from_sig(self, sig: str) -> list[str]:
        """Extract params from signature string."""
        if "(" not in sig:
            return []
        try:
            params_str = sig.split("(", 1)[1].rsplit(")", 1)[0]
            return [p.strip() for p in params_str.split(",") if p.strip()]
        except:
            return []

    def _detect_language(self, file_path: Path) -> str:
        """Detect language from file extension."""
        ext_map = {
            ".py": "python", ".pyx": "python", ".pyi": "python",
            ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
            ".ts": "typescript", ".tsx": "typescript",
            ".go": "go", ".rs": "rust", ".rb": "ruby",
            ".java": "java", ".kt": "kotlin",
            ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
            ".cs": "csharp", ".swift": "swift",
            ".scala": "scala", ".sc": "scala",
            ".lua": "lua",
        }
        return ext_map.get(file_path.suffix.lower(), "unknown")


def extract_directory(
    directory: str | Path,
    extensions: set[str] | None = None,
    recursive: bool = True,
) -> dict[str, Any]:
    """
    Extract code structure from all files in a directory.

    Args:
        directory: Directory to scan
        extensions: File extensions to include (default: all supported)
        recursive: Whether to scan subdirectories

    Returns:
        Combined extraction results
    """
    directory = Path(directory)
    extractor = HybridExtractor()

    if extensions is None:
        extensions = (
            HybridExtractor.PYTHON_EXTENSIONS |
            HybridExtractor.TREE_SITTER_EXTENSIONS |
            {".go", ".rs", ".rb", ".java", ".kt", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".scala", ".sc"}
        )

    results = {
        "directory": str(directory),
        "files": [],
    }

    pattern = "**/*" if recursive else "*"
    for file_path in directory.glob(pattern):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in extensions:
            continue
        if file_path.name.startswith("."):
            continue

        try:
            module_info = extractor.extract(file_path)
            results["files"].append(module_info.to_compact())
        except Exception as e:
            logger.warning(f"Failed to extract {file_path}: {e}")

    return results


# === CLI ===

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hybrid_extractor.py <file_or_directory> [--compact] [--recursive] [--cfg]")
        sys.exit(1)

    target = Path(sys.argv[1])
    compact = "--compact" in sys.argv
    show_cfg = "--cfg" in sys.argv

    if target.is_dir():
        result = extract_directory(target)
        print(json.dumps(result, indent=2))
    else:
        extractor = HybridExtractor()
        info = extractor.extract(target)

        if show_cfg:
            # Extract CFG for all functions
            from .cfg_extractor import (
                extract_python_cfg, extract_typescript_cfg,
                extract_go_cfg, extract_rust_cfg,
                TREE_SITTER_AVAILABLE, TREE_SITTER_GO_AVAILABLE, TREE_SITTER_RUST_AVAILABLE
            )

            source = target.read_text()
            suffix = target.suffix.lower()
            cfg_results = []

            for func in info.functions:
                try:
                    if suffix == ".py":
                        cfg = extract_python_cfg(source, func.name)
                    elif suffix in {".ts", ".tsx", ".js", ".jsx"} and TREE_SITTER_AVAILABLE:
                        cfg = extract_typescript_cfg(source, func.name)
                    elif suffix == ".go" and TREE_SITTER_GO_AVAILABLE:
                        cfg = extract_go_cfg(source, func.name)
                    elif suffix == ".rs" and TREE_SITTER_RUST_AVAILABLE:
                        cfg = extract_rust_cfg(source, func.name)
                    else:
                        continue

                    cfg_results.append(cfg.to_dict())
                except Exception as e:
                    logger.debug(f"CFG extraction failed for {func.name}: {e}")

            output = {
                "file": str(target),
                "module": info.to_compact() if compact else info.to_dict(),
                "cfg": cfg_results,
            }
            print(json.dumps(output, indent=2))
        else:
            output = info.to_compact() if compact else info.to_dict()
            print(json.dumps(output, indent=2))
