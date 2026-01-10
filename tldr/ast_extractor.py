#!/usr/bin/env python3
"""
AST-based code structure extractor with full metadata.

Extracts:
- Function signatures WITH return types
- Class hierarchy (inheritance)
- Import dependencies
- Docstrings

Supports:
- Python (via ast module) - full support
- TypeScript/JavaScript (via tree-sitter) - planned
- Other languages (via tree-sitter) - planned
"""

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    """Extracted function/method information."""
    name: str
    params: list[str]
    return_type: str | None
    docstring: str | None
    is_method: bool = False
    is_async: bool = False
    decorators: list[str] = field(default_factory=list)
    line_number: int = 0

    def signature(self) -> str:
        """Return full signature string."""
        async_prefix = "async " if self.is_async else ""
        params_str = ", ".join(self.params)
        ret = f" -> {self.return_type}" if self.return_type else ""
        return f"{async_prefix}def {self.name}({params_str}){ret}"


@dataclass
class ClassInfo:
    """Extracted class information."""
    name: str
    bases: list[str]
    docstring: str | None
    methods: list[FunctionInfo] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    line_number: int = 0

    def signature(self) -> str:
        """Return class definition signature."""
        bases_str = ", ".join(self.bases) if self.bases else ""
        return f"class {self.name}({bases_str})" if bases_str else f"class {self.name}"


@dataclass
class ImportInfo:
    """Extracted import information."""
    module: str
    names: list[str]  # Empty for 'import x', filled for 'from x import y, z'
    is_from: bool = False
    line_number: int = 0

    def statement(self) -> str:
        """Return import statement string."""
        if self.is_from:
            names_str = ", ".join(self.names)
            return f"from {self.module} import {names_str}"
        return f"import {self.module}"


@dataclass
class CallGraphInfo:
    """Call graph showing function relationships."""
    calls: dict[str, list[str]] = field(default_factory=dict)  # func -> [called funcs]
    called_by: dict[str, list[str]] = field(default_factory=dict)  # func -> [callers]

    def add_call(self, caller: str, callee: str):
        """Record a function call."""
        if caller not in self.calls:
            self.calls[caller] = []
        if callee not in self.calls[caller]:
            self.calls[caller].append(callee)

        if callee not in self.called_by:
            self.called_by[callee] = []
        if caller not in self.called_by[callee]:
            self.called_by[callee].append(caller)


@dataclass
class ModuleInfo:
    """Complete module extraction result."""
    file_path: str
    language: str
    docstring: str | None
    imports: list[ImportInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    call_graph: CallGraphInfo = field(default_factory=CallGraphInfo)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "language": self.language,
            "docstring": self.docstring,
            "imports": [
                {"module": i.module, "names": i.names, "is_from": i.is_from}
                for i in self.imports
            ],
            "classes": [
                {
                    "name": c.name,
                    "line_number": c.line_number,
                    "signature": c.signature(),
                    "bases": c.bases,
                    "docstring": c.docstring,
                    "decorators": c.decorators,
                    "methods": [
                        {
                            "name": m.name,
                            "line_number": m.line_number,
                            "signature": m.signature(),
                            "params": m.params,
                            "return_type": m.return_type,
                            "docstring": m.docstring,
                            "is_async": m.is_async,
                            "decorators": m.decorators,
                        }
                        for m in c.methods
                    ],
                }
                for c in self.classes
            ],
            "functions": [
                {
                    "name": f.name,
                    "line_number": f.line_number,
                    "signature": f.signature(),
                    "params": f.params,
                    "return_type": f.return_type,
                    "docstring": f.docstring,
                    "is_async": f.is_async,
                    "decorators": f.decorators,
                }
                for f in self.functions
            ],
            "call_graph": {
                "calls": self.call_graph.calls,
                "called_by": self.call_graph.called_by,
            } if self.call_graph.calls else {},
        }

    def to_compact(self) -> dict[str, Any]:
        """Compact format optimized for LLM context."""
        result: dict[str, Any] = {
            "file": Path(self.file_path).name,
            "lang": self.language,
        }

        if self.docstring:
            # Truncate long docstrings
            doc = self.docstring[:200] + "..." if len(self.docstring) > 200 else self.docstring
            result["doc"] = doc

        if self.imports:
            result["imports"] = [i.statement() for i in self.imports]

        if self.classes:
            result["classes"] = {}
            for c in self.classes:
                class_info: dict[str, Any] = {"bases": c.bases} if c.bases else {}
                if c.docstring:
                    class_info["doc"] = c.docstring[:100] + "..." if len(c.docstring) > 100 else c.docstring
                if c.methods:
                    class_info["methods"] = [m.signature() for m in c.methods]
                result["classes"][c.name] = class_info

        if self.functions:
            result["functions"] = [f.signature() for f in self.functions]

        if self.call_graph.calls:
            result["calls"] = self.call_graph.calls

        return result


class PythonASTExtractor:
    """Extract code structure from Python files using AST."""

    def extract(self, file_path: str | Path) -> ModuleInfo:
        """Extract module information from a Python file."""
        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()

        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return ModuleInfo(
                file_path=str(file_path),
                language="python",
                docstring=None,
            )

        module_info = ModuleInfo(
            file_path=str(file_path),
            language="python",
            docstring=ast.get_docstring(tree),
        )

        # First pass: collect all defined function/method names
        defined_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined_names.add(node.name)

        # Second pass: extract structure and calls
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_info.imports.append(ImportInfo(
                        module=alias.name,
                        names=[],
                        is_from=False,
                        line_number=node.lineno,
                    ))

            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                names = [alias.name for alias in node.names]
                module_info.imports.append(ImportInfo(
                    module=module_name,
                    names=names,
                    is_from=True,
                    line_number=node.lineno,
                ))

            elif isinstance(node, ast.ClassDef):
                class_info = self._extract_class(
                    node,
                    call_graph=module_info.call_graph,
                    defined_names=defined_names,
                    module_info=module_info,
                )
                module_info.classes.append(class_info)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = self._extract_function(node)
                module_info.functions.append(func_info)
                # Extract calls from this function
                self._extract_calls(node, node.name, module_info.call_graph, defined_names)
                # Extract nested functions
                self._extract_nested_functions(node, module_info, defined_names)

        return module_info

    def _extract_nested_functions(
        self,
        parent_node: ast.FunctionDef | ast.AsyncFunctionDef,
        module_info: ModuleInfo,
        defined_names: set[str],
    ):
        """Extract nested functions from a function body."""
        for node in ast.walk(parent_node):
            if node is parent_node:
                continue  # Skip the parent itself
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = self._extract_function(node)
                # Mark as nested for context
                func_info.decorators.insert(0, f"nested_in:{parent_node.name}")
                module_info.functions.append(func_info)
                # Extract calls from this nested function
                self._extract_calls(node, node.name, module_info.call_graph, defined_names)

    def _extract_class(
        self,
        node: ast.ClassDef,
        call_graph: CallGraphInfo | None = None,
        defined_names: set[str] | None = None,
        module_info: ModuleInfo | None = None,
        parent_path: str = "",
    ) -> ClassInfo:
        """Extract class information, including nested classes."""
        bases = []
        for base in node.bases:
            bases.append(self._node_to_str(base))

        decorators = [self._node_to_str(d) for d in node.decorator_list]

        # Build qualified name for nested class tracking
        qualified_name = f"{parent_path}.{node.name}" if parent_path else node.name

        class_info = ClassInfo(
            name=node.name,
            bases=bases,
            docstring=ast.get_docstring(node),
            decorators=decorators,
            line_number=node.lineno,
        )

        # Extract methods and nested classes
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method = self._extract_function(item, is_method=True)
                class_info.methods.append(method)
                # Extract calls from this method
                if call_graph and defined_names:
                    caller_name = f"{qualified_name}.{item.name}"
                    self._extract_calls(item, caller_name, call_graph, defined_names)

            elif isinstance(item, ast.ClassDef):
                # Recursively extract nested classes
                nested_class = self._extract_class(
                    item,
                    call_graph=call_graph,
                    defined_names=defined_names,
                    module_info=module_info,
                    parent_path=qualified_name,
                )
                # Add nested class to module's classes list
                if module_info is not None:
                    module_info.classes.append(nested_class)
                # Also add nested class methods to module's functions list for discoverability
                for method in nested_class.methods:
                    if module_info is not None:
                        # Create a copy with qualified name
                        nested_method = FunctionInfo(
                            name=method.name,
                            params=method.params,
                            return_type=method.return_type,
                            docstring=method.docstring,
                            is_method=True,
                            is_async=method.is_async,
                            decorators=[f"nested_in:{qualified_name}.{nested_class.name}"] + method.decorators,
                            line_number=method.line_number,
                        )
                        module_info.functions.append(nested_method)

        return class_info

    def _extract_calls(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        caller_name: str,
        call_graph: CallGraphInfo,
        defined_names: set[str],
    ):
        """Extract function calls from a function body."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee = self._get_call_name(child)
                if callee and callee in defined_names:
                    call_graph.add_call(caller_name, callee)

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Get the name of a called function."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # For method calls like self.method() or obj.method()
            # We only track the method name for simplicity
            return node.func.attr
        return None

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_method: bool = False
    ) -> FunctionInfo:
        """Extract function/method information."""
        params = self._extract_params(node.args)
        return_type = self._node_to_str(node.returns) if node.returns else None
        decorators = [self._node_to_str(d) for d in node.decorator_list]

        return FunctionInfo(
            name=node.name,
            params=params,
            return_type=return_type,
            docstring=ast.get_docstring(node),
            is_method=is_method,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            decorators=decorators,
            line_number=node.lineno,
        )

    def _extract_params(self, args: ast.arguments) -> list[str]:
        """Extract parameter list with type annotations."""
        params = []

        # Positional-only params (before /)
        for arg in args.posonlyargs:
            params.append(self._format_arg(arg))

        if args.posonlyargs:
            params.append("/")

        # Regular positional/keyword params
        defaults_start = len(args.args) - len(args.defaults)
        for i, arg in enumerate(args.args):
            param = self._format_arg(arg)
            # Add default value indicator
            if i >= defaults_start:
                default_idx = i - defaults_start
                default = args.defaults[default_idx]
                param += f" = {self._node_to_str(default)}"
            params.append(param)

        # *args
        if args.vararg:
            params.append(f"*{self._format_arg(args.vararg)}")
        elif args.kwonlyargs:
            params.append("*")

        # Keyword-only params
        kw_defaults_map = {i: d for i, d in enumerate(args.kw_defaults) if d is not None}
        for i, arg in enumerate(args.kwonlyargs):
            param = self._format_arg(arg)
            if i in kw_defaults_map:
                param += f" = {self._node_to_str(kw_defaults_map[i])}"
            params.append(param)

        # **kwargs
        if args.kwarg:
            params.append(f"**{self._format_arg(args.kwarg)}")

        return params

    def _format_arg(self, arg: ast.arg) -> str:
        """Format a single argument with optional type annotation."""
        if arg.annotation:
            return f"{arg.arg}: {self._node_to_str(arg.annotation)}"
        return arg.arg

    def _node_to_str(self, node: ast.AST | None) -> str:
        """Convert AST node to string representation."""
        if node is None:
            return ""

        # Python 3.9+ has ast.unparse
        try:
            return ast.unparse(node)
        except AttributeError:
            # Fallback for older Python
            return self._manual_unparse(node)

    def _manual_unparse(self, node: ast.AST) -> str:
        """Manual unparse for older Python versions."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._manual_unparse(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._manual_unparse(node.value)}[{self._manual_unparse(node.slice)}]"
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Tuple):
            elts = ", ".join(self._manual_unparse(e) for e in node.elts)
            return f"({elts})"
        elif isinstance(node, ast.List):
            elts = ", ".join(self._manual_unparse(e) for e in node.elts)
            return f"[{elts}]"
        elif isinstance(node, ast.BinOp):
            # Handle Union types like X | Y
            if isinstance(node.op, ast.BitOr):
                return f"{self._manual_unparse(node.left)} | {self._manual_unparse(node.right)}"
        elif isinstance(node, ast.Call):
            func = self._manual_unparse(node.func)
            args = ", ".join(self._manual_unparse(a) for a in node.args)
            return f"{func}({args})"

        return "<unknown>"


def extract_python(file_path: str | Path) -> ModuleInfo:
    """Convenience function to extract Python module info."""
    extractor = PythonASTExtractor()
    return extractor.extract(file_path)


def extract_file(file_path: str | Path) -> ModuleInfo:
    """
    Extract code structure from any supported file.

    Supports:
    - Python (.py, .pyx, .pyi) via native AST
    - TypeScript/JavaScript (.ts, .tsx, .js, .jsx) via tree-sitter
    - Go (.go) via tree-sitter-go
    - Rust (.rs) via tree-sitter-rust
    - Other languages via Pygments fallback (signatures only)
    """
    # Use HybridExtractor which handles all languages
    from tldr.hybrid_extractor import HybridExtractor

    extractor = HybridExtractor()
    return extractor.extract(file_path)


# === CLI ===

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ast_extractor.py <file_path> [--compact]")
        sys.exit(1)

    file_path = sys.argv[1]
    compact = "--compact" in sys.argv

    try:
        info = extract_file(file_path)
        output = info.to_compact() if compact else info.to_dict()
        print(json.dumps(output, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
