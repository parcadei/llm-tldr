"""
Data Flow Graph (DFG) extraction for multi-language code analysis.

Provides DFG extraction for:
- Python (via ast module)
- TypeScript/JavaScript (via tree-sitter)
- Go (via tree-sitter)
- Rust (via tree-sitter)

Based on pflux/python-program-analysis architecture:
- DEFINITION: variable assignments (x = ...)
- UPDATE: in-place modifications (x += ..., x.append())
- USE: variable reads

Uses reaching definitions analysis on CFG to build def-use chains.
"""
import ast
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VarRef:
    """
    A reference to a variable (definition or use).

    Reference types:
    - "definition": variable assignment (x = ...)
    - "update": in-place modification (x += ..., x.append())
    - "use": variable read
    """
    name: str
    ref_type: str  # "definition", "update", "use"
    line: int
    column: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.ref_type,
            "line": self.line,
            "column": self.column,
        }


@dataclass
class DataflowEdge:
    """
    A def-use relationship connecting a definition to a use.

    Represents that the value defined at def_ref may flow to use_ref.
    """
    def_ref: VarRef
    use_ref: VarRef

    @property
    def var_name(self) -> str:
        return self.def_ref.name

    def to_dict(self) -> dict:
        return {
            "var": self.var_name,
            "def_line": self.def_ref.line,
            "use_line": self.use_ref.line,
            "def": self.def_ref.to_dict(),
            "use": self.use_ref.to_dict(),
        }


@dataclass
class DFGInfo:
    """
    Data flow graph for a function.

    Provides:
    - All variable references (definitions and uses)
    - Def-use chains (dataflow edges)
    - Variable grouping for quick lookup
    """
    function_name: str
    var_refs: list[VarRef]
    dataflow_edges: list[DataflowEdge]

    @property
    def variables(self) -> dict[str, list[VarRef]]:
        """Group references by variable name."""
        result: dict[str, list[VarRef]] = {}
        for ref in self.var_refs:
            if ref.name not in result:
                result[ref.name] = []
            result[ref.name].append(ref)
        return result

    def to_dict(self) -> dict:
        return {
            "function": self.function_name,
            "refs": [r.to_dict() for r in self.var_refs],
            "edges": [e.to_dict() for e in self.dataflow_edges],
            "variables": list(self.variables.keys()),
        }


# =============================================================================
# Python DFG Extraction (using ast module)
# =============================================================================

class PythonDefUseVisitor(ast.NodeVisitor):
    """
    Extract variable definitions and uses from Python AST.

    Builds a list of VarRefs for all variable references.
    """

    def __init__(self):
        self.refs: list[VarRef] = []
        self.scope_stack: list[set[str]] = [set()]  # Track local scope

    def _add_ref(self, name: str, ref_type: str, node: ast.AST):
        """Add a variable reference."""
        ref = VarRef(
            name=name,
            ref_type=ref_type,
            line=node.lineno,
            column=node.col_offset,
        )
        self.refs.append(ref)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Handle function definition - parameters are definitions."""
        # Add parameters as definitions
        for arg in node.args.args:
            self._add_ref(arg.arg, "definition", arg)

        # Process body
        for stmt in node.body:
            self.visit(stmt)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Handle async function - same as regular function."""
        for arg in node.args.args:
            self._add_ref(arg.arg, "definition", arg)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Assign(self, node: ast.Assign):
        """Handle assignment: target = value."""
        # First visit value side (uses)
        self.visit(node.value)

        # Then add definitions for targets
        for target in node.targets:
            self._visit_target(target, "definition")

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Handle annotated assignment: target: type = value."""
        if node.value:
            self.visit(node.value)
        if node.target:
            self._visit_target(node.target, "definition")

    def visit_AugAssign(self, node: ast.AugAssign):
        """Handle augmented assignment: target += value."""
        # Value side
        self.visit(node.value)

        # Target is both used and updated
        if isinstance(node.target, ast.Name):
            self._add_ref(node.target.id, "use", node.target)
            self._add_ref(node.target.id, "update", node.target)
        else:
            self._visit_target(node.target, "update")

    def _visit_target(self, target: ast.AST, ref_type: str):
        """Visit an assignment target."""
        if isinstance(target, ast.Name):
            self._add_ref(target.id, ref_type, target)
        elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
            for elt in target.elts:
                self._visit_target(elt, ref_type)
        elif isinstance(target, ast.Starred):
            self._visit_target(target.value, ref_type)
        # Attribute/Subscript targets don't introduce new definitions

    def visit_For(self, node: ast.For):
        """Handle for loop - target is definition."""
        # Iterator is used
        self.visit(node.iter)

        # Target is defined
        self._visit_target(node.target, "definition")

        # Body
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_With(self, node: ast.With):
        """Handle with statement - optional_vars are definitions."""
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self._visit_target(item.optional_vars, "definition")

        for stmt in node.body:
            self.visit(stmt)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        """Handle except - name is definition."""
        if node.name:
            self._add_ref(node.name, "definition", node)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Name(self, node: ast.Name):
        """Handle name reference - context determines if it's a use."""
        if isinstance(node.ctx, ast.Load):
            self._add_ref(node.id, "use", node)
        # Store context is handled by parent (Assign, etc.)

    def visit_comprehension(self, node: ast.comprehension):
        """Handle comprehension - target is definition, iter/ifs are uses."""
        self.visit(node.iter)
        self._visit_target(node.target, "definition")
        for if_clause in node.ifs:
            self.visit(if_clause)

    def visit_ListComp(self, node: ast.ListComp):
        """Handle list comprehension."""
        for gen in node.generators:
            self.visit_comprehension(gen)
        self.visit(node.elt)

    def visit_SetComp(self, node: ast.SetComp):
        """Handle set comprehension."""
        for gen in node.generators:
            self.visit_comprehension(gen)
        self.visit(node.elt)

    def visit_DictComp(self, node: ast.DictComp):
        """Handle dict comprehension."""
        for gen in node.generators:
            self.visit_comprehension(gen)
        self.visit(node.key)
        self.visit(node.value)

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        """Handle generator expression."""
        for gen in node.generators:
            self.visit_comprehension(gen)
        self.visit(node.elt)

    def visit_Lambda(self, node: ast.Lambda):
        """Handle lambda - parameters are definitions."""
        for arg in node.args.args:
            self._add_ref(arg.arg, "definition", arg)
        self.visit(node.body)

    # Generic visitor for expressions with children
    def generic_visit(self, node: ast.AST):
        """Visit all child nodes."""
        for child in ast.iter_child_nodes(node):
            self.visit(child)


class PythonReachingDefsAnalyzer:
    """
    Reaching definitions analysis for Python code.

    Uses worklist algorithm on basic blocks to compute
    which definitions reach each use.
    """

    def __init__(self, refs: list[VarRef]):
        self.refs = refs
        self.defs_by_line: dict[int, list[VarRef]] = {}
        self.uses_by_line: dict[int, list[VarRef]] = {}

        # Group refs by line
        for ref in refs:
            if ref.ref_type in ("definition", "update"):
                if ref.line not in self.defs_by_line:
                    self.defs_by_line[ref.line] = []
                self.defs_by_line[ref.line].append(ref)
            if ref.ref_type in ("use", "update"):
                if ref.line not in self.uses_by_line:
                    self.uses_by_line[ref.line] = []
                self.uses_by_line[ref.line].append(ref)

    def compute_def_use_chains(self) -> list[DataflowEdge]:
        """
        Compute def-use chains using reaching definitions.

        Simplified algorithm:
        1. Process lines in order
        2. Track which definition of each var is "active"
        3. When we see a use, link to active definition
        4. When we see a def, it becomes the active definition
        """
        edges: list[DataflowEdge] = []

        # Active definitions: var_name -> VarRef
        active_defs: dict[str, list[VarRef]] = {}

        # Get all lines in order
        all_lines = sorted(set(self.defs_by_line.keys()) | set(self.uses_by_line.keys()))

        for line in all_lines:
            # First process uses at this line
            if line in self.uses_by_line:
                for use_ref in self.uses_by_line[line]:
                    # Link to active definitions of this variable
                    if use_ref.name in active_defs:
                        for def_ref in active_defs[use_ref.name]:
                            edges.append(DataflowEdge(def_ref=def_ref, use_ref=use_ref))

            # Then process definitions at this line (kill old defs)
            if line in self.defs_by_line:
                for def_ref in self.defs_by_line[line]:
                    # New definition kills old ones (for now, simple case)
                    # In reality, we'd need CFG for proper reaching defs
                    active_defs[def_ref.name] = [def_ref]

        return edges


class CFGReachingDefsAnalyzer:
    """
    CFG-based reaching definitions analysis.

    Uses worklist algorithm on CFG blocks to properly handle branches.
    Definitions from both if/else branches reach uses after merge point.
    """

    def __init__(self, refs: list[VarRef], cfg):
        self.refs = refs
        self.cfg = cfg

        # Group refs by line
        self.defs_by_line: dict[int, list[VarRef]] = {}
        self.uses_by_line: dict[int, list[VarRef]] = {}

        for ref in refs:
            if ref.ref_type in ("definition", "update"):
                if ref.line not in self.defs_by_line:
                    self.defs_by_line[ref.line] = []
                self.defs_by_line[ref.line].append(ref)
            if ref.ref_type in ("use", "update"):
                if ref.line not in self.uses_by_line:
                    self.uses_by_line[ref.line] = []
                self.uses_by_line[ref.line].append(ref)

        # Build block adjacency for predecessor lookup
        self.predecessors: dict[int, list[int]] = {b.id: [] for b in cfg.blocks}
        for edge in cfg.edges:
            self.predecessors[edge.target_id].append(edge.source_id)

    def _get_block_lines(self, block) -> range:
        """Get line range covered by a block."""
        return range(block.start_line, block.end_line + 1)

    def compute_def_use_chains(self) -> list[DataflowEdge]:
        """
        Compute def-use chains using CFG-based reaching definitions.

        Algorithm:
        1. Initialize reaching_in[block] = {} for all blocks
        2. For entry block, reaching_in = parameter defs
        3. Iterate until fixed point:
           - reaching_in[block] = merge(reaching_out[pred] for pred in predecessors)
           - reaching_out[block] = gen[block] ∪ (reaching_in[block] - kill[block])
        4. Build edges from reaching defs to uses
        """
        edges: list[DataflowEdge] = []

        # reaching_in[block_id] = {var_name: [VarRef, ...]}
        reaching_in: dict[int, dict[str, list[VarRef]]] = {
            b.id: {} for b in self.cfg.blocks
        }
        reaching_out: dict[int, dict[str, list[VarRef]]] = {
            b.id: {} for b in self.cfg.blocks
        }

        # Compute gen/kill sets for each block
        # Key: assign each line to exactly ONE block (the smallest/most specific)
        gen: dict[int, dict[str, list[VarRef]]] = {b.id: {} for b in self.cfg.blocks}
        kill: dict[int, set[str]] = {b.id: set() for b in self.cfg.blocks}

        # Map each line to its owning block (smallest range wins, but prefer non-exit blocks)
        line_to_block: dict[int, int] = {}
        for block in self.cfg.blocks:
            for line in self._get_block_lines(block):
                if line not in line_to_block:
                    line_to_block[line] = block.id
                else:
                    existing = next(b for b in self.cfg.blocks if b.id == line_to_block[line])
                    # Skip exit blocks without predecessors for line ownership
                    if block.block_type == "exit" and block.id not in self.predecessors:
                        continue
                    if block.block_type == "exit" and not self.predecessors.get(block.id):
                        continue
                    # Prefer the block with smaller line range (more specific)
                    existing_range = existing.end_line - existing.start_line
                    new_range = block.end_line - block.start_line
                    if new_range < existing_range:
                        line_to_block[line] = block.id

        for block in self.cfg.blocks:
            for line in self._get_block_lines(block):
                # Only count this line's defs if this block owns the line
                if line_to_block.get(line) != block.id:
                    continue
                if line in self.defs_by_line:
                    for def_ref in self.defs_by_line[line]:
                        # Gen: this block generates this definition
                        gen[block.id][def_ref.name] = [def_ref]
                        # Kill: this block kills prior definitions of same var
                        kill[block.id].add(def_ref.name)

        # Worklist algorithm - iterate until fixed point
        changed = True
        max_iterations = 100  # Safety limit
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for block in self.cfg.blocks:
                # Compute reaching_in = merge of predecessor reaching_out
                new_reaching_in: dict[str, list[VarRef]] = {}

                for pred_id in self.predecessors[block.id]:
                    for var_name, defs in reaching_out[pred_id].items():
                        if var_name not in new_reaching_in:
                            new_reaching_in[var_name] = []
                        # Merge: add all reaching defs (may have duplicates)
                        for d in defs:
                            if d not in new_reaching_in[var_name]:
                                new_reaching_in[var_name].append(d)

                # Compute reaching_out = gen ∪ (reaching_in - kill)
                new_reaching_out: dict[str, list[VarRef]] = {}

                # First: pass through defs that aren't killed
                for var_name, defs in new_reaching_in.items():
                    if var_name not in kill[block.id]:
                        new_reaching_out[var_name] = defs.copy()

                # Then: add generated defs (these override)
                for var_name, defs in gen[block.id].items():
                    new_reaching_out[var_name] = defs.copy()

                # Check if changed
                if new_reaching_in != reaching_in[block.id]:
                    changed = True
                    reaching_in[block.id] = new_reaching_in

                if new_reaching_out != reaching_out[block.id]:
                    changed = True
                    reaching_out[block.id] = new_reaching_out

        # Build edges from reaching defs to uses
        for block in self.cfg.blocks:
            # Get reaching defs at start of block - DEEP COPY
            block_reaching = {k: list(v) for k, v in reaching_in[block.id].items()}

            for line in self._get_block_lines(block):
                # Only process lines that this block owns
                if line_to_block.get(line) != block.id:
                    continue

                # Process uses - link to reaching defs
                if line in self.uses_by_line:
                    for use_ref in self.uses_by_line[line]:
                        if use_ref.name in block_reaching:
                            for def_ref in block_reaching[use_ref.name]:
                                edges.append(DataflowEdge(
                                    def_ref=def_ref,
                                    use_ref=use_ref
                                ))

                # Process defs - update reaching for subsequent lines
                if line in self.defs_by_line:
                    for def_ref in self.defs_by_line[line]:
                        block_reaching[def_ref.name] = [def_ref]

        return edges


def extract_python_dfg(code: str, function_name: str) -> DFGInfo:
    """
    Extract DFG for a Python function using CFG-based reaching definitions.

    This properly handles control flow - definitions from both if/else
    branches will reach uses after the merge point.

    Args:
        code: Python source code
        function_name: Name of function to analyze

    Returns:
        DFGInfo with variable references and def-use chains
    """
    from tldr.cfg_extractor import extract_python_cfg

    tree = ast.parse(code)

    # Find the function
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                func_node = node
                break

    if func_node is None:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Extract definitions and uses
    visitor = PythonDefUseVisitor()
    visitor.visit(func_node)

    # Get CFG for the function and compute CFG-aware def-use chains
    try:
        cfg = extract_python_cfg(code, function_name)
        analyzer = CFGReachingDefsAnalyzer(visitor.refs, cfg)
    except ValueError:
        # Fall back to simple analysis if CFG extraction fails
        analyzer = PythonReachingDefsAnalyzer(visitor.refs)

    edges = analyzer.compute_def_use_chains()

    return DFGInfo(
        function_name=function_name,
        var_refs=visitor.refs,
        dataflow_edges=edges,
    )


def extract_python_dfg_with_cfg(code: str, function_name: str) -> DFGInfo:
    """
    Extract DFG for a Python function using CFG-based reaching definitions.

    This properly handles control flow - definitions from both if/else
    branches will reach uses after the merge point.

    Args:
        code: Python source code
        function_name: Name of function to analyze

    Returns:
        DFGInfo with variable references and def-use chains
    """
    from tldr.cfg_extractor import extract_python_cfg

    tree = ast.parse(code)

    # Find the function
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                func_node = node
                break

    if func_node is None:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Extract definitions and uses
    visitor = PythonDefUseVisitor()
    visitor.visit(func_node)

    # Get CFG for the function
    try:
        cfg = extract_python_cfg(code, function_name)
    except ValueError:
        # Fall back to simple analysis if CFG extraction fails
        analyzer = PythonReachingDefsAnalyzer(visitor.refs)
        edges = analyzer.compute_def_use_chains()
        return DFGInfo(
            function_name=function_name,
            var_refs=visitor.refs,
            dataflow_edges=edges,
        )

    # Compute def-use chains using CFG
    analyzer = CFGReachingDefsAnalyzer(visitor.refs, cfg)
    edges = analyzer.compute_def_use_chains()

    return DFGInfo(
        function_name=function_name,
        var_refs=visitor.refs,
        dataflow_edges=edges,
    )


# =============================================================================
# Tree-sitter based DFG extraction (TypeScript, Go, Rust)
# =============================================================================

# Tree-sitter imports (optional)
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

TREE_SITTER_JAVA_AVAILABLE = False
try:
    import tree_sitter_java
    TREE_SITTER_JAVA_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_C_AVAILABLE = False
try:
    import tree_sitter_c
    TREE_SITTER_C_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_RUBY_AVAILABLE = False
try:
    import tree_sitter_ruby
    TREE_SITTER_RUBY_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_PHP_AVAILABLE = False
try:
    import tree_sitter_php
    TREE_SITTER_PHP_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_CPP_AVAILABLE = False
try:
    import tree_sitter_cpp
    TREE_SITTER_CPP_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_SWIFT_AVAILABLE = False
try:
    import tree_sitter_swift
    TREE_SITTER_SWIFT_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_CSHARP_AVAILABLE = False
try:
    import tree_sitter_c_sharp
    TREE_SITTER_CSHARP_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_KOTLIN_AVAILABLE = False
try:
    import tree_sitter_kotlin
    TREE_SITTER_KOTLIN_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_SCALA_AVAILABLE = False
try:
    import tree_sitter_scala
    TREE_SITTER_SCALA_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_LUA_AVAILABLE = False
try:
    import tree_sitter_lua
    TREE_SITTER_LUA_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_ELIXIR_AVAILABLE = False
try:
    import tree_sitter_elixir
    TREE_SITTER_ELIXIR_AVAILABLE = True
except ImportError:
    pass


class TreeSitterDefUseVisitor:
    """
    Extract variable definitions and uses from tree-sitter parse tree.

    Works for TypeScript/JavaScript, Go, and Rust.
    """

    # Assignment node types per language
    ASSIGNMENT_TYPES = {
        # TypeScript/JavaScript
        "variable_declaration",
        "lexical_declaration",
        "assignment_expression",
        "augmented_assignment_expression",
        # Go
        "short_var_declaration",
        "assignment_statement",
        "var_declaration",
        # Rust
        "let_declaration",
        # Java
        "local_variable_declaration",
        "field_declaration",
        # Ruby
        "assignment",
        "operator_assignment",
        # Swift
        "property_declaration",
        "value_binding_pattern",
        "directly_assignable_expression",
        # Kotlin
        "variable_declaration",  # val/var
    }

    # Parameter node types
    PARAM_TYPES = {
        "formal_parameters",
        "parameter_list",
        "parameters",
        "required_parameter",
        "optional_parameter",
        "rest_parameter",
        "parameter_declaration",
        "parameter",  # Swift
    }

    # Identifier types
    IDENTIFIER_TYPES = {
        "identifier",
        "property_identifier",
        "shorthand_property_identifier",
        "shorthand_property_identifier_pattern",
        "simple_identifier",  # Swift
    }

    def __init__(self, source: bytes, language: str):
        self.source = source
        self.language = language
        self.refs: list[VarRef] = []

    def get_node_text(self, node) -> str:
        """Get source text for a node."""
        return self.source[node.start_byte:node.end_byte].decode('utf-8')

    def visit(self, node):
        """Visit a node and its children."""
        self._visit_node(node)

    def _visit_node(self, node):
        """Process a node based on its type."""
        node_type = node.type

        # Handle assignments
        if node_type in self.ASSIGNMENT_TYPES:
            self._handle_assignment(node)
            return

        # Handle parameters (definitions)
        if node_type in self.PARAM_TYPES:
            self._handle_parameters(node)
            return

        # Handle for-loop (iterator is definition)
        if node_type in ("for_statement", "for_in_statement", "for_of_statement"):
            self._handle_for_loop(node)
            return

        # Handle identifiers (uses) - but not when they're being defined
        if node_type in self.IDENTIFIER_TYPES:
            # Only track as use if we're reading, not defining
            parent = node.parent
            if parent and not self._is_definition_context(parent, node):
                name = self.get_node_text(node)
                if self._is_valid_var_name(name):
                    self._add_ref(name, "use", node)
            return

        # Recurse into children
        for child in node.children:
            self._visit_node(child)

    def _is_definition_context(self, parent, node) -> bool:
        """Check if node is being defined (not used)."""
        parent_type = parent.type

        # Assignment targets
        if parent_type == "assignment_expression":
            # Left side of assignment
            if parent.child_by_field_name("left") == node:
                return True
        if parent_type == "augmented_assignment_expression":
            if parent.child_by_field_name("left") == node:
                return True
        if parent_type == "variable_declarator":
            # The name being declared
            if parent.child_by_field_name("name") == node:
                return True
        if parent_type == "short_var_declaration":
            # Go := declarations - left side
            left = parent.child_by_field_name("left")
            if left and self._node_contains(left, node):
                return True
        if parent_type == "let_declaration":
            # Rust let - pattern is definition
            pattern = parent.child_by_field_name("pattern")
            if pattern and self._node_contains(pattern, node):
                return True
        # Ruby: assignment and operator_assignment
        if parent_type == "assignment":
            # Left side of assignment (first child is the target identifier)
            left = parent.child_by_field_name("left")
            if left and self._node_contains(left, node):
                return True
            # Sometimes the left side is not a field, check first child
            if parent.children and len(parent.children) > 0:
                if parent.children[0] == node or self._node_contains(parent.children[0], node):
                    return True
        if parent_type == "operator_assignment":
            # Left side of augmented assignment
            left = parent.child_by_field_name("left")
            if left and self._node_contains(left, node):
                return True
            if parent.children and len(parent.children) > 0:
                if parent.children[0] == node or self._node_contains(parent.children[0], node):
                    return True

        return False

    def _node_contains(self, parent, target) -> bool:
        """Check if parent contains target node."""
        if parent == target:
            return True
        for child in parent.children:
            if self._node_contains(child, target):
                return True
        return False

    def _is_valid_var_name(self, name: str) -> bool:
        """Check if name is a valid variable name (not keyword, etc.)."""
        keywords = {
            "if", "else", "for", "while", "return", "function", "const", "let", "var",
            "true", "false", "null", "undefined", "this", "super", "new",
            "func", "package", "import", "type", "struct", "interface",
            "fn", "pub", "mod", "use", "impl", "trait", "struct", "enum",
            "self", "Self", "mut", "ref", "match", "loop", "break", "continue",
        }
        return name not in keywords and not name.startswith("_")

    def _add_ref(self, name: str, ref_type: str, node):
        """Add a variable reference."""
        ref = VarRef(
            name=name,
            ref_type=ref_type,
            line=node.start_point[0] + 1,  # tree-sitter is 0-indexed
            column=node.start_point[1],
        )
        self.refs.append(ref)

    def _handle_assignment(self, node):
        """Handle assignment nodes."""
        node_type = node.type

        if node_type == "variable_declaration" or node_type == "lexical_declaration":
            # TypeScript: const x = 1 or let x = 1
            for child in node.children:
                if child.type == "variable_declarator":
                    name_node = child.child_by_field_name("name")
                    value_node = child.child_by_field_name("value")

                    # Visit value first (uses)
                    if value_node:
                        self._visit_node(value_node)

                    # Then add definition
                    if name_node and name_node.type in self.IDENTIFIER_TYPES:
                        self._add_ref(self.get_node_text(name_node), "definition", name_node)

        elif node_type == "assignment_expression":
            # x = y
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")

            # Visit right side first (uses)
            if right:
                self._visit_node(right)

            # Add definition for left side
            if left and left.type in self.IDENTIFIER_TYPES:
                self._add_ref(self.get_node_text(left), "definition", left)

        elif node_type == "augmented_assignment_expression":
            # x += y
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")

            # Visit right side first (uses)
            if right:
                self._visit_node(right)

            # Left is both used and updated
            if left and left.type in self.IDENTIFIER_TYPES:
                name = self.get_node_text(left)
                self._add_ref(name, "use", left)
                self._add_ref(name, "update", left)

        elif node_type == "short_var_declaration":
            # Go: x := 1
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")

            # Visit right first (uses)
            if right:
                self._visit_node(right)

            # Add definitions for left side
            if left:
                for child in left.children:
                    if child.type == "identifier":
                        self._add_ref(self.get_node_text(child), "definition", child)

        elif node_type == "assignment_statement":
            # Go: x = 1
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")

            # Visit right first (uses)
            if right:
                self._visit_node(right)

            # Add definitions for left side
            if left:
                for child in left.children:
                    if child.type == "identifier":
                        self._add_ref(self.get_node_text(child), "definition", child)

        elif node_type == "let_declaration":
            # Rust: let x = 1
            pattern = node.child_by_field_name("pattern")
            value = node.child_by_field_name("value")

            # Visit value first (uses)
            if value:
                self._visit_node(value)

            # Add definition for pattern
            if pattern:
                self._extract_pattern_names(pattern, "definition")

        elif node_type == "assignment":
            # Ruby: x = y (first child is identifier, then =, then value)
            if len(node.children) >= 3:
                left = node.children[0]  # identifier
                # Skip = operator
                right = node.children[2] if len(node.children) > 2 else None

                # Visit right side first (uses)
                if right:
                    self._visit_node(right)

                # Add definition for left side
                if left.type == "identifier":
                    self._add_ref(self.get_node_text(left), "definition", left)

        elif node_type == "operator_assignment":
            # Ruby: x += y (first child is identifier, then +=, then value)
            if len(node.children) >= 3:
                left = node.children[0]  # identifier
                # Skip operator
                right = node.children[2] if len(node.children) > 2 else None

                # Visit right side first (uses)
                if right:
                    self._visit_node(right)

                # Left is both used and updated
                if left.type == "identifier":
                    name = self.get_node_text(left)
                    self._add_ref(name, "use", left)
                    self._add_ref(name, "update", left)

    def _extract_pattern_names(self, pattern, ref_type: str):
        """Extract variable names from a pattern (Rust, destructuring)."""
        if pattern.type == "identifier":
            self._add_ref(self.get_node_text(pattern), ref_type, pattern)
        else:
            for child in pattern.children:
                self._extract_pattern_names(child, ref_type)

    def _handle_parameters(self, node):
        """Handle function parameters as definitions."""
        for child in node.children:
            if child.type == "identifier":
                self._add_ref(self.get_node_text(child), "definition", child)
            elif child.type in ("required_parameter", "optional_parameter",
                               "parameter_declaration", "parameter"):
                # Look for identifier inside
                for inner in child.children:
                    if inner.type == "identifier":
                        self._add_ref(self.get_node_text(inner), "definition", inner)
                        break
            else:
                # Recurse for nested structures
                self._handle_parameters(child)

    def _handle_for_loop(self, node):
        """Handle for loop - iterator variable is definition."""
        # TypeScript/JavaScript: for (let i = 0; ...) or for (x of arr)
        # Go: for i, v := range arr

        for child in node.children:
            if child.type in ("lexical_declaration", "variable_declaration"):
                self._handle_assignment(child)
            elif child.type == "for_clause":
                # Go for clause
                init = child.child_by_field_name("initializer")
                if init:
                    self._visit_node(init)
            elif child.type in ("block", "statement_block"):
                # Body
                self._visit_node(child)
            else:
                # Other parts (condition, update, iterable)
                self._visit_node(child)


def extract_typescript_dfg(code: str, function_name: str) -> DFGInfo:
    """
    Extract DFG for a TypeScript/JavaScript function.

    Args:
        code: TypeScript/JavaScript source code
        function_name: Name of function to analyze

    Returns:
        DFGInfo with variable references and def-use chains
    """
    if not TREE_SITTER_AVAILABLE:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Parse with tree-sitter
    ts_lang = Language(tree_sitter_typescript.language_typescript())
    parser = Parser(ts_lang)
    source_bytes = code.encode('utf-8')
    tree = parser.parse(source_bytes)

    # Find the function
    func_node = _find_function_by_name(tree.root_node, function_name, source_bytes)
    if func_node is None:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Extract definitions and uses
    visitor = TreeSitterDefUseVisitor(source_bytes, "typescript")
    visitor.visit(func_node)

    # Compute def-use chains
    analyzer = PythonReachingDefsAnalyzer(visitor.refs)  # Same algorithm works
    edges = analyzer.compute_def_use_chains()

    return DFGInfo(
        function_name=function_name,
        var_refs=visitor.refs,
        dataflow_edges=edges,
    )


def extract_go_dfg(code: str, function_name: str) -> DFGInfo:
    """
    Extract DFG for a Go function.

    Args:
        code: Go source code
        function_name: Name of function to analyze

    Returns:
        DFGInfo with variable references and def-use chains
    """
    if not TREE_SITTER_GO_AVAILABLE:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Parse with tree-sitter
    go_lang = Language(tree_sitter_go.language())
    parser = Parser(go_lang)
    source_bytes = code.encode('utf-8')
    tree = parser.parse(source_bytes)

    # Find the function
    func_node = _find_function_by_name(tree.root_node, function_name, source_bytes)
    if func_node is None:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Extract definitions and uses
    visitor = TreeSitterDefUseVisitor(source_bytes, "go")
    visitor.visit(func_node)

    # Compute def-use chains
    analyzer = PythonReachingDefsAnalyzer(visitor.refs)
    edges = analyzer.compute_def_use_chains()

    return DFGInfo(
        function_name=function_name,
        var_refs=visitor.refs,
        dataflow_edges=edges,
    )


def extract_rust_dfg(code: str, function_name: str) -> DFGInfo:
    """
    Extract DFG for a Rust function.

    Args:
        code: Rust source code
        function_name: Name of function to analyze

    Returns:
        DFGInfo with variable references and def-use chains
    """
    if not TREE_SITTER_RUST_AVAILABLE:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Parse with tree-sitter
    rust_lang = Language(tree_sitter_rust.language())
    parser = Parser(rust_lang)
    source_bytes = code.encode('utf-8')
    tree = parser.parse(source_bytes)

    # Find the function
    func_node = _find_function_by_name(tree.root_node, function_name, source_bytes)
    if func_node is None:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Extract definitions and uses
    visitor = TreeSitterDefUseVisitor(source_bytes, "rust")
    visitor.visit(func_node)

    # Compute def-use chains
    analyzer = PythonReachingDefsAnalyzer(visitor.refs)
    edges = analyzer.compute_def_use_chains()

    return DFGInfo(
        function_name=function_name,
        var_refs=visitor.refs,
        dataflow_edges=edges,
    )


def extract_java_dfg(code: str, function_name: str) -> DFGInfo:
    """
    Extract DFG for a Java method.

    Args:
        code: Java source code
        function_name: Name of method to analyze

    Returns:
        DFGInfo with variable references and def-use chains
    """
    if not TREE_SITTER_JAVA_AVAILABLE:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Parse with tree-sitter
    java_lang = Language(tree_sitter_java.language())
    parser = Parser(java_lang)
    source_bytes = code.encode('utf-8')
    tree = parser.parse(source_bytes)

    # Find the method
    func_node = _find_function_by_name(tree.root_node, function_name, source_bytes)
    if func_node is None:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Extract definitions and uses
    visitor = TreeSitterDefUseVisitor(source_bytes, "java")
    visitor.visit(func_node)

    # Compute def-use chains
    analyzer = PythonReachingDefsAnalyzer(visitor.refs)
    edges = analyzer.compute_def_use_chains()

    return DFGInfo(
        function_name=function_name,
        var_refs=visitor.refs,
        dataflow_edges=edges,
    )


def extract_c_dfg(code: str, function_name: str) -> DFGInfo:
    """
    Extract DFG for a C function.

    Args:
        code: C source code
        function_name: Name of function to analyze

    Returns:
        DFGInfo with variable references and def-use chains
    """
    if not TREE_SITTER_C_AVAILABLE:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Parse with tree-sitter
    c_lang = Language(tree_sitter_c.language())
    parser = Parser(c_lang)
    source_bytes = code.encode('utf-8')
    tree = parser.parse(source_bytes)

    # Find the function using C-specific logic
    func_node = _find_c_function_by_name(tree.root_node, function_name, source_bytes)
    if func_node is None:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Extract definitions and uses
    visitor = TreeSitterDefUseVisitor(source_bytes, "c")
    visitor.visit(func_node)

    # Compute def-use chains
    analyzer = PythonReachingDefsAnalyzer(visitor.refs)
    edges = analyzer.compute_def_use_chains()

    return DFGInfo(
        function_name=function_name,
        var_refs=visitor.refs,
        dataflow_edges=edges,
    )


def _find_c_function_by_name(root, name: str, source: bytes):
    """Find a C function node by name in tree-sitter tree."""
    def search(node):
        if node.type == "function_definition":
            declarator = node.child_by_field_name("declarator")
            # Handle pointer_declarator wrapping function_declarator
            if declarator and declarator.type == "pointer_declarator":
                for child in declarator.children:
                    if child.type == "function_declarator":
                        declarator = child
                        break
            if declarator and declarator.type == "function_declarator":
                inner_decl = declarator.child_by_field_name("declarator")
                if inner_decl and inner_decl.type == "identifier":
                    func_name = source[inner_decl.start_byte:inner_decl.end_byte].decode('utf-8')
                    if func_name == name:
                        return node

        for child in node.children:
            result = search(child)
            if result:
                return result
        return None

    return search(root)


def extract_cpp_dfg(code: str, function_name: str) -> DFGInfo:
    """
    Extract DFG for a C++ function.

    Args:
        code: C++ source code
        function_name: Name of function to analyze

    Returns:
        DFGInfo with variable references and def-use chains
    """
    if not TREE_SITTER_CPP_AVAILABLE:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Parse with tree-sitter
    cpp_lang = Language(tree_sitter_cpp.language())
    parser = Parser(cpp_lang)
    source_bytes = code.encode('utf-8')
    tree = parser.parse(source_bytes)

    # Find the function using C++-specific logic (same as C)
    func_node = _find_cpp_function_by_name(tree.root_node, function_name, source_bytes)
    if func_node is None:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Extract definitions and uses
    visitor = TreeSitterDefUseVisitor(source_bytes, "cpp")
    visitor.visit(func_node)

    # Compute def-use chains
    analyzer = PythonReachingDefsAnalyzer(visitor.refs)
    edges = analyzer.compute_def_use_chains()

    return DFGInfo(
        function_name=function_name,
        var_refs=visitor.refs,
        dataflow_edges=edges,
    )


def _find_cpp_function_by_name(root, name: str, source: bytes):
    """Find a C++ function node by name in tree-sitter tree.

    Handles both standalone functions (identifier) and class methods (field_identifier).
    """
    def search(node):
        if node.type == "function_definition":
            declarator = node.child_by_field_name("declarator")
            # Handle pointer_declarator wrapping function_declarator
            if declarator and declarator.type == "pointer_declarator":
                for child in declarator.children:
                    if child.type == "function_declarator":
                        declarator = child
                        break
            if declarator and declarator.type == "function_declarator":
                inner_decl = declarator.child_by_field_name("declarator")
                # Check both identifier (standalone functions) and field_identifier (class methods)
                if inner_decl and inner_decl.type in ("identifier", "field_identifier"):
                    func_name = source[inner_decl.start_byte:inner_decl.end_byte].decode('utf-8')
                    if func_name == name:
                        return node

        for child in node.children:
            result = search(child)
            if result:
                return result
        return None

    return search(root)


def extract_ruby_dfg(code: str, function_name: str) -> DFGInfo:
    """
    Extract DFG for a Ruby function.

    Args:
        code: Ruby source code
        function_name: Name of method to analyze

    Returns:
        DFGInfo with variable references and def-use chains
    """
    if not TREE_SITTER_RUBY_AVAILABLE:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Parse with tree-sitter
    ruby_lang = Language(tree_sitter_ruby.language())
    parser = Parser(ruby_lang)
    source_bytes = code.encode('utf-8')
    tree = parser.parse(source_bytes)

    # Find the function
    func_node = _find_ruby_function_by_name(tree.root_node, function_name, source_bytes)
    if func_node is None:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Extract definitions and uses
    visitor = TreeSitterDefUseVisitor(source_bytes, "ruby")
    visitor.visit(func_node)

    # Compute def-use chains
    analyzer = PythonReachingDefsAnalyzer(visitor.refs)
    edges = analyzer.compute_def_use_chains()

    return DFGInfo(
        function_name=function_name,
        var_refs=visitor.refs,
        dataflow_edges=edges,
    )


def _find_ruby_function_by_name(root, name: str, source: bytes):
    """Find a Ruby method node by name in tree-sitter tree."""
    def search(node):
        if node.type == "method":
            # Get the method name from the name field
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = source[name_node.start_byte:name_node.end_byte].decode('utf-8')
                if func_name == name:
                    return node

        for child in node.children:
            result = search(child)
            if result:
                return result
        return None

    return search(root)


def _find_function_by_name(root, name: str, source: bytes):
    """Find a function node by name in tree-sitter tree."""
    FUNCTION_TYPES = {
        "function_declaration",
        "function_definition",
        "function_item",
        "method_definition",
        "method_declaration",  # Java
        "arrow_function",
        "method",  # Ruby: def method_name ... end
    }

    def search(node):
        if node.type in FUNCTION_TYPES:
            # Try to find the function name
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = source[name_node.start_byte:name_node.end_byte].decode('utf-8')
                if func_name == name:
                    return node

        for child in node.children:
            result = search(child)
            if result:
                return result
        return None

    return search(root)


def extract_php_dfg(code: str, function_name: str) -> DFGInfo:
    """
    Extract DFG for a PHP function.

    Args:
        code: PHP source code (may include <?php tag)
        function_name: Name of function to analyze

    Returns:
        DFGInfo with variable references and def-use chains
    """
    if not TREE_SITTER_PHP_AVAILABLE:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Parse with tree-sitter
    php_lang = Language(tree_sitter_php.language_php())
    parser = Parser(php_lang)
    source_bytes = code.encode('utf-8')
    tree = parser.parse(source_bytes)

    # Find the function
    func_node = _find_php_function_by_name(tree.root_node, function_name, source_bytes)
    if func_node is None:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Extract definitions and uses
    visitor = PHPDefUseVisitor(source_bytes)
    visitor.visit(func_node)

    # Compute def-use chains
    analyzer = PythonReachingDefsAnalyzer(visitor.refs)
    edges = analyzer.compute_def_use_chains()

    return DFGInfo(
        function_name=function_name,
        var_refs=visitor.refs,
        dataflow_edges=edges,
    )


def _find_php_function_by_name(root, name: str, source: bytes):
    """Find a PHP function node by name in tree-sitter tree."""
    def search(node):
        # PHP function_definition has name child
        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = source[name_node.start_byte:name_node.end_byte].decode('utf-8')
                if func_name == name:
                    return node

        # Also check method_declaration for class methods
        if node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = source[name_node.start_byte:name_node.end_byte].decode('utf-8')
                if func_name == name:
                    return node

        for child in node.children:
            result = search(child)
            if result:
                return result
        return None

    return search(root)


class PHPDefUseVisitor:
    """
    Extract variable definitions and uses from PHP tree-sitter parse tree.

    PHP variables start with $, e.g., $x, $total, $result.
    """

    def __init__(self, source: bytes):
        self.source = source
        self.refs: list[VarRef] = []

    def get_node_text(self, node) -> str:
        """Get source text for a node."""
        return self.source[node.start_byte:node.end_byte].decode('utf-8')

    def visit(self, node):
        """Visit a node and its children."""
        self._visit_node(node)

    def _visit_node(self, node):
        """Process a node based on its type."""
        node_type = node.type

        # Handle assignments: $x = ...
        if node_type == "assignment_expression":
            self._handle_assignment(node)
            return

        # Handle augmented assignments: $x += ...
        if node_type == "augmented_assignment_expression":
            self._handle_augmented_assignment(node)
            return

        # Handle function parameters (definitions)
        if node_type == "formal_parameters" or node_type == "simple_parameter":
            self._handle_parameters(node)
            return

        # Handle foreach as both definition and use
        if node_type == "foreach_statement":
            self._handle_foreach(node)
            return

        # Handle variable_name (uses)
        if node_type == "variable_name":
            # Check if this is a use context (not an assignment target)
            parent = node.parent
            if parent and not self._is_definition_context(parent, node):
                name = self.get_node_text(node)
                # Strip $ prefix if present for consistency
                if name.startswith('$'):
                    name = name[1:]
                self._add_ref(name, "use", node)
            return

        # Recurse into children
        for child in node.children:
            self._visit_node(child)

    def _is_definition_context(self, parent, node) -> bool:
        """Check if node is being defined (not used)."""
        parent_type = parent.type

        # Left side of assignment
        if parent_type == "assignment_expression":
            left = parent.child_by_field_name("left")
            if left == node:
                return True
        if parent_type == "augmented_assignment_expression":
            left = parent.child_by_field_name("left")
            if left == node:
                return True

        return False

    def _add_ref(self, name: str, ref_type: str, node):
        """Add a variable reference."""
        ref = VarRef(
            name=name,
            ref_type=ref_type,
            line=node.start_point[0] + 1,  # tree-sitter is 0-indexed
            column=node.start_point[1],
        )
        self.refs.append(ref)

    def _handle_assignment(self, node):
        """Handle PHP assignment: $x = ..."""
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")

        # Visit right side first (uses)
        if right:
            self._visit_node(right)

        # Add definition for left side
        if left and left.type == "variable_name":
            name = self.get_node_text(left)
            if name.startswith('$'):
                name = name[1:]
            self._add_ref(name, "definition", left)

    def _handle_augmented_assignment(self, node):
        """Handle PHP augmented assignment: $x += ..."""
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")

        # Visit right side first (uses)
        if right:
            self._visit_node(right)

        # Left is both used and updated
        if left and left.type == "variable_name":
            name = self.get_node_text(left)
            if name.startswith('$'):
                name = name[1:]
            self._add_ref(name, "use", left)
            self._add_ref(name, "update", left)

    def _handle_parameters(self, node):
        """Handle PHP function parameters."""
        if node.type == "formal_parameters":
            for child in node.children:
                if child.type == "simple_parameter":
                    self._handle_parameters(child)
        elif node.type == "simple_parameter":
            # Find variable_name child
            for child in node.children:
                if child.type == "variable_name":
                    name = self.get_node_text(child)
                    if name.startswith('$'):
                        name = name[1:]
                    self._add_ref(name, "definition", child)
                    break

    def _handle_foreach(self, node):
        """Handle PHP foreach loop: foreach ($items as $key => $value)."""
        # The 'as' clause defines variables
        for child in node.children:
            if child.type == "foreach_clause":
                # foreach_clause has the iteration variable definitions
                for fc_child in child.children:
                    if fc_child.type == "pair":
                        # $key => $value pattern
                        for pair_child in fc_child.children:
                            if pair_child.type == "variable_name":
                                name = self.get_node_text(pair_child)
                                if name.startswith('$'):
                                    name = name[1:]
                                self._add_ref(name, "definition", pair_child)
                    elif fc_child.type == "variable_name":
                        # Simple $value pattern
                        name = self.get_node_text(fc_child)
                        if name.startswith('$'):
                            name = name[1:]
                        self._add_ref(name, "definition", fc_child)

        # Visit the body
        body = node.child_by_field_name("body")
        if body:
            self._visit_node(body)


def extract_swift_dfg(code: str, function_name: str) -> DFGInfo:
    """
    Extract DFG for a Swift function.

    Args:
        code: Swift source code
        function_name: Name of function to analyze

    Returns:
        DFGInfo with variable references and def-use chains
    """
    if not TREE_SITTER_SWIFT_AVAILABLE:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Parse with tree-sitter
    swift_lang = Language(tree_sitter_swift.language())
    parser = Parser(swift_lang)
    source_bytes = code.encode('utf-8')
    tree = parser.parse(source_bytes)

    # Find the function
    func_node = _find_swift_function_by_name(tree.root_node, function_name, source_bytes)
    if func_node is None:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Extract definitions and uses
    visitor = SwiftDefUseVisitor(source_bytes)
    visitor.visit(func_node)

    # Compute def-use chains
    analyzer = PythonReachingDefsAnalyzer(visitor.refs)
    edges = analyzer.compute_def_use_chains()

    return DFGInfo(
        function_name=function_name,
        var_refs=visitor.refs,
        dataflow_edges=edges,
    )


def _find_swift_function_by_name(root, name: str, source: bytes):
    """Find a Swift function node by name in tree-sitter tree."""
    def search(node):
        if node.type == "function_declaration":
            # Get the function name from the name field
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = source[name_node.start_byte:name_node.end_byte].decode('utf-8')
                if func_name == name:
                    return node

        for child in node.children:
            result = search(child)
            if result:
                return result
        return None

    return search(root)


class SwiftDefUseVisitor:
    """
    Extract variable definitions and uses from Swift tree-sitter parse tree.

    Swift variables are declared with let (immutable) or var (mutable).
    """

    def __init__(self, source: bytes):
        self.source = source
        self.refs: list[VarRef] = []

    def get_node_text(self, node) -> str:
        """Get source text for a node."""
        return self.source[node.start_byte:node.end_byte].decode('utf-8')

    def visit(self, node):
        """Visit a node and its children."""
        self._visit_node(node)

    def _visit_node(self, node):
        """Process a node based on its type."""
        node_type = node.type

        # Handle property declarations (let/var)
        if node_type == "property_declaration":
            self._handle_property_declaration(node)
            return

        # Handle direct assignments
        if node_type == "directly_assignable_expression":
            self._handle_assignment(node)
            return

        # Handle function parameters (definitions)
        if node_type == "parameter":
            self._handle_parameter(node)
            return

        # Handle simple_identifier (uses)
        if node_type == "simple_identifier":
            # Check if this is a use context (not a definition)
            parent = node.parent
            if parent and not self._is_definition_context(parent, node):
                name = self.get_node_text(node)
                self._add_ref(name, "use", node)
            return

        # Recurse into children
        for child in node.children:
            self._visit_node(child)

    def _is_definition_context(self, parent, node) -> bool:
        """Check if node is being defined (not used)."""
        parent_type = parent.type

        # Pattern in property_declaration
        if parent_type == "pattern":
            return True
        # Value binding pattern
        if parent_type == "value_binding_pattern":
            return True
        # Left side of assignment
        if parent_type == "directly_assignable_expression":
            # Check if node is the left side (first child)
            if parent.children and parent.children[0] == node:
                return True
        # Parameter name
        if parent_type == "parameter":
            return True

        return False

    def _add_ref(self, name: str, ref_type: str, node):
        """Add a variable reference."""
        ref = VarRef(
            name=name,
            ref_type=ref_type,
            line=node.start_point[0] + 1,  # tree-sitter is 0-indexed
            column=node.start_point[1],
        )
        self.refs.append(ref)

    def _handle_property_declaration(self, node):
        """Handle Swift property declaration: let x = ... or var x = ..."""
        # Find pattern child which contains the identifier
        for child in node.children:
            if child.type == "pattern":
                # Pattern contains the identifier being defined
                for pattern_child in child.children:
                    if pattern_child.type == "simple_identifier":
                        name = self.get_node_text(pattern_child)
                        self._add_ref(name, "definition", pattern_child)
            elif child.type == "value_binding_pattern":
                # value_binding_pattern: "let x" or "var x"
                for vbp_child in child.children:
                    if vbp_child.type == "pattern":
                        for pattern_child in vbp_child.children:
                            if pattern_child.type == "simple_identifier":
                                name = self.get_node_text(pattern_child)
                                self._add_ref(name, "definition", pattern_child)
            # Visit value side for uses
            if child.type not in ("pattern", "value_binding_pattern", "type_annotation"):
                self._visit_node(child)

    def _handle_assignment(self, node):
        """Handle Swift assignment: x = ..."""
        children = node.children
        if len(children) >= 3:
            left = children[0]
            # Skip = operator
            right = children[2] if len(children) > 2 else None

            # Visit right side first (uses)
            if right:
                self._visit_node(right)

            # Add definition for left side
            if left.type == "simple_identifier":
                name = self.get_node_text(left)
                self._add_ref(name, "definition", left)

    def _handle_parameter(self, node):
        """Handle Swift function parameter."""
        # Find simple_identifier child for parameter name
        for child in node.children:
            if child.type == "simple_identifier":
                name = self.get_node_text(child)
                self._add_ref(name, "definition", child)
                break


def extract_csharp_dfg(code: str, function_name: str) -> DFGInfo:
    """
    Extract DFG for a C# method.

    Args:
        code: C# source code
        function_name: Name of method to analyze

    Returns:
        DFGInfo with variable references and def-use chains
    """
    if not TREE_SITTER_CSHARP_AVAILABLE:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Parse with tree-sitter
    csharp_lang = Language(tree_sitter_c_sharp.language())
    parser = Parser(csharp_lang)
    source_bytes = code.encode('utf-8')
    tree = parser.parse(source_bytes)

    # Find the method
    func_node = _find_csharp_function_by_name(tree.root_node, function_name, source_bytes)
    if func_node is None:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Extract definitions and uses
    visitor = CSharpDefUseVisitor(source_bytes)
    visitor.visit(func_node)

    # Compute def-use chains
    analyzer = PythonReachingDefsAnalyzer(visitor.refs)
    edges = analyzer.compute_def_use_chains()

    return DFGInfo(
        function_name=function_name,
        var_refs=visitor.refs,
        dataflow_edges=edges,
    )


def _find_csharp_function_by_name(root, name: str, source: bytes):
    """Find a C# method node by name in tree-sitter tree."""
    def search(node):
        if node.type == "method_declaration":
            # Get the method name from the name field
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = source[name_node.start_byte:name_node.end_byte].decode('utf-8')
                if func_name == name:
                    return node

        for child in node.children:
            result = search(child)
            if result:
                return result
        return None

    return search(root)


class CSharpDefUseVisitor:
    """
    Extract variable definitions and uses from C# tree-sitter parse tree.

    C# variables are declared with var or explicit types.
    """

    def __init__(self, source: bytes):
        self.source = source
        self.refs: list[VarRef] = []

    def get_node_text(self, node) -> str:
        """Get source text for a node."""
        return self.source[node.start_byte:node.end_byte].decode('utf-8')

    def visit(self, node):
        """Visit a node and its children."""
        self._visit_node(node)

    def _add_ref(self, name: str, ref_type: str, node):
        """Add a variable reference."""
        ref = VarRef(
            name=name,
            ref_type=ref_type,
            line=node.start_point[0] + 1,  # tree-sitter is 0-indexed
            column=node.start_point[1],
        )
        self.refs.append(ref)

    def _visit_node(self, node):
        """Process a node based on its type."""
        node_type = node.type

        # Handle variable declarations: int x = 5; or var x = 5;
        if node_type == "local_declaration_statement":
            self._handle_local_declaration(node)
            return

        # Handle assignments: x = 5;
        if node_type == "assignment_expression":
            self._handle_assignment(node)
            return

        # Handle parameters
        if node_type == "parameter":
            self._handle_parameter(node)
            return

        # Handle identifiers (uses) - but not when they're being defined
        if node_type == "identifier":
            parent = node.parent
            if parent and not self._is_definition_context(parent, node):
                name = self.get_node_text(node)
                if self._is_valid_var_name(name):
                    self._add_ref(name, "use", node)
            return

        # Recurse into children
        for child in node.children:
            self._visit_node(child)

    def _is_definition_context(self, parent, node) -> bool:
        """Check if node is being defined (not used)."""
        parent_type = parent.type

        # Variable declarator - the variable being declared
        if parent_type == "variable_declarator":
            # The name being declared
            name_child = parent.child_by_field_name("name")
            if name_child and name_child == node:
                return True

        # Assignment - left side is definition
        if parent_type == "assignment_expression":
            left = parent.child_by_field_name("left")
            if left and left == node:
                return True

        return False

    def _is_valid_var_name(self, name: str) -> bool:
        """Check if name is a valid variable name (not keyword, etc.)."""
        keywords = {
            "if", "else", "for", "foreach", "while", "return", "switch", "case",
            "true", "false", "null", "this", "base", "new", "class", "struct",
            "void", "int", "string", "bool", "var", "const", "static", "public",
            "private", "protected", "internal", "try", "catch", "finally",
            "throw", "break", "continue", "default", "using", "namespace",
        }
        return name not in keywords and not name.startswith("_")

    def _handle_local_declaration(self, node):
        """Handle C# local variable declaration: int x = 5; or var x = 5;"""
        for child in node.children:
            if child.type == "variable_declaration":
                for decl_child in child.children:
                    if decl_child.type == "variable_declarator":
                        # Get the variable name
                        name_node = decl_child.child_by_field_name("name")
                        if name_node:
                            # This is a definition
                            self._add_ref(self.get_node_text(name_node), "definition", name_node)
                        # Get the initializer and visit for uses
                        init = decl_child.child_by_field_name("initializer")
                        if init:
                            self._visit_node(init)

    def _handle_assignment(self, node):
        """Handle C# assignment: x = ..."""
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")

        # Visit right side first (uses)
        if right:
            self._visit_node(right)

        # Add definition for left side
        if left and left.type == "identifier":
            name = self.get_node_text(left)
            self._add_ref(name, "definition", left)

    def _handle_parameter(self, node):
        """Handle C# method parameter."""
        # Find the parameter name
        name_node = node.child_by_field_name("name")
        if name_node:
            name = self.get_node_text(name_node)
            self._add_ref(name, "definition", name_node)


def extract_kotlin_dfg(code: str, function_name: str) -> DFGInfo:
    """
    Extract DFG for a Kotlin function.

    Args:
        code: Kotlin source code
        function_name: Name of function to analyze

    Returns:
        DFGInfo with variable references and def-use chains
    """
    if not TREE_SITTER_KOTLIN_AVAILABLE:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Parse with tree-sitter
    kotlin_lang = Language(tree_sitter_kotlin.language())
    parser = Parser(kotlin_lang)
    source_bytes = code.encode('utf-8')
    tree = parser.parse(source_bytes)

    # Find the function
    func_node = _find_kotlin_function_by_name(tree.root_node, function_name, source_bytes)
    if func_node is None:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Extract definitions and uses
    visitor = KotlinDefUseVisitor(source_bytes)
    visitor.visit(func_node)

    # Compute def-use chains
    analyzer = PythonReachingDefsAnalyzer(visitor.refs)
    edges = analyzer.compute_def_use_chains()

    return DFGInfo(
        function_name=function_name,
        var_refs=visitor.refs,
        dataflow_edges=edges,
    )


def _find_kotlin_function_by_name(root, name: str, source: bytes):
    """Find a Kotlin function node by name in tree-sitter tree."""
    def search(node):
        if node.type == "function_declaration":
            # Get the function name from the identifier child
            for child in node.children:
                if child.type == "identifier":
                    func_name = source[child.start_byte:child.end_byte].decode('utf-8')
                    if func_name == name:
                        return node
                    break

        for child in node.children:
            result = search(child)
            if result:
                return result
        return None

    return search(root)


class KotlinDefUseVisitor:
    """
    Extract variable definitions and uses from Kotlin tree-sitter parse tree.

    Kotlin variables are declared with val or var.
    """

    def __init__(self, source: bytes):
        self.source = source
        self.refs: list[VarRef] = []

    def get_node_text(self, node) -> str:
        """Get source text for a node."""
        return self.source[node.start_byte:node.end_byte].decode('utf-8')

    def visit(self, node):
        """Visit a node and its children."""
        self._visit_node(node)

    def _add_ref(self, name: str, ref_type: str, node):
        """Add a variable reference."""
        ref = VarRef(
            name=name,
            ref_type=ref_type,
            line=node.start_point[0] + 1,  # tree-sitter is 0-indexed
            column=node.start_point[1],
        )
        self.refs.append(ref)

    def _visit_node(self, node):
        """Process a node based on its type."""
        node_type = node.type

        # Handle property/variable declarations: val x = ... or var x = ...
        if node_type == "property_declaration":
            self._handle_property_declaration(node)
            return

        # Handle assignments
        if node_type == "assignment":
            self._handle_assignment(node)
            return

        # Handle parameters
        if node_type == "parameter":
            self._handle_parameter(node)
            return

        # Handle identifier (uses) - tree-sitter-kotlin uses "identifier" not "simple_identifier"
        if node_type == "identifier":
            parent = node.parent
            if parent and not self._is_definition_context(parent, node):
                name = self.get_node_text(node)
                if self._is_valid_var_name(name):
                    self._add_ref(name, "use", node)
            return

        # Recurse into children
        for child in node.children:
            self._visit_node(child)

    def _is_definition_context(self, parent, node) -> bool:
        """Check if node is being defined (not used)."""
        parent_type = parent.type

        # Left side of assignment
        if parent_type == "assignment":
            # First identifier child is the target
            for child in parent.children:
                if child.type == "identifier":
                    return child == node
                break

        # Property declaration name (inside variable_declaration)
        if parent_type == "variable_declaration":
            for child in parent.children:
                if child.type == "identifier":
                    return child == node
                break

        # Parameter name
        if parent_type == "parameter":
            for child in parent.children:
                if child.type == "identifier":
                    return child == node
                break

        # user_type (type annotation) is not a variable use
        if parent_type == "user_type":
            return True

        return False

    def _is_valid_var_name(self, name: str) -> bool:
        """Check if name is a valid variable name (not keyword, etc.)."""
        keywords = {
            "if", "else", "for", "while", "return", "when", "is", "in", "as",
            "true", "false", "null", "this", "super", "class", "object", "fun",
            "val", "var", "const", "public", "private", "protected", "internal",
            "try", "catch", "finally", "throw", "break", "continue", "package",
            "import", "interface", "abstract", "override", "open", "sealed",
        }
        return name not in keywords and not name.startswith("_")

    def _handle_property_declaration(self, node):
        """Handle Kotlin property declaration: val x = ... or var x = ..."""
        # Find variable_declaration child which contains the name
        for child in node.children:
            if child.type == "variable_declaration":
                # Get the variable name from identifier
                for decl_child in child.children:
                    if decl_child.type == "identifier":
                        self._add_ref(self.get_node_text(decl_child), "definition", decl_child)
                        break
            # Visit the expression for uses (identifier is the initializer value)
            elif child.type in ("call_expression", "identifier", "binary_expression",
                               "string_literal", "integer_literal"):
                self._visit_node(child)

    def _handle_assignment(self, node):
        """Handle Kotlin assignment: x = ..."""
        children = list(node.children)
        if len(children) >= 3:
            # Format: identifier = expression
            left = children[0]
            # Skip = operator
            right = children[2] if len(children) > 2 else None

            # Visit right side first (uses)
            if right:
                self._visit_node(right)

            # Add definition for left side
            if left.type == "identifier":
                name = self.get_node_text(left)
                self._add_ref(name, "definition", left)

    def _handle_parameter(self, node):
        """Handle Kotlin function parameter."""
        # Find the parameter name (identifier)
        for child in node.children:
            if child.type == "identifier":
                name = self.get_node_text(child)
                self._add_ref(name, "definition", child)
                break


def extract_scala_dfg(code: str, function_name: str) -> DFGInfo:
    """
    Extract DFG for a Scala function.

    Args:
        code: Scala source code
        function_name: Name of function to analyze

    Returns:
        DFGInfo with variable references and def-use chains
    """
    if not TREE_SITTER_SCALA_AVAILABLE:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Parse with tree-sitter
    scala_lang = Language(tree_sitter_scala.language())
    parser = Parser(scala_lang)
    source_bytes = code.encode('utf-8')
    tree = parser.parse(source_bytes)

    # Find the function
    func_node = _find_scala_function_by_name(tree.root_node, function_name, source_bytes)
    if func_node is None:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Extract definitions and uses
    visitor = ScalaDefUseVisitor(source_bytes)
    visitor.visit(func_node)

    # Compute def-use chains
    analyzer = PythonReachingDefsAnalyzer(visitor.refs)
    edges = analyzer.compute_def_use_chains()

    return DFGInfo(
        function_name=function_name,
        var_refs=visitor.refs,
        dataflow_edges=edges,
    )


def _find_scala_function_by_name(root, name: str, source: bytes):
    """Find a Scala function node by name in tree-sitter tree."""
    def search(node):
        if node.type == "function_definition":
            # Get the function name from the name field
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = source[name_node.start_byte:name_node.end_byte].decode('utf-8')
                if func_name == name:
                    return node

        for child in node.children:
            result = search(child)
            if result:
                return result
        return None

    return search(root)


class ScalaDefUseVisitor:
    """
    Extract variable definitions and uses from Scala tree-sitter parse tree.

    Scala variables are declared with val (immutable) or var (mutable).
    """

    def __init__(self, source: bytes):
        self.source = source
        self.refs: list[VarRef] = []

    def get_node_text(self, node) -> str:
        """Get source text for a node."""
        return self.source[node.start_byte:node.end_byte].decode('utf-8')

    def visit(self, node):
        """Visit a node and its children."""
        self._visit_node(node)

    def _add_ref(self, name: str, ref_type: str, node):
        """Add a variable reference."""
        ref = VarRef(
            name=name,
            ref_type=ref_type,
            line=node.start_point[0] + 1,  # tree-sitter is 0-indexed
            column=node.start_point[1],
        )
        self.refs.append(ref)

    def _visit_node(self, node):
        """Process a node based on its type."""
        node_type = node.type

        # Handle val/var declarations: val x = ... or var x = ...
        if node_type == "val_definition":
            self._handle_val_definition(node)
            return

        if node_type == "var_definition":
            self._handle_var_definition(node)
            return

        # Handle assignments (var reassignment)
        if node_type == "assignment_expression":
            self._handle_assignment(node)
            return

        # Handle augmented assignments
        if node_type == "compound_assignment_expression":
            self._handle_compound_assignment(node)
            return

        # Handle parameters
        if node_type == "parameter":
            self._handle_parameter(node)
            return

        # Handle identifier (uses)
        if node_type == "identifier":
            parent = node.parent
            if parent and not self._is_definition_context(parent, node):
                name = self.get_node_text(node)
                if self._is_valid_var_name(name):
                    self._add_ref(name, "use", node)
            return

        # Recurse into children
        for child in node.children:
            self._visit_node(child)

    def _is_definition_context(self, parent, node) -> bool:
        """Check if node is being defined (not used)."""
        parent_type = parent.type

        # Left side of assignment
        if parent_type == "assignment_expression":
            left = parent.child_by_field_name("left")
            if left and left == node:
                return True

        # Val/var definition - pattern contains the name
        if parent_type == "val_definition" or parent_type == "var_definition":
            pattern = parent.child_by_field_name("pattern")
            if pattern and self._node_contains(pattern, node):
                return True

        # Parameter name
        if parent_type == "parameter":
            return True

        # Type annotation (not a variable use)
        if parent_type in ("type_identifier", "simple_type", "generic_type"):
            return True

        return False

    def _node_contains(self, parent, target) -> bool:
        """Check if parent contains target node."""
        if parent == target:
            return True
        for child in parent.children:
            if self._node_contains(child, target):
                return True
        return False

    def _is_valid_var_name(self, name: str) -> bool:
        """Check if name is a valid variable name (not keyword, etc.)."""
        keywords = {
            "if", "else", "for", "while", "return", "match", "case",
            "true", "false", "null", "this", "super", "class", "object", "def",
            "val", "var", "new", "override", "abstract", "sealed", "final",
            "try", "catch", "finally", "throw", "import", "package", "extends",
            "with", "trait", "type", "lazy", "yield", "implicit", "private",
            "protected", "public", "Int", "String", "Boolean", "Unit", "Any",
        }
        return name not in keywords and not name.startswith("_")

    def _handle_val_definition(self, node):
        """Handle Scala val definition: val x = ..."""
        # Find pattern child which contains the identifier
        pattern = node.child_by_field_name("pattern")
        value = node.child_by_field_name("value")

        # Visit value first (uses)
        if value:
            self._visit_node(value)

        # Add definition for pattern
        if pattern:
            self._extract_pattern_names(pattern, "definition")

    def _handle_var_definition(self, node):
        """Handle Scala var definition: var x = ..."""
        # Same structure as val
        pattern = node.child_by_field_name("pattern")
        value = node.child_by_field_name("value")

        # Visit value first (uses)
        if value:
            self._visit_node(value)

        # Add definition for pattern
        if pattern:
            self._extract_pattern_names(pattern, "definition")

    def _extract_pattern_names(self, pattern, ref_type: str):
        """Extract variable names from a pattern."""
        if pattern.type == "identifier":
            name = self.get_node_text(pattern)
            if self._is_valid_var_name(name):
                self._add_ref(name, ref_type, pattern)
        else:
            for child in pattern.children:
                self._extract_pattern_names(child, ref_type)

    def _handle_assignment(self, node):
        """Handle Scala assignment: x = ..."""
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")

        # Visit right side first (uses)
        if right:
            self._visit_node(right)

        # Add definition for left side
        if left and left.type == "identifier":
            name = self.get_node_text(left)
            self._add_ref(name, "definition", left)

    def _handle_compound_assignment(self, node):
        """Handle Scala compound assignment: x += ..."""
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")

        # Visit right side first (uses)
        if right:
            self._visit_node(right)

        # Left is both used and updated
        if left and left.type == "identifier":
            name = self.get_node_text(left)
            self._add_ref(name, "use", left)
            self._add_ref(name, "definition", left)

    def _handle_parameter(self, node):
        """Handle Scala function parameter."""
        # Find the parameter name (identifier)
        name_node = node.child_by_field_name("name")
        if name_node:
            name = self.get_node_text(name_node)
            self._add_ref(name, "definition", name_node)


# =============================================================================
# Lua DFG Extraction
# =============================================================================

def extract_lua_dfg(code: str, function_name: str) -> DFGInfo:
    """
    Extract DFG for a Lua function.

    Args:
        code: Lua source code
        function_name: Name of function to analyze

    Returns:
        DFGInfo with variable references and def-use chains
    """
    if not TREE_SITTER_LUA_AVAILABLE:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Parse with tree-sitter
    lua_lang = Language(tree_sitter_lua.language())
    parser = Parser(lua_lang)
    source_bytes = code.encode('utf-8')
    tree = parser.parse(source_bytes)

    # Find the function
    func_node = _find_lua_function_by_name(tree.root_node, function_name, source_bytes)
    if func_node is None:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Extract definitions and uses
    visitor = LuaDefUseVisitor(source_bytes)
    visitor.visit(func_node)

    # Compute def-use chains
    analyzer = PythonReachingDefsAnalyzer(visitor.refs)
    edges = analyzer.compute_def_use_chains()

    return DFGInfo(
        function_name=function_name,
        var_refs=visitor.refs,
        dataflow_edges=edges,
    )


def _find_lua_function_by_name(root, name: str, source: bytes):
    """Find a Lua function node by name in tree-sitter tree.

    Handles both:
    - function name() ... end (function_declaration)
    - local function name() ... end (function_declaration with local)
    """
    def search(node):
        # Check function_declaration: function name() end or local function name() end
        if node.type == "function_declaration":
            # Find the identifier child (the function name)
            for child in node.children:
                if child.type == "identifier":
                    func_name = source[child.start_byte:child.end_byte].decode('utf-8')
                    if func_name == name:
                        return node
                    break
                elif child.type in ("dot_index_expression", "method_index_expression"):
                    # Table.method - get the field name
                    field = child.child_by_field_name("field")
                    if field:
                        func_name = source[field.start_byte:field.end_byte].decode('utf-8')
                        if func_name == name:
                            return node
                    break

        for child in node.children:
            result = search(child)
            if result:
                return result
        return None

    return search(root)


class LuaDefUseVisitor:
    """
    Extract variable definitions and uses from Lua tree-sitter parse tree.

    Lua variables:
    - local x = value (local declaration)
    - x = value (assignment, may create global)
    - function parameters
    - for loop variables
    """

    def __init__(self, source: bytes):
        self.source = source
        self.refs: list[VarRef] = []

    def get_node_text(self, node) -> str:
        """Get source text for a node."""
        return self.source[node.start_byte:node.end_byte].decode('utf-8')

    def visit(self, node):
        """Visit a node and its children."""
        self._visit_node(node)

    def _add_ref(self, name: str, ref_type: str, node):
        """Add a variable reference."""
        ref = VarRef(
            name=name,
            ref_type=ref_type,
            line=node.start_point[0] + 1,  # tree-sitter is 0-indexed
            column=node.start_point[1],
        )
        self.refs.append(ref)

    def _visit_node(self, node):
        """Process a node based on its type."""
        node_type = node.type

        # Handle local variable declarations: local x = ...
        if node_type == "variable_declaration":
            self._handle_local_declaration(node)
            return

        # Handle assignments: x = y (may be definition or update)
        if node_type == "assignment_statement":
            self._handle_assignment(node)
            return

        # Handle function parameters
        if node_type == "parameters":
            self._handle_parameters(node)
            return

        # Handle for loops (numeric and generic)
        if node_type == "for_statement":
            self._handle_for_loop(node)
            return

        # Handle identifiers (uses) - but not when they're being defined
        if node_type == "identifier":
            parent = node.parent
            if parent and not self._is_definition_context(parent, node):
                name = self.get_node_text(node)
                if self._is_valid_var_name(name):
                    self._add_ref(name, "use", node)
            return

        # Recurse into children
        for child in node.children:
            self._visit_node(child)

    def _is_definition_context(self, parent, node) -> bool:
        """Check if node is being defined (not used)."""
        parent_type = parent.type

        # Left side of assignment_statement
        if parent_type == "assignment_statement":
            # First child before '=' is the target
            for i, child in enumerate(parent.children):
                if child.type == "identifier" and child == node:
                    # Check if there's an '=' after this
                    for j in range(i + 1, len(parent.children)):
                        if parent.children[j].type == "=":
                            return True
                    # If this is the first identifier and equals is somewhere
                    return i == 0

        # Variable declaration: local x = ...
        if parent_type == "variable_declaration":
            return True

        # Variable_list in assignment (left side)
        if parent_type == "variable_list":
            # All identifiers in variable_list are definitions
            return True

        # For loop variables
        if parent_type == "for_generic_clause" or parent_type == "for_numeric_clause":
            return True

        # Parameters
        if parent_type == "parameters":
            return True

        return False

    def _is_valid_var_name(self, name: str) -> bool:
        """Check if name is a valid variable name (not keyword, etc.)."""
        keywords = {
            "if", "then", "else", "elseif", "end", "for", "while", "do",
            "repeat", "until", "break", "return", "local", "function",
            "true", "false", "nil", "and", "or", "not", "in",
            "self",  # convention for method receiver
        }
        return name not in keywords and not name.startswith("_")

    def _handle_local_declaration(self, node):
        """Handle Lua local variable declaration: local x = ... or local x, y = ..."""
        # variable_declaration contains an assignment_statement
        for child in node.children:
            if child.type == "assignment_statement":
                # Handle the assignment inside
                self._handle_assignment(child)
                return

        # Fallback: look for variable_list directly (local x without assignment)
        names = []
        for child in node.children:
            if child.type == "variable_list":
                for var_child in child.children:
                    if var_child.type == "identifier":
                        names.append(var_child)

        # Add definitions for all variable names
        for name_node in names:
            name = self.get_node_text(name_node)
            self._add_ref(name, "definition", name_node)

    def _handle_assignment(self, node):
        """Handle Lua assignment: x = y or x, y = a, b"""
        # Find targets (left of =) and values (right of =)
        targets = []
        values_node = None

        found_equals = False
        for child in node.children:
            if child.type == "=":
                found_equals = True
            elif not found_equals:
                if child.type == "variable_list":
                    for var_child in child.children:
                        if var_child.type == "identifier":
                            targets.append(var_child)
                elif child.type == "identifier":
                    targets.append(child)
            else:
                if child.type == "expression_list":
                    values_node = child

        # Visit values first (uses)
        if values_node:
            self._visit_node(values_node)

        # Add definitions for targets
        for target_node in targets:
            name = self.get_node_text(target_node)
            self._add_ref(name, "definition", target_node)

    def _handle_parameters(self, node):
        """Handle Lua function parameters."""
        for child in node.children:
            if child.type == "identifier":
                name = self.get_node_text(child)
                self._add_ref(name, "definition", child)

    def _handle_for_loop(self, node):
        """Handle Lua for loops (numeric and generic)."""
        # Find the loop clause (for_numeric_clause or for_generic_clause)
        clause = None
        body = None

        for child in node.children:
            if child.type in ("for_numeric_clause", "for_generic_clause"):
                clause = child
            elif child.type == "block":
                body = child

        if clause:
            # Extract loop variables
            for var_child in clause.children:
                if var_child.type == "identifier":
                    name = self.get_node_text(var_child)
                    self._add_ref(name, "definition", var_child)
                elif var_child.type == "variable_list":
                    # Generic for: for i, v in pairs(t) do
                    for id_child in var_child.children:
                        if id_child.type == "identifier":
                            name = self.get_node_text(id_child)
                            self._add_ref(name, "definition", id_child)

            # Visit expressions in clause (uses)
            for expr_child in clause.children:
                if expr_child.type == "expression_list":
                    self._visit_node(expr_child)

        # Visit loop body
        if body:
            self._visit_node(body)


# =============================================================================
# Elixir DFG Extraction
# =============================================================================

class ElixirDefUseVisitor:
    """
    Extract variable definitions and uses from Elixir tree-sitter parse tree.

    Elixir uses pattern matching for variable binding:
    - x = value (assignment/match)
    - {a, b} = tuple (destructuring)
    - def func(x, y) (parameters)
    """

    def __init__(self, source: bytes):
        self.source = source
        self.refs: list[VarRef] = []

    def get_node_text(self, node) -> str:
        """Get source text for a node."""
        return self.source[node.start_byte:node.end_byte].decode('utf-8')

    def visit(self, node):
        """Visit a node and its children."""
        self._visit_node(node)

    def _add_ref(self, name: str, ref_type: str, node):
        """Add a variable reference."""
        ref = VarRef(
            name=name,
            ref_type=ref_type,
            line=node.start_point[0] + 1,  # tree-sitter is 0-indexed
            column=node.start_point[1],
        )
        self.refs.append(ref)

    def _visit_node(self, node):
        """Process a node based on its type."""
        node_type = node.type

        # Handle binary_operator (includes = for match)
        if node_type == "binary_operator":
            self._handle_binary_operator(node)
            return

        # Handle call nodes (includes function definitions)
        if node_type == "call":
            self._handle_call(node)
            return

        # Handle identifier
        if node_type == "identifier":
            parent = node.parent
            if parent and not self._is_definition_context(parent, node):
                name = self.get_node_text(node)
                if self._is_valid_var_name(name):
                    self._add_ref(name, "use", node)
            return

        # Recurse into children
        for child in node.children:
            self._visit_node(child)

    def _is_definition_context(self, parent, node) -> bool:
        """Check if node is being defined (not used)."""
        parent_type = parent.type

        # Left side of = match operator
        if parent_type == "binary_operator":
            # Check if this is a match/assignment (=)
            for child in parent.children:
                if child.type == "=" or self.get_node_text(child) == "=":
                    left = None
                    for c in parent.children:
                        if c.is_named:
                            left = c
                            break
                    if left and self._node_contains(left, node):
                        return True

        # Part of function arguments pattern
        if parent_type == "arguments":
            return True

        # Part of tuple/list pattern on left of match
        if parent_type in ("tuple", "list"):
            grandparent = parent.parent
            if grandparent and grandparent.type == "binary_operator":
                # Check if parent is on left side of =
                for child in grandparent.children:
                    if child.type == "=" or self.get_node_text(child) == "=":
                        left = None
                        for c in grandparent.children:
                            if c.is_named:
                                left = c
                                break
                        if left and self._node_contains(left, parent):
                            return True

        return False

    def _node_contains(self, parent, target) -> bool:
        """Check if parent contains target node."""
        if parent == target:
            return True
        for child in parent.children:
            if self._node_contains(child, target):
                return True
        return False

    def _is_valid_var_name(self, name: str) -> bool:
        """Check if name is a valid variable name."""
        # Elixir variables start with lowercase or underscore
        if not name or name[0].isupper():
            return False
        keywords = {
            "def", "defp", "defmodule", "do", "end", "if", "else", "unless",
            "case", "cond", "with", "fn", "when", "true", "false", "nil",
            "and", "or", "not", "in", "import", "alias", "use", "require",
            "for", "raise", "try", "catch", "rescue", "after", "receive",
            "quote", "unquote",
        }
        return name not in keywords

    def _handle_binary_operator(self, node):
        """Handle Elixir binary operators, particularly match (=)."""
        # Find the operator
        operator = None
        left = None
        right = None

        for i, child in enumerate(node.children):
            if not child.is_named and self.get_node_text(child) in ("=", "|>", "<-"):
                operator = self.get_node_text(child)
            elif child.is_named:
                if left is None:
                    left = child
                else:
                    right = child

        if operator == "=":
            # Match operator - left side is definition, right is use
            if right:
                self._visit_node(right)
            if left:
                self._extract_pattern_definitions(left)
        elif operator == "<-":
            # Comprehension binding
            if right:
                self._visit_node(right)
            if left:
                self._extract_pattern_definitions(left)
        else:
            # Other operators - visit both sides
            if left:
                self._visit_node(left)
            if right:
                self._visit_node(right)

    def _extract_pattern_definitions(self, pattern):
        """Extract variable definitions from a pattern."""
        if pattern.type == "identifier":
            name = self.get_node_text(pattern)
            if self._is_valid_var_name(name):
                self._add_ref(name, "definition", pattern)
        elif pattern.type in ("tuple", "list", "map"):
            for child in pattern.children:
                if child.is_named:
                    self._extract_pattern_definitions(child)
        else:
            # For other patterns, recurse
            for child in pattern.children:
                if child.is_named:
                    self._extract_pattern_definitions(child)

    def _handle_call(self, node):
        """Handle Elixir calls, including function definitions."""
        call_name = None
        for child in node.children:
            if child.type == "identifier":
                call_name = self.get_node_text(child)
                break

        if call_name in ("def", "defp"):
            # Function definition - extract parameter names
            args = node.child_by_field_name("arguments")
            if args:
                for arg_child in args.children:
                    if arg_child.type == "call":
                        # The function signature: def func_name(params)
                        inner_args = arg_child.child_by_field_name("arguments")
                        if inner_args:
                            for param in inner_args.children:
                                if param.is_named:
                                    self._extract_pattern_definitions(param)
            # Visit the do_block for body
            for child in node.children:
                if child.type == "do_block":
                    self._visit_node(child)
        else:
            # Regular call - visit children
            for child in node.children:
                if child.is_named:
                    self._visit_node(child)


def _find_elixir_function_by_name(root, name: str, source: bytes):
    """Find an Elixir function node by name in tree-sitter tree.

    Elixir functions are defined with def/defp macros:
    - def function_name(args) do ... end
    - defp private_function(args) do ... end
    """
    def search(node):
        if node.type == "call":
            # Check if this is a def/defp call
            call_name = None
            for child in node.children:
                if child.type == "identifier":
                    call_name = source[child.start_byte:child.end_byte].decode('utf-8')
                    break

            if call_name in ("def", "defp"):
                # Find the function name in arguments
                # Note: tree-sitter-elixir uses direct children, not field names
                args = None
                for child in node.children:
                    if child.type == "arguments":
                        args = child
                        break
                if args:
                    for arg_child in args.children:
                        if arg_child.type == "call":
                            # Function with params: def func_name(args)
                            for c in arg_child.children:
                                if c.type == "identifier":
                                    func_name = source[c.start_byte:c.end_byte].decode('utf-8')
                                    if func_name == name:
                                        return node
                        elif arg_child.type == "identifier":
                            # Function without params: def func_name do
                            func_name = source[arg_child.start_byte:arg_child.end_byte].decode('utf-8')
                            if func_name == name:
                                return node

        for child in node.children:
            result = search(child)
            if result:
                return result
        return None

    return search(root)


def extract_elixir_dfg(code: str, function_name: str) -> DFGInfo:
    """
    Extract DFG for an Elixir function.

    Args:
        code: Elixir source code
        function_name: Name of function to analyze

    Returns:
        DFGInfo with variable references and def-use chains
    """
    if not TREE_SITTER_ELIXIR_AVAILABLE:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Parse with tree-sitter
    elixir_lang = Language(tree_sitter_elixir.language())
    parser = Parser(elixir_lang)
    source_bytes = code.encode('utf-8')
    tree = parser.parse(source_bytes)

    # Find the function
    func_node = _find_elixir_function_by_name(tree.root_node, function_name, source_bytes)
    if func_node is None:
        return DFGInfo(
            function_name=function_name,
            var_refs=[],
            dataflow_edges=[],
        )

    # Extract definitions and uses
    visitor = ElixirDefUseVisitor(source_bytes)
    visitor.visit(func_node)

    # Build def-use chains
    defs: dict[str, list[VarRef]] = {}
    uses: dict[str, list[VarRef]] = {}

    for ref in visitor.refs:
        if ref.ref_type == "definition":
            if ref.name not in defs:
                defs[ref.name] = []
            defs[ref.name].append(ref)
        elif ref.ref_type == "use":
            if ref.name not in uses:
                uses[ref.name] = []
            uses[ref.name].append(ref)

    # Create dataflow edges - each use links to the most recent def
    edges: list[DataflowEdge] = []
    for var_name, use_list in uses.items():
        if var_name in defs:
            for use_ref in use_list:
                # Find the most recent definition before this use
                recent_def = None
                for def_ref in defs[var_name]:
                    if def_ref.line <= use_ref.line:
                        if recent_def is None or def_ref.line > recent_def.line:
                            recent_def = def_ref
                if recent_def:
                    edges.append(DataflowEdge(def_ref=recent_def, use_ref=use_ref))

    return DFGInfo(
        function_name=function_name,
        var_refs=visitor.refs,
        dataflow_edges=edges,
    )
