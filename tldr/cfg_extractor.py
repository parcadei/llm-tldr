"""
Control Flow Graph (CFG) extraction for multi-language code analysis.

Provides CFG extraction for:
- Python (via ast module)
- TypeScript/JavaScript (via tree-sitter)
- Go (via tree-sitter)
- Rust (via tree-sitter)

Based on staticfg pattern but simplified for TLDR-code use case.
"""
import ast
from dataclasses import dataclass, field
from typing import Any

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

TREE_SITTER_ELIXIR_AVAILABLE = False
try:
    import tree_sitter_elixir
    TREE_SITTER_ELIXIR_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_ZIG_AVAILABLE = False
try:
    import tree_sitter_zig
    TREE_SITTER_ZIG_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_LUA_AVAILABLE = False
try:
    import tree_sitter_lua
    TREE_SITTER_LUA_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_LUAU_AVAILABLE = False
try:
    import tree_sitter_luau
    TREE_SITTER_LUAU_AVAILABLE = True
except ImportError:
    pass


@dataclass
class CFGBlock:
    """
    Basic block - sequential statements with no internal branches.

    A basic block is a sequence of statements where:
    - Control enters only at the first statement
    - Control leaves only at the last statement
    """
    id: int
    start_line: int
    end_line: int
    block_type: str  # "entry", "branch", "loop_header", "loop_body", "return", "exit", "body"
    statements: list[str] = field(default_factory=list)  # Optional statement summaries
    func_calls: list[str] = field(default_factory=list)  # Functions called in this block
    predecessors: list[int] = field(default_factory=list)  # IDs of blocks that lead here

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "type": self.block_type,
            "lines": [self.start_line, self.end_line],
        }
        if self.func_calls:
            d["calls"] = self.func_calls
        return d

    def get_source(self, source_code: str) -> str:
        """Get source code for this block from the original code.

        Args:
            source_code: The full source code of the file/function.

        Returns:
            The source code lines for this block.
        """
        lines = source_code.splitlines()
        # Line numbers are 1-indexed, list is 0-indexed
        start_idx = max(0, self.start_line - 1)
        end_idx = min(len(lines), self.end_line)
        return '\n'.join(lines[start_idx:end_idx])


@dataclass
class CFGEdge:
    """
    Edge between blocks with optional branch condition.

    Edge types:
    - "true" / "false": conditional branch outcomes
    - "unconditional": direct flow (sequential, function call)
    - "back_edge": loop iteration (body -> guard)
    - "break": exit loop early
    - "continue": skip to next iteration
    - "iterate": for-loop enters body
    - "exhausted": for-loop exits when iterator is done
    """
    source_id: int
    target_id: int
    edge_type: str
    condition: str | None = None  # Human-readable condition like "x > 0"

    def to_dict(self) -> dict:
        d = {
            "from": self.source_id,
            "to": self.target_id,
            "type": self.edge_type,
        }
        if self.condition:
            d["condition"] = self.condition
        return d


@dataclass
class CFGInfo:
    """
    Control flow graph for a function.

    Provides:
    - Basic block structure
    - Control flow edges with conditions
    - Entry/exit points
    - Cyclomatic complexity metric
    - Nested function CFGs (closures, inner functions)
    """
    function_name: str
    blocks: list[CFGBlock]
    edges: list[CFGEdge]
    entry_block_id: int
    exit_block_ids: list[int]
    cyclomatic_complexity: int  # edges - nodes + 2
    nested_cfgs: dict[str, "CFGInfo"] = field(default_factory=dict)  # name -> CFG for nested functions

    def to_dict(self) -> dict:
        d = {
            "function": self.function_name,
            "blocks": [b.to_dict() for b in self.blocks],
            "edges": [e.to_dict() for e in self.edges],
            "entry_block": self.entry_block_id,
            "exit_blocks": self.exit_block_ids,
            "cyclomatic_complexity": self.cyclomatic_complexity,
        }
        if self.nested_cfgs:
            d["nested_functions"] = {name: cfg.to_dict() for name, cfg in self.nested_cfgs.items()}
        return d


# =============================================================================
# Python CFG Extraction (using ast module)
# =============================================================================

class PythonCFGBuilder(ast.NodeVisitor):
    """
    Build CFG from Python AST.

    Based on staticfg pattern but simplified for our needs.
    We track:
    - Basic blocks with line numbers
    - Edges with types and conditions
    - Loop guards for back edges
    - Decision points for cyclomatic complexity
    """

    def __init__(self):
        self.blocks: list[CFGBlock] = []
        self.edges: list[CFGEdge] = []
        self.current_block_id = 0
        self.current_block: CFGBlock | None = None
        self.entry_block_id: int | None = None
        self.exit_block_ids: list[int] = []

        # Loop tracking for break/continue
        self.loop_guard_stack: list[int] = []  # Block IDs of loop guards
        self.after_loop_stack: list[int] = []  # Block IDs after loop

        # Decision points for complexity calculation
        self.decision_points: int = 0

        # Nested function CFGs (closures, inner functions)
        self.nested_cfgs: dict[str, CFGInfo] = {}

    def new_block(self, block_type: str, start_line: int, end_line: int | None = None) -> CFGBlock:
        """Create a new block and add it to the graph."""
        block = CFGBlock(
            id=self.current_block_id,
            start_line=start_line,
            end_line=end_line or start_line,
            block_type=block_type,
        )
        self.blocks.append(block)
        self.current_block_id += 1
        return block

    def add_edge(self, source_id: int, target_id: int, edge_type: str, condition: str | None = None):
        """Add an edge between blocks."""
        edge = CFGEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            condition=condition,
        )
        self.edges.append(edge)

    def build(self, func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> CFGInfo:
        """Build CFG from a function definition node."""
        # Create entry block
        entry = self.new_block("entry", func_node.lineno)
        self.entry_block_id = entry.id
        self.current_block = entry

        # Visit function body
        for stmt in func_node.body:
            # Extract calls before visiting (which may create new blocks)
            if self.current_block:
                self._add_calls_to_block(self.current_block, stmt)
            self.visit(stmt)

        # If current block hasn't been marked as exit, mark it
        if self.current_block and self.current_block.id not in self.exit_block_ids:
            # Implicit return at end of function
            self.exit_block_ids.append(self.current_block.id)
            self.current_block.block_type = "exit"

        # Compute predecessors from edges
        block_map = {b.id: b for b in self.blocks}
        for edge in self.edges:
            target = block_map.get(edge.target_id)
            if target and edge.source_id not in target.predecessors:
                target.predecessors.append(edge.source_id)

        # Calculate cyclomatic complexity: decision points + 1
        # This is simpler and more accurate than E - N + 2 for disconnected graphs
        complexity = self.decision_points + 1

        return CFGInfo(
            function_name=func_node.name,
            blocks=self.blocks,
            edges=self.edges,
            entry_block_id=self.entry_block_id,
            exit_block_ids=self.exit_block_ids,
            cyclomatic_complexity=complexity,
            nested_cfgs=self.nested_cfgs,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Handle nested function definitions - build sub-CFG."""
        # Build CFG for the nested function
        nested_builder = PythonCFGBuilder()
        nested_cfg = nested_builder.build(node)
        self.nested_cfgs[node.name] = nested_cfg

        # The function definition itself is just a statement in the current block
        # No control flow change - the def doesn't execute the function

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Handle nested async function definitions - build sub-CFG."""
        nested_builder = PythonCFGBuilder()
        nested_cfg = nested_builder.build(node)
        self.nested_cfgs[node.name] = nested_cfg

    def _get_condition_str(self, node: ast.expr) -> str:
        """Convert AST condition node to readable string."""
        try:
            return ast.unparse(node)
        except Exception:
            return "<condition>"

    def _extract_calls_shallow(self, node: ast.AST) -> list[str]:
        """Extract function call names from a single statement (not nested blocks).

        For compound statements (if, for, while), only extracts from the condition,
        not from the body. Body calls are extracted when those blocks are visited.
        """
        calls = []

        # For compound statements, only look at non-body parts
        if isinstance(node, ast.If):
            # Only the test expression
            for child in ast.walk(node.test):
                if isinstance(child, ast.Call):
                    calls.extend(self._get_call_name(child))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            # Only the iter expression
            for child in ast.walk(node.iter):
                if isinstance(child, ast.Call):
                    calls.extend(self._get_call_name(child))
        elif isinstance(node, ast.While):
            # Only the test expression
            for child in ast.walk(node.test):
                if isinstance(child, ast.Call):
                    calls.extend(self._get_call_name(child))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            # Only the context expressions
            for item in node.items:
                for child in ast.walk(item.context_expr):
                    if isinstance(child, ast.Call):
                        calls.extend(self._get_call_name(child))
        else:
            # Simple statements - walk everything
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    calls.extend(self._get_call_name(child))

        return calls

    def _get_call_name(self, call_node: ast.Call) -> list[str]:
        """Extract the function name from a Call node."""
        if isinstance(call_node.func, ast.Name):
            return [call_node.func.id]
        elif isinstance(call_node.func, ast.Attribute):
            return [call_node.func.attr]
        return []

    def _add_calls_to_block(self, block: CFGBlock, stmt: ast.AST):
        """Extract and add function calls from statement to block."""
        calls = self._extract_calls_shallow(stmt)
        for call in calls:
            if call not in block.func_calls:
                block.func_calls.append(call)

    def visit_If(self, node: ast.If):
        """Handle if/elif/else statements - creates diamond pattern."""
        # Track decision point for complexity
        self.decision_points += 1

        # Current block becomes branch block
        if self.current_block:
            self.current_block.block_type = "branch"
            self.current_block.end_line = node.lineno
            branch_block_id = self.current_block.id
        else:
            branch = self.new_block("branch", node.lineno)
            branch_block_id = branch.id

        condition = self._get_condition_str(node.test)

        # Create block for true branch (if body)
        if_body_start = node.body[0].lineno if node.body else node.lineno
        if_body_end = node.body[-1].end_lineno if node.body else node.lineno
        true_block = self.new_block("body", if_body_start, if_body_end)
        self.add_edge(branch_block_id, true_block.id, "true", condition)

        # Create after-if block for merging
        after_if = self.new_block("body", node.end_lineno or node.lineno)

        # Process true branch
        self.current_block = true_block
        for stmt in node.body:
            if self.current_block:
                self._add_calls_to_block(self.current_block, stmt)
            self.visit(stmt)
        # Connect true branch to after-if (unless it ends with return)
        if self.current_block and self.current_block.id not in self.exit_block_ids:
            self.add_edge(self.current_block.id, after_if.id, "unconditional")

        # Process false branch (else/elif)
        if node.orelse:
            else_body_start = node.orelse[0].lineno
            else_body_end = node.orelse[-1].end_lineno if hasattr(node.orelse[-1], 'end_lineno') else node.orelse[-1].lineno
            false_block = self.new_block("body", else_body_start, else_body_end)
            self.add_edge(branch_block_id, false_block.id, "false", f"not ({condition})")

            self.current_block = false_block
            for stmt in node.orelse:
                if self.current_block:
                    self._add_calls_to_block(self.current_block, stmt)
                self.visit(stmt)
            # Connect else branch to after-if (unless it ends with return)
            if self.current_block and self.current_block.id not in self.exit_block_ids:
                self.add_edge(self.current_block.id, after_if.id, "unconditional")
        else:
            # No else - false edge goes directly to after-if
            self.add_edge(branch_block_id, after_if.id, "false", f"not ({condition})")

        # Continue with after-if block
        self.current_block = after_if

    def visit_While(self, node: ast.While):
        """Handle while loops - creates loop with back edge."""
        # Track decision point for complexity
        self.decision_points += 1

        # Create loop guard block
        guard = self.new_block("loop_header", node.lineno)

        # Connect current block to guard
        if self.current_block:
            self.add_edge(self.current_block.id, guard.id, "unconditional")

        condition = self._get_condition_str(node.test)

        # Create after-loop block
        after_loop = self.new_block("body", node.end_lineno or node.lineno)

        # Track for break/continue
        self.loop_guard_stack.append(guard.id)
        self.after_loop_stack.append(after_loop.id)

        # Create loop body block
        if node.body:
            body_start = node.body[0].lineno
            body_end = node.body[-1].end_lineno if hasattr(node.body[-1], 'end_lineno') else node.body[-1].lineno
        else:
            body_start = node.lineno
            body_end = node.lineno
        body = self.new_block("loop_body", body_start, body_end)

        # Edge from guard to body (condition true)
        self.add_edge(guard.id, body.id, "true", condition)

        # Edge from guard to after (condition false)
        self.add_edge(guard.id, after_loop.id, "false", f"not ({condition})")

        # Process loop body
        self.current_block = body
        for stmt in node.body:
            if self.current_block:
                self._add_calls_to_block(self.current_block, stmt)
            self.visit(stmt)

        # Back edge from end of body to guard
        if self.current_block and self.current_block.id not in self.exit_block_ids:
            self.add_edge(self.current_block.id, guard.id, "back_edge")

        # Pop loop tracking
        self.loop_guard_stack.pop()
        self.after_loop_stack.pop()

        # Continue after loop
        self.current_block = after_loop

    def visit_For(self, node: ast.For):
        """Handle for loops - similar to while but with iterator."""
        # Track decision point for complexity
        self.decision_points += 1

        # Create loop guard block
        guard = self.new_block("loop_header", node.lineno)

        # Connect current block to guard
        if self.current_block:
            self.add_edge(self.current_block.id, guard.id, "unconditional")

        # Create after-loop block
        after_loop = self.new_block("body", node.end_lineno or node.lineno)

        # Track for break/continue
        self.loop_guard_stack.append(guard.id)
        self.after_loop_stack.append(after_loop.id)

        # Create loop body block
        if node.body:
            body_start = node.body[0].lineno
            body_end = node.body[-1].end_lineno if hasattr(node.body[-1], 'end_lineno') else node.body[-1].lineno
        else:
            body_start = node.lineno
            body_end = node.lineno
        body = self.new_block("loop_body", body_start, body_end)

        # Edge from guard to body (iterator has next)
        target_str = self._get_condition_str(node.target) if isinstance(node.target, ast.expr) else str(node.target)
        iter_str = self._get_condition_str(node.iter)
        self.add_edge(guard.id, body.id, "iterate", f"{target_str} in {iter_str}")

        # Edge from guard to after (iterator exhausted)
        self.add_edge(guard.id, after_loop.id, "exhausted")

        # Process loop body
        self.current_block = body
        for stmt in node.body:
            if self.current_block:
                self._add_calls_to_block(self.current_block, stmt)
            self.visit(stmt)

        # Back edge from end of body to guard
        if self.current_block and self.current_block.id not in self.exit_block_ids:
            self.add_edge(self.current_block.id, guard.id, "back_edge")

        # Pop loop tracking
        self.loop_guard_stack.pop()
        self.after_loop_stack.pop()

        # Continue after loop
        self.current_block = after_loop

    def visit_Return(self, node: ast.Return):
        """Handle return statements - marks exit block."""
        if self.current_block:
            self.current_block.block_type = "return"
            self.current_block.end_line = node.lineno
            self.exit_block_ids.append(self.current_block.id)

        # Create unreachable block for any code after return
        self.current_block = self.new_block("body", node.lineno)

    def visit_Break(self, node: ast.Break):
        """Handle break - edge to after-loop block."""
        if self.after_loop_stack and self.current_block:
            self.add_edge(self.current_block.id, self.after_loop_stack[-1], "break")
            # Create unreachable block for any code after break
            self.current_block = self.new_block("body", node.lineno)

    def visit_Continue(self, node: ast.Continue):
        """Handle continue - edge to loop guard."""
        if self.loop_guard_stack and self.current_block:
            self.add_edge(self.current_block.id, self.loop_guard_stack[-1], "continue")
            # Create unreachable block for any code after continue
            self.current_block = self.new_block("body", node.lineno)

    def generic_visit(self, node: ast.AST):
        """Visit children for compound statements we don't handle specially."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                self.visit(child)


def extract_python_cfg(source: str, function_name: str) -> CFGInfo:
    """
    Extract CFG for a specific function from Python source code.

    Args:
        source: Python source code as string
        function_name: Name of the function to extract CFG for

    Returns:
        CFGInfo dataclass with blocks, edges, and complexity

    Raises:
        ValueError: If function not found in source
    """
    tree = ast.parse(source)

    # Find the function
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                builder = PythonCFGBuilder()
                return builder.build(node)

    raise ValueError(f"Function '{function_name}' not found in source")


# =============================================================================
# Tree-sitter based CFG extraction (TypeScript, Go, Rust)
# =============================================================================

class TreeSitterCFGBuilder:
    """
    Build CFG from tree-sitter parse tree.

    Works for TypeScript, JavaScript, Go, Rust, Ruby, and Swift.
    """

    # Node type mappings per language
    IF_TYPES = {"if_statement", "if_expression", "if"}  # Ruby uses "if"
    WHILE_TYPES = {"while_statement", "while_expression", "while"}  # Ruby uses "while"
    FOR_TYPES = {"for_statement", "for_expression", "for_in_statement", "for", "foreach_statement", "for_range_loop", "for_generic_clause", "for_numeric_clause"}  # Ruby uses "for", PHP uses "foreach_statement", C++ uses "for_range_loop", Lua uses for_generic/for_numeric
    LOOP_TYPES = {"loop_expression"}  # Rust's infinite loop
    REPEAT_TYPES = {"repeat_statement"}  # Lua's repeat-until loop
    RETURN_TYPES = {"return_statement", "return_expression", "return"}  # Ruby uses "return"
    BREAK_TYPES = {"break_statement", "break_expression", "break"}  # Ruby uses "break"
    CONTINUE_TYPES = {"continue_statement", "continue_expression", "next"}  # Ruby uses "next" for continue
    CASE_TYPES = {"case_statement", "switch_statement", "case", "when_expression"}  # Ruby uses "case", Kotlin uses "when_expression"
    BEGIN_RESCUE_TYPES = {"begin", "try_statement", "do_statement"}  # Ruby uses "begin" for try/rescue, Swift uses "do_statement" for do/catch
    GUARD_TYPES = {"guard_statement"}  # Swift guard statement
    FUNCTION_TYPES = {
        "function_declaration", "function_definition", "function_item",
        "method_definition", "arrow_function", "function_expression",
        "method",  # Ruby uses "method"
    }

    def __init__(self, source: bytes, language: str):
        self.source = source
        self.language = language
        self.blocks: list[CFGBlock] = []
        self.edges: list[CFGEdge] = []
        self.current_block_id = 0
        self.current_block: CFGBlock | None = None
        self.entry_block_id: int | None = None
        self.exit_block_ids: list[int] = []

        # Loop tracking
        self.loop_guard_stack: list[int] = []
        self.after_loop_stack: list[int] = []

        # Decision points for complexity calculation
        self.decision_points: int = 0

    def new_block(self, block_type: str, start_line: int, end_line: int | None = None) -> CFGBlock:
        block = CFGBlock(
            id=self.current_block_id,
            start_line=start_line,
            end_line=end_line or start_line,
            block_type=block_type,
        )
        self.blocks.append(block)
        self.current_block_id += 1
        return block

    def add_edge(self, source_id: int, target_id: int, edge_type: str, condition: str | None = None):
        edge = CFGEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            condition=condition,
        )
        self.edges.append(edge)

    def get_node_text(self, node) -> str:
        """Get source text for a node."""
        return self.source[node.start_byte:node.end_byte].decode('utf-8')

    def build(self, func_node, func_name: str) -> CFGInfo:
        """Build CFG from a function node."""
        # Create entry block
        entry = self.new_block("entry", func_node.start_point[0] + 1)
        self.entry_block_id = entry.id
        self.current_block = entry

        # Find function body
        body = self._find_function_body(func_node)
        if body:
            self._visit_node(body)
            # Update current block's end line to cover the body
            if self.current_block:
                self.current_block.end_line = max(
                    self.current_block.end_line,
                    body.end_point[0] + 1
                )

        # Mark final block as exit if not already
        if self.current_block and self.current_block.id not in self.exit_block_ids:
            self.exit_block_ids.append(self.current_block.id)
            self.current_block.block_type = "exit"
            # Ensure exit block covers the whole function for simple functions
            self.current_block.end_line = max(
                self.current_block.end_line,
                func_node.end_point[0] + 1
            )

        # Calculate cyclomatic complexity: decision points + 1
        complexity = self.decision_points + 1

        return CFGInfo(
            function_name=func_name,
            blocks=self.blocks,
            edges=self.edges,
            entry_block_id=self.entry_block_id,
            exit_block_ids=self.exit_block_ids,
            cyclomatic_complexity=complexity,
        )

    def _find_function_body(self, node):
        """Find the body/block child of a function node."""
        for child in node.children:
            if child.type in {"statement_block", "block", "compound_statement", "expression_statement", "code_block"}:
                return child
            if child.type == "body":
                return child
            # Ruby: method body is body_statement
            if child.type == "body_statement":
                return child
            # Swift: function_body contains the statements
            if child.type == "function_body":
                return child
        # For arrow functions, the body might be an expression
        return node

    def _visit_node(self, node):
        """Visit a tree-sitter node and build CFG."""
        if node.type in self.IF_TYPES:
            self._visit_if(node)
        elif node.type in self.WHILE_TYPES:
            self._visit_while(node)
        elif node.type in self.FOR_TYPES:
            self._visit_for(node)
        elif node.type in self.LOOP_TYPES:
            self._visit_loop(node)
        elif node.type in self.REPEAT_TYPES:
            self._visit_repeat(node)
        elif node.type in self.RETURN_TYPES:
            self._visit_return(node)
        elif node.type in self.BREAK_TYPES:
            self._visit_break(node)
        elif node.type in self.CONTINUE_TYPES:
            self._visit_continue(node)
        elif node.type in self.GUARD_TYPES:
            self._visit_guard(node)
        elif node.type in self.CASE_TYPES:
            self._visit_switch(node)
        elif node.type == "call" and self.language == "ruby":
            # Ruby: check for .each do |x| ... end pattern (iterators)
            self._visit_ruby_call(node)
        else:
            # Visit children for compound nodes
            for child in node.children:
                if child.is_named:
                    self._visit_node(child)

    def _visit_ruby_call(self, node):
        """Handle Ruby method calls, especially iterators like .each."""
        # Check if this call has a do_block (iterator pattern)
        do_block = None
        for child in node.children:
            if child.type == "do_block":
                do_block = child
                break

        if do_block:
            # This is an iterator like .each, .map, etc.
            self.decision_points += 1

            guard = self.new_block("loop_header", node.start_point[0] + 1)
            if self.current_block:
                self.add_edge(self.current_block.id, guard.id, "unconditional")

            after_loop = self.new_block("body", node.end_point[0] + 1)

            self.loop_guard_stack.append(guard.id)
            self.after_loop_stack.append(after_loop.id)

            # Find body_statement inside do_block
            body = None
            for child in do_block.children:
                if child.type == "body_statement":
                    body = child
                    break

            if body:
                loop_body = self.new_block("loop_body", body.start_point[0] + 1, body.end_point[0] + 1)
                self.add_edge(guard.id, loop_body.id, "iterate")
                self.add_edge(guard.id, after_loop.id, "exhausted")

                self.current_block = loop_body
                self._visit_node(body)

                if self.current_block and self.current_block.id not in self.exit_block_ids:
                    self.add_edge(self.current_block.id, guard.id, "back_edge")

            self.loop_guard_stack.pop()
            self.after_loop_stack.pop()
            self.current_block = after_loop
        else:
            # Regular method call, just visit children
            for child in node.children:
                if child.is_named:
                    self._visit_node(child)

    def _find_child_by_type(self, node, types: set[str]):
        """Find first child matching any of the given types."""
        for child in node.children:
            if child.type in types:
                return child
        return None

    def _find_child_by_field(self, node, field_name: str):
        """Find child by field name."""
        return node.child_by_field_name(field_name)

    def _visit_if(self, node):
        """Handle if/else statements."""
        # Track decision point for complexity
        self.decision_points += 1

        if self.current_block:
            self.current_block.block_type = "branch"
            self.current_block.end_line = node.start_point[0] + 1
            branch_block_id = self.current_block.id
        else:
            branch = self.new_block("branch", node.start_point[0] + 1)
            branch_block_id = branch.id

        # Get condition text
        condition_node = self._find_child_by_field(node, "condition")
        if not condition_node:
            # Try parenthesized_expression for JS/TS
            for child in node.children:
                if child.type == "parenthesized_expression":
                    condition_node = child
                    break
                # Ruby: condition is a binary expression directly under if
                if child.type == "binary":
                    condition_node = child
                    break
        condition = self.get_node_text(condition_node) if condition_node else "<condition>"

        # Create after-if block
        after_if = self.new_block("body", node.end_point[0] + 1)

        # Find consequence (if body)
        consequence = self._find_child_by_field(node, "consequence")
        if not consequence:
            # Try statement_block for JS/TS
            for child in node.children:
                if child.type in {"statement_block", "block", "compound_statement", "statements"}:
                    consequence = child
                    break
                # Ruby: if body is in "then" child
                if child.type == "then":
                    consequence = child
                    break

        if consequence:
            true_block = self.new_block("body", consequence.start_point[0] + 1, consequence.end_point[0] + 1)
            self.add_edge(branch_block_id, true_block.id, "true", condition)

            self.current_block = true_block
            self._visit_node(consequence)

            if self.current_block and self.current_block.id not in self.exit_block_ids:
                self.add_edge(self.current_block.id, after_if.id, "unconditional")

        # Find alternative (else body)
        alternative = self._find_child_by_field(node, "alternative")
        if not alternative:
            for child in node.children:
                if child.type in {"else_clause", "else"}:
                    alternative = child
                    break

        if alternative:
            false_block = self.new_block("body", alternative.start_point[0] + 1, alternative.end_point[0] + 1)
            self.add_edge(branch_block_id, false_block.id, "false", f"not ({condition})")

            self.current_block = false_block
            self._visit_node(alternative)

            if self.current_block and self.current_block.id not in self.exit_block_ids:
                self.add_edge(self.current_block.id, after_if.id, "unconditional")
        else:
            self.add_edge(branch_block_id, after_if.id, "false", f"not ({condition})")

        self.current_block = after_if

    def _visit_while(self, node):
        """Handle while loops."""
        # Track decision point for complexity
        self.decision_points += 1

        guard = self.new_block("loop_header", node.start_point[0] + 1)

        if self.current_block:
            self.add_edge(self.current_block.id, guard.id, "unconditional")

        # Get condition
        condition_node = self._find_child_by_field(node, "condition")
        if not condition_node:
            # Ruby: condition is a binary expression directly under while
            for child in node.children:
                if child.type == "binary":
                    condition_node = child
                    break
        condition = self.get_node_text(condition_node) if condition_node else "<condition>"

        after_loop = self.new_block("body", node.end_point[0] + 1)

        self.loop_guard_stack.append(guard.id)
        self.after_loop_stack.append(after_loop.id)

        # Find body
        body = self._find_child_by_field(node, "body")
        if not body:
            for child in node.children:
                if child.type in {"statement_block", "block", "compound_statement", "statements"}:
                    body = child
                    break
                # Ruby: while body is in "do" child
                if child.type == "do":
                    body = child
                    break

        if body:
            loop_body = self.new_block("loop_body", body.start_point[0] + 1, body.end_point[0] + 1)
            self.add_edge(guard.id, loop_body.id, "true", condition)
            self.add_edge(guard.id, after_loop.id, "false", f"not ({condition})")

            self.current_block = loop_body
            self._visit_node(body)

            if self.current_block and self.current_block.id not in self.exit_block_ids:
                self.add_edge(self.current_block.id, guard.id, "back_edge")

        self.loop_guard_stack.pop()
        self.after_loop_stack.pop()
        self.current_block = after_loop

    def _visit_for(self, node):
        """Handle for loops."""
        # Track decision point for complexity
        self.decision_points += 1

        guard = self.new_block("loop_header", node.start_point[0] + 1)

        if self.current_block:
            self.add_edge(self.current_block.id, guard.id, "unconditional")

        after_loop = self.new_block("body", node.end_point[0] + 1)

        self.loop_guard_stack.append(guard.id)
        self.after_loop_stack.append(after_loop.id)

        # Find body
        body = self._find_child_by_field(node, "body")
        if not body:
            for child in node.children:
                if child.type in {"statement_block", "block", "compound_statement", "statements"}:
                    body = child
                    break

        if body:
            loop_body = self.new_block("loop_body", body.start_point[0] + 1, body.end_point[0] + 1)
            self.add_edge(guard.id, loop_body.id, "iterate")
            self.add_edge(guard.id, after_loop.id, "exhausted")

            self.current_block = loop_body
            self._visit_node(body)

            if self.current_block and self.current_block.id not in self.exit_block_ids:
                self.add_edge(self.current_block.id, guard.id, "back_edge")

        self.loop_guard_stack.pop()
        self.after_loop_stack.pop()
        self.current_block = after_loop

    def _visit_loop(self, node):
        """Handle Rust's infinite loop (loop {})."""
        # Track decision point for complexity (the loop has implicit condition)
        self.decision_points += 1

        guard = self.new_block("loop_header", node.start_point[0] + 1)

        if self.current_block:
            self.add_edge(self.current_block.id, guard.id, "unconditional")

        after_loop = self.new_block("body", node.end_point[0] + 1)

        self.loop_guard_stack.append(guard.id)
        self.after_loop_stack.append(after_loop.id)

        # Find body (block in Rust loop)
        body = None
        for child in node.children:
            if child.type == "block":
                body = child
                break

        if body:
            loop_body = self.new_block("loop_body", body.start_point[0] + 1, body.end_point[0] + 1)
            # Infinite loop - always enters body
            self.add_edge(guard.id, loop_body.id, "unconditional")

            self.current_block = loop_body
            self._visit_node(body)

            # Back edge (unless break happened)
            if self.current_block and self.current_block.id not in self.exit_block_ids:
                self.add_edge(self.current_block.id, guard.id, "back_edge")

        self.loop_guard_stack.pop()
        self.after_loop_stack.pop()
        self.current_block = after_loop

    def _visit_repeat(self, node):
        """Handle Lua's repeat-until loops.

        repeat-until is unique: body executes at least once, then condition is checked.
        Unlike while (condition checked first), the body always runs first.
        """
        # Track decision point for complexity
        self.decision_points += 1

        # Create body block first (executes before condition check)
        body_start = node.start_point[0] + 1
        body = self.new_block("loop_body", body_start)

        if self.current_block:
            self.add_edge(self.current_block.id, body.id, "unconditional")

        # Create after-loop block
        after_loop = self.new_block("body", node.end_point[0] + 1)

        # Track for break
        self.loop_guard_stack.append(body.id)  # For repeat, body is the target for continue
        self.after_loop_stack.append(after_loop.id)

        # Visit body statements
        self.current_block = body
        for child in node.children:
            if child.type == "block":
                self._visit_node(child)
                break

        # Get condition (until clause)
        condition_node = node.child_by_field_name("condition")
        condition = self.get_node_text(condition_node) if condition_node else "<condition>"

        # After body, check condition: if true -> exit, if false -> repeat
        if self.current_block and self.current_block.id not in self.exit_block_ids:
            # False condition -> back to body
            self.add_edge(self.current_block.id, body.id, "back_edge", f"not ({condition})")
            # True condition -> exit loop
            self.add_edge(self.current_block.id, after_loop.id, "true", condition)

        self.loop_guard_stack.pop()
        self.after_loop_stack.pop()
        self.current_block = after_loop

    def _visit_return(self, node):
        """Handle return statements."""
        if self.current_block:
            self.current_block.block_type = "return"
            self.current_block.end_line = node.start_point[0] + 1
            self.exit_block_ids.append(self.current_block.id)

        self.current_block = self.new_block("body", node.start_point[0] + 1)

    def _visit_break(self, node):
        """Handle break statements."""
        if self.after_loop_stack and self.current_block:
            self.add_edge(self.current_block.id, self.after_loop_stack[-1], "break")
            self.current_block = self.new_block("body", node.start_point[0] + 1)

    def _visit_continue(self, node):
        """Handle continue statements."""
        if self.loop_guard_stack and self.current_block:
            self.add_edge(self.current_block.id, self.loop_guard_stack[-1], "continue")
            self.current_block = self.new_block("body", node.start_point[0] + 1)

    def _visit_guard(self, node):
        """Handle Swift guard statements - guard let/guard else pattern."""
        # Track decision point for complexity
        self.decision_points += 1

        if self.current_block:
            self.current_block.block_type = "branch"
            self.current_block.end_line = node.start_point[0] + 1
            branch_block_id = self.current_block.id
        else:
            branch = self.new_block("branch", node.start_point[0] + 1)
            branch_block_id = branch.id

        # Get condition text (guard condition)
        condition = self.get_node_text(node)

        # Create after-guard block for normal flow (guard passes)
        after_guard = self.new_block("body", node.end_point[0] + 1)

        # Find the else block (guard failure case)
        else_body = None
        for child in node.children:
            if child.type == "code_block":
                else_body = child
                break

        if else_body:
            false_block = self.new_block("body", else_body.start_point[0] + 1, else_body.end_point[0] + 1)
            self.add_edge(branch_block_id, false_block.id, "false", "guard failed")

            self.current_block = false_block
            self._visit_node(else_body)

            # Guard else typically returns/throws, but connect to after if it doesn't
            if self.current_block and self.current_block.id not in self.exit_block_ids:
                self.add_edge(self.current_block.id, after_guard.id, "unconditional")
        else:
            self.add_edge(branch_block_id, after_guard.id, "false", "guard failed")

        # True edge goes to after_guard (guard passes, execution continues)
        self.add_edge(branch_block_id, after_guard.id, "true", "guard passed")

        self.current_block = after_guard

    def _visit_switch(self, node):
        """Handle switch/case statements."""
        # Track decision point for complexity (each case is a decision)
        self.decision_points += 1

        if self.current_block:
            self.current_block.block_type = "branch"
            self.current_block.end_line = node.start_point[0] + 1
            switch_block_id = self.current_block.id
        else:
            switch_block = self.new_block("branch", node.start_point[0] + 1)
            switch_block_id = switch_block.id

        # Create after-switch block for merging
        after_switch = self.new_block("body", node.end_point[0] + 1)

        # Find all case clauses
        # C/C++: case_statement inside compound_statement
        # PHP: case_statement/default_statement inside switch_block
        # Others: switch_case, switch_entry, case_item directly under switch
        case_types = ("case_statement", "default_statement", "switch_case", "switch_entry", "case_item")

        def find_cases(parent_node):
            """Recursively find case nodes (they may be inside compound_statement/switch_block)."""
            cases = []
            for child in parent_node.children:
                if child.type in case_types:
                    cases.append(child)
                elif child.type in ("compound_statement", "switch_block"):
                    cases.extend(find_cases(child))
            return cases

        prev_case_block = None
        for child in find_cases(node):
            # Each case is a new decision point
            self.decision_points += 1
            case_block = self.new_block("body", child.start_point[0] + 1, child.end_point[0] + 1)
            self.add_edge(switch_block_id, case_block.id, "case")

            # Handle fallthrough from previous case (if no break/return)
            if prev_case_block and prev_case_block.id not in self.exit_block_ids:
                has_break = any(
                    c.type in ("break_statement", "return_statement")
                    for c in child.children if c.is_named
                )
                if not has_break:
                    self.add_edge(prev_case_block.id, case_block.id, "fallthrough")

            self.current_block = case_block
            # Visit case body
            for case_child in child.children:
                if case_child.is_named and case_child.type not in ("case_pattern", "default_keyword", "number_literal"):
                    self._visit_node(case_child)

            # Connect case to after_switch (unless it returns/breaks)
            if self.current_block and self.current_block.id not in self.exit_block_ids:
                self.add_edge(self.current_block.id, after_switch.id, "unconditional")

            prev_case_block = self.current_block

        self.current_block = after_switch


def _get_ts_parser(language: str):
    """Get or create a tree-sitter parser for the given language."""
    from tree_sitter import Language, Parser

    parser = Parser()

    # Handle each language with its own availability check
    if language in ("typescript", "tsx"):
        if not TREE_SITTER_AVAILABLE:
            raise ImportError("tree-sitter-typescript not available")
        import tree_sitter_typescript
        if language == "tsx":
            parser.language = Language(tree_sitter_typescript.language_tsx())
        else:
            parser.language = Language(tree_sitter_typescript.language_typescript())
    elif language == "javascript":
        if not TREE_SITTER_AVAILABLE:
            raise ImportError("tree-sitter-javascript not available")
        import tree_sitter_javascript
        parser.language = Language(tree_sitter_javascript.language())
    elif language == "go":
        if not TREE_SITTER_GO_AVAILABLE:
            raise ImportError("tree-sitter-go not available")
        import tree_sitter_go
        parser.language = Language(tree_sitter_go.language())
    elif language == "rust":
        if not TREE_SITTER_RUST_AVAILABLE:
            raise ImportError("tree-sitter-rust not available")
        import tree_sitter_rust
        parser.language = Language(tree_sitter_rust.language())
    elif language == "java":
        if not TREE_SITTER_JAVA_AVAILABLE:
            raise ImportError("tree-sitter-java not available")
        import tree_sitter_java
        parser.language = Language(tree_sitter_java.language())
    elif language == "c":
        if not TREE_SITTER_C_AVAILABLE:
            raise ImportError("tree-sitter-c not available")
        import tree_sitter_c
        parser.language = Language(tree_sitter_c.language())
    elif language == "ruby":
        if not TREE_SITTER_RUBY_AVAILABLE:
            raise ImportError("tree-sitter-ruby not available")
        import tree_sitter_ruby
        parser.language = Language(tree_sitter_ruby.language())
    elif language == "php":
        if not TREE_SITTER_PHP_AVAILABLE:
            raise ImportError("tree-sitter-php not available")
        import tree_sitter_php
        # Use language_php() which handles PHP with embedded HTML
        parser.language = Language(tree_sitter_php.language_php())
    elif language == "cpp":
        if not TREE_SITTER_CPP_AVAILABLE:
            raise ImportError("tree-sitter-cpp not available")
        import tree_sitter_cpp
        parser.language = Language(tree_sitter_cpp.language())
    elif language == "swift":
        if not TREE_SITTER_SWIFT_AVAILABLE:
            raise ImportError("tree-sitter-swift not available")
        import tree_sitter_swift
        parser.language = Language(tree_sitter_swift.language())
    elif language == "csharp":
        if not TREE_SITTER_CSHARP_AVAILABLE:
            raise ImportError("tree-sitter-c-sharp not available")
        import tree_sitter_c_sharp
        parser.language = Language(tree_sitter_c_sharp.language())
    elif language == "kotlin":
        if not TREE_SITTER_KOTLIN_AVAILABLE:
            raise ImportError("tree-sitter-kotlin not available")
        import tree_sitter_kotlin
        parser.language = Language(tree_sitter_kotlin.language())
    elif language == "scala":
        if not TREE_SITTER_SCALA_AVAILABLE:
            raise ImportError("tree-sitter-scala not available")
        import tree_sitter_scala
        parser.language = Language(tree_sitter_scala.language())
    elif language == "elixir":
        if not TREE_SITTER_ELIXIR_AVAILABLE:
            raise ImportError("tree-sitter-elixir not available")
        import tree_sitter_elixir
        parser.language = Language(tree_sitter_elixir.language())
    elif language == "lua":
        if not TREE_SITTER_LUA_AVAILABLE:
            raise ImportError("tree-sitter-lua not available")
        import tree_sitter_lua
        parser.language = Language(tree_sitter_lua.language())
    elif language == "luau":
        if not TREE_SITTER_LUAU_AVAILABLE:
            raise ImportError("tree-sitter-luau not available")
        import tree_sitter_luau
        parser.language = Language(tree_sitter_luau.language())
    elif language == "zig":
        if not TREE_SITTER_ZIG_AVAILABLE:
            raise ImportError("tree-sitter-zig not available")
        import tree_sitter_zig
        parser.language = Language(tree_sitter_zig.language())
    else:
        raise ValueError(f"Unsupported language: {language}")

    return parser


def _find_function_node(tree, function_name: str, language: str):
    """Find function node in tree-sitter tree by name."""
    func_types = {
        "function_declaration", "function_definition", "function_item",
        "method_definition", "arrow_function", "function_expression",
        "method_declaration",
        "method",  # Ruby: def method_name ... end
    }

    def find_in_node(node):
        if node.type in func_types:
            # Try to get function name
            name_node = node.child_by_field_name("name")
            if name_node:
                name = name_node.text.decode('utf-8') if hasattr(name_node, 'text') else str(name_node)
                if name == function_name:
                    return node
            # For C/C++, the function name is in declarator.declarator (function_declarator -> identifier)
            # Also handles pointer_declarator wrapping function_declarator (e.g., char* get_day())
            # For C++ class methods, the name is a field_identifier instead of identifier
            if language in ("c", "cpp"):
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
                        name = inner_decl.text.decode('utf-8') if hasattr(inner_decl, 'text') else str(inner_decl)
                        if name == function_name:
                            return node
            # For Go, check identifier child
            for child in node.children:
                if child.type == "identifier":
                    name = child.text.decode('utf-8') if hasattr(child, 'text') else str(child)
                    if name == function_name:
                        return node
            # For Swift/Kotlin, check simple_identifier child
            for child in node.children:
                if child.type == "simple_identifier":
                    name = child.text.decode('utf-8') if hasattr(child, 'text') else str(child)
                    if name == function_name:
                        return node

        for child in node.children:
            result = find_in_node(child)
            if result:
                return result
        return None

    return find_in_node(tree.root_node)


def extract_typescript_cfg(source: str, function_name: str) -> CFGInfo:
    """Extract CFG for a TypeScript/JavaScript function."""
    if not TREE_SITTER_AVAILABLE:
        raise ImportError("tree-sitter not available for TypeScript parsing")

    source_bytes = source.encode('utf-8')
    parser = _get_ts_parser("typescript")
    tree = parser.parse(source_bytes)

    func_node = _find_function_node(tree, function_name, "typescript")
    if not func_node:
        raise ValueError(f"Function '{function_name}' not found in source")

    builder = TreeSitterCFGBuilder(source_bytes, "typescript")
    return builder.build(func_node, function_name)


def extract_go_cfg(source: str, function_name: str) -> CFGInfo:
    """Extract CFG for a Go function."""
    if not TREE_SITTER_GO_AVAILABLE:
        raise ImportError("tree-sitter-go not available")

    source_bytes = source.encode('utf-8')
    parser = _get_ts_parser("go")
    tree = parser.parse(source_bytes)

    func_node = _find_function_node(tree, function_name, "go")
    if not func_node:
        raise ValueError(f"Function '{function_name}' not found in source")

    builder = TreeSitterCFGBuilder(source_bytes, "go")
    return builder.build(func_node, function_name)


def extract_rust_cfg(source: str, function_name: str) -> CFGInfo:
    """Extract CFG for a Rust function."""
    if not TREE_SITTER_RUST_AVAILABLE:
        raise ImportError("tree-sitter-rust not available")

    source_bytes = source.encode('utf-8')
    parser = _get_ts_parser("rust")
    tree = parser.parse(source_bytes)

    func_node = _find_function_node(tree, function_name, "rust")
    if not func_node:
        raise ValueError(f"Function '{function_name}' not found in source")

    builder = TreeSitterCFGBuilder(source_bytes, "rust")
    return builder.build(func_node, function_name)


def extract_java_cfg(source: str, function_name: str) -> CFGInfo:
    """Extract CFG for a Java method."""
    if not TREE_SITTER_JAVA_AVAILABLE:
        raise ImportError("tree-sitter-java not available")

    source_bytes = source.encode('utf-8')
    parser = _get_ts_parser("java")
    tree = parser.parse(source_bytes)

    func_node = _find_function_node(tree, function_name, "java")
    if not func_node:
        raise ValueError(f"Function '{function_name}' not found in source")

    builder = TreeSitterCFGBuilder(source_bytes, "java")
    return builder.build(func_node, function_name)


def extract_c_cfg(source: str, function_name: str) -> CFGInfo:
    """Extract CFG for a C function."""
    if not TREE_SITTER_C_AVAILABLE:
        raise ImportError("tree-sitter-c not available")

    source_bytes = source.encode('utf-8')
    parser = _get_ts_parser("c")
    tree = parser.parse(source_bytes)

    func_node = _find_function_node(tree, function_name, "c")
    if not func_node:
        raise ValueError(f"Function '{function_name}' not found in source")

    builder = TreeSitterCFGBuilder(source_bytes, "c")
    return builder.build(func_node, function_name)


def extract_cpp_cfg(source: str, function_name: str) -> CFGInfo:
    """Extract CFG for a C++ function."""
    if not TREE_SITTER_CPP_AVAILABLE:
        raise ImportError("tree-sitter-cpp not available")

    source_bytes = source.encode('utf-8')
    parser = _get_ts_parser("cpp")
    tree = parser.parse(source_bytes)

    func_node = _find_function_node(tree, function_name, "cpp")
    if not func_node:
        raise ValueError(f"Function '{function_name}' not found in source")

    builder = TreeSitterCFGBuilder(source_bytes, "cpp")
    return builder.build(func_node, function_name)


def extract_php_cfg(source: str, function_name: str) -> CFGInfo:
    """Extract CFG for a PHP function.

    Args:
        source: PHP source code (may include <?php tag)
        function_name: Name of function to extract CFG for

    Returns:
        CFGInfo with blocks, edges, and complexity

    Raises:
        ImportError: If tree-sitter-php is not available
        ValueError: If function not found in source
    """
    if not TREE_SITTER_PHP_AVAILABLE:
        raise ImportError("tree-sitter-php not available")

    source_bytes = source.encode('utf-8')
    parser = _get_ts_parser("php")
    tree = parser.parse(source_bytes)

    func_node = _find_function_node(tree, function_name, "php")
    if not func_node:
        raise ValueError(f"Function '{function_name}' not found in source")

    builder = TreeSitterCFGBuilder(source_bytes, "php")
    return builder.build(func_node, function_name)


def extract_ruby_cfg(source: str, function_name: str) -> CFGInfo:
    """Extract CFG for a Ruby method."""
    if not TREE_SITTER_RUBY_AVAILABLE:
        raise ImportError("tree-sitter-ruby not available")

    source_bytes = source.encode('utf-8')
    parser = _get_ts_parser("ruby")
    tree = parser.parse(source_bytes)

    func_node = _find_function_node(tree, function_name, "ruby")
    if not func_node:
        raise ValueError(f"Function '{function_name}' not found in source")

    builder = TreeSitterCFGBuilder(source_bytes, "ruby")
    return builder.build(func_node, function_name)


def extract_swift_cfg(source: str, function_name: str) -> CFGInfo:
    """Extract CFG for a Swift function.

    Args:
        source: Swift source code
        function_name: Name of function to extract CFG for

    Returns:
        CFGInfo with blocks, edges, and complexity

    Raises:
        ImportError: If tree-sitter-swift is not available
        ValueError: If function not found in source
    """
    if not TREE_SITTER_SWIFT_AVAILABLE:
        raise ImportError("tree-sitter-swift not available")

    source_bytes = source.encode('utf-8')
    parser = _get_ts_parser("swift")
    tree = parser.parse(source_bytes)

    func_node = _find_function_node(tree, function_name, "swift")
    if not func_node:
        raise ValueError(f"Function '{function_name}' not found in source")

    builder = TreeSitterCFGBuilder(source_bytes, "swift")
    return builder.build(func_node, function_name)


def extract_csharp_cfg(source: str, function_name: str) -> CFGInfo:
    """Extract CFG for a C# method.

    Args:
        source: C# source code
        function_name: Name of method to extract CFG for

    Returns:
        CFGInfo with blocks, edges, and complexity

    Raises:
        ImportError: If tree-sitter-c-sharp is not available
        ValueError: If method not found in source
    """
    if not TREE_SITTER_CSHARP_AVAILABLE:
        raise ImportError("tree-sitter-c-sharp not available")

    source_bytes = source.encode('utf-8')
    parser = _get_ts_parser("csharp")
    tree = parser.parse(source_bytes)

    func_node = _find_function_node(tree, function_name, "csharp")
    if not func_node:
        raise ValueError(f"Function '{function_name}' not found in source")

    builder = TreeSitterCFGBuilder(source_bytes, "csharp")
    return builder.build(func_node, function_name)


def extract_kotlin_cfg(source: str, function_name: str) -> CFGInfo:
    """Extract CFG for a Kotlin function.

    Args:
        source: Kotlin source code
        function_name: Name of function to extract CFG for

    Returns:
        CFGInfo with blocks, edges, and complexity

    Raises:
        ImportError: If tree-sitter-kotlin is not available
        ValueError: If function not found in source
    """
    if not TREE_SITTER_KOTLIN_AVAILABLE:
        raise ImportError("tree-sitter-kotlin not available")

    source_bytes = source.encode('utf-8')
    parser = _get_ts_parser("kotlin")
    tree = parser.parse(source_bytes)

    func_node = _find_function_node(tree, function_name, "kotlin")
    if not func_node:
        raise ValueError(f"Function '{function_name}' not found in source")

    builder = TreeSitterCFGBuilder(source_bytes, "kotlin")
    return builder.build(func_node, function_name)


def extract_scala_cfg(source: str, function_name: str) -> CFGInfo:
    """Extract CFG for a Scala function.

    Args:
        source: Scala source code
        function_name: Name of function to extract CFG for

    Returns:
        CFGInfo with blocks, edges, and complexity

    Raises:
        ImportError: If tree-sitter-scala is not available
        ValueError: If function not found in source
    """
    if not TREE_SITTER_SCALA_AVAILABLE:
        raise ImportError("tree-sitter-scala not available")

    source_bytes = source.encode('utf-8')
    parser = _get_ts_parser("scala")
    tree = parser.parse(source_bytes)

    func_node = _find_function_node(tree, function_name, "scala")
    if not func_node:
        raise ValueError(f"Function '{function_name}' not found in source")

    builder = TreeSitterCFGBuilder(source_bytes, "scala")
    return builder.build(func_node, function_name)


def extract_lua_cfg(source: str, function_name: str) -> CFGInfo:
    """Extract CFG for a Lua function.

    Args:
        source: Lua source code
        function_name: Name of function to extract CFG for

    Returns:
        CFGInfo with blocks, edges, and complexity

    Raises:
        ImportError: If tree-sitter-lua is not available
        ValueError: If function not found in source
    """
    if not TREE_SITTER_LUA_AVAILABLE:
        raise ImportError("tree-sitter-lua not available")

    source_bytes = source.encode('utf-8')
    parser = _get_ts_parser("lua")
    tree = parser.parse(source_bytes)

    func_node = _find_lua_function_by_name(tree.root_node, function_name, source_bytes)
    if not func_node:
        raise ValueError(f"Function '{function_name}' not found in source")

    builder = TreeSitterCFGBuilder(source_bytes, "lua")
    return builder.build(func_node, function_name)


def _find_lua_function_by_name(root, name: str, source: bytes):
    """Find a Lua function node by name in tree-sitter tree.

    Handles both:
    - function name() ... end (function_declaration)
    - local function name() ... end (function_declaration with local)
    - function Table.name() ... end (table method)
    - function Table:name() ... end (table method with self)
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
                    break  # Only check first identifier
                elif child.type in ("dot_index_expression", "method_index_expression"):
                    # Table.method or Table:method - get the field name
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


# =============================================================================
# Luau CFG Extraction
# =============================================================================


def extract_luau_cfg(source: str, function_name: str) -> CFGInfo:
    """Extract CFG for a Luau function.

    Luau is syntactically similar to Lua but with type annotations
    and some additional features like continue, compound assignment.

    Args:
        source: Luau source code
        function_name: Name of function to extract CFG for

    Returns:
        CFGInfo with blocks, edges, and complexity

    Raises:
        ImportError: If tree-sitter-luau is not available
        ValueError: If function not found in source
    """
    if not TREE_SITTER_LUAU_AVAILABLE:
        raise ImportError("tree-sitter-luau not available")

    source_bytes = source.encode('utf-8')
    parser = _get_ts_parser("luau")
    tree = parser.parse(source_bytes)

    func_node = _find_luau_function_by_name(tree.root_node, function_name, source_bytes)
    if not func_node:
        raise ValueError(f"Function '{function_name}' not found in source")

    # Reuse TreeSitterCFGBuilder with luau language setting
    builder = TreeSitterCFGBuilder(source_bytes, "luau")
    return builder.build(func_node, function_name)


def _find_luau_function_by_name(root, name: str, source: bytes):
    """Find a Luau function node by name in tree-sitter tree.

    Handles both:
    - function name() ... end (function_declaration)
    - local function name() ... end (function_declaration with local)
    - function Table.name() ... end (table method)
    - function Table:name() ... end (table method with self)

    Luau uses similar AST structure to Lua.
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
                    break  # Only check first identifier
                elif child.type in ("dot_index_expression", "method_index_expression"):
                    # Table.method or Table:method - get the last identifier
                    for subchild in child.children:
                        if subchild.type == "identifier":
                            last_id = source[subchild.start_byte:subchild.end_byte].decode('utf-8')
                    if last_id == name:
                        return node
                    break

        for child in node.children:
            result = search(child)
            if result:
                return result
        return None

    return search(root)


# =============================================================================
# Elixir CFG Extraction
# =============================================================================

class ElixirCFGBuilder:
    """
    Build CFG from Elixir tree-sitter parse tree.

    Elixir has a unique AST where everything is a macro/call:
    - def/defp are function definitions (macros)
    - if/case/cond/with are control flow (macros)
    - Function body is in do_block
    """

    def __init__(self, source: bytes):
        self.source = source
        self.blocks: list[CFGBlock] = []
        self.edges: list[CFGEdge] = []
        self.current_block_id = 0
        self.current_block: CFGBlock | None = None
        self.entry_block_id: int | None = None
        self.exit_block_ids: list[int] = []

        # Loop tracking (for Enum.each, etc.)
        self.loop_guard_stack: list[int] = []
        self.after_loop_stack: list[int] = []

        # Decision points for complexity
        self.decision_points: int = 0

    def new_block(self, block_type: str, start_line: int, end_line: int | None = None) -> CFGBlock:
        block = CFGBlock(
            id=self.current_block_id,
            start_line=start_line,
            end_line=end_line or start_line,
            block_type=block_type,
        )
        self.blocks.append(block)
        self.current_block_id += 1
        return block

    def add_edge(self, source_id: int, target_id: int, edge_type: str, condition: str | None = None):
        edge = CFGEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            condition=condition,
        )
        self.edges.append(edge)

    def get_node_text(self, node) -> str:
        """Get source text for a node."""
        return self.source[node.start_byte:node.end_byte].decode('utf-8')

    def _get_call_name(self, node) -> str | None:
        """Get the name of a call node (e.g., 'if', 'def', 'case')."""
        for child in node.children:
            if child.type == "identifier":
                return self.get_node_text(child)
            if child.type == "dot":
                # For qualified calls like Enum.reduce
                return None
        return None

    def build(self, func_node, func_name: str) -> CFGInfo:
        """Build CFG from an Elixir function node."""
        # Create entry block
        entry = self.new_block("entry", func_node.start_point[0] + 1)
        self.entry_block_id = entry.id
        self.current_block = entry

        # Find do_block (function body)
        do_block = None
        for child in func_node.children:
            if child.type == "do_block":
                do_block = child
                break

        if do_block:
            self._visit_node(do_block)
            if self.current_block:
                self.current_block.end_line = max(
                    self.current_block.end_line,
                    do_block.end_point[0] + 1
                )

        # Mark final block as exit
        if self.current_block and self.current_block.id not in self.exit_block_ids:
            self.exit_block_ids.append(self.current_block.id)
            self.current_block.block_type = "exit"
            self.current_block.end_line = max(
                self.current_block.end_line,
                func_node.end_point[0] + 1
            )

        # If no edges were created (simple straight-line function), create a separate
        # exit block and add an unconditional edge from entry to exit
        if len(self.edges) == 0 and len(self.blocks) == 1 and self.entry_block_id is not None:
            # Change entry block back to "body" type
            self.blocks[0].block_type = "entry"
            # Create a proper exit block
            exit_block = self.new_block("exit", func_node.end_point[0] + 1)
            # Add edge from entry to exit
            self.add_edge(self.entry_block_id, exit_block.id, "unconditional")
            # Update exit block list
            self.exit_block_ids = [exit_block.id]

        complexity = self.decision_points + 1

        return CFGInfo(
            function_name=func_name,
            blocks=self.blocks,
            edges=self.edges,
            entry_block_id=self.entry_block_id,
            exit_block_ids=self.exit_block_ids,
            cyclomatic_complexity=complexity,
        )

    def _visit_node(self, node):
        """Visit a tree-sitter node and build CFG."""
        if node.type == "call":
            call_name = self._get_call_name(node)
            if call_name == "if":
                self._visit_if(node)
            elif call_name == "case":
                self._visit_case(node)
            elif call_name == "cond":
                self._visit_cond(node)
            elif call_name == "with":
                self._visit_with(node)
            else:
                # Regular call - visit children
                for child in node.children:
                    if child.is_named:
                        self._visit_node(child)
        elif node.type == "do_block":
            # Visit children of do_block
            for child in node.children:
                if child.is_named:
                    self._visit_node(child)
        elif node.type == "stab_clause":
            # Anonymous function clause - visit body
            for child in node.children:
                if child.type == "body":
                    self._visit_node(child)
        else:
            # Visit children for other nodes
            for child in node.children:
                if child.is_named:
                    self._visit_node(child)

    def _visit_if(self, node):
        """Handle Elixir if expressions."""
        self.decision_points += 1

        if self.current_block:
            self.current_block.block_type = "branch"
            self.current_block.end_line = node.start_point[0] + 1
            branch_block_id = self.current_block.id
        else:
            branch = self.new_block("branch", node.start_point[0] + 1)
            branch_block_id = branch.id

        # Get condition (first argument to if)
        condition = "<condition>"
        args = node.child_by_field_name("arguments")
        if args and args.children:
            for child in args.children:
                if child.is_named:
                    condition = self.get_node_text(child)
                    break

        after_if = self.new_block("body", node.end_point[0] + 1)

        # Find do_block for true branch
        do_block = None
        else_block_node = None
        for child in node.children:
            if child.type == "do_block":
                do_block = child
                # Look for else_block inside do_block
                for dc in do_block.children:
                    if dc.type == "else_block":
                        else_block_node = dc
                        break

        if do_block:
            true_block = self.new_block("body", do_block.start_point[0] + 1, do_block.end_point[0] + 1)
            self.add_edge(branch_block_id, true_block.id, "true", condition)

            self.current_block = true_block
            # Visit do_block children except else_block
            for child in do_block.children:
                if child.is_named and child.type != "else_block":
                    self._visit_node(child)

            if self.current_block and self.current_block.id not in self.exit_block_ids:
                self.add_edge(self.current_block.id, after_if.id, "unconditional")

        if else_block_node:
            false_block = self.new_block("body", else_block_node.start_point[0] + 1, else_block_node.end_point[0] + 1)
            self.add_edge(branch_block_id, false_block.id, "false", f"not ({condition})")

            self.current_block = false_block
            self._visit_node(else_block_node)

            if self.current_block and self.current_block.id not in self.exit_block_ids:
                self.add_edge(self.current_block.id, after_if.id, "unconditional")
        else:
            self.add_edge(branch_block_id, after_if.id, "false", f"not ({condition})")

        self.current_block = after_if

    def _visit_case(self, node):
        """Handle Elixir case expressions."""
        self.decision_points += 1

        if self.current_block:
            self.current_block.block_type = "branch"
            self.current_block.end_line = node.start_point[0] + 1
            case_block_id = self.current_block.id
        else:
            case_block = self.new_block("branch", node.start_point[0] + 1)
            case_block_id = case_block.id

        after_case = self.new_block("body", node.end_point[0] + 1)

        # Find do_block with stab_clause children (case arms)
        do_block = None
        for child in node.children:
            if child.type == "do_block":
                do_block = child
                break

        if do_block:
            for child in do_block.children:
                if child.type == "stab_clause":
                    self.decision_points += 1
                    clause_block = self.new_block("body", child.start_point[0] + 1, child.end_point[0] + 1)
                    self.add_edge(case_block_id, clause_block.id, "case")

                    self.current_block = clause_block
                    self._visit_node(child)

                    if self.current_block and self.current_block.id not in self.exit_block_ids:
                        self.add_edge(self.current_block.id, after_case.id, "unconditional")

        self.current_block = after_case

    def _visit_cond(self, node):
        """Handle Elixir cond expressions."""
        self.decision_points += 1

        if self.current_block:
            self.current_block.block_type = "branch"
            self.current_block.end_line = node.start_point[0] + 1
            cond_block_id = self.current_block.id
        else:
            cond_block = self.new_block("branch", node.start_point[0] + 1)
            cond_block_id = cond_block.id

        after_cond = self.new_block("body", node.end_point[0] + 1)

        # Find do_block with stab_clause children
        do_block = None
        for child in node.children:
            if child.type == "do_block":
                do_block = child
                break

        if do_block:
            for child in do_block.children:
                if child.type == "stab_clause":
                    self.decision_points += 1
                    clause_block = self.new_block("body", child.start_point[0] + 1, child.end_point[0] + 1)
                    self.add_edge(cond_block_id, clause_block.id, "case")

                    self.current_block = clause_block
                    self._visit_node(child)

                    if self.current_block and self.current_block.id not in self.exit_block_ids:
                        self.add_edge(self.current_block.id, after_cond.id, "unconditional")

        self.current_block = after_cond

    def _visit_with(self, node):
        """Handle Elixir with expressions."""
        self.decision_points += 1

        if self.current_block:
            self.current_block.block_type = "branch"
            self.current_block.end_line = node.start_point[0] + 1
            with_block_id = self.current_block.id
        else:
            with_block = self.new_block("branch", node.start_point[0] + 1)
            with_block_id = with_block.id

        after_with = self.new_block("body", node.end_point[0] + 1)

        # Find do_block
        do_block = None
        else_block_node = None
        for child in node.children:
            if child.type == "do_block":
                do_block = child
                for dc in do_block.children:
                    if dc.type == "else_block":
                        else_block_node = dc
                        break

        if do_block:
            # Success path
            success_block = self.new_block("body", do_block.start_point[0] + 1, do_block.end_point[0] + 1)
            self.add_edge(with_block_id, success_block.id, "true", "with success")

            self.current_block = success_block
            for child in do_block.children:
                if child.is_named and child.type != "else_block":
                    self._visit_node(child)

            if self.current_block and self.current_block.id not in self.exit_block_ids:
                self.add_edge(self.current_block.id, after_with.id, "unconditional")

        if else_block_node:
            # Failure path
            fail_block = self.new_block("body", else_block_node.start_point[0] + 1, else_block_node.end_point[0] + 1)
            self.add_edge(with_block_id, fail_block.id, "false", "with failure")

            self.current_block = fail_block
            self._visit_node(else_block_node)

            if self.current_block and self.current_block.id not in self.exit_block_ids:
                self.add_edge(self.current_block.id, after_with.id, "unconditional")
        else:
            self.add_edge(with_block_id, after_with.id, "false", "with failure")

        self.current_block = after_with


def _find_elixir_function_node(tree, function_name: str):
    """Find Elixir function node by name.

    In Elixir, functions are defined with `def` or `defp` macros:
    - def function_name(args) do ... end
    - defp private_function(args) do ... end

    The tree-sitter AST represents these as:
    - call node with identifier "def" or "defp"
    - arguments contain another call node with the function name
    """
    def find_in_node(node):
        if node.type == "call":
            # Check if this is a def/defp call
            call_name = None
            for child in node.children:
                if child.type == "identifier":
                    call_name = tree.root_node.text[child.start_byte:child.end_byte].decode('utf-8')
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
                            # This is the function call pattern: def func_name(args)
                            for c in arg_child.children:
                                if c.type == "identifier":
                                    name = tree.root_node.text[c.start_byte:c.end_byte].decode('utf-8')
                                    if name == function_name:
                                        return node
                        elif arg_child.type == "identifier":
                            # Simple function with no args: def func_name do
                            name = tree.root_node.text[arg_child.start_byte:arg_child.end_byte].decode('utf-8')
                            if name == function_name:
                                return node

        for child in node.children:
            result = find_in_node(child)
            if result:
                return result
        return None

    return find_in_node(tree.root_node)


def extract_elixir_cfg(source: str, function_name: str) -> CFGInfo:
    """Extract CFG for an Elixir function.

    Args:
        source: Elixir source code
        function_name: Name of function to extract CFG for

    Returns:
        CFGInfo with blocks, edges, and complexity

    Raises:
        ImportError: If tree-sitter-elixir is not available
        ValueError: If function not found in source
    """
    if not TREE_SITTER_ELIXIR_AVAILABLE:
        raise ImportError("tree-sitter-elixir not available")

    source_bytes = source.encode('utf-8')
    parser = _get_ts_parser("elixir")
    tree = parser.parse(source_bytes)

    func_node = _find_elixir_function_node(tree, function_name)
    if not func_node:
        raise ValueError(f"Function '{function_name}' not found in source")

    builder = ElixirCFGBuilder(source_bytes)
    return builder.build(func_node, function_name)


class ZigCFGBuilder:
    """Build a control-flow graph from Zig source bytes.

    Maintains blocks, edges, entry/exit block IDs, loop guard and
    after-loop stacks, and a decision_points counter used to compute
    cyclomatic complexity.
    """

    def __init__(self, source: bytes):
        self.source = source
        self.blocks: list[CFGBlock] = []
        self.edges: list[CFGEdge] = []
        self.current_block_id = 0
        self.current_block: CFGBlock | None = None
        self.entry_block_id: int | None = None
        self.exit_block_ids: list[int] = []
        self.loop_guard_stack: list[int] = []
        self.after_loop_stack: list[int] = []
        self.decision_points: int = 0

    def new_block(self, block_type: str, start_line: int, end_line: int | None = None) -> CFGBlock:
        block = CFGBlock(
            id=self.current_block_id,
            start_line=start_line,
            end_line=end_line or start_line,
            block_type=block_type,
        )
        self.blocks.append(block)
        self.current_block_id += 1
        return block

    def add_edge(self, source_id: int, target_id: int, edge_type: str, condition: str | None = None):
        edge = CFGEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            condition=condition,
        )
        self.edges.append(edge)

    def get_node_text(self, node) -> str:
        return self.source[node.start_byte:node.end_byte].decode('utf-8')

    def build(self, func_node, func_name: str) -> CFGInfo:
        entry = self.new_block("entry", func_node.start_point[0] + 1)
        self.entry_block_id = entry.id
        self.current_block = entry

        body = None
        for child in func_node.children:
            if child.type == "block":
                body = child
                break

        if body:
            self._visit_node(body)
            if self.current_block:
                self.current_block.end_line = max(
                    self.current_block.end_line,
                    body.end_point[0] + 1
                )

        if self.current_block and self.current_block.id not in self.exit_block_ids:
            self.exit_block_ids.append(self.current_block.id)
            self.current_block.block_type = "exit"
            self.current_block.end_line = max(
                self.current_block.end_line,
                func_node.end_point[0] + 1
            )

        if not self.edges and len(self.blocks) == 1 and self.entry_block_id is not None:
            self.blocks[0].block_type = "entry"
            exit_block = self.new_block("exit", func_node.end_point[0] + 1)
            self.add_edge(self.entry_block_id, exit_block.id, "unconditional")
            self.exit_block_ids = [exit_block.id]

        return CFGInfo(
            function_name=func_name,
            blocks=self.blocks,
            edges=self.edges,
            entry_block_id=self.entry_block_id,
            exit_block_ids=self.exit_block_ids,
            cyclomatic_complexity=self.decision_points + 1,
        )

    def _visit_node(self, node):
        node_type = node.type
        if node_type in ("if_expression", "if_statement"):
            self._visit_if(node)
        elif node_type in ("while_expression", "while_statement"):
            self._visit_while(node)
        elif node_type in ("for_expression", "for_statement"):
            self._visit_for(node)
        elif node_type == "switch_expression":
            self._visit_switch(node)
        elif node_type == "return_statement":
            self._visit_return(node)
        elif node_type == "break_statement":
            self._visit_break(node)
        elif node_type == "continue_statement":
            self._visit_continue(node)
        else:
            for child in node.children:
                if child.is_named:
                    self._visit_node(child)

    def _visit_if(self, node):
        self.decision_points += 1

        if self.current_block:
            self.current_block.block_type = "branch"
            self.current_block.end_line = node.start_point[0] + 1
            branch_block_id = self.current_block.id
        else:
            branch = self.new_block("branch", node.start_point[0] + 1)
            branch_block_id = branch.id

        cond_node = node.child_by_field_name("condition")
        condition = self.get_node_text(cond_node) if cond_node else "<condition>"
        after_if = self.new_block("body", node.end_point[0] + 1)

        consequence = node.child_by_field_name("consequence")
        if not consequence:
            for child in node.children:
                if child.type == "block":
                    consequence = child
                    break

        alternative = node.child_by_field_name("alternative")
        if not alternative:
            blocks_seen = 0
            for child in node.children:
                if child.type == "block":
                    blocks_seen += 1
                    if blocks_seen == 2:
                        alternative = child
                        break
                elif child.type in ("else_clause", "else"):
                    alternative = child
                    break

        if not consequence:
            for child in node.children:
                if child.is_named and child.type not in ("else_clause", "else", "identifier", "binary_expression"):
                    if child != alternative:
                        consequence = child
                        break

        if consequence:
            true_block = self.new_block("body", consequence.start_point[0] + 1, consequence.end_point[0] + 1)
            self.add_edge(branch_block_id, true_block.id, "true", condition)
            self.current_block = true_block
            self._visit_node(consequence)
            if self.current_block and self.current_block.id not in self.exit_block_ids:
                self.add_edge(self.current_block.id, after_if.id, "unconditional")

        if alternative:
            false_block = self.new_block("body", alternative.start_point[0] + 1, alternative.end_point[0] + 1)
            self.add_edge(branch_block_id, false_block.id, "false", f"not ({condition})")
            self.current_block = false_block
            self._visit_node(alternative)
            if self.current_block and self.current_block.id not in self.exit_block_ids:
                self.add_edge(self.current_block.id, after_if.id, "unconditional")
        else:
            self.add_edge(branch_block_id, after_if.id, "false", f"not ({condition})")

        self.current_block = after_if

    def _visit_while(self, node):
        self.decision_points += 1

        guard = self.new_block("loop_header", node.start_point[0] + 1)
        if self.current_block:
            self.add_edge(self.current_block.id, guard.id, "unconditional")

        after_loop = self.new_block("body", node.end_point[0] + 1)

        self.loop_guard_stack.append(guard.id)
        self.after_loop_stack.append(after_loop.id)

        body = node.child_by_field_name("body")

        if body:
            loop_body = self.new_block("loop_body", body.start_point[0] + 1, body.end_point[0] + 1)
            cond_node = node.child_by_field_name("condition")
            while_cond = self.get_node_text(cond_node) if cond_node else "<while_cond>"
            self.add_edge(guard.id, loop_body.id, "true", while_cond)
            self.add_edge(guard.id, after_loop.id, "false", f"not ({while_cond})")

            self.current_block = loop_body
            self._visit_node(body)

            if self.current_block and self.current_block.id not in self.exit_block_ids:
                self.add_edge(self.current_block.id, guard.id, "back_edge")

        self.loop_guard_stack.pop()
        self.after_loop_stack.pop()
        self.current_block = after_loop

    def _visit_for(self, node):
        self.decision_points += 1

        guard = self.new_block("loop_header", node.start_point[0] + 1)
        if self.current_block:
            self.add_edge(self.current_block.id, guard.id, "unconditional")

        after_loop = self.new_block("body", node.end_point[0] + 1)

        self.loop_guard_stack.append(guard.id)
        self.after_loop_stack.append(after_loop.id)

        body = node.child_by_field_name("body")

        if body:
            loop_body = self.new_block("loop_body", body.start_point[0] + 1, body.end_point[0] + 1)
            self.add_edge(guard.id, loop_body.id, "iterate")
            self.add_edge(guard.id, after_loop.id, "exhausted")

            self.current_block = loop_body
            self._visit_node(body)

            if self.current_block and self.current_block.id not in self.exit_block_ids:
                self.add_edge(self.current_block.id, guard.id, "back_edge")

        self.loop_guard_stack.pop()
        self.after_loop_stack.pop()
        self.current_block = after_loop

    def _visit_return(self, node):
        if self.current_block:
            self.current_block.block_type = "return"
            self.current_block.end_line = node.start_point[0] + 1
            self.exit_block_ids.append(self.current_block.id)
        self.current_block = self.new_block("body", node.start_point[0] + 1)

    def _visit_break(self, node):
        if self.after_loop_stack and self.current_block:
            self.add_edge(self.current_block.id, self.after_loop_stack[-1], "break")
            self.current_block = self.new_block("body", node.start_point[0] + 1)

    def _visit_continue(self, node):
        if self.loop_guard_stack and self.current_block:
            self.add_edge(self.current_block.id, self.loop_guard_stack[-1], "continue")
            self.current_block = self.new_block("body", node.start_point[0] + 1)

    def _visit_switch(self, node):
        self.decision_points += 1

        if self.current_block:
            self.current_block.block_type = "branch"
            self.current_block.end_line = node.start_point[0] + 1
            switch_block_id = self.current_block.id
        else:
            switch_block = self.new_block("branch", node.start_point[0] + 1)
            switch_block_id = switch_block.id

        after_switch = self.new_block("body", node.end_point[0] + 1)

        for child in node.children:
            if child.type in ("switch_arm", "switch_prong"):
                self.decision_points += 1
                arm_block = self.new_block("body", child.start_point[0] + 1, child.end_point[0] + 1)
                self.add_edge(switch_block_id, arm_block.id, "case")
                self.current_block = arm_block
                self._visit_node(child)
                if self.current_block and self.current_block.id not in self.exit_block_ids:
                    self.add_edge(self.current_block.id, after_switch.id, "unconditional")

        self.current_block = after_switch


def _find_zig_function_node(root, function_name: str, source: bytes):
    def find_in_node(node):
        if node.type in ("function_declaration", "test_declaration"):
            for child in node.children:
                if child.type == "identifier":
                    name = source[child.start_byte:child.end_byte].decode('utf-8')
                    if name == function_name:
                        return node
                    break

        for child in node.children:
            result = find_in_node(child)
            if result:
                return result
        return None

    return find_in_node(root)


def extract_zig_cfg(source: str, function_name: str) -> CFGInfo:
    if not TREE_SITTER_ZIG_AVAILABLE:
        raise ImportError("tree-sitter-zig not available")

    source_bytes = source.encode('utf-8')
    parser = _get_ts_parser("zig")
    tree = parser.parse(source_bytes)

    func_node = _find_zig_function_node(tree.root_node, function_name, source_bytes)
    if not func_node:
        raise ValueError(f"Function '{function_name}' not found in source")

    builder = ZigCFGBuilder(source_bytes)
    return builder.build(func_node, function_name)
