"""
Program Dependence Graph (PDG) extraction for multi-language code analysis.

PDG combines CFG (control flow) and DFG (data flow) into a unified graph where:
- Control dependencies: "this statement executes only if that condition is true"
- Data dependencies: "this statement uses a value computed by that statement"

Why it helps LLMs:
- Program slicing: "what code affects variable X at line Y?"
- Code similarity detection
- Refactoring impact analysis: "what breaks if I change this line?"
- Better semantic structure than AST alone

Architecture (following ARISTODE pattern):
- All 3 graphs accessible separately (CFG, DFG, PDG)
- Unified edge labeling (control vs data)
- Support for forward/backward slicing
"""

from collections import deque
from dataclasses import dataclass, field

from .cfg_extractor import CFGInfo, extract_python_cfg
from .dfg_extractor import DFGInfo, extract_python_dfg


# =============================================================================
# PDG Data Structures
# =============================================================================


@dataclass(slots=True)
class PDGNode:
    """
    A node in the PDG representing a statement or expression.

    Maps to CFG blocks but also tracks data flow through the node.
    """

    id: int
    node_type: str  # "statement", "branch", "loop", "entry", "exit"
    start_line: int
    end_line: int

    # Data flow at this node
    definitions: list[str] = field(default_factory=list)  # Variables defined here
    uses: list[str] = field(default_factory=list)  # Variables used here

    # CFG block reference
    cfg_block_id: int | None = None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "type": self.node_type,
            "lines": [self.start_line, self.end_line],
        }
        if self.definitions:
            d["defs"] = self.definitions
        if self.uses:
            d["uses"] = self.uses
        return d


@dataclass(slots=True)
class PDGEdge:
    """
    An edge in the PDG with dependency type labeling.

    Edge types:
    - "control": Control dependency (from CFG)
      - "control:true" / "control:false": Branch conditions
      - "control:unconditional": Sequential flow
      - "control:back_edge": Loop back
    - "data": Data dependency (from DFG)
      - "data:<varname>": Def-use chain for variable
    """

    source_id: int
    target_id: int
    dep_type: str  # "control" or "data"
    label: str  # e.g., "true", "false", "unconditional", or variable name

    def to_dict(self) -> dict:
        return {
            "from": self.source_id,
            "to": self.target_id,
            "type": self.dep_type,
            "label": self.label,
        }

    @property
    def full_type(self) -> str:
        """Get full type string like 'control:true' or 'data:x'."""
        return f"{self.dep_type}:{self.label}"


@dataclass
class PDGInfo:
    """
    Program Dependence Graph combining CFG and DFG.

    Provides:
    - Access to underlying CFG and DFG separately
    - Unified node/edge view with labeled edges
    - Program slicing operations
    """

    function_name: str

    # Underlying graphs (accessible separately per ARISTODE pattern)
    cfg: CFGInfo
    dfg: DFGInfo

    # Unified PDG representation
    nodes: list[PDGNode] = field(default_factory=list)
    edges: list[PDGEdge] = field(default_factory=list)

    # Internal cache for O(1) node lookups (built lazily, excluded from repr/eq)
    _node_by_id_cache: dict[int, PDGNode] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def _node_by_id(self) -> dict[int, PDGNode]:
        """
        Lazily build and cache node lookup dict for O(1) access by ID.

        Replaces O(n) linear searches in slicing operations with O(1) dict lookups,
        providing 100x+ speedup for large PDGs during BFS traversal.
        """
        if self._node_by_id_cache is None:
            self._node_by_id_cache = {n.id: n for n in self.nodes}
        return self._node_by_id_cache

    def to_dict(self) -> dict:
        """Export full PDG with all layers."""
        return {
            "function": self.function_name,
            "pdg": {
                "nodes": [n.to_dict() for n in self.nodes],
                "edges": [e.to_dict() for e in self.edges],
            },
            "cfg": self.cfg.to_dict(),
            "dfg": self.dfg.to_dict(),
        }

    def to_compact_dict(self) -> dict:
        """Export compact PDG summary (for agent context)."""
        # Count edge types
        control_edges = sum(1 for e in self.edges if e.dep_type == "control")
        data_edges = sum(1 for e in self.edges if e.dep_type == "data")

        return {
            "function": self.function_name,
            "nodes": len(self.nodes),
            "control_edges": control_edges,
            "data_edges": data_edges,
            "complexity": self.cfg.cyclomatic_complexity,
            "variables": list(self.dfg.variables.keys()),
        }

    # =========================================================================
    # Program Slicing Operations
    # =========================================================================

    def backward_slice(self, line: int, variable: str | None = None) -> set[int]:
        """
        Compute backward slice: all statements that can affect the given line.

        Args:
            line: Line number to slice from
            variable: Optional specific variable to trace (traces all if None)

        Returns:
            Set of line numbers in the backward slice
        """
        # Find nodes at the target line
        target_nodes = [n for n in self.nodes if n.start_line <= line <= n.end_line]
        if not target_nodes:
            return set()

        # Build reverse edge map
        incoming: dict[int, list[PDGEdge]] = {}
        for edge in self.edges:
            if edge.target_id not in incoming:
                incoming[edge.target_id] = []
            incoming[edge.target_id].append(edge)

        # BFS backward through dependencies
        slice_lines: set[int] = set()
        visited: set[int] = set()
        worklist: deque[int] = deque(n.id for n in target_nodes)

        while worklist:
            node_id = worklist.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)

            # Add this node's lines to slice (O(1) lookup via cached dict)
            node = self._node_by_id.get(node_id)
            if node:
                for line_num in range(node.start_line, node.end_line + 1):
                    slice_lines.add(line_num)

            # Follow incoming edges
            for edge in incoming.get(node_id, []):
                # If filtering by variable, only follow relevant data edges
                if variable and edge.dep_type == "data" and edge.label != variable:
                    continue
                worklist.append(edge.source_id)

        return slice_lines

    def forward_slice(self, line: int, variable: str | None = None) -> set[int]:
        """
        Compute forward slice: all statements that can be affected by the given line.

        Args:
            line: Line number to slice from
            variable: Optional specific variable to trace (traces all if None)

        Returns:
            Set of line numbers in the forward slice
        """
        # Find nodes at the source line
        source_nodes = [n for n in self.nodes if n.start_line <= line <= n.end_line]
        if not source_nodes:
            return set()

        # Build forward edge map
        outgoing: dict[int, list[PDGEdge]] = {}
        for edge in self.edges:
            if edge.source_id not in outgoing:
                outgoing[edge.source_id] = []
            outgoing[edge.source_id].append(edge)

        # BFS forward through dependencies
        slice_lines: set[int] = set()
        visited: set[int] = set()
        worklist: deque[int] = deque(n.id for n in source_nodes)

        while worklist:
            node_id = worklist.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)

            # Add this node's lines to slice (O(1) lookup via cached dict)
            node = self._node_by_id.get(node_id)
            if node:
                for line_num in range(node.start_line, node.end_line + 1):
                    slice_lines.add(line_num)

            # Follow outgoing edges
            for edge in outgoing.get(node_id, []):
                # If filtering by variable, only follow relevant data edges
                if variable and edge.dep_type == "data" and edge.label != variable:
                    continue
                worklist.append(edge.target_id)

        return slice_lines

    def get_dependencies(self, line: int) -> dict[str, list[dict]]:
        """
        Get all dependencies for a line (both incoming and outgoing).

        Returns:
            Dict with 'control_in', 'control_out', 'data_in', 'data_out' keys
        """
        # Find nodes at the line
        target_nodes = [n for n in self.nodes if n.start_line <= line <= n.end_line]
        if not target_nodes:
            return {"control_in": [], "control_out": [], "data_in": [], "data_out": []}

        target_ids = {n.id for n in target_nodes}

        result = {
            "control_in": [],
            "control_out": [],
            "data_in": [],
            "data_out": [],
        }

        for edge in self.edges:
            edge_dict = edge.to_dict()

            if edge.target_id in target_ids:
                key = f"{edge.dep_type}_in"
                result[key].append(edge_dict)

            if edge.source_id in target_ids:
                key = f"{edge.dep_type}_out"
                result[key].append(edge_dict)

        return result


# =============================================================================
# PDG Construction
# =============================================================================


class PDGBuilder:
    """
    Build PDG by merging CFG and DFG.

    Steps:
    1. Build CFG for control dependencies
    2. Build DFG for data dependencies
    3. Create unified nodes from CFG blocks
    4. Add control edges from CFG
    5. Add data edges from DFG, mapping line numbers to nodes
    """

    def __init__(self, cfg: CFGInfo, dfg: DFGInfo):
        self.cfg = cfg
        self.dfg = dfg
        self.nodes: list[PDGNode] = []
        self.edges: list[PDGEdge] = []

        # Map from line to node ID for data flow edge mapping
        self._line_to_node: dict[int, int] = {}
        # Map from node ID to node for O(1) lookups during construction
        self._node_by_id: dict[int, PDGNode] = {}

    def build(self) -> PDGInfo:
        """Build the PDG from CFG and DFG."""
        self._create_nodes_from_cfg()
        self._add_control_edges()
        self._add_data_edges()

        return PDGInfo(
            function_name=self.cfg.function_name,
            cfg=self.cfg,
            dfg=self.dfg,
            nodes=self.nodes,
            edges=self.edges,
        )

    def _create_nodes_from_cfg(self):
        """Create PDG nodes from CFG blocks."""
        # Map CFG block types to PDG node types
        type_map = {
            "entry": "entry",
            "exit": "exit",
            "branch": "branch",
            "loop_header": "loop",
            "loop_body": "statement",
            "body": "statement",
            "return": "statement",
        }

        for block in self.cfg.blocks:
            node = PDGNode(
                id=block.id,
                node_type=type_map.get(block.block_type, "statement"),
                start_line=block.start_line,
                end_line=block.end_line,
                cfg_block_id=block.id,
            )
            self.nodes.append(node)
            # Build node ID -> node mapping for O(1) lookups
            self._node_by_id[block.id] = node

            # Build line -> node mapping
            for line in range(block.start_line, block.end_line + 1):
                self._line_to_node[line] = block.id

        # Add variable refs to nodes (O(1) lookup via dict)
        for ref in self.dfg.var_refs:
            node_id = self._line_to_node.get(ref.line)
            if node_id is not None:
                node = self._node_by_id.get(node_id)
                if node:
                    if ref.ref_type in ("definition", "update"):
                        if ref.name not in node.definitions:
                            node.definitions.append(ref.name)
                    elif ref.ref_type == "use":
                        if ref.name not in node.uses:
                            node.uses.append(ref.name)

    def _add_control_edges(self):
        """Add control dependency edges from CFG."""
        for cfg_edge in self.cfg.edges:
            edge = PDGEdge(
                source_id=cfg_edge.source_id,
                target_id=cfg_edge.target_id,
                dep_type="control",
                label=cfg_edge.edge_type,
            )
            self.edges.append(edge)

    def _add_data_edges(self):
        """Add data dependency edges from DFG."""
        for df_edge in self.dfg.dataflow_edges:
            # Map def line to source node
            source_node_id = self._line_to_node.get(df_edge.def_ref.line)
            # Map use line to target node
            target_node_id = self._line_to_node.get(df_edge.use_ref.line)

            if source_node_id is not None and target_node_id is not None:
                # Avoid self-loops (def and use in same block)
                if source_node_id != target_node_id:
                    edge = PDGEdge(
                        source_id=source_node_id,
                        target_id=target_node_id,
                        dep_type="data",
                        label=df_edge.var_name,
                    )
                    self.edges.append(edge)


# =============================================================================
# Python PDG Extraction
# =============================================================================


def extract_python_pdg(source_code: str, function_name: str) -> PDGInfo | None:
    """
    Extract PDG for a Python function.

    Args:
        source_code: Python source code containing the function
        function_name: Name of the function to analyze

    Returns:
        PDGInfo with CFG, DFG, and merged PDG, or None if extraction fails
    """
    try:
        # Extract CFG
        cfg = extract_python_cfg(source_code, function_name)
        if cfg is None:
            return None

        # Extract DFG
        dfg = extract_python_dfg(source_code, function_name)
        if dfg is None:
            return None

        # Build PDG
        builder = PDGBuilder(cfg, dfg)
        return builder.build()
    except ValueError:
        # Function not found
        return None


# =============================================================================
# TypeScript/JavaScript PDG Extraction
# =============================================================================


def extract_typescript_pdg(source_code: str, function_name: str) -> PDGInfo | None:
    """
    Extract PDG for a TypeScript/JavaScript function.
    """
    try:
        from .cfg_extractor import extract_typescript_cfg
        from .dfg_extractor import extract_typescript_dfg

        cfg = extract_typescript_cfg(source_code, function_name)
        if cfg is None:
            return None

        dfg = extract_typescript_dfg(source_code, function_name)
        if dfg is None:
            return None

        builder = PDGBuilder(cfg, dfg)
        return builder.build()
    except ValueError:
        return None


def extract_javascript_pdg(source_code: str, function_name: str) -> PDGInfo | None:
    """
    Extract PDG for a JavaScript function.

    Uses TypeScript extractors since tree-sitter parses JS/TS identically.
    """
    try:
        from .cfg_extractor import extract_typescript_cfg
        from .dfg_extractor import extract_typescript_dfg

        cfg = extract_typescript_cfg(source_code, function_name)
        if cfg is None:
            return None

        dfg = extract_typescript_dfg(source_code, function_name)
        if dfg is None:
            return None

        builder = PDGBuilder(cfg, dfg)
        return builder.build()
    except ValueError:
        return None


# =============================================================================
# Go PDG Extraction
# =============================================================================


def extract_go_pdg(source_code: str, function_name: str) -> PDGInfo | None:
    """
    Extract PDG for a Go function.
    """
    try:
        from .cfg_extractor import extract_go_cfg
        from .dfg_extractor import extract_go_dfg

        cfg = extract_go_cfg(source_code, function_name)
        if cfg is None:
            return None

        dfg = extract_go_dfg(source_code, function_name)
        if dfg is None:
            return None

        builder = PDGBuilder(cfg, dfg)
        return builder.build()
    except ValueError:
        return None


# =============================================================================
# Rust PDG Extraction
# =============================================================================


def extract_rust_pdg(source_code: str, function_name: str) -> PDGInfo | None:
    """
    Extract PDG for a Rust function.
    """
    try:
        from .cfg_extractor import extract_rust_cfg
        from .dfg_extractor import extract_rust_dfg

        cfg = extract_rust_cfg(source_code, function_name)
        if cfg is None:
            return None

        dfg = extract_rust_dfg(source_code, function_name)
        if dfg is None:
            return None

        builder = PDGBuilder(cfg, dfg)
        return builder.build()
    except ValueError:
        return None


# =============================================================================
# Java PDG Extraction
# =============================================================================


def extract_java_pdg(source_code: str, function_name: str) -> PDGInfo | None:
    """
    Extract PDG for a Java function.
    """
    try:
        from .cfg_extractor import extract_java_cfg
        from .dfg_extractor import extract_java_dfg

        cfg = extract_java_cfg(source_code, function_name)
        if cfg is None:
            return None

        dfg = extract_java_dfg(source_code, function_name)
        if dfg is None:
            return None

        builder = PDGBuilder(cfg, dfg)
        return builder.build()
    except ValueError:
        return None


# =============================================================================
# C PDG Extraction
# =============================================================================


def extract_c_pdg(source_code: str, function_name: str) -> PDGInfo | None:
    """
    Extract PDG for a C function.
    """
    try:
        from .cfg_extractor import extract_c_cfg
        from .dfg_extractor import extract_c_dfg

        cfg = extract_c_cfg(source_code, function_name)
        if cfg is None:
            return None

        dfg = extract_c_dfg(source_code, function_name)
        if dfg is None:
            return None

        builder = PDGBuilder(cfg, dfg)
        return builder.build()
    except ValueError:
        return None


# =============================================================================
# C++ PDG Extraction
# =============================================================================


def extract_cpp_pdg(source_code: str, function_name: str) -> PDGInfo | None:
    """
    Extract PDG for a C++ function.
    """
    try:
        from .cfg_extractor import extract_cpp_cfg
        from .dfg_extractor import extract_cpp_dfg

        cfg = extract_cpp_cfg(source_code, function_name)
        if cfg is None:
            return None

        dfg = extract_cpp_dfg(source_code, function_name)
        if dfg is None:
            return None

        builder = PDGBuilder(cfg, dfg)
        return builder.build()
    except ValueError:
        return None


# =============================================================================
# Ruby PDG Extraction
# =============================================================================


def extract_ruby_pdg(source_code: str, function_name: str) -> PDGInfo | None:
    """
    Extract PDG for a Ruby function.
    """
    try:
        from .cfg_extractor import extract_ruby_cfg
        from .dfg_extractor import extract_ruby_dfg

        cfg = extract_ruby_cfg(source_code, function_name)
        if cfg is None:
            return None

        dfg = extract_ruby_dfg(source_code, function_name)
        if dfg is None:
            return None

        builder = PDGBuilder(cfg, dfg)
        return builder.build()
    except ValueError:
        return None


# =============================================================================
# PHP PDG Extraction
# =============================================================================


def extract_php_pdg(source_code: str, function_name: str) -> PDGInfo | None:
    """
    Extract PDG for a PHP function.

    Args:
        source_code: PHP source code (may include <?php tag)
        function_name: Name of function to analyze

    Returns:
        PDGInfo with combined control/data flow, or None if function not found
    """
    try:
        from .cfg_extractor import extract_php_cfg
        from .dfg_extractor import extract_php_dfg

        cfg = extract_php_cfg(source_code, function_name)
        if cfg is None:
            return None

        dfg = extract_php_dfg(source_code, function_name)
        if dfg is None:
            return None

        builder = PDGBuilder(cfg, dfg)
        return builder.build()
    except ValueError:
        return None


# =============================================================================
# Kotlin PDG Extraction
# =============================================================================


def extract_kotlin_pdg(source_code: str, function_name: str) -> PDGInfo | None:
    """
    Extract PDG for a Kotlin function.

    Args:
        source_code: Kotlin source code
        function_name: Name of function to analyze

    Returns:
        PDGInfo with combined control/data flow, or None if function not found
    """
    try:
        from .cfg_extractor import extract_kotlin_cfg
        from .dfg_extractor import extract_kotlin_dfg

        cfg = extract_kotlin_cfg(source_code, function_name)
        if cfg is None:
            return None

        dfg = extract_kotlin_dfg(source_code, function_name)
        if dfg is None:
            return None

        builder = PDGBuilder(cfg, dfg)
        return builder.build()
    except ValueError:
        return None


# =============================================================================
# Swift PDG Extraction
# =============================================================================


def extract_swift_pdg(source_code: str, function_name: str) -> PDGInfo | None:
    """
    Extract PDG for a Swift function.

    Args:
        source_code: Swift source code
        function_name: Name of function to analyze

    Returns:
        PDGInfo with combined control/data flow, or None if function not found
    """
    try:
        from .cfg_extractor import extract_swift_cfg
        from .dfg_extractor import extract_swift_dfg

        cfg = extract_swift_cfg(source_code, function_name)
        if cfg is None:
            return None

        dfg = extract_swift_dfg(source_code, function_name)
        if dfg is None:
            return None

        builder = PDGBuilder(cfg, dfg)
        return builder.build()
    except ValueError:
        return None


def extract_csharp_pdg(source_code: str, function_name: str) -> PDGInfo | None:
    """
    Extract PDG for a C# method.

    Args:
        source_code: C# source code
        function_name: Name of method to analyze

    Returns:
        PDGInfo with combined control/data flow, or None if method not found
    """
    try:
        from .cfg_extractor import extract_csharp_cfg
        from .dfg_extractor import extract_csharp_dfg

        cfg = extract_csharp_cfg(source_code, function_name)
        if cfg is None:
            return None

        dfg = extract_csharp_dfg(source_code, function_name)
        if dfg is None:
            return None

        builder = PDGBuilder(cfg, dfg)
        return builder.build()
    except ValueError:
        return None


# =============================================================================
# Scala PDG Extraction
# =============================================================================


def extract_scala_pdg(source_code: str, function_name: str) -> PDGInfo | None:
    """
    Extract PDG for a Scala function.

    Args:
        source_code: Scala source code
        function_name: Name of function to analyze

    Returns:
        PDGInfo with combined control/data flow, or None if function not found
    """
    try:
        from .cfg_extractor import extract_scala_cfg
        from .dfg_extractor import extract_scala_dfg

        cfg = extract_scala_cfg(source_code, function_name)
        if cfg is None:
            return None

        dfg = extract_scala_dfg(source_code, function_name)
        if dfg is None:
            return None

        builder = PDGBuilder(cfg, dfg)
        return builder.build()
    except ValueError:
        return None


# =============================================================================
# Lua PDG Extraction
# =============================================================================


def extract_lua_pdg(source_code: str, function_name: str) -> PDGInfo | None:
    """
    Extract PDG for a Lua function.

    Args:
        source_code: Lua source code
        function_name: Name of function to analyze

    Returns:
        PDGInfo with combined control/data flow, or None if function not found
    """
    try:
        from .cfg_extractor import extract_lua_cfg
        from .dfg_extractor import extract_lua_dfg

        cfg = extract_lua_cfg(source_code, function_name)
        if cfg is None:
            return None

        dfg = extract_lua_dfg(source_code, function_name)
        if dfg is None:
            return None

        builder = PDGBuilder(cfg, dfg)
        return builder.build()
    except ValueError:
        return None


# =============================================================================
# Elixir PDG Extraction
# =============================================================================


def extract_elixir_pdg(source_code: str, function_name: str) -> PDGInfo | None:
    """
    Extract PDG for an Elixir function.

    Args:
        source_code: Elixir source code
        function_name: Name of function to analyze

    Returns:
        PDGInfo with combined control/data flow, or None if function not found
    """
    try:
        from .cfg_extractor import extract_elixir_cfg
        from .dfg_extractor import extract_elixir_dfg

        cfg = extract_elixir_cfg(source_code, function_name)
        if cfg is None:
            return None

        dfg = extract_elixir_dfg(source_code, function_name)
        if dfg is None:
            return None

        builder = PDGBuilder(cfg, dfg)
        return builder.build()
    except ValueError:
        return None


# =============================================================================
# Multi-language convenience function
# =============================================================================


def extract_pdg(source_code: str, function_name: str, language: str) -> PDGInfo | None:
    """
    Extract PDG for any supported language.

    Args:
        source_code: Source code containing the function
        function_name: Name of the function to analyze
        language: One of "python", "typescript", "javascript", "go", "rust", "java", "c", "ruby", "php", "csharp", "elixir"

    Returns:
        PDGInfo or None if extraction fails
    """
    extractors = {
        "python": extract_python_pdg,
        "typescript": extract_typescript_pdg,
        "javascript": extract_javascript_pdg,
        "go": extract_go_pdg,
        "rust": extract_rust_pdg,
        "java": extract_java_pdg,
        "c": extract_c_pdg,
        "cpp": extract_cpp_pdg,
        "ruby": extract_ruby_pdg,
        "php": extract_php_pdg,
        "kotlin": extract_kotlin_pdg,
        "swift": extract_swift_pdg,
        "csharp": extract_csharp_pdg,
        "scala": extract_scala_pdg,
        "lua": extract_lua_pdg,
        "elixir": extract_elixir_pdg,
    }

    extractor = extractors.get(language.lower())
    if extractor is None:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported: {', '.join(extractors.keys())}"
        )

    return extractor(source_code, function_name)
