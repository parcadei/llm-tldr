"""
TLDR-Code: Multi-layer Code Intelligence for Token-Efficient Agent Context

Provides 5 layers of code analysis:
- Layer 1: AST - Signatures, types, classes
- Layer 2: Call Graph - Who calls what, entry points
- Layer 3: CFG - Control flow, branches, loops, complexity
- Layer 4: DFG - Data flow, def-use chains
- Layer 5: PDG - Program dependencies, slicing

All layers accessible separately (ARISTODE pattern) or combined.
"""

try:
    from importlib.metadata import version
    __version__ = version("tldr-code")
except Exception:
    __version__ = "1.0.0"
__author__ = "parcadei"

# Original exports
from .signature_extractor_pygments import SignatureExtractor

# Layer 1: AST
from .ast_extractor import extract_python, extract_file

# Layer 2: Call Graph (hybrid extractor has multiple exports)
try:
    from .hybrid_extractor import extract_call_graph
except ImportError:
    extract_call_graph = None  # Optional dependency

# Layer 3: CFG
from .cfg_extractor import (
    CFGInfo,
    CFGBlock,
    CFGEdge,
    extract_python_cfg,
)

# Layer 4: DFG
from .dfg_extractor import (
    DFGInfo,
    VarRef,
    DataflowEdge,
    extract_python_dfg,
)

# Layer 5: PDG (combines CFG + DFG)
from .pdg_extractor import (
    PDGInfo,
    PDGNode,
    PDGEdge,
    extract_python_pdg,
    extract_pdg,
)

__all__ = [
    # Original
    "SignatureExtractor",
    # Layer 1: AST
    "extract_python",
    "extract_file",
    # Layer 2: Call Graph
    "extract_call_graph",
    # Layer 3: CFG
    "CFGInfo",
    "CFGBlock",
    "CFGEdge",
    "extract_python_cfg",
    # Layer 4: DFG
    "DFGInfo",
    "VarRef",
    "DataflowEdge",
    "extract_python_dfg",
    # Layer 5: PDG (multi-language via extract_pdg)
    "PDGInfo",
    "PDGNode",
    "PDGEdge",
    "extract_python_pdg",
    "extract_pdg",  # Multi-language convenience function
]