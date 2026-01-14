"""Tests for Luau Program Dependency Graph extraction (TDD - write tests first).

Luau PDG combines:
- Control Flow Graph (CFG) - branches, loops, continue
- Data Flow Graph (DFG) - variable definitions and uses

These tests define expected behavior for extract_luau_pdg().
"""

import pytest


class TestLuauPDGBasic:
    """Tests for basic Luau PDG extraction."""

    def test_luau_pdg_simple_function(self):
        """Should extract PDG for simple typed function."""
        from tldr.pdg_extractor import extract_luau_pdg

        code = """function simple(x: number): number
    local y = x + 1
    return y
end
"""
        pdg = extract_luau_pdg(code, "simple")

        assert pdg is not None
        # PDG should have data edges (control edges are only for branches)
        data_edges = [e for e in pdg.edges if e.dep_type == "data"]
        assert len(data_edges) > 0
        # Data edge from x parameter to y definition
        data_vars = {edge.label for edge in data_edges}
        assert "x" in data_vars or "y" in data_vars

    def test_luau_pdg_with_continue(self):
        """Should handle continue statement (Luau-specific) in PDG."""
        from tldr.pdg_extractor import extract_luau_pdg

        code = """function filterOdd(n: number): number
    local sum = 0
    for i = 1, n do
        if i % 2 == 0 then continue end
        sum += i
    end
    return sum
end
"""
        pdg = extract_luau_pdg(code, "filterOdd")

        assert pdg is not None
        # Continue should create control flow edges
        control_edges = [e for e in pdg.edges if e.dep_type == "control"]
        assert len(control_edges) >= 3  # At least entry, loop, continue
        # Data edges for sum at DFG level (PDG is block-level)
        sum_edges = [e for e in pdg.dfg.dataflow_edges if e.var_name == "sum"]
        assert len(sum_edges) > 0


class TestLuauPDGCompoundAssignment:
    """Tests for compound assignment operators (Luau-specific)."""

    def test_luau_pdg_compound_assignment_chain(self):
        """Should track data flow through compound assignments."""
        from tldr.pdg_extractor import extract_luau_pdg

        code = """function accumulate(): number
    local x = 0
    x += 1
    x += 2
    return x
end
"""
        pdg = extract_luau_pdg(code, "accumulate")

        assert pdg is not None
        # Compound assignment creates both USE and DEF at DFG level
        # x(init) -> x(+=1) -> x(+=2) -> return
        # Check the underlying DFG for the chain
        x_dfg_edges = [e for e in pdg.dfg.dataflow_edges if e.var_name == "x"]
        # Should have multiple data edges for x showing the chain
        assert len(x_dfg_edges) >= 2


class TestLuauPDGNotFound:
    """Tests for error handling."""

    def test_luau_pdg_function_not_found(self):
        """Should return None for non-existent function."""
        from tldr.pdg_extractor import extract_luau_pdg

        code = """function exists(): ()
end
"""
        pdg = extract_luau_pdg(code, "nonexistent")

        assert pdg is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
