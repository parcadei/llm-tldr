"""Tests for Luau DFG extraction (TDD - tests written before implementation).

These tests define the expected behavior for Data Flow Graph extraction
from Luau code, including Luau-specific features:
- compound assignment operators (+=, -=, *=) must create USE and DEF
- type annotations should not affect data flow
- continue statement interaction with variables

The function `extract_luau_dfg` does NOT exist yet - these tests must FAIL.
"""

import pytest


# =============================================================================
# Test 1: Basic Definition and Use
# =============================================================================

def test_luau_dfg_basic_def_use():
    """Basic variable definition and use should be tracked."""
    from tldr.dfg_extractor import extract_luau_dfg

    code = '''
function example(): number
    local x = 10
    local y = x + 5
    return y
end
'''
    dfg = extract_luau_dfg(code, "example")

    assert dfg is not None
    assert dfg.function_name == "example"

    # Find x definition
    x_defs = [r for r in dfg.var_refs if r.name == "x" and r.ref_type == "definition"]
    assert len(x_defs) >= 1, "x should have at least one definition"

    # Find x use
    x_uses = [r for r in dfg.var_refs if r.name == "x" and r.ref_type == "use"]
    assert len(x_uses) >= 1, "x should have at least one use"

    # Find y definition and use
    y_defs = [r for r in dfg.var_refs if r.name == "y" and r.ref_type == "definition"]
    y_uses = [r for r in dfg.var_refs if r.name == "y" and r.ref_type == "use"]
    assert len(y_defs) >= 1, "y should have at least one definition"
    assert len(y_uses) >= 1, "y should have at least one use (in return)"


# =============================================================================
# Test 2: Typed Variable Declaration
# =============================================================================

def test_luau_dfg_typed_declaration():
    """Type annotations should not affect DFG structure."""
    from tldr.dfg_extractor import extract_luau_dfg

    code = '''
function typed(): ()
    local name: string = "test"
    local count: number = 0
    print(name, count)
end
'''
    dfg = extract_luau_dfg(code, "typed")

    assert dfg is not None

    # name and count should be tracked despite type annotations
    names = {r.name for r in dfg.var_refs}
    assert "name" in names, "name variable should be tracked"
    assert "count" in names, "count variable should be tracked"


# =============================================================================
# Test 3: Compound Assignment (Luau-specific, critical)
# =============================================================================

def test_luau_dfg_compound_assignment():
    """Compound assignment (+=, -=, *=) must create both USE and DEF.

    This is CRITICAL for Luau - compound assignment is syntactic sugar for
    x = x + value, so it must be tracked as:
    1. USE of x (read current value)
    2. DEF of x (write new value)
    """
    from tldr.dfg_extractor import extract_luau_dfg

    code = '''
function compound(): number
    local x = 5
    x += 3
    x -= 1
    x *= 2
    return x
end
'''
    dfg = extract_luau_dfg(code, "compound")

    assert dfg is not None

    x_refs = [r for r in dfg.var_refs if r.name == "x"]
    x_defs = [r for r in x_refs if r.ref_type == "definition"]
    x_uses = [r for r in x_refs if r.ref_type == "use"]

    # Initial def + 3 compound assignments = 4 definitions
    assert len(x_defs) >= 4, f"Expected 4 definitions of x, got {len(x_defs)}"

    # 3 compound assignments + 1 return = 4 uses
    assert len(x_uses) >= 4, f"Expected 4 uses of x, got {len(x_uses)}"

    # Check dataflow edges: each compound creates use->def chain
    # At minimum, there should be edges connecting the uses to defs
    assert len(dfg.dataflow_edges) >= 3, "Should have dataflow edges for compound ops"


# =============================================================================
# Test 4: Function Parameters as Definitions
# =============================================================================

def test_luau_dfg_parameters():
    """Typed function parameters should be tracked as definitions."""
    from tldr.dfg_extractor import extract_luau_dfg

    code = '''
function greet(name: string, count: number): ()
    for i = 1, count do
        print(name)
    end
end
'''
    dfg = extract_luau_dfg(code, "greet")

    assert dfg is not None

    # Parameters are definitions
    name_defs = [r for r in dfg.var_refs if r.name == "name" and r.ref_type == "definition"]
    count_defs = [r for r in dfg.var_refs if r.name == "count" and r.ref_type == "definition"]

    assert len(name_defs) >= 1, "name parameter should be a definition"
    assert len(count_defs) >= 1, "count parameter should be a definition"

    # Parameters used in body
    name_uses = [r for r in dfg.var_refs if r.name == "name" and r.ref_type == "use"]
    count_uses = [r for r in dfg.var_refs if r.name == "count" and r.ref_type == "use"]

    assert len(name_uses) >= 1, "name should be used in print"
    assert len(count_uses) >= 1, "count should be used in for loop"


# =============================================================================
# Test 5: For Loop Variable
# =============================================================================

def test_luau_dfg_for_variable():
    """Numeric for loop variable should be tracked."""
    from tldr.dfg_extractor import extract_luau_dfg

    code = '''
function sumRange(): number
    local total = 0
    for i = 1, 10 do
        total += i
    end
    return total
end
'''
    dfg = extract_luau_dfg(code, "sumRange")

    assert dfg is not None

    # i is defined by for loop
    i_defs = [r for r in dfg.var_refs if r.name == "i" and r.ref_type == "definition"]
    assert len(i_defs) >= 1, "i should be defined by for loop"

    # i is used in compound assignment
    i_uses = [r for r in dfg.var_refs if r.name == "i" and r.ref_type == "use"]
    assert len(i_uses) >= 1, "i should be used in total += i"


# =============================================================================
# Test 6: Generic For Variables
# =============================================================================

def test_luau_dfg_generic_for():
    """Generic for-in loop should define iterator variables."""
    from tldr.dfg_extractor import extract_luau_dfg

    code = '''
function process(items: {Item}): ()
    for index, item in items do
        item:process()
        print(index)
    end
end
'''
    dfg = extract_luau_dfg(code, "process")

    assert dfg is not None

    # index and item are defined by for-in
    index_defs = [r for r in dfg.var_refs if r.name == "index" and r.ref_type == "definition"]
    item_defs = [r for r in dfg.var_refs if r.name == "item" and r.ref_type == "definition"]

    assert len(index_defs) >= 1, "index should be defined by for-in"
    assert len(item_defs) >= 1, "item should be defined by for-in"


# =============================================================================
# Test 7: Table Field Access
# =============================================================================

def test_luau_dfg_table_access():
    """Table field access should track the table variable."""
    from tldr.dfg_extractor import extract_luau_dfg

    code = '''
function updatePlayer(player: Player): ()
    local oldHealth = player.health
    player.health = 100
    print(oldHealth)
end
'''
    dfg = extract_luau_dfg(code, "updatePlayer")

    assert dfg is not None

    # player is used (accessing .health)
    player_uses = [r for r in dfg.var_refs if r.name == "player" and r.ref_type == "use"]
    assert len(player_uses) >= 1, "player should be used when accessing fields"

    # oldHealth is defined and used
    old_defs = [r for r in dfg.var_refs if r.name == "oldHealth" and r.ref_type == "definition"]
    old_uses = [r for r in dfg.var_refs if r.name == "oldHealth" and r.ref_type == "use"]
    assert len(old_defs) >= 1
    assert len(old_uses) >= 1


# =============================================================================
# Test 8: Closure Capture
# =============================================================================

def test_luau_dfg_closure():
    """Closure should capture variables from outer scope."""
    from tldr.dfg_extractor import extract_luau_dfg

    code = '''
function makeCounter(): () -> number
    local count = 0
    return function()
        count += 1
        return count
    end
end
'''
    dfg = extract_luau_dfg(code, "makeCounter")

    assert dfg is not None

    # count is defined in outer function
    count_defs = [r for r in dfg.var_refs if r.name == "count" and r.ref_type == "definition"]
    assert len(count_defs) >= 1, "count should be defined"

    # count is used in inner function (closure capture)
    count_uses = [r for r in dfg.var_refs if r.name == "count" and r.ref_type == "use"]
    assert len(count_uses) >= 1, "count should be used in closure"


# =============================================================================
# Test 9: Multiple Assignment
# =============================================================================

def test_luau_dfg_multiple_assignment():
    """Multiple assignment should track all variables."""
    from tldr.dfg_extractor import extract_luau_dfg

    code = '''
function swap(): ()
    local a, b = 1, 2
    a, b = b, a
    print(a, b)
end
'''
    dfg = extract_luau_dfg(code, "swap")

    assert dfg is not None

    # Both a and b should have definitions
    a_defs = [r for r in dfg.var_refs if r.name == "a" and r.ref_type == "definition"]
    b_defs = [r for r in dfg.var_refs if r.name == "b" and r.ref_type == "definition"]

    # Initial declaration + swap = 2 definitions each
    assert len(a_defs) >= 2, f"a should have 2 definitions, got {len(a_defs)}"
    assert len(b_defs) >= 2, f"b should have 2 definitions, got {len(b_defs)}"


# =============================================================================
# Test 10: Optional Type Annotation
# =============================================================================

def test_luau_dfg_optional_type():
    """Optional type (number?) should not affect DFG."""
    from tldr.dfg_extractor import extract_luau_dfg

    code = '''
function maybeValue(x: number?): number
    if x then
        return x
    else
        return 0
    end
end
'''
    dfg = extract_luau_dfg(code, "maybeValue")

    assert dfg is not None

    # x is defined as parameter
    x_defs = [r for r in dfg.var_refs if r.name == "x" and r.ref_type == "definition"]
    assert len(x_defs) >= 1, "x should be defined as parameter"

    # x is used in condition and return
    x_uses = [r for r in dfg.var_refs if r.name == "x" and r.ref_type == "use"]
    assert len(x_uses) >= 2, "x should be used in if condition and return"


# =============================================================================
# Test: Function Not Found
# =============================================================================

def test_luau_dfg_function_not_found():
    """Should return empty DFG when function not found (not raise)."""
    from tldr.dfg_extractor import extract_luau_dfg

    code = '''
function exists(): ()
end
'''
    dfg = extract_luau_dfg(code, "nonexistent")

    # Following Lua pattern: return empty DFG, not raise
    assert dfg is not None
    assert dfg.function_name == "nonexistent"
    assert len(dfg.var_refs) == 0


# =============================================================================
# Test: Continue Does Not Break DFG
# =============================================================================

def test_luau_dfg_with_continue():
    """Continue statement should not break variable tracking."""
    from tldr.dfg_extractor import extract_luau_dfg

    code = '''
function sumOdd(n: number): number
    local total = 0
    for i = 1, n do
        if i % 2 == 0 then continue end
        total += i
    end
    return total
end
'''
    dfg = extract_luau_dfg(code, "sumOdd")

    assert dfg is not None

    # total should have def (initial) and uses (compound, return)
    total_defs = [r for r in dfg.var_refs if r.name == "total" and r.ref_type == "definition"]
    total_uses = [r for r in dfg.var_refs if r.name == "total" and r.ref_type == "use"]

    assert len(total_defs) >= 1, "total should be defined"
    assert len(total_uses) >= 1, "total should be used"
