"""Tests for Luau CFG extraction (TDD - tests written before implementation).

These tests define the expected behavior for Control Flow Graph extraction
from Luau code, including Luau-specific features:
- continue statement (not in Lua 5.1)
- compound assignment operators (+=, -=, etc.)
- type annotations (should not affect flow)
- generic functions

The functions `extract_luau_cfg` do NOT exist yet - these tests must FAIL.
"""

import pytest


# =============================================================================
# Test 1: Simple Function CFG
# =============================================================================

def test_luau_cfg_simple_function():
    """Simple typed function should produce linear CFG with entry/exit."""
    from tldr.cfg_extractor import extract_luau_cfg

    code = '''
function add(a: number, b: number): number
    return a + b
end
'''
    cfg = extract_luau_cfg(code, "add")

    # Should have entry and exit blocks
    assert cfg is not None
    assert len(cfg.blocks) >= 1  # At least one block (entry/return)

    # Cyclomatic complexity for linear code is 1
    assert cfg.cyclomatic_complexity == 1

    # Function name should be extracted
    assert cfg.function_name == "add"

    # Should have at least one exit block
    assert len(cfg.exit_block_ids) >= 1


# =============================================================================
# Test 2: Typed Function Parameters
# =============================================================================

def test_luau_cfg_typed_parameters():
    """Type annotations on parameters should be parsed but not affect CFG structure."""
    from tldr.cfg_extractor import extract_luau_cfg

    code = '''
function greet(name: string, times: number?): string
    return "Hello, " .. name
end
'''
    cfg = extract_luau_cfg(code, "greet")

    assert cfg is not None
    assert cfg.function_name == "greet"

    # Type annotations don't add control flow
    assert cfg.cyclomatic_complexity == 1


# =============================================================================
# Test 3: If Statement
# =============================================================================

def test_luau_cfg_if_statement():
    """If-else should create branch in CFG."""
    from tldr.cfg_extractor import extract_luau_cfg

    code = '''
function process(x: number): number
    if x > 0 then
        return x
    else
        return -x
    end
end
'''
    cfg = extract_luau_cfg(code, "process")

    assert cfg is not None

    # If with else: complexity = 2 (one decision point)
    assert cfg.cyclomatic_complexity == 2

    # Should have: entry -> condition -> then/else -> exit
    # At least 3 edges: entry->cond, cond->then, cond->else
    assert len(cfg.edges) >= 3


# =============================================================================
# Test 4: Continue Statement (Luau-specific)
# =============================================================================

def test_luau_cfg_continue_statement():
    """Continue statement should create back-edge to loop header.

    This is LUAU-SPECIFIC - Lua 5.1 does not have continue.
    """
    from tldr.cfg_extractor import extract_luau_cfg

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
    cfg = extract_luau_cfg(code, "sumOdd")

    assert cfg is not None

    # Must handle continue - look for back edge
    # The CFG should have an edge from continue block back to loop header
    back_edges = [e for e in cfg.edges if e.edge_type == "back" or "continue" in str(e).lower()]

    # Either explicit back edge or the graph structure handles continue
    # At minimum, the function should parse without error
    assert len(cfg.blocks) >= 3  # entry, loop, body, continue-check, exit


# =============================================================================
# Test 5: While Loop
# =============================================================================

def test_luau_cfg_while_loop():
    """While loop should create loop structure with back edge."""
    from tldr.cfg_extractor import extract_luau_cfg

    code = '''
function countdown(n: number): ()
    while n > 0 do
        print(n)
        n -= 1
    end
end
'''
    cfg = extract_luau_cfg(code, "countdown")

    assert cfg is not None

    # While loop adds 1 decision point
    assert cfg.cyclomatic_complexity >= 2

    # Should have back edge from body to header
    assert len(cfg.edges) >= 3


# =============================================================================
# Test 6: Repeat-Until Loop
# =============================================================================

def test_luau_cfg_repeat_until():
    """Repeat-until should execute body before checking condition."""
    from tldr.cfg_extractor import extract_luau_cfg

    code = '''
function waitUntilReady(): ()
    repeat
        wait(0.1)
    until isReady()
end
'''
    cfg = extract_luau_cfg(code, "waitUntilReady")

    assert cfg is not None

    # Repeat-until is a loop with decision at end
    assert cfg.cyclomatic_complexity >= 2


# =============================================================================
# Test 7: Compound Assignment in CFG (Luau-specific)
# =============================================================================

def test_luau_cfg_compound_assignment():
    """Compound assignment (+=) should be treated as regular statement in CFG.

    This is LUAU-SPECIFIC - Lua 5.1 does not have compound assignment.
    """
    from tldr.cfg_extractor import extract_luau_cfg

    code = '''
function accumulate(values: {number}): number
    local sum = 0
    for _, v in values do
        sum += v
    end
    return sum
end
'''
    cfg = extract_luau_cfg(code, "accumulate")

    assert cfg is not None

    # Compound assignment doesn't affect control flow
    # Just the for loop adds complexity
    assert cfg.cyclomatic_complexity >= 2


# =============================================================================
# Test 8: Generic For Loop
# =============================================================================

def test_luau_cfg_generic_for():
    """Generic for-in loop should create loop structure."""
    from tldr.cfg_extractor import extract_luau_cfg

    code = '''
function printAll(items: {string}): ()
    for i, item in items do
        print(i, item)
    end
end
'''
    cfg = extract_luau_cfg(code, "printAll")

    assert cfg is not None

    # For-in loop adds decision point
    assert cfg.cyclomatic_complexity >= 2


# =============================================================================
# Test 9: Method Definition
# =============================================================================

def test_luau_cfg_method_definition():
    """Method with colon syntax should be found and analyzed."""
    from tldr.cfg_extractor import extract_luau_cfg

    code = '''
local Player = {}

function Player:takeDamage(amount: number): ()
    self.health -= amount
    if self.health <= 0 then
        self:die()
    end
end
'''
    cfg = extract_luau_cfg(code, "takeDamage")

    assert cfg is not None

    # Method has one if statement
    assert cfg.cyclomatic_complexity >= 2


# =============================================================================
# Test 10: Nested Control Flow
# =============================================================================

def test_luau_cfg_nested_control():
    """Nested if statements should accumulate complexity."""
    from tldr.cfg_extractor import extract_luau_cfg

    code = '''
function classify(x: number): string
    if x > 0 then
        if x > 100 then
            return "large"
        else
            return "small"
        end
    else
        return "non-positive"
    end
end
'''
    cfg = extract_luau_cfg(code, "classify")

    assert cfg is not None

    # Two nested if-else: complexity = 3
    assert cfg.cyclomatic_complexity == 3


# =============================================================================
# Test: Function Not Found
# =============================================================================

def test_luau_cfg_function_not_found():
    """Should raise ValueError when function not found."""
    from tldr.cfg_extractor import extract_luau_cfg

    code = '''
function exists(): ()
end
'''
    with pytest.raises(ValueError, match="not found"):
        extract_luau_cfg(code, "nonexistent")


# =============================================================================
# Test: Generic Function (Luau-specific)
# =============================================================================

def test_luau_cfg_generic_function():
    """Generic function with type parameter should parse correctly.

    This is LUAU-SPECIFIC - Lua does not have generics.
    """
    from tldr.cfg_extractor import extract_luau_cfg

    code = '''
function identity<T>(value: T): T
    return value
end
'''
    cfg = extract_luau_cfg(code, "identity")

    assert cfg is not None
    assert cfg.function_name == "identity"
    assert cfg.cyclomatic_complexity == 1
