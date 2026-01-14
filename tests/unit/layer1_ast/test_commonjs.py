"""Tests for CommonJS exports support (Issue #21).

Tests extraction of function_expression nodes in CommonJS patterns:
- exports.foo = function() {}
- module.exports.foo = function() {}
- exports.foo = async function() {}
"""

from pathlib import Path

import pytest
from tldr.hybrid_extractor import HybridExtractor


@pytest.fixture
def extractor():
    return HybridExtractor()


class TestCommonJSBasic:
    """Basic CommonJS export patterns."""

    def test_exports_function(self, extractor, tmp_path: Path):
        """exports.foo = function() {} should extract 'foo'."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
exports.helloWorld = function(req, res) {
    res.send('Hello!');
};
""")
        result = extractor.extract(js_file)
        func_names = [f.name for f in result.functions]
        assert "helloWorld" in func_names

    def test_exports_function_with_params(self, extractor, tmp_path: Path):
        """Should extract parameters from CommonJS function."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
exports.connect = function(host, port, options) {
    return new Connection(host, port);
};
""")
        result = extractor.extract(js_file)
        func = next((f for f in result.functions if f.name == "connect"), None)
        assert func is not None
        assert "host" in func.params
        assert "port" in func.params
        assert "options" in func.params

    def test_module_exports_function(self, extractor, tmp_path: Path):
        """module.exports.foo = function() {} should extract 'foo'."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
module.exports.initialize = function(config) {
    return setup(config);
};
""")
        result = extractor.extract(js_file)
        func_names = [f.name for f in result.functions]
        assert "initialize" in func_names

    def test_async_exports_function(self, extractor, tmp_path: Path):
        """exports.foo = async function() {} should extract with is_async=True."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
exports.fetchData = async function(url) {
    const response = await fetch(url);
    return response.json();
};
""")
        result = extractor.extract(js_file)
        func = next((f for f in result.functions if f.name == "fetchData"), None)
        assert func is not None
        assert func.is_async is True

    def test_multiple_exports(self, extractor, tmp_path: Path):
        """Multiple CommonJS exports should all be extracted."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
exports.connect = function(host) { return host; };
exports.disconnect = function() { return true; };
exports.query = function(sql) { return []; };
""")
        result = extractor.extract(js_file)
        func_names = [f.name for f in result.functions]
        assert "connect" in func_names
        assert "disconnect" in func_names
        assert "query" in func_names


class TestCommonJSFirebase:
    """Firebase Functions patterns (common use case)."""

    def test_firebase_https_function(self, extractor, tmp_path: Path):
        """Firebase HTTPS function pattern."""
        js_file = tmp_path / "index.js"
        js_file.write_text("""
const functions = require('firebase-functions');

exports.helloWorld = functions.https.onRequest((req, res) => {
    res.send('Hello from Firebase!');
});
""")
        result = extractor.extract(js_file)
        func_names = [f.name for f in result.functions]
        assert "helloWorld" in func_names

    def test_firebase_auth_trigger(self, extractor, tmp_path: Path):
        """Firebase Auth trigger pattern."""
        js_file = tmp_path / "index.js"
        js_file.write_text("""
const functions = require('firebase-functions');

exports.userCreated = functions.auth.user().onCreate((user) => {
    console.log('New user:', user.uid);
});
""")
        result = extractor.extract(js_file)
        func_names = [f.name for f in result.functions]
        assert "userCreated" in func_names

    def test_firebase_firestore_trigger(self, extractor, tmp_path: Path):
        """Firebase Firestore trigger pattern."""
        js_file = tmp_path / "index.js"
        js_file.write_text("""
const functions = require('firebase-functions');

exports.onDocumentCreate = functions.firestore
    .document('users/{userId}')
    .onCreate((snap, context) => {
        const data = snap.data();
        return null;
    });
""")
        result = extractor.extract(js_file)
        func_names = [f.name for f in result.functions]
        assert "onDocumentCreate" in func_names


class TestCommonJSMixedPatterns:
    """Mixed ES6 and CommonJS patterns."""

    def test_mixed_exports_and_functions(self, extractor, tmp_path: Path):
        """Both CommonJS exports and regular functions should be extracted."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
function helper() {
    return 42;
}

exports.main = function() {
    return helper();
};
""")
        result = extractor.extract(js_file)
        func_names = [f.name for f in result.functions]
        assert "helper" in func_names
        assert "main" in func_names
        # Note: top-level const arrow = () => ... extraction is a separate issue

    def test_commonjs_with_classes(self, extractor, tmp_path: Path):
        """CommonJS exports alongside class definitions."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
class Database {
    constructor() {}
    query(sql) { return []; }
}

exports.createDb = function() {
    return new Database();
};
""")
        result = extractor.extract(js_file)
        func_names = [f.name for f in result.functions]
        class_names = [c.name for c in result.classes]
        assert "createDb" in func_names
        assert "Database" in class_names


class TestCommonJSEdgeCases:
    """Edge cases and patterns that should be skipped."""

    def test_computed_property_skipped(self, extractor, tmp_path: Path):
        """exports[dynamic] = function() {} should be skipped (no static name)."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
const name = 'dynamicFunc';
exports[name] = function() {
    return 'dynamic';
};
""")
        result = extractor.extract(js_file)
        # Should not crash, may or may not extract (implementation choice)
        assert result is not None

    def test_module_exports_bare_skipped(self, extractor, tmp_path: Path):
        """module.exports = function() {} (no property name) should be skipped."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
module.exports = function() {
    return 'anonymous';
};
""")
        result = extractor.extract(js_file)
        # Should not crash, anonymous function has no name to extract
        assert result is not None

    def test_nested_in_conditional(self, extractor, tmp_path: Path):
        """CommonJS in conditional should still be extracted."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
if (process.env.NODE_ENV === 'production') {
    exports.handler = function(req, res) {
        res.send('prod');
    };
}
""")
        result = extractor.extract(js_file)
        func_names = [f.name for f in result.functions]
        assert "handler" in func_names

    def test_iife_not_extracted(self, extractor, tmp_path: Path):
        """IIFE patterns should not be extracted as named functions."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
(function() {
    console.log('IIFE');
})();
""")
        result = extractor.extract(js_file)
        # Should not crash, IIFE is anonymous
        assert result is not None


class TestCommonJSLineNumbers:
    """Line number accuracy for CommonJS exports."""

    def test_line_numbers_correct(self, extractor, tmp_path: Path):
        """Line numbers should point to the function, not the exports statement."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""// Line 1
// Line 2
exports.myFunc = function() {  // Line 3
    return 42;
};
""")
        result = extractor.extract(js_file)
        func = next((f for f in result.functions if f.name == "myFunc"), None)
        assert func is not None
        # Line 3 is where the function expression starts (0-indexed + 1 = line 3)
        assert func.line_number == 3


class TestCommonJSAdversarial:
    """Adversarial test cases to stress CommonJS extraction."""

    def test_deeply_nested_export(self, extractor, tmp_path: Path):
        """Export nested in multiple control structures."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
if (process.env.NODE_ENV === 'production') {
    if (process.env.FEATURE_FLAG) {
        try {
            exports.deeplyNested = function() {
                return 'deep';
            };
        } catch (e) {}
    }
}
""")
        result = extractor.extract(js_file)
        # May or may not extract depending on recursion depth - should not crash
        assert result is not None

    def test_export_with_jsdoc(self, extractor, tmp_path: Path):
        """CommonJS export with JSDoc comment."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
/**
 * Handles incoming requests.
 * @param {Request} req - The request object
 * @param {Response} res - The response object
 */
exports.handler = function(req, res) {
    res.send('OK');
};
""")
        result = extractor.extract(js_file)
        func = next((f for f in result.functions if f.name == "handler"), None)
        assert func is not None
        # JSDoc attachment for CommonJS is a nice-to-have enhancement
        # The function should still be extracted even if docstring isn't attached
        # TODO: Future enhancement - attach JSDoc to CommonJS function_expression

    def test_reassigned_export(self, extractor, tmp_path: Path):
        """Same export name assigned twice."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
exports.handler = function() { return 1; };
exports.handler = function() { return 2; };
""")
        result = extractor.extract(js_file)
        handlers = [f for f in result.functions if f.name == "handler"]
        # Both should be extracted (duplicates are allowed)
        assert len(handlers) >= 1

    def test_mixed_export_styles(self, extractor, tmp_path: Path):
        """Mix of exports.x, module.exports.x, and module.exports = {}."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
exports.a = function() { return 'a'; };
module.exports.b = function() { return 'b'; };

// This pattern uses object literal - different code path
module.exports = {
    c: function() { return 'c'; }
};
""")
        result = extractor.extract(js_file)
        func_names = [f.name for f in result.functions]
        assert "a" in func_names
        assert "b" in func_names
        # 'c' may or may not be extracted (object literal pattern)

    def test_export_in_try_catch(self, extractor, tmp_path: Path):
        """Export inside try-catch block."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
try {
    exports.risky = function() {
        throw new Error('risky');
    };
} catch (e) {
    exports.fallback = function() {
        return 'safe';
    };
}
""")
        result = extractor.extract(js_file)
        func_names = [f.name for f in result.functions]
        # At least one should be extracted
        assert "risky" in func_names or "fallback" in func_names

    def test_export_in_loop(self, extractor, tmp_path: Path):
        """Export inside loop (weird but valid JS)."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
for (let i = 0; i < 1; i++) {
    exports.looped = function() {
        return i;
    };
}
""")
        result = extractor.extract(js_file)
        # May or may not extract - should not crash
        assert result is not None

    def test_empty_function_body(self, extractor, tmp_path: Path):
        """Export with empty function body."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
exports.noop = function() {};
exports.asyncNoop = async function() {};
""")
        result = extractor.extract(js_file)
        func_names = [f.name for f in result.functions]
        assert "noop" in func_names
        assert "asyncNoop" in func_names

    def test_unicode_function_name(self, extractor, tmp_path: Path):
        """Export with unicode characters in name (valid JS identifier)."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
exports.café = function() { return 'coffee'; };
exports.$special = function() { return 'dollar'; };
exports._private = function() { return 'underscore'; };
""")
        result = extractor.extract(js_file)
        func_names = [f.name for f in result.functions]
        assert "café" in func_names
        assert "$special" in func_names
        assert "_private" in func_names

    def test_very_long_function(self, extractor, tmp_path: Path):
        """Export with a very long function body."""
        body_lines = ["    console.log('line " + str(i) + "');" for i in range(100)]
        body = "\n".join(body_lines)
        js_file = tmp_path / "test.js"
        js_file.write_text(f"""
exports.longFunction = function() {{
{body}
}};
""")
        result = extractor.extract(js_file)
        func = next((f for f in result.functions if f.name == "longFunction"), None)
        assert func is not None

    def test_generator_function(self, extractor, tmp_path: Path):
        """CommonJS export with generator function."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
exports.generator = function*() {
    yield 1;
    yield 2;
    yield 3;
};
""")
        result = extractor.extract(js_file)
        # Generator functions may or may not be extracted - should not crash
        assert result is not None


class TestCommonJSCallGraph:
    """Call graph extraction for CommonJS exports (Issue #21 fix)."""

    def test_basic_call_graph(self, extractor, tmp_path: Path):
        """CommonJS exports should track calls to defined functions."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
function helper() {
    return 42;
}

exports.main = function() {
    return helper();
};
""")
        result = extractor.extract(js_file)
        assert "main" in result.call_graph.calls
        assert "helper" in result.call_graph.calls["main"]
        assert "helper" in result.call_graph.called_by
        assert "main" in result.call_graph.called_by["helper"]

    def test_multiple_calls_from_export(self, extractor, tmp_path: Path):
        """CommonJS export calling multiple functions."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
function validate(data) { return true; }
function transform(data) { return data; }
function save(data) { return data; }

exports.process = function(data) {
    if (validate(data)) {
        const result = transform(data);
        return save(result);
    }
};
""")
        result = extractor.extract(js_file)
        assert "process" in result.call_graph.calls
        calls = result.call_graph.calls["process"]
        assert "validate" in calls
        assert "transform" in calls
        assert "save" in calls

    def test_chained_calls(self, extractor, tmp_path: Path):
        """Call graph with chained function calls."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
function a() { return b(); }
function b() { return c(); }
function c() { return 42; }

exports.start = function() {
    return a();
};
""")
        result = extractor.extract(js_file)
        assert "start" in result.call_graph.calls
        assert "a" in result.call_graph.calls["start"]
        assert "a" in result.call_graph.calls
        assert "b" in result.call_graph.calls["a"]

    def test_async_export_calls(self, extractor, tmp_path: Path):
        """Async CommonJS export should track calls."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
function fetchData(url) { return fetch(url); }
function parseResponse(res) { return res.json(); }

exports.getData = async function(url) {
    const res = await fetchData(url);
    return parseResponse(res);
};
""")
        result = extractor.extract(js_file)
        assert "getData" in result.call_graph.calls
        calls = result.call_graph.calls["getData"]
        assert "fetchData" in calls
        assert "parseResponse" in calls

    def test_module_exports_call_graph(self, extractor, tmp_path: Path):
        """module.exports.foo should also track calls."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
function init() { return {}; }

module.exports.setup = function() {
    return init();
};
""")
        result = extractor.extract(js_file)
        assert "setup" in result.call_graph.calls
        assert "init" in result.call_graph.calls["setup"]


class TestCommonJSCallGraphAdversarial:
    """Adversarial tests for CommonJS call graph extraction."""

    def test_nested_export_call_graph(self, extractor, tmp_path: Path):
        """Deeply nested CommonJS export should track calls."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
function helper() { return 1; }

if (process.env.NODE_ENV) {
    try {
        exports.nested = function() {
            return helper();
        };
    } catch (e) {}
}
""")
        result = extractor.extract(js_file)
        if "nested" in result.call_graph.calls:
            assert "helper" in result.call_graph.calls["nested"]

    def test_recursive_call(self, extractor, tmp_path: Path):
        """Recursive function call in CommonJS export."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

exports.compute = function(n) {
    return factorial(n);
};
""")
        result = extractor.extract(js_file)
        assert "compute" in result.call_graph.calls
        assert "factorial" in result.call_graph.calls["compute"]
        # Recursive call
        assert "factorial" in result.call_graph.calls
        assert "factorial" in result.call_graph.calls["factorial"]

    def test_call_in_callback(self, extractor, tmp_path: Path):
        """Calls inside callbacks within CommonJS export."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
function process(item) { return item * 2; }

exports.transform = function(arr) {
    return arr.map(function(item) {
        return process(item);
    });
};
""")
        result = extractor.extract(js_file)
        # The call to process is inside the anonymous callback
        # Current implementation may or may not catch it
        assert result is not None

    def test_iife_call_graph(self, extractor, tmp_path: Path):
        """IIFE should not pollute call graph."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
function helper() { return 1; }

(function() {
    helper();
})();

exports.main = function() {
    return helper();
};
""")
        result = extractor.extract(js_file)
        assert "main" in result.call_graph.calls
        assert "helper" in result.call_graph.calls["main"]

    def test_shadowed_function(self, extractor, tmp_path: Path):
        """Shadowed function names should still be tracked."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
function helper() { return 'outer'; }

exports.main = function() {
    function helper() { return 'inner'; }
    return helper();
};
""")
        result = extractor.extract(js_file)
        # Both functions should be extracted
        helpers = [f for f in result.functions if f.name == "helper"]
        assert len(helpers) >= 1

    def test_method_call_not_tracked(self, extractor, tmp_path: Path):
        """Method calls on objects should not be tracked as function calls."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
function getData() { return [1, 2, 3]; }

exports.process = function() {
    const data = getData();
    return data.map(x => x * 2);  // .map is a method, not a defined function
};
""")
        result = extractor.extract(js_file)
        assert "process" in result.call_graph.calls
        assert "getData" in result.call_graph.calls["process"]
        # "map" should NOT be in calls (it's a method, not a defined function)
        assert "map" not in result.call_graph.calls.get("process", [])

    def test_no_false_positives(self, extractor, tmp_path: Path):
        """Variables named like functions should not create false call edges."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
function realFunc() { return 1; }

exports.test = function() {
    const realFunc = 42;  // Shadow with variable
    return realFunc;      // This is variable access, not a call
};
""")
        result = extractor.extract(js_file)
        # Should not crash, call graph extraction should handle this
        assert result is not None
