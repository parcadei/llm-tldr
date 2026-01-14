"""
Python parser for cross-file call analysis.
"""

import ast
from typing import Dict, List, Optional

from tldr.cross_file_calls.parsers.base import BaseParser


class PythonParser(BaseParser):
    """Parser for Python files."""
    
    def extract_calls(self, file_path: str, timeout: Optional[float] = None) -> List[Dict]:
        """Extract function calls from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            calls = []
            
            # Walk the AST to find function calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    call_info = self._extract_call_info(node, file_path)
                    if call_info:
                        calls.append(call_info)
            
            return calls
            
        except Exception as e:
            return []
    
    def parse_imports(self, file_path: str) -> List[Dict]:
        """Parse import statements from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            'type': 'import',
                            'module': alias.name,
                            'name': alias.asname or alias.name,
                            'asname': alias.asname,
                            'line': node.lineno,
                            'column': node.col_offset
                        })
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append({
                            'type': 'from_import',
                            'module': module,
                            'name': alias.name,
                            'asname': alias.asname,
                            'line': node.lineno,
                            'column': node.col_offset
                        })
            
            return imports
            
        except Exception as e:
            return []
    
    def _extract_call_info(self, node: ast.Call, file_path: str) -> Optional[Dict]:
        """Extract information about a function call."""
        try:
            call_info = {
                'file': file_path,
                'line': node.lineno,
                'column': node.col_offset,
                'type': 'function_call'
            }
            
            # Handle different types of call expressions
            if isinstance(node.func, ast.Name):
                # Simple function call: func()
                call_info['function'] = node.func.id
                call_info['module'] = None
                call_info['object'] = None
                
            elif isinstance(node.func, ast.Attribute):
                # Method call: obj.method() or module.func()
                call_info.update(self._extract_attribute_call(node.func))
                
            elif isinstance(node.func, ast.Subscript):
                # Subscript call: obj[key]()
                call_info['function'] = None
                call_info['module'] = None
                call_info['object'] = ast.unparse(node.func) if hasattr(ast, 'unparse') else None
                
            else:
                # Other types of calls
                call_info['function'] = None
                call_info['module'] = None
                call_info['object'] = None
            
            # Extract arguments
            call_info['args'] = self._extract_arguments(node)
            
            return call_info
            
        except Exception:
            return None
    
    def _extract_attribute_call(self, node: ast.Attribute) -> Dict:
        """Extract information from an attribute call."""
        result = {
            'function': node.attr,
            'object': None,
            'module': None
        }
        
        # Walk the attribute chain to find the base object
        current = node.value
        parts = [node.attr]
        
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        
        # Now current should be a Name or other expression
        if isinstance(current, ast.Name):
            parts.append(current.id)
            result['object'] = current.id
            
            # Check if this might be a module import
            if len(parts) >= 2:
                # Use the base object as module (e.g., 'a' in 'a.b.c')
                # But keep the chain logic if needed. Original code used parts[-2].
                # If we want the root module, it's parts[0] (before reverse) -> parts[-1] (after reverse)
                # Let's trust the requirement: "result['module'] refers to the base object"
                result['module'] = parts[0]
        
        # Reverse the parts to get the full expression
        parts.reverse()
        result['full_expression'] = '.'.join(parts)
        
        return result
    
    def _extract_arguments(self, node: ast.Call) -> List[Dict]:
        """Extract argument information from a function call."""
        args = []
        
        # Positional arguments
        for i, arg in enumerate(node.args):
            arg_info = {
                'position': i,
                'type': 'positional',
                'value': self._ast_to_string(arg),
                'name': None
            }
            args.append(arg_info)
        
        # Keyword arguments
        for keyword in node.keywords:
            arg_info = {
                'position': None,
                'type': 'keyword',
                'value': self._ast_to_string(keyword.value),
                'name': keyword.arg
            }
            args.append(arg_info)
        
        return args
    
    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert an AST node to a string representation."""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                # Fallback for older Python versions
                return ast.dump(node)
        except Exception:
            return str(type(node).__name__)


def parse_imports(file_path: str) -> List[Dict]:
    """Parse import statements from a Python file (backward compatibility)."""
    parser = PythonParser()
    return parser.parse_imports(file_path)


def _extract_file_calls(file_path: str) -> List[Dict]:
    """Extract function calls from a Python file (backward compatibility)."""
    parser = PythonParser()
    return parser.extract_calls(file_path)


def _index_python_file(file_path: str) -> Dict:
    """Index a Python file and extract all relevant information."""
    parser = PythonParser()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
    except Exception:
        # Fallback for read/parse errors
        return {
            'file': file_path,
            'imports': [],
            'calls': [],
            'definitions': []
        }
    
    # Helper to avoid re-parsing
    def get_imports(tree):
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'name': alias.asname or alias.name,
                        'asname': alias.asname,
                        'line': node.lineno,
                        'column': node.col_offset
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': module,
                        'name': alias.name,
                        'asname': alias.asname,
                        'line': node.lineno,
                        'column': node.col_offset
                    })
        return imports

    def get_calls(tree, file_path):
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_info = parser._extract_call_info(node, file_path)
                if call_info:
                    calls.append(call_info)
        return calls

    return {
        'file': file_path,
        'imports': get_imports(tree),
        'calls': get_calls(tree, file_path),
        'definitions': _extract_definitions_from_ast(tree)
    }


def _extract_definitions_from_ast(tree: ast.AST) -> List[Dict]:
    """Extract function and class definitions from an AST."""
    definitions = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.append({
                'type': 'async_function' if isinstance(node, ast.AsyncFunctionDef) else 'function',
                'name': node.name,
                'line': node.lineno,
                'column': node.col_offset,
                'args': [arg.arg for arg in node.args.args],
                'decorators': [ast.unparse(d) for d in node.decorator_list] if hasattr(ast, 'unparse') else []
            })
        
        elif isinstance(node, ast.ClassDef):
            definitions.append({
                'type': 'class',
                'name': node.name,
                'line': node.lineno,
                'column': node.col_offset,
                'methods': [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))],
                'decorators': [ast.unparse(d) for d in node.decorator_list] if hasattr(ast, 'unparse') else []
            })
    
    return definitions


def _extract_definitions(file_path: str) -> List[Dict]:
    """Extract function and class definitions from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
        return _extract_definitions_from_ast(tree)
    except Exception:
        return []
