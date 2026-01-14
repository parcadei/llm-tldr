"""
TypeScript/JavaScript parser for cross-file call analysis.
"""

import re
from typing import Dict, List, Optional

from tldr.cross_file_calls.parsers.base import BaseParser
from tldr.cross_file_calls.core import HAS_TS_PARSER, _get_ts_parser


class TypeScriptParser(BaseParser):
    """Parser for TypeScript and JavaScript files."""
    
    def extract_calls(self, file_path: str, timeout: Optional[float] = None) -> List[Dict]:
        """Extract function calls from a TypeScript/JavaScript file."""
        if not HAS_TS_PARSER:
            return self._extract_calls_regex(file_path)
        
        try:
            parser = _get_ts_parser()
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = parser.parse(bytes(content, 'utf-8'))
            calls = []
            
            # Walk the tree to find function calls
            def walk_node(node, depth=0):
                if depth > 100:  # Prevent infinite recursion
                    return
                
                if node.type == 'call_expression':
                    call_info = self._extract_ts_call_info(node, file_path, content)
                    if call_info:
                        calls.append(call_info)
                
                # Recursively check children
                for child in node.children:
                    walk_node(child, depth + 1)
            
            walk_node(tree.root_node)
            return calls
            
        except Exception:
            return self._extract_calls_regex(file_path)
    
    def parse_imports(self, file_path: str) -> List[Dict]:
        """Parse import statements from a TypeScript/JavaScript file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            imports = []
            
            # Regex patterns for different import types
            patterns = [
                # import * as name from 'module'
                (r'import\s+\*\s+as\s+(\w+)\s+from\s+[\'"]([^\'"]+)[\'"]', 'import_all'),
                # import { items } from 'module'
                (r'import\s+\{([^}]+)\}\s+from\s+[\'"]([^\'"]+)[\'"]', 'import_named'),
                # import default from 'module'
                (r'import\s+(\w+)\s+from\s+[\'"]([^\'"]+)[\'"]', 'import_default'),
                # import name = require('module')
                (r'import\s+(\w+)\s*=\s*require\s*\([\'"]([^\'"]+)[\'"]\)', 'import_require'),
                # const { items } = require('module')
                (r'const\s+\{([^}]+)\}\s*=\s*require\s*\([\'"]([^\'"]+)[\'"]\)', 'require_named'),
                # const name = require('module')
                (r'const\s+(\w+)\s*=\s*require\s*\([\'"]([^\'"]+)[\'"]\)', 'require_default'),
            ]
            
            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern, import_type in patterns:
                    match = re.search(pattern, line)
                    if match:
                        if import_type in ['import_named', 'require_named']:
                            # Handle named imports/exports
                            names = [name.strip() for name in match.group(1).split(',')]
                            for name in names:
                                if ' as ' in name:
                                    original, alias = name.split(' as ')
                                    imports.append({
                                        'type': import_type,
                                        'module': match.group(2),
                                        'name': original.strip(),
                                        'asname': alias.strip(),
                                        'line': line_num,
                                        'column': line.find(match.group(0))
                                    })
                                else:
                                    imports.append({
                                        'type': import_type,
                                        'module': match.group(2),
                                        'name': name.strip(),
                                        'asname': None,
                                        'line': line_num,
                                        'column': line.find(match.group(0))
                                    })
                        else:
                            imports.append({
                                'type': import_type,
                                'module': match.group(2),
                                'name': match.group(1),
                                'asname': None,
                                'line': line_num,
                                'column': line.find(match.group(0))
                            })
            
            return imports
            
        except Exception as e:
            return []
    
    def _extract_ts_call_info(self, node, file_path: str, content: str) -> Optional[Dict]:
        """Extract call information from a tree-sitter node."""
        try:
            call_info = {
                'file': file_path,
                'line': node.start_point[0] + 1,
                'column': node.start_point[1],
                'type': 'function_call'
            }
            
            # Find the function being called
            function_node = node.child_by_field_name('function')
            if function_node:
                call_info.update(self._extract_function_info(function_node, content))
            
            # Extract arguments
            call_info['args'] = self._extract_ts_arguments(node, content)
            
            return call_info
            
        except Exception:
            return None
    
    def _extract_function_info(self, node, content: str) -> Dict:
        """Extract function information from a tree-sitter node."""
        result: Dict[str, Optional[str]] = {
            'function': None,
            'module': None,
            'object': None,
            'full_expression': None
        }
        
        try:
            if node.type == 'identifier':
                result['function'] = content[node.start_byte:node.end_byte]
            
            elif node.type == 'member_expression':
                # Extract object and property
                object_node = node.child_by_field_name('object')
                property_node = node.child_by_field_name('property')
                
                if property_node:
                    result['function'] = content[property_node.start_byte:property_node.end_byte]
                
                if object_node:
                    result['object'] = content[object_node.start_byte:object_node.end_byte]
                    
                    # Check if object might be a module
                    if object_node.type == 'identifier':
                        result['module'] = result['object']
                
                # Full expression
                result['full_expression'] = content[node.start_byte:node.end_byte]
            
            elif node.type in ['call_expression', 'new_expression']:
                # Nested call or constructor
                func_node = node.child_by_field_name('function')
                if func_node:
                    return self._extract_function_info(func_node, content)
            
        except Exception:
            pass
        
        return result
    
    def _extract_ts_arguments(self, node, content: str) -> List[Dict]:
        """Extract arguments from a function call node."""
        args = []
        
        try:
            arguments_node = node.child_by_field_name('arguments')
            if arguments_node:
                # Filter out comma nodes and count position correctly
                position = 0
                for child in arguments_node.children:
                    if child.type == ',':
                        continue
                    
                    arg_info: Dict[str, Optional[object]] = {
                        'position': position,
                        'type': 'positional',
                        'value': None,
                        'name': None
                    }
                    
                    # Check if this is a named argument (object property)
                    if child.type == 'object':
                        for prop in child.children:
                            if prop.type == 'pair':
                                key_node = prop.child_by_field_name('key')
                                if key_node:
                                    arg_info['name'] = content[key_node.start_byte:key_node.end_byte]
                                    break
                    
                    args.append(arg_info)
                    position += 1
        except Exception:
            pass
        
        return args
    
    def _extract_calls_regex(self, file_path: str) -> List[Dict]:
        """Fallback regex-based call extraction."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            calls = []
            seen = set()  # Track seen calls to avoid duplicates
            
            # Regex patterns for function calls
            patterns = [
                # function()
                r'(\b\w+)\s*\(',
                # object.method()
                r'(\b\w+\.\w+)\s*\(',
                # this.method()
                r'(this\.\w+)\s*\(',
            ]
            
            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern in patterns:
                    for match in re.finditer(pattern, line):
                        # Create a unique key for deduplication
                        key = (line_num, match.start(), match.group(1))
                        if key in seen:
                            continue
                        seen.add(key)
                        
                        call_info = {
                            'file': file_path,
                            'line': line_num,
                            'column': match.start(),
                            'type': 'function_call',
                            'function': match.group(1).split('.')[-1] if '.' in match.group(1) else match.group(1),
                            'object': match.group(1).split('.')[0] if '.' in match.group(1) else None,
                            'module': None,
                            'full_expression': match.group(1),
                            'args': []
                        }
                        calls.append(call_info)
            
            return calls
            
        except Exception:
            return []


# Backward compatibility functions
def parse_ts_imports(file_path: str) -> List[Dict]:
    """Parse TypeScript imports (backward compatibility)."""
    parser = TypeScriptParser()
    return parser.parse_imports(file_path)


def _extract_ts_file_calls(file_path: str) -> List[Dict]:
    """Extract TypeScript file calls (backward compatibility)."""
    parser = TypeScriptParser()
    return parser.extract_calls(file_path)


def _index_typescript_file(file_path: str) -> Dict:
    """Index a TypeScript file."""
    parser = TypeScriptParser()
    
    return {
        'file': file_path,
        'imports': parser.parse_imports(file_path),
        'calls': parser.extract_calls(file_path),
        'definitions': _extract_ts_definitions(file_path)
    }


def _extract_ts_definitions(file_path: str) -> List[Dict]:
    """Extract function and class definitions from a TypeScript file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        definitions = []
        
        # Regex patterns for definitions
        patterns = [
            # function name() {}
            (r'function\s+(\w+)\s*\(', 'function'),
            # const name = () => {}
            (r'const\s+(\w+)\s*=\s*\(', 'arrow_function'),
            # class name {}
            (r'class\s+(\w+)', 'class'),
            # interface name {}
            (r'interface\s+(\w+)', 'interface'),
            # export function name() {}
            (r'export\s+function\s+(\w+)\s*\(', 'export_function'),
            # export const name = () => {}
            (r'export\s+const\s+(\w+)\s*=\s*\(', 'export_arrow'),
            # export class name {}
            (r'export\s+class\s+(\w+)', 'export_class'),
        ]
        
        for line_num, line in enumerate(content.split('\n'), 1):
            for pattern, def_type in patterns:
                match = re.search(pattern, line)
                if match:
                    definitions.append({
                        'type': def_type,
                        'name': match.group(1),
                        'line': line_num,
                        'column': line.find(match.group(0))
                    })
        
        return definitions
        
    except Exception:
        return []
