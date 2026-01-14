"""
Rust parser for cross-file call analysis.
"""

import re
from typing import Dict, List, Optional

from tldr.cross_file_calls.parsers.base import BaseParser
from tldr.cross_file_calls.core import HAS_RUST_PARSER, _get_rust_parser


class RustParser(BaseParser):
    """Parser for Rust files."""
    
    def extract_calls(self, file_path: str, timeout: Optional[float] = None) -> List[Dict]:
        """Extract function calls from a Rust file."""
        if not HAS_RUST_PARSER:
            return self._extract_calls_regex(file_path)
        
        try:
            parser = _get_rust_parser()
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = parser.parse(bytes(content, 'utf-8'))
            calls = []
            
            def walk_node(node, depth=0):
                if depth > 100:
                    return
                
                if node.type == 'call_expression':
                    call_info = self._extract_rust_call_info(node, file_path, content)
                    if call_info:
                        calls.append(call_info)
                
                for child in node.children:
                    walk_node(child, depth + 1)
            
            walk_node(tree.root_node)
            return calls
            
        except Exception:
            return self._extract_calls_regex(file_path)
    
    def parse_imports(self, file_path: str) -> List[Dict]:
        """Parse import statements from a Rust file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            imports = []
            
            # Rust import patterns
            patterns = [
                # use module::item;
                (r'use\s+([^;]+);', 'use'),
                # extern crate name;
                (r'extern\s+crate\s+(\w+);', 'extern_crate'),
            ]
            
            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern, import_type in patterns:
                    match = re.search(pattern, line)
                    if match:
                        if import_type == 'use':
                            path = match.group(1).strip()
                            if ' as ' in path:
                                module_path, alias = path.split(' as ')
                                imports.append({
                                    'type': import_type,
                                    'module': module_path.strip(),
                                    'name': module_path.split('::')[-1],
                                    'asname': alias.strip(),
                                    'line': line_num,
                                    'column': line.find(match.group(0))
                                })
                            else:
                                imports.append({
                                    'type': import_type,
                                    'module': path,
                                    'name': path.split('::')[-1],
                                    'asname': None,
                                    'line': line_num,
                                    'column': line.find(match.group(0))
                                })
                        else:  # extern_crate
                            imports.append({
                                'type': import_type,
                                'module': match.group(1),
                                'name': match.group(1),
                                'asname': None,
                                'line': line_num,
                                'column': line.find(match.group(0))
                            })
            
            return imports
            
        except Exception:
            return []
    
    def _extract_rust_call_info(self, node, file_path: str, content: str) -> Optional[Dict]:
        """Extract call information from a Rust tree-sitter node."""
        try:
            call_info = {
                'file': file_path,
                'line': node.start_point[0] + 1,
                'column': node.start_point[1],
                'type': 'function_call'
            }
            
            function_node = node.child_by_field_name('function')
            if function_node:
                call_info.update(self._extract_function_info(function_node, content))
            
            call_info['args'] = self._extract_rust_arguments(node)
            return call_info
            
        except Exception:
            return None
    
    def _extract_function_info(self, node, content: str) -> Dict[str, Optional[str]]:
        """Extract function information from a Rust tree-sitter node."""
        result: Dict[str, Optional[str]] = {
            'function': None,
            'module': None,
            'object': None,
            'full_expression': None
        }
        
        try:
            if node.type == 'identifier':
                result['function'] = content[node.start_byte:node.end_byte]
            elif node.type == 'field_expression':
                # object::method() or object.method()
                object_node = node.child_by_field_name('value')
                field_node = node.child_by_field_name('field')
                
                if field_node:
                    result['function'] = content[field_node.start_byte:field_node.end_byte]
                if object_node:
                    result['object'] = content[object_node.start_byte:object_node.end_byte]
                    result['module'] = result['object']
                
                result['full_expression'] = content[node.start_byte:node.end_byte]
            elif node.type == 'path_expression':
                result['full_expression'] = content[node.start_byte:node.end_byte]
                parts = result['full_expression'].split('::')
                if parts:
                    result['function'] = parts[-1]
                    if len(parts) > 1:
                        result['module'] = parts[-2]
        except Exception:
            pass
        
        return result
    
    def _extract_rust_arguments(self, node) -> List[Dict]:
        """Extract arguments from a Rust function call."""
        args = []
        
        try:
            arguments_node = node.child_by_field_name('arguments')
            if arguments_node:
                # Filter out comma nodes and count position correctly
                position = 0
                for child in arguments_node.children:
                    if child.type != ',':
                        args.append({
                            'position': position,
                            'type': 'positional',
                            'value': None,
                            'name': None
                        })
                        position += 1
        except Exception:
            pass
        
        return args
    
    def _extract_calls_regex(self, file_path: str) -> List[Dict]:
        """Fallback regex-based call extraction for Rust."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            calls = []
            
            # Rust function call patterns
            patterns = [
                # function()
                (r'(\b[a-z]\w*)\s*\(', 'snake_case'),
                # Type::method()
                (r'(\b[A-Z]\w*::[a-z]\w*)\s*\(', 'method'),
                # module::function()
                (r'(\b[a-z]\w*::[a-z]\w*)\s*\(', 'module'),
            ]
            
            seen_calls = set()
            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern, _call_type in patterns:
                    for match in re.finditer(pattern, line):
                        full_expr = match.group(1)
                        # Deduplicate based on position and expression
                        call_key = (line_num, match.start(), full_expr)
                        if call_key in seen_calls:
                            continue
                        seen_calls.add(call_key)

                        call_info = {
                            'file': file_path,
                            'line': line_num,
                            'column': match.start(),
                            'type': 'function_call',
                            'function': full_expr.split('::')[-1],
                            'object': full_expr.split('::')[0] if '::' in full_expr else None,
                            'module': full_expr.split('::')[0] if '::' in full_expr else None,
                            'full_expression': full_expr,
                            'args': []
                        }
                        calls.append(call_info)
            
            return calls
            
        except Exception:
            return []


# Backward compatibility
def _extract_rust_file_calls(file_path: str) -> List[Dict]:
    """Extract Rust file calls (backward compatibility)."""
    parser = RustParser()
    return parser.extract_calls(file_path)
