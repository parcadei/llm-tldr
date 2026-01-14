"""
Go parser for cross-file call analysis.
"""

import re
from typing import Dict, List, Optional

from tldr.cross_file_calls.parsers.base import BaseParser
from tldr.cross_file_calls.core import HAS_GO_PARSER, _get_go_parser


class GoParser(BaseParser):
    """Parser for Go files."""
    
    def extract_calls(self, file_path: str, timeout: Optional[float] = None) -> List[Dict]:
        """
        Extract function calls from a Go file.
        
        Note: The timeout parameter is accepted for API consistency but is not
        currently implemented for tree-sitter parsing. Falls back to regex-based
        extraction if tree-sitter is unavailable.
        """
        if not HAS_GO_PARSER:
            return self._extract_calls_regex(file_path)
        
        try:
            parser = _get_go_parser()
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = parser.parse(bytes(content, 'utf-8'))
            calls = []
            
            def walk_node(node, depth=0):
                if depth > 100:
                    return
                
                if node.type == 'call_expression':
                    call_info = self._extract_go_call_info(node, file_path, content)
                    if call_info:
                        calls.append(call_info)
                
                for child in node.children:
                    walk_node(child, depth + 1)
            
            walk_node(tree.root_node)
            return calls
            
        except Exception:
            return self._extract_calls_regex(file_path)
    
    def parse_imports(self, file_path: str) -> List[Dict]:
        """Parse import statements from a Go file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            imports = []
            
            # Go import patterns
            patterns = [
                # import "module"
                (r'import\s+[\'"]([^\'"]+)[\'"]', 'single'),
                # import alias "module"
                (r'import\s+(\w+)\s+[\'"]([^\'"]+)[\'"]', 'alias'),
            ]
            
            in_multi_import = False
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                
                # Check for multi-line import block
                if line == 'import (':
                    in_multi_import = True
                    continue
                elif line == ')' and in_multi_import:
                    in_multi_import = False
                    continue
                elif in_multi_import and line and not line.startswith('//'):
                    # Handle lines within import block
                    if line.startswith('"'):
                        # Simple import: "module"
                        match = re.match(r'[\'"]([^\'"]+)[\'"]', line)
                        if match:
                            imports.append({
                                'type': 'multi',
                                'module': match.group(1),
                                'name': None,
                                'asname': None,
                                'line': line_num,
                                'column': 0
                            })
                    else:
                        # Alias import: alias "module"
                        match = re.match(r'(\w+)\s+[\'"]([^\'"]+)[\'"]', line)
                        if match:
                            imports.append({
                                'type': 'multi',
                                'module': match.group(2),
                                'name': match.group(1),
                                'asname': None,
                                'line': line_num,
                                'column': 0
                            })
                    continue
                
                # Regular import patterns
                for pattern, import_type in patterns:
                    match = re.search(pattern, line)
                    if match:
                        if import_type == 'single':
                            imports.append({
                                'type': import_type,
                                'module': match.group(1),
                                'name': None,
                                'asname': None,
                                'line': line_num,
                                'column': line.find(match.group(0))
                            })
                        elif import_type == 'alias':
                            imports.append({
                                'type': import_type,
                                'module': match.group(2),
                                'name': match.group(1),
                                'asname': None,
                                'line': line_num,
                                'column': line.find(match.group(0))
                            })
            
            return imports
            
        except Exception:
            return []
    
    def _extract_go_call_info(self, node, file_path: str, content: str) -> Optional[Dict]:
        """Extract call information from a Go tree-sitter node."""
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
            
            call_info['args'] = self._extract_go_arguments(node)
            return call_info
            
        except Exception:
            return None
    
    def _extract_function_info(self, node, content: str) -> Dict:
        """Extract function information from a Go tree-sitter node."""
        result = {
            'function': None,
            'module': None,
            'object': None,
            'full_expression': None
        }
        
        try:
            if node.type == 'identifier':
                result['function'] = content[node.start_byte:node.end_byte]
            elif node.type == 'selector_expression':
                # object.method()
                object_node = node.child_by_field_name('operand')
                field_node = node.child_by_field_name('field')
                
                if field_node:
                    result['function'] = content[field_node.start_byte:field_node.end_byte]
                if object_node:
                    result['object'] = content[object_node.start_byte:object_node.end_byte]
                    result['module'] = result['object']  # In Go, object is often the package
                
                result['full_expression'] = content[node.start_byte:node.end_byte]
        except Exception:
            pass
        
        return result
    
    def _extract_go_arguments(self, node) -> List[Dict]:
        """Extract arguments from a Go function call."""
        args = []
        
        try:
            arguments_node = node.child_by_field_name('arguments')
            if arguments_node:
                # Filter out comma nodes to get correct position indices
                non_comma_children = [
                    child for child in arguments_node.children 
                    if child.type != ','
                ]
                for position, child in enumerate(non_comma_children):
                    args.append({
                        'position': position,
                        'type': 'positional',
                        'value': None,
                        'name': None
                    })
        except Exception:
            pass
        
        return args
    
    def _extract_calls_regex(self, file_path: str) -> List[Dict]:
        """Fallback regex-based call extraction for Go."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            calls = []
            
            # Go function call patterns
            patterns = [
                # Function()
                (r'(\b[A-Z]\w*)\s*\(', 'exported'),
                # package.Function()
                (r'(\b[a-z]\w+\.[A-Z]\w*)\s*\(', 'package'),
                # method()
                (r'(\b[a-z]\w+)\s*\(', 'local'),
            ]
            
            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern, _call_type in patterns:
                    for match in re.finditer(pattern, line):
                        call_info = {
                            'file': file_path,
                            'line': line_num,
                            'column': match.start(),
                            'type': 'function_call',
                            'function': match.group(1).split('.')[-1] if '.' in match.group(1) else match.group(1),
                            'object': match.group(1).split('.')[0] if '.' in match.group(1) else None,
                            'module': match.group(1).split('.')[0] if '.' in match.group(1) else None,
                            'full_expression': match.group(1),
                            'args': []
                        }
                        calls.append(call_info)
            
            return calls
            
        except Exception:
            return []


# Backward compatibility
def _extract_go_file_calls(file_path: str) -> List[Dict]:
    """Extract Go file calls (backward compatibility)."""
    parser = GoParser()
    return parser.extract_calls(file_path)
