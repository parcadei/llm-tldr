"""
C parser for cross-file call analysis.
"""

import re
from typing import Dict, List, Optional

from tldr.cross_file_calls.parsers.base import BaseParser
from tldr.cross_file_calls.core import HAS_C_PARSER, _get_c_parser


class CParser(BaseParser):
    """Parser for C files."""
    
    def extract_calls(self, file_path: str, timeout: Optional[float] = None) -> List[Dict]:
        """Extract function calls from a C file."""
        if not HAS_C_PARSER:
            return self._extract_calls_regex(file_path)
        
        try:
            parser = _get_c_parser()
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = parser.parse(bytes(content, 'utf-8'))
            calls = []
            
            def walk_node(node, depth=0):
                if depth > 100:
                    return
                
                if node.type == 'call_expression':
                    call_info = self._extract_c_call_info(node, file_path, content)
                    if call_info:
                        calls.append(call_info)
                
                for child in node.children:
                    walk_node(child, depth + 1)
            
            walk_node(tree.root_node)
            return calls
            
        except Exception:
            return self._extract_calls_regex(file_path)
    
    def parse_imports(self, file_path: str) -> List[Dict]:
        """Parse include statements from a C file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            imports = []
            
            # C include patterns
            patterns = [
                # #include <header.h>
                (r'#include\s+<([^>]+)>', 'system'),
                # #include "header.h"
                (r'#include\s+"([^"]+)"', 'local'),
            ]
            
            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern, import_type in patterns:
                    match = re.search(pattern, line)
                    if match:
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
    
    def _extract_c_call_info(self, node, file_path: str, content: str) -> Optional[Dict]:
        """Extract call information from a C tree-sitter node."""
        try:
            call_info = {
                'file': file_path,
                'line': node.start_point[0] + 1,
                'column': node.start_point[1],
                'type': 'function_call'
            }
            
            function_node = node.child_by_field_name('function')
            if function_node:
                call_info['function'] = content[function_node.start_byte:function_node.end_byte]
            
            call_info['args'] = self._extract_c_arguments(node)
            return call_info
            
        except Exception:
            return None
    
    def _extract_c_arguments(self, node) -> List[Dict]:
        """Extract arguments from a C function call."""
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
        """Fallback regex-based call extraction for C."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            calls = []
            
            # C function call patterns
            pattern = r'(\b[a-zA-Z_]\w*)\s*\('
            
            for line_num, line in enumerate(content.split('\n'), 1):
                for match in re.finditer(pattern, line):
                    calls.append({
                        'file': file_path,
                        'line': line_num,
                        'column': match.start(),
                        'type': 'function_call',
                        'function': match.group(1),
                        'object': None,
                        'module': None,
                        'full_expression': match.group(1),
                        'args': []
                    })
            
            return calls
            
        except Exception:
            return []


# Backward compatibility
def _extract_c_file_calls(file_path: str) -> List[Dict]:
    """Extract C file calls (backward compatibility)."""
    parser = CParser()
    return parser.extract_calls(file_path)
