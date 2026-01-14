"""
Java parser for cross-file call analysis.
"""

import re
import signal
from typing import Dict, List, Optional

from tldr.cross_file_calls.parsers.base import BaseParser
from tldr.cross_file_calls.core import HAS_JAVA_PARSER, _get_java_parser


class JavaParser(BaseParser):
    """Parser for Java files."""
    
    def extract_calls(self, file_path: str, timeout: Optional[float] = None) -> List[Dict]:
        """
        Extract function calls from a Java file.
        
        Note: The timeout parameter uses signal.SIGALRM (Unix only).
        Falls back to regex-based extraction if tree-sitter is unavailable or on timeout.
        """
        if not HAS_JAVA_PARSER:
            return self._extract_calls_regex(file_path)
        
        # Timeout handler
        def handler(signum, frame):
            raise TimeoutError("Parsing timed out")
        
        try:
            # Set timeout if provided and supported
            if timeout and hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(int(timeout) or 1)  # Ensure at least 1s
            
            try:
                parser = _get_java_parser()
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = parser.parse(bytes(content, 'utf-8'))
            finally:
                if timeout and hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Disable alarm
            
            calls = []
            
            def walk_node(node, depth=0):
                if depth > 100:
                    return
                
                if node.type == 'method_invocation':
                    call_info = self._extract_java_call_info(node, file_path, content)
                    if call_info:
                        calls.append(call_info)
                
                for child in node.children:
                    walk_node(child, depth + 1)
            
            walk_node(tree.root_node)
            return calls
            
        except Exception:
            return self._extract_calls_regex(file_path)
    
    def parse_imports(self, file_path: str) -> List[Dict]:
        """Parse import statements from a Java file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            imports = []
            
            # Java import patterns
            patterns = [
                # import package.Class;
                (r'import\s+([^;]+);', 'import'),
                # import static package.Class.*;
                (r'import\s+static\s+([^;]+);', 'static_import'),
            ]
            
            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern, import_type in patterns:
                    match = re.search(pattern, line)
                    if match:
                        path = match.group(1).strip()
                        imports.append({
                            'type': import_type,
                            'module': path,
                            'name': path.split('.')[-1],
                            'asname': None,
                            'line': line_num,
                            'column': line.find(match.group(0))
                        })
            
            return imports
            
        except Exception:
            return []
    
    def _extract_java_call_info(self, node, file_path: str, content: str) -> Optional[Dict]:
        """Extract call information from a Java tree-sitter node."""
        try:
            call_info = {
                'file': file_path,
                'line': node.start_point[0] + 1,
                'column': node.start_point[1],
                'type': 'method_call'
            }
            
            # Extract method name
            name_node = node.child_by_field_name('name')
            if name_node:
                call_info['function'] = content[name_node.start_byte:name_node.end_byte]
            
            # Extract object (if any)
            object_node = node.child_by_field_name('object')
            if object_node:
                call_info['object'] = content[object_node.start_byte:object_node.end_byte]
                call_info['module'] = call_info['object']
            
            call_info['args'] = self._extract_java_arguments(node)
            return call_info
            
        except Exception:
            return None
    
    def _extract_java_arguments(self, node) -> List[Dict]:
        """Extract arguments from a Java method call."""
        args = []
        
        try:
            arguments_node = node.child_by_field_name('arguments')
            if arguments_node:
                # Filter out comma nodes to get correct position indices
                non_comma_children = [
                    child for child in arguments_node.children 
                    if child.type != ','
                ]
                for position, _ in enumerate(non_comma_children):
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
        """Fallback regex-based call extraction for Java."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            calls = []
            
            # Java method call patterns
            patterns = [
                # method()
                (r'(\b[a-z]\w*)\s*\(', 'local'),
                # Class.method()
                (r'(\b[A-Z]\w*\.[a-z]\w*)\s*\(', 'static'),
                # object.method()
                (r'(\b[a-z]\w*\.[a-z]\w*)\s*\(', 'instance'),
            ]
            
            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern, _ in patterns:
                    for match in re.finditer(pattern, line):
                        call_info = {
                            'file': file_path,
                            'line': line_num,
                            'column': match.start(),
                            'type': 'method_call',
                            'function': match.group(1).split('.')[-1],
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
def _extract_java_file_calls(file_path: str) -> List[Dict]:
    """Extract Java file calls (backward compatibility)."""
    parser = JavaParser()
    return parser.extract_calls(file_path)
