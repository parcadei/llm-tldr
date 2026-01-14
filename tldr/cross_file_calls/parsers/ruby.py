"""
Ruby parser for cross-file call analysis.
"""

import re
from typing import Dict, List, Optional

from tldr.cross_file_calls.parsers.base import BaseParser


class RubyParser(BaseParser):
    """Parser for Ruby files."""
    
    def extract_calls(self, file_path: str, timeout: Optional[float] = None) -> List[Dict]:
        """Extract function calls from a Ruby file."""
        return self._extract_calls_regex(file_path)
    
    def parse_imports(self, file_path: str) -> List[Dict]:
        """Parse require statements from a Ruby file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            imports = []
            
            patterns = [
                (r'require\s+[\'"]([^\'"]+)[\'"]', 'require'),
                (r'require_relative\s+[\'"]([^\'"]+)[\'"]', 'require_relative'),
                (r'include\s+(\w+)', 'include'),
            ]
            
            for line_num, line in enumerate(content.split('\n'), 1):
                # Skip comment-only lines
                if line.strip().startswith('#'):
                    continue
                
                for pattern, import_type in patterns:
                    # For 'include' pattern, skip if it's inside a string
                    if import_type == 'include':
                        # Check if 'include' is inside quotes
                        if re.search(r'(["\']).*include\s+\w+.*\1', line):
                            continue
                    
                    match = re.search(pattern, line)
                    if match:
                        imports.append({
                            'type': import_type,
                            'module': match.group(1).strip(),
                            'name': match.group(1).strip(),
                            'asname': None,
                            'line': line_num,
                            'column': line.find(match.group(0))
                        })
            
            return imports
            
        except Exception:
            return []
    
    def _extract_calls_regex(self, file_path: str) -> List[Dict]:
        """Regex-based call extraction for Ruby."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            calls = []
            
            for line_num, line in enumerate(content.split('\n'), 1):
                # Matches: method(args), Receiver.method(args), Module::Class.method(args)
                for match in re.finditer(r'(\b(?:[A-Za-z_]\w*(?:\.|::))*[A-Za-z_]\w*[!?]?)\s*\(', line):
                    full_expr = match.group(1)
                    
                    # Split into object/module and function name
                    if '.' in full_expr:
                        parts = full_expr.rsplit('.', 1)
                        obj = parts[0]
                        func = parts[1]
                    elif '::' in full_expr:
                        parts = full_expr.rsplit('::', 1)
                        obj = parts[0]
                        func = parts[1]
                    else:
                        obj = None
                        func = full_expr
                        
                    calls.append({
                        'file': file_path,
                        'line': line_num,
                        'column': match.start(),
                        'type': 'method_call',
                        'function': func,
                        'object': obj,
                        'module': obj,
                        'full_expression': full_expr,
                        'args': []
                    })
            
            return calls
            
        except Exception:
            return []


# Backward compatibility
def _extract_ruby_file_calls(file_path: str) -> List[Dict]:
    """Extract Ruby file calls (backward compatibility)."""
    parser = RubyParser()
    return parser.extract_calls(file_path)
