"""
PHP parser for cross-file call analysis.
"""

import re
from typing import Dict, List, Optional

from tldr.cross_file_calls.parsers.base import BaseParser


class PhpParser(BaseParser):
    """Parser for PHP files."""
    
    def extract_calls(self, file_path: str, timeout: Optional[float] = None) -> List[Dict]:
        """Extract function calls from a PHP file."""
        return self._extract_calls_regex(file_path)
    
    def parse_imports(self, file_path: str) -> List[Dict]:
        """Parse include/require statements from a PHP file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            imports = []
            
            patterns = [
                (r'include\s+[\'"]([^\'"]+)[\'"]', 'include'),
                (r'include_once\s+[\'"]([^\'"]+)[\'"]', 'include_once'),
                (r'require\s+[\'"]([^\'"]+)[\'"]', 'require'),
                (r'require_once\s+[\'"]([^\'"]+)[\'"]', 'require_once'),
                (r'use\s+([^;]+);', 'use'),
            ]
            
            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern, import_type in patterns:
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
        """Regex-based call extraction for PHP."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            calls = []
            
            for line_num, line in enumerate(content.split('\n'), 1):
                # Matches: func(), $obj->method(), Class::method(), \NS\func()
                # Group 1: Optional qualifier ($var->, Class::, \NS\)
                # Group 2: Function/Method name
                for match in re.finditer(r'(?:(\$?\w+(?:->|::)|\\(?:[\w\\]+\\)?))?(\w+)\s*\(', line):
                    qualifier = match.group(1)
                    func_name = match.group(2)
                    full_expr = match.group(0)[:-1] # strip '('
                    
                    obj = None
                    module = None
                    
                    if qualifier:
                        if '->' in qualifier:
                            obj = qualifier[:-2] # strip ->
                        elif '::' in qualifier:
                            obj = qualifier[:-2] # strip ::
                            module = obj
                        else:
                            # Namespace qualifier
                            module = qualifier.rstrip('\\')

                    calls.append({
                        'file': file_path,
                        'line': line_num,
                        'column': match.start(),
                        'type': 'function_call',
                        'function': func_name,
                        'object': obj,
                        'module': module,
                        'full_expression': full_expr,
                        'args': []
                    })
            
            return calls
            
        except Exception:
            return []


# Backward compatibility
def _extract_php_file_calls(file_path: str) -> List[Dict]:
    """Extract PHP file calls (backward compatibility)."""
    parser = PhpParser()
    return parser.extract_calls(file_path)