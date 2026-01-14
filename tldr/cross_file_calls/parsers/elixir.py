"""
Elixir parser for cross-file call analysis.
"""

import re
from typing import Dict, List, Optional

from tldr.cross_file_calls.parsers.base import BaseParser


class ElixirParser(BaseParser):
    """Parser for Elixir files."""
    
    def extract_calls(self, file_path: str, timeout: Optional[float] = None) -> List[Dict]:
        """Extract function calls from an Elixir file."""
        return self._extract_calls_regex(file_path)
    
    def parse_imports(self, file_path: str) -> List[Dict]:
        """Parse import/use statements from an Elixir file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            imports = []
            
            patterns = [
                (r'import\s+([A-Z][\w\.]*)', 'import'),
                (r'alias\s+([A-Z][\w\.]*)', 'alias'),
                (r'use\s+([A-Z][\w\.]*)', 'use'),
                (r'require\s+([A-Z][\w\.]*)', 'require'),
            ]
            
            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern, import_type in patterns:
                    match = re.search(pattern, line)
                    if match:
                        module = match.group(1)
                        imports.append({
                            'type': import_type,
                            'module': module,
                            'name': module.split('.')[-1],
                            'asname': None,
                            'line': line_num,
                            'column': line.find(match.group(0))
                        })
            
            return imports
            
        except Exception:
            return []
    
    def _extract_calls_regex(self, file_path: str) -> List[Dict]:
        """Regex-based call extraction for Elixir."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            calls = []
            
            # Elixir function call patterns
            patterns = [
                # Module.function()
                r'([A-Z][\w\.]*\.[a-z_]\w*)\s*\(',
                # function()
                r'(?<![A-Z])(\b[a-z_]\w*)\s*\(',
            ]
            
            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern in patterns:
                    for match in re.finditer(pattern, line):
                        full_expr = match.group(1)
                        parts = full_expr.split('.')
                        
                        calls.append({
                            'file': file_path,
                            'line': line_num,
                            'column': match.start(),
                            'type': 'function_call',
                            'function': parts[-1],
                            'object': parts[0] if len(parts) > 1 else None,
                            'module': '.'.join(parts[:-1]) if len(parts) > 1 else None,
                            'full_expression': full_expr,
                            'args': []
                        })
            
            return calls
            
        except Exception:
            return []


# Backward compatibility
def _extract_elixir_file_calls(file_path: str) -> List[Dict]:
    """Extract Elixir file calls (backward compatibility)."""
    parser = ElixirParser()
    return parser.extract_calls(file_path)
