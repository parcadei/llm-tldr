"""
C# parser for cross-file call analysis.
"""

import re
from typing import Dict, List, Optional

from tldr.cross_file_calls.parsers.base import BaseParser


class CSharpParser(BaseParser):
    """Parser for C# files."""
    
    _CSHARP_KEYWORDS = frozenset({
        'if', 'else', 'while', 'for', 'foreach', 'switch', 'catch', 'using',
        'lock', 'fixed', 'checked', 'unchecked', 'typeof', 'sizeof', 'nameof',
        'default', 'new', 'return', 'throw', 'await',
    })

    def extract_calls(self, file_path: str, timeout: Optional[float] = None) -> List[Dict]:
        """Extract function calls from a C# file."""
        return self._extract_calls_regex(file_path)
    
    def parse_imports(self, file_path: str) -> List[Dict]:
        """Parse using statements from a C# file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            imports = []
            
            for line_num, line in enumerate(content.split('\n'), 1):
                # global using X;
                match = re.search(r'global\s+using\s+([^;]+);', line)
                if match:
                    imports.append({
                        'type': 'global_using',
                        'module': match.group(1).strip(),
                        'name': match.group(1).strip(),
                        'asname': None,
                        'line': line_num,
                        'column': line.find(match.group(0))
                    })
                    continue
                
                # using static X;
                match = re.search(r'using\s+static\s+([^;]+);', line)
                if match:
                    imports.append({
                        'type': 'using_static',
                        'module': match.group(1).strip(),
                        'name': match.group(1).strip(),
                        'asname': None,
                        'line': line_num,
                        'column': line.find(match.group(0))
                    })
                    continue
                
                # using Alias = X;
                match = re.search(r'using\s+(\w+)\s*=\s*([^;]+);', line)
                if match:
                    imports.append({
                        'type': 'using_alias',
                        'module': match.group(2).strip(),
                        'name': match.group(2).strip(),
                        'asname': match.group(1).strip(),
                        'line': line_num,
                        'column': line.find(match.group(0))
                    })
                    continue
                
                # using Namespace;
                match = re.search(r'using\s+([^;=]+);', line)
                if match:
                    # Skip if this is a using statement (resource disposal)
                    if 'using\s*\(' in line:
                        continue
                    imports.append({
                        'type': 'using',
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
        """Regex-based call extraction for C#."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            calls = []
            
            for line_num, line in enumerate(content.split('\n'), 1):
                for match in re.finditer(r'(\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(', line):
                    func_name = match.group(1).split('.')[-1]
                    if func_name in self._CSHARP_KEYWORDS:
                        continue
                        
                    calls.append({
                        'file': file_path,
                        'line': line_num,
                        'column': match.start(),
                        'type': 'method_call',
                        'function': func_name,
                        'object': match.group(1).split('.')[0] if '.' in match.group(1) else None,
                        'module': match.group(1).split('.')[0] if '.' in match.group(1) else None,
                        'full_expression': match.group(1),
                        'args': []
                    })
            
            return calls
            
        except Exception:
            return []


# Backward compatibility
def _extract_csharp_file_calls(file_path: str) -> List[Dict]:
    """Extract C# file calls (backward compatibility)."""
    parser = CSharpParser()
    return parser.extract_calls(file_path)
