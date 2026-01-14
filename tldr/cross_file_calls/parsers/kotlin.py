"""
Kotlin parser for cross-file call analysis.
"""

import re
from typing import Dict, List, Optional

from tldr.cross_file_calls.parsers.base import BaseParser


class KotlinParser(BaseParser):
    """Parser for Kotlin files."""
    
    def extract_calls(self, file_path: str, timeout: Optional[float] = None) -> List[Dict]:
        """Extract function calls from a Kotlin file."""
        return self._extract_calls_regex(file_path)
    
    def parse_imports(self, file_path: str) -> List[Dict]:
        """Parse import statements from a Kotlin file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            imports = []
            
            for line_num, line in enumerate(content.split('\n'), 1):
                match = re.search(r'import\s+([^;]+)', line)
                if match:
                    # Extract path and strip inline comments
                    path = match.group(1).strip()
                    # Remove inline comments (// style)
                    if '//' in path:
                        path = path.split('//')[0].strip()
                    # Remove any trailing whitespace
                    path = path.strip()
                    
                    if path:  # Only add if path is not empty
                        imports.append({
                            'type': 'import',
                            'module': path,
                            'name': path.split('.')[-1],
                            'asname': None,
                            'line': line_num,
                            'column': line.find(match.group(0))
                        })
            
            return imports
            
        except Exception:
            return []
    
    def _extract_calls_regex(self, file_path: str) -> List[Dict]:
        """Regex-based call extraction for Kotlin."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            calls = []
            
            for line_num, line in enumerate(content.split('\n'), 1):
                for match in re.finditer(r'(\b[a-zA-Z_]\w*(?:\.\w+)*)\s*\(', line):
                    calls.append({
                        'file': file_path,
                        'line': line_num,
                        'column': match.start(),
                        'type': 'function_call',
                        'function': match.group(1).split('.')[-1],
                        'object': match.group(1).split('.')[0] if '.' in match.group(1) else None,
                        'module': match.group(1).split('.')[0] if '.' in match.group(1) else None,
                        'full_expression': match.group(1),
                        'args': []
                    })
            
            return calls
            
        except Exception:
            return []


# Backward compatibility
def _extract_kotlin_file_calls(file_path: str) -> List[Dict]:
    """Extract Kotlin file calls (backward compatibility)."""
    parser = KotlinParser()
    return parser.extract_calls(file_path)
