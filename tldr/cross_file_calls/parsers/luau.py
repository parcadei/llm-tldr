"""
Luau parser for cross-file call analysis.
"""

import re
from typing import Dict, List, Optional

from tldr.cross_file_calls.parsers.lua import LuaBaseParser


class LuauParser(LuaBaseParser):
    """Parser for Luau files."""
    
    def extract_calls(self, file_path: str, timeout: Optional[float] = None) -> List[Dict]:
        """Extract function calls from a Luau file."""
        return self._extract_calls_regex(file_path)
    
    def parse_imports(self, file_path: str) -> List[Dict]:
        """Parse require statements from a Luau file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            imports = []
            
            for line_num, line in enumerate(content.split('\n'), 1):
                match = re.search(r'require\s*\(\s*script\s*:\s*([^\)]+)\s*\)', line)
                if match:
                    imports.append({
                        'type': 'require',
                        'module': match.group(1).strip(),
                        'name': match.group(1).strip(),
                        'asname': None,
                        'line': line_num,
                        'column': line.find(match.group(0))
                    })
            
            return imports
            
        except Exception:
            return []


# Backward compatibility
def _extract_luau_file_calls(file_path: str) -> List[Dict]:
    """Extract Luau file calls (backward compatibility)."""
    parser = LuauParser()
    return parser.extract_calls(file_path)
