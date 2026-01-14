"""
Lua base parser and Lua parser for cross-file call analysis.
"""

import re
from typing import Dict, List, Optional

from tldr.cross_file_calls.parsers.base import BaseParser


class LuaBaseParser(BaseParser):
    """Base parser with shared logic for Lua and Luau files."""
    
    def _extract_calls_regex(self, file_path: str) -> List[Dict]:
        """Shared regex-based call extraction for Lua/Luau."""
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


class LuaParser(LuaBaseParser):
    """Parser for Lua files."""
    
    def extract_calls(self, file_path: str, timeout: Optional[float] = None) -> List[Dict]:
        """Extract function calls from a Lua file."""
        return self._extract_calls_regex(file_path)
    
    def parse_imports(self, file_path: str) -> List[Dict]:
        """Parse require statements from a Lua file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            imports = []
            
            for line_num, line in enumerate(content.split('\n'), 1):
                # Support require "mod" and require("mod")
                match = re.search(r'require\s*\(?\s*[\'"]([^\'"]+)[\'"]', line)
                if match:
                    imports.append({
                        'type': 'require',
                        'module': match.group(1),
                        'name': match.group(1),
                        'asname': None,
                        'line': line_num,
                        'column': line.find(match.group(0))
                    })
            
            return imports
            
        except Exception:
            return []


# Backward compatibility
def _extract_lua_file_calls(file_path: str) -> List[Dict]:
    """Extract Lua file calls (backward compatibility)."""
    parser = LuaParser()
    return parser.extract_calls(file_path)
