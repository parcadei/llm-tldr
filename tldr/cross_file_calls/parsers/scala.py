"""
Scala parser for cross-file call analysis.
"""

import re
from typing import Dict, List, Optional

from tldr.cross_file_calls.parsers.base import BaseParser


class ScalaParser(BaseParser):
    """Parser for Scala files."""
    
    def extract_calls(self, file_path: str, timeout: Optional[float] = None) -> List[Dict]:
        """
        Extract function calls from a Scala file.
        
        Note: The timeout parameter is accepted for API consistency but is not
        currently implemented for regex-based parsing.
        """
        return self._extract_calls_regex(file_path)
    
    def parse_imports(self, file_path: str) -> List[Dict]:
        """Parse import statements from a Scala file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            imports = []
            
            for line_num, line in enumerate(content.split('\n'), 1):
                # Check for grouped imports: import foo.{Bar, Baz}
                grouped_match = re.search(r'import\s+(.+)\.{([^}]+)}', line)
                if grouped_match:
                    base_module = grouped_match.group(1).strip()
                    symbols = grouped_match.group(2).strip()
                    
                    for symbol in symbols.split(','):
                        symbol = symbol.strip()
                        # Handle alias: Bar => B
                        if '=>' in symbol:
                            parts = symbol.split('=>')
                            name = parts[0].strip()
                            alias = parts[1].strip()
                            imports.append({
                                'type': 'import_grouped',
                                'module': base_module,
                                'name': name,
                                'asname': alias,
                                'line': line_num,
                                'column': line.find(grouped_match.group(0))
                            })
                        else:
                            imports.append({
                                'type': 'import_grouped',
                                'module': base_module,
                                'name': symbol,
                                'asname': None,
                                'line': line_num,
                                'column': line.find(grouped_match.group(0))
                            })
                    continue
                
                # Check for wildcard imports: import foo._
                wildcard_match = re.search(r'import\s+(.+)\._', line)
                if wildcard_match:
                    module = wildcard_match.group(1).strip()
                    imports.append({
                        'type': 'import_wildcard',
                        'module': module,
                        'name': '_',
                        'asname': None,
                        'line': line_num,
                        'column': line.find(wildcard_match.group(0))
                    })
                    continue
                
                # Regular import
                match = re.search(r'import\s+([^;{]+)', line)
                if match:
                    path = match.group(1).strip()
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
        """Regex-based call extraction for Scala."""
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
def _extract_scala_file_calls(file_path: str) -> List[Dict]:
    """Extract Scala file calls (backward compatibility)."""
    parser = ScalaParser()
    return parser.extract_calls(file_path)
