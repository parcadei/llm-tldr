"""
Swift parser for cross-file call analysis.
"""

import re
from typing import Dict, List, Optional

from tldr.cross_file_calls.parsers.base import BaseParser


class SwiftParser(BaseParser):
    """Parser for Swift files."""
    
    def extract_calls(self, file_path: str, timeout: Optional[float] = None) -> List[Dict]:
        """Extract function calls from a Swift file."""
        return self._extract_calls_regex(file_path)
    
    def parse_imports(self, file_path: str) -> List[Dict]:
        """Parse import statements from a Swift file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            imports = []
            
            for line_num, line in enumerate(content.split('\n'), 1):
                # Handle kind-qualified imports: import class/struct/enum/protocol/func/var/let Module.Symbol
                kind_match = re.search(
                    r'import\s+(class|struct|enum|protocol|func|var|let)\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)', 
                    line
                )
                if kind_match:
                    kind = kind_match.group(1)
                    full_path = kind_match.group(2)
                    parts = full_path.split('.')
                    imports.append({
                        'type': f'import_{kind}',
                        'module': parts[0] if parts else full_path,
                        'name': full_path,
                        'asname': None,
                        'kind': kind,
                        'line': line_num,
                        'column': line.find(kind_match.group(0))
                    })
                    continue
                
                # Handle regular imports with submodules: import UIKit.UIView
                submodule_match = re.search(r'import\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)', line)
                if submodule_match:
                    full_path = submodule_match.group(1)
                    parts = full_path.split('.')
                    imports.append({
                        'type': 'import_submodule',
                        'module': parts[0],
                        'name': full_path,
                        'asname': None,
                        'line': line_num,
                        'column': line.find(submodule_match.group(0))
                    })
                    continue
                
                # Handle simple imports: import UIKit
                simple_match = re.search(r'import\s+([A-Za-z_]\w*)', line)
                if simple_match:
                    module = simple_match.group(1)
                    imports.append({
                        'type': 'import',
                        'module': module,
                        'name': module,
                        'asname': None,
                        'line': line_num,
                        'column': line.find(simple_match.group(0))
                    })
            
            return imports
            
        except Exception:
            return []
    
    def _extract_calls_regex(self, file_path: str) -> List[Dict]:
        """Regex-based call extraction for Swift."""
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
def _extract_swift_file_calls(file_path: str) -> List[Dict]:
    """Extract Swift file calls (backward compatibility)."""
    parser = SwiftParser()
    return parser.extract_calls(file_path)
