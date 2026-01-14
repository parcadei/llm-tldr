"""
Base parser interface for cross-file call analysis.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseParser(ABC):
    """Base interface for all language parsers."""
    
    @abstractmethod
    def extract_calls(self, file_path: str, timeout: Optional[float] = None) -> List[Dict]:
        """Extract function calls from a file."""
        pass
    
    @abstractmethod
    def parse_imports(self, file_path: str) -> List[Dict]:
        """Parse import statements from a file."""
        pass
