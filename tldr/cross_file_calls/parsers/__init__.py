"""
Language-specific parsers for cross-file call analysis.
"""

import os
from typing import Dict, List, Optional, Type

from tldr.cross_file_calls.parsers.base import BaseParser
from tldr.cross_file_calls.parsers.python import PythonParser
from tldr.cross_file_calls.parsers.typescript import TypeScriptParser
from tldr.cross_file_calls.parsers.go import GoParser
from tldr.cross_file_calls.parsers.rust import RustParser
from tldr.cross_file_calls.parsers.java import JavaParser
from tldr.cross_file_calls.parsers.c import CParser
from tldr.cross_file_calls.parsers.cpp import CppParser
from tldr.cross_file_calls.parsers.csharp import CSharpParser
from tldr.cross_file_calls.parsers.php import PhpParser
from tldr.cross_file_calls.parsers.ruby import RubyParser
from tldr.cross_file_calls.parsers.swift import SwiftParser
from tldr.cross_file_calls.parsers.kotlin import KotlinParser
from tldr.cross_file_calls.parsers.scala import ScalaParser
from tldr.cross_file_calls.parsers.lua import LuaParser
from tldr.cross_file_calls.parsers.luau import LuauParser
from tldr.cross_file_calls.parsers.elixir import ElixirParser


PARSERS: Dict[str, Type[BaseParser]] = {
    'python': PythonParser,
    'typescript': TypeScriptParser,
    'javascript': TypeScriptParser,
    'go': GoParser,
    'rust': RustParser,
    'java': JavaParser,
    'c': CParser,
    'cpp': CppParser,
    'csharp': CSharpParser,
    'php': PhpParser,
    'ruby': RubyParser,
    'swift': SwiftParser,
    'kotlin': KotlinParser,
    'scala': ScalaParser,
    'lua': LuaParser,
    'luau': LuauParser,
    'elixir': ElixirParser,
}


EXTENSION_MAP: Dict[str, str] = {
    '.py': 'python',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.go': 'go',
    '.rs': 'rust',
    '.java': 'java',
    '.c': 'c',
    '.h': 'c',
    '.cpp': 'cpp',
    '.cxx': 'cpp',
    '.cc': 'cpp',
    '.hpp': 'cpp',
    '.hxx': 'cpp',
    '.hh': 'cpp',
    '.cs': 'csharp',
    '.php': 'php',
    '.rb': 'ruby',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.kts': 'kotlin',
    '.scala': 'scala',
    '.sc': 'scala',
    '.lua': 'lua',
    '.luau': 'luau',
    '.ex': 'elixir',
    '.exs': 'elixir',
}


def get_parser_for_file(file_path: str) -> Optional[BaseParser]:
    _, ext = os.path.splitext(file_path.lower())
    
    if ext not in EXTENSION_MAP:
        return None
    
    language = EXTENSION_MAP[ext]
    # Redundant check removed: PARSERS is derived from EXTENSION_MAP logic usually, 
    # but here we just lookup directly.
    # If we want to be safe:
    parser_class = PARSERS.get(language)
    if parser_class:
        return parser_class()
    return None


def get_parser_for_language(language: str) -> Optional[BaseParser]:
    if language not in PARSERS:
        return None
    
    parser_class = PARSERS[language]
    return parser_class()


def list_supported_languages() -> List[str]:
    return list(PARSERS.keys())


def list_supported_extensions() -> List[str]:
    return list(EXTENSION_MAP.keys())
