"""
Call graph builder for cross-file call analysis.
"""

import os
from typing import Dict, List, Optional, Set

from tldr.cross_file_calls.core import ProjectCallGraph
from tldr.cross_file_calls.scanner import scan_project


def build_project_call_graph(
    root_dir: str,
    languages: Optional[List[str]] = None,
    verbose: bool = False,
    exclude_dirs: Optional[Set[str]] = None,
    include_dirs: Optional[Set[str]] = None,
    max_file_size: Optional[int] = None,
    follow_symlinks: bool = False,
    respect_gitignore: bool = True,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> ProjectCallGraph:
    """
    Build a complete project call graph.
    
    This is the main entry point for building a call graph from a project.
    
    Args:
        root_dir: Root directory to scan
        languages: List of languages to process (None for all)
        verbose: Enable verbose output
        exclude_dirs: Directories to exclude
        include_dirs: Directories to include
        max_file_size: Maximum file size to process (bytes)
        follow_symlinks: Whether to follow symbolic links
        respect_gitignore: Whether to respect .gitignore files
        parallel: Enable parallel processing
        max_workers: Maximum number of worker threads
        
    Returns:
        ProjectCallGraph containing all discovered calls and relationships
    """
    # Use scan_project to do the heavy lifting
    call_graph = scan_project(
        root_dir=root_dir,
        languages=languages,
        verbose=verbose,
        exclude_dirs=exclude_dirs,
        include_dirs=include_dirs,
        max_file_size=max_file_size,
        follow_symlinks=follow_symlinks,
        respect_gitignore=respect_gitignore,
        parallel=parallel,
        max_workers=max_workers,
    )
    
    # Resolve cross-file calls
    _resolve_cross_file_calls(call_graph, verbose)
    
    return call_graph


def _resolve_cross_file_calls(call_graph: ProjectCallGraph, verbose: bool = False) -> None:
    """Resolve cross-file call relationships."""
    if verbose:
        print("Resolving cross-file calls...")
    
    # Build a map of definitions for quick lookup
    definition_map = {}
    for file_path, file_info in call_graph.files.items():
        for definition in file_info.get('definitions', []):
            key = (definition.get('name'), definition.get('type'))
            if key not in definition_map:
                definition_map[key] = []
            definition_map[key].append({
                'file': file_path,
                'definition': definition
            })
    
    # Resolve each call to its definition
    resolved_count = 0
    for file_path, file_info in call_graph.files.items():
        for call in file_info.get('calls', []):
            func_name = call.get('function')
            if not func_name:
                continue
            
            # Try to find matching definition
            for def_type in ['function', 'method', 'class']:
                key = (func_name, def_type)
                if key in definition_map:
                    # Found potential matches
                    matches = definition_map[key]
                    
                    # Prefer definitions from the same file
                    same_file_matches = [m for m in matches if m['file'] == file_path]
                    if same_file_matches:
                        call['resolved_to'] = same_file_matches[0]
                    else:
                        # Use first match from other files
                        call['resolved_to'] = matches[0]
                    
                    resolved_count += 1
                    break
    
    if verbose:
        print(f"Resolved {resolved_count} calls")


def _build_python_call_graph(root_dir: str, verbose: bool = False) -> ProjectCallGraph:
    """Build call graph for Python files only."""
    return build_project_call_graph(
        root_dir=root_dir,
        languages=['python'],
        verbose=verbose
    )


def _build_typescript_call_graph(root_dir: str, verbose: bool = False) -> ProjectCallGraph:
    """Build call graph for TypeScript files only."""
    return build_project_call_graph(
        root_dir=root_dir,
        languages=['typescript', 'javascript'],
        verbose=verbose
    )


def _build_go_call_graph(root_dir: str, verbose: bool = False) -> ProjectCallGraph:
    """Build call graph for Go files only."""
    return build_project_call_graph(
        root_dir=root_dir,
        languages=['go'],
        verbose=verbose
    )


def _build_rust_call_graph(root_dir: str, verbose: bool = False) -> ProjectCallGraph:
    """Build call graph for Rust files only."""
    return build_project_call_graph(
        root_dir=root_dir,
        languages=['rust'],
        verbose=verbose
    )


def _build_java_call_graph(root_dir: str, verbose: bool = False) -> ProjectCallGraph:
    """Build call graph for Java files only."""
    return build_project_call_graph(
        root_dir=root_dir,
        languages=['java'],
        verbose=verbose
    )


def _build_c_call_graph(root_dir: str, verbose: bool = False) -> ProjectCallGraph:
    """Build call graph for C files only."""
    return build_project_call_graph(
        root_dir=root_dir,
        languages=['c'],
        verbose=verbose
    )


def _build_cpp_call_graph(root_dir: str, verbose: bool = False) -> ProjectCallGraph:
    """Build call graph for C++ files only."""
    return build_project_call_graph(
        root_dir=root_dir,
        languages=['cpp'],
        verbose=verbose
    )


def extract_call_graph(
    root_dir: str,
    language: Optional[str] = None,
    verbose: bool = False,
    **kwargs
) -> ProjectCallGraph:
    """
    Extract call graph from a project (backward compatibility).
    
    This function provides backward compatibility with the old API.
    """
    languages = [language] if language else None
    return build_project_call_graph(
        root_dir=root_dir,
        languages=languages,
        verbose=verbose,
        **kwargs
    )
