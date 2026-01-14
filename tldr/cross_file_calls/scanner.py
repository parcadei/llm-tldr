"""
Project scanning functionality for cross-file call analysis.
"""

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tldr.cross_file_calls.core import ProjectCallGraph
from tldr.parse_helpers import find_files_by_extension

logger = logging.getLogger(__name__)


def scan_project(
    root_dir: str,
    languages: Optional[List[str]] = None,
    verbose: bool = False,
    exclude_dirs: Optional[Set[str]] = None,
    include_dirs: Optional[Set[str]] = None,
    exclude_patterns: Optional[Set[str]] = None,
    include_patterns: Optional[Set[str]] = None,
    max_file_size: Optional[int] = None,
    follow_symlinks: bool = False,
    respect_gitignore: bool = True,
    incremental: bool = False,
    cache_dir: Optional[str] = None,
    file_timeout: Optional[float] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> ProjectCallGraph:
    """
    Scan a project directory and build a call graph.
    
    Args:
        root_dir: Root directory to scan
        languages: List of languages to process (None for all)
        verbose: Enable verbose output
        exclude_dirs: Directories to exclude
        include_dirs: Directories to include (exclusive with exclude_dirs)
        exclude_patterns: File patterns to exclude
        include_patterns: File patterns to include (exclusive with exclude_patterns)
        max_file_size: Maximum file size to process (bytes)
        follow_symlinks: Whether to follow symbolic links
        respect_gitignore: Whether to respect .gitignore files
        incremental: Enable incremental scanning (not yet implemented)
        cache_dir: Directory for cache files (not yet implemented)
        file_timeout: Timeout per file (seconds)
        parallel: Enable parallel processing
        max_workers: Maximum number of worker threads
        
    Returns:
        ProjectCallGraph containing all discovered calls and relationships
    """
    # Warn about unimplemented parameters
    if incremental or cache_dir:
        warnings.warn(
            "incremental and cache_dir parameters are not yet implemented",
            stacklevel=2
        )
    
    start_time = time.time()
    
    if verbose:
        print(f"Scanning project: {root_dir}")
        if languages:
            print(f"Languages: {', '.join(languages)}")
    
    # Initialize project call graph
    call_graph = ProjectCallGraph()
    
    # Find all relevant files
    files_to_scan = _find_files_to_scan(
        root_dir=root_dir,
        languages=languages,
        exclude_dirs=exclude_dirs,
        include_dirs=include_dirs,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        max_file_size=max_file_size,
        follow_symlinks=follow_symlinks,
        respect_gitignore=respect_gitignore,
        verbose=verbose
    )
    
    if verbose:
        print(f"Found {len(files_to_scan)} files to scan")
    
    # Process files and extract calls
    _process_files(
        files_to_scan=files_to_scan,
        call_graph=call_graph,
        verbose=verbose,
        file_timeout=file_timeout,
        parallel=parallel,
        max_workers=max_workers
    )
    
    # Build call relationships
    _build_call_relationships(call_graph)
    
    end_time = time.time()
    
    if verbose:
        print(f"Scanning completed in {end_time - start_time:.2f}s")
        print(f"Processed {len(call_graph.files)} files")
        print(f"Found {len(call_graph.calls)} calls")
        print(f"Discovered {len(call_graph.definitions)} definitions")
    
    return call_graph


def _find_files_to_scan(
    root_dir: str,
    languages: Optional[List[str]],
    exclude_dirs: Optional[Set[str]],
    include_dirs: Optional[Set[str]],
    exclude_patterns: Optional[Set[str]],
    include_patterns: Optional[Set[str]],
    max_file_size: Optional[int],
    follow_symlinks: bool,
    respect_gitignore: bool,
    verbose: bool
) -> List[str]:
    """Find all files that should be scanned."""
    
    # Default exclude directories
    default_exclude_dirs = {
        '.git', '.svn', '.hg', '__pycache__', 'node_modules',
        '.vscode', '.idea', 'build', 'dist', 'target', 'vendor',
        '.venv', 'venv', 'env', '.env', '.tox', '.pytest_cache'
    }
    
    if exclude_dirs:
        default_exclude_dirs.update(exclude_dirs)
    
    # Language-specific file extensions
    lang_extensions = {
        'python': ['.py'],
        'typescript': ['.ts', '.tsx'],
        'javascript': ['.js', '.jsx'],
        'go': ['.go'],
        'rust': ['.rs'],
        'java': ['.java'],
        'c': ['.c', '.h'],
        'cpp': ['.cpp', '.cxx', '.cc', '.hpp', '.hxx', '.hh'],
        'csharp': ['.cs'],
        'php': ['.php'],
        'ruby': ['.rb'],
        'swift': ['.swift'],
        'kotlin': ['.kt', '.kts'],
        'scala': ['.scala', '.sc'],
        'lua': ['.lua'],
        'luau': ['.luau'],
        'elixir': ['.ex', '.exs']
    }
    
    # Determine which extensions to look for
    extensions = set()
    if languages:
        for lang in languages:
            if lang in lang_extensions:
                extensions.update(lang_extensions[lang])
    else:
        for exts in lang_extensions.values():
            extensions.update(exts)
    
    # Find files using the helper function
    # Merge exclude dirs
    effective_exclude_dirs = set(default_exclude_dirs)
    if exclude_dirs:
        effective_exclude_dirs.update(exclude_dirs)
        
    # If include_dirs is specified, we still want to apply exclusions like .git or node_modules
    # The original code disabled exclusions if include_dirs was set, which is unsafe.
    
    files = find_files_by_extension(
        root_dir=root_dir,
        extensions=list(extensions),
        exclude_dirs=effective_exclude_dirs,
        include_dirs=include_dirs,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        max_file_size=max_file_size,
        follow_symlinks=follow_symlinks,
        respect_gitignore=respect_gitignore,
        verbose=verbose
    )
    
    return files


def _process_files(
    files_to_scan: List[str],
    call_graph: ProjectCallGraph,
    verbose: bool,
    file_timeout: Optional[float],
    parallel: bool,
    max_workers: Optional[int]
) -> None:
    """Process all files and extract calls."""
    
    if parallel and len(files_to_scan) > 1:
        _process_files_parallel(
            files_to_scan, call_graph, verbose, file_timeout, max_workers
        )
    else:
        _process_files_sequential(
            files_to_scan, call_graph, verbose, file_timeout
        )


def _process_files_sequential(
    files_to_scan: List[str],
    call_graph: ProjectCallGraph,
    verbose: bool,
    file_timeout: Optional[float]
) -> None:
    """Process files sequentially."""
    
    from tldr.cross_file_calls.parsers import get_parser_for_file
    
    for i, file_path in enumerate(files_to_scan):
        if verbose and (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(files_to_scan)} files")
        
        try:
            # Get appropriate parser for this file
            parser = get_parser_for_file(file_path)
            if not parser:
                continue
            
            # Extract calls from file
            file_calls = parser.extract_calls(file_path, timeout=file_timeout)
            
            # Add to call graph
            call_graph.add_file(file_path, file_calls)
            
        except Exception as e:
            if verbose:
                print(f"Error processing {file_path}: {e}")
            continue


def _process_files_parallel(
    files_to_scan: List[str],
    call_graph: ProjectCallGraph,
    verbose: bool,
    file_timeout: Optional[float],
    max_workers: Optional[int]
) -> None:
    """Process files in parallel."""
    
    import concurrent.futures
    import threading
    from tldr.cross_file_calls.parsers import get_parser_for_file
    
    # Capture verbose and file_timeout for thread-safe access
    _verbose = verbose
    _file_timeout = file_timeout
    
    # Thread-safe error collection
    errors: List[str] = []
    errors_lock = threading.Lock()
    
    def process_file(file_path: str, timeout: Optional[float], log_verbose: bool) -> Tuple[str, List]:
        """Process a single file and return results."""
        try:
            parser = get_parser_for_file(file_path)
            if not parser:
                return file_path, []
            
            file_calls = parser.extract_calls(file_path, timeout=timeout)
            return file_path, file_calls
            
        except Exception as e:
            if log_verbose:
                with errors_lock:
                    errors.append(f"Error processing {file_path}: {e}")
            return file_path, []
    
    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_file, file_path, _file_timeout, _verbose): file_path
            for file_path in files_to_scan
        }
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
            if _verbose and (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(files_to_scan)} files")
            
            file_path, file_calls = future.result()
            if file_calls:
                call_graph.add_file(file_path, file_calls)
    
    # Report errors after processing
    if _verbose:
        for error in errors:
            print(error)


def _build_call_relationships(call_graph: ProjectCallGraph) -> None:
    """
    Build relationships between calls and definitions.
    
    Note: This is a placeholder for backward compatibility.
    The actual resolution logic is implemented in builder._resolve_cross_file_calls.
    """
    # The actual cross-file resolution is done in builder._resolve_cross_file_calls
    # This function is kept for API compatibility
    pass
