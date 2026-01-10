#!/usr/bin/env python3
"""
tldr_code.py - Main entry point for TLDR function signature extractor

This script automatically detects whether the input is a GitHub URL or local directory path
and processes it accordingly to generate TLDR JSON files with function signatures.

Usage:
    python tldr_code.py <github_url_or_directory_path> [output_filename]
    python tldr_code.py https://github.com/user/repo
    python tldr_code.py ./src/my_project
    python tldr_code.py ./src/my_project my_output.json
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from urllib.parse import urlparse

from tldr.github_adapter import GitHubAdapter
from tldr.tldr_file_creator import TLDRFileCreator

def is_github_url(input_string: str) -> bool:
    """
    Determine if the input string is a GitHub URL.
    
    Args:
        input_string (str): Input string to check
        
    Returns:
        bool: True if it's a GitHub URL, False otherwise
    """
    try:
        parsed = urlparse(input_string)
        
        # Check if it has a scheme (http/https) and is a GitHub domain
        if parsed.scheme in ['http', 'https'] and parsed.netloc.lower() in ['github.com', 'www.github.com']:
            # Check if path has at least user/repo format
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) >= 2 and path_parts[0] and path_parts[1]:
                return True
        
        return False
        
    except Exception:
        return False

def process_github_url(github_url: str, github_temp_dir: str, output_filename: str = None, terse_output: bool = False) -> str:
    """
    Process a GitHub URL to create a TLDR file.
    
    Args:
        github_url (str): GitHub repository URL
        output_filename (str): Optional output filename
        terse_output (bool): Exclude files with 0 signatures
        
    Returns:
        str: Path to the generated TLDR file
        :param github_temp_dir:
    """
    logging.info(f"Processing GitHub repository: {github_url}")
    
    adapter = GitHubAdapter(terse_output=terse_output)
    
    # Determine output directory - use current directory if no specific output file given
    if github_temp_dir:
        output_dir = os.path.abspath(github_temp_dir)
        if not output_dir:
            raise NotADirectoryError(f"Temporary directory '{github_temp_dir}' is not a valid directory.")
    else:
        output_dir = os.path.abspath(".") # use current directory by default
    
    tldr_file = adapter.process_github_repo(
        github_url=github_url,
        output_dir=output_dir,
        cleanup=True
    )
    
    # If user specified a custom output filename, rename the file
    if output_filename and output_filename != tldr_file:
        os.rename(tldr_file, output_filename)
        return output_filename
    
    return tldr_file

def process_local_path(directory_path: str, output_filename: str = None, terse_output: bool = False) -> str:
    """
    Process a local directory path to create a TLDR file.
    
    Args:
        directory_path (str): Local directory path
        output_filename (str): Optional output filename
        terse_output (bool): Exclude files with 0 signatures
        
    Returns:
        str: Path to the generated TLDR file
    """
    logging.info(f"Processing local directory: {directory_path}")
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' not found.")
    
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a directory.")
    
    creator = TLDRFileCreator(terse_output=terse_output)
    
    # Set default output filename if not provided
    if output_filename is None:
        output_filename = os.path.join(directory_path, 'tldr.json')
    
    return creator.create_tldr_file(directory_path, output_filename)

def main():
    """
    Main function to handle command line arguments and route to appropriate processor.
    """
    parser = argparse.ArgumentParser(
        description='TLDR - Extract function signatures from GitHub repositories or local directories',
        epilog="""
Examples:
  python tldr_code.py https://github.com/user/repo
  python tldr_code.py ./src/my_project
  python tldr_code.py ./src/my_project custom_output.json
  python tldr_code.py https://github.com/user/repo /path/for/downloaded/repo
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'input',
        help='GitHub repository URL (https://github.com/fastapi/fastapi) or local directory path to process (/path/to/directory))'
    )
    parser.add_argument(
        'output_filename', 
        nargs='?', 
        help='Optional output filename (defaults to tldr.json in the target directory)'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--terse-output',
        action='store_true',
        help='Exclude files with 0 signatures from output'
    )
    parser.add_argument(
        '--github-temp-dir',
        type=Path,
        help='location to store temporary files for GitHub repos (default: current directory), ignored if input is a local directory'
    )

    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    # logging.info(f"Setting log level to: {logging.getLevelName(log_level)}")
    logging.getLogger().setLevel(log_level)

    try:
        logging.debug(f"args: {args}")
        # Detect input type and route accordingly
        if is_github_url(args.input):
            tldr_file = process_github_url(args.input, args.github_temp_dir, args.output_filename, args.terse_output)
            logging.info(f"✓ GitHub repository processed successfully!")
        else:
            tldr_file = process_local_path(args.input, args.output_filename, args.terse_output)
            logging.info(f"✓ Local directory processed successfully!")
        
        logging.info(f"TLDR file created: {tldr_file}")
        
    except KeyboardInterrupt:
        logging.info("\n✗ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logging.info(f"✗ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()