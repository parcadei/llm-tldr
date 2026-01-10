import logging
import os

from pygments_tldr import highlight
from pygments_tldr.formatters.tldr import TLDRFormatter
from pygments_tldr.lexers import get_lexer_for_filename, get_lexer_by_name
from pygments_tldr.util import ClassNotFound


class SignatureExtractor():
    def get_signatures(self, filename):
        """
        Extracts function signatures from the provided code.
        """
        # Parse command line options
        show_linenos = False
        full_document = False

        # Check if file exists
        if not os.path.exists(filename):
            logging.error(f"Error: File '{filename}' not found.")
            raise FileNotFoundError(f"File '{filename}' does not exist.")

        try:
            # Read the file
            with open(filename, 'r', encoding='utf-8') as f:
                code = f.read()

            # Get appropriate lexer for the file
            try:
                lexer = get_lexer_for_filename(filename)
            except ClassNotFound:
                # Fallback to text lexer if file type not recognized
                logging.error(f"Warning: Could not determine lexer for '{filename}', using text")
                lexer = get_lexer_by_name('text')

            # Create formatter with options
            formatter_options = {
                'highlight_functions': True,
                'linenos': show_linenos,
                'full': full_document
            }

            # Auto-detect language from lexer
            if hasattr(lexer, 'aliases') and lexer.aliases:
                formatter_options['lang'] = lexer.aliases[0]

            # Set title for full documents
            if full_document:
                formatter_options['title'] = f'Code Analysis: {os.path.basename(filename)}'

            formatter = TLDRFormatter(**formatter_options)

            # Generate the highlighted code
            result = highlight(code, lexer, formatter)

            # Output the result
            logging.debug(f"Result:\n{result}")
            return result

        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}")
            raise
