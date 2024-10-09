from pathlib import Path
import pandas as pd
import argparse

from pagewise_utils import assemble_pagewise_raw_parser_output

def main():
    """
    Program that compiles a dataset of pages (rather than documents) by fuzzy string matching page texts of PyMuPDF with
    entire-document strings produced by Nougat, Marker, etc.

    Output is a `raw` table, i.e. has columns:
    - `path` (rel. path of source PDF)
    - `html`, `marker`, `nougat`, etc. (text columns of each parser or groundtruth [html]), 
    - `page` (id of page from which the text comes for a given document path)
    """
    
    # argument parser setup
    parser = argparse.ArgumentParser(description="Assemble pagewise raw parser output")
    
    # arguments
    parser.add_argument('-i', '--index', type=int, default=0, help="Index of the chunk (of the PDFs) being processed. Default is 0.")
    parser.add_argument('-p', '--parsers', nargs='+', default=['html', 'nougat', 'pymupdf', 'pypdf', 'marker', 'grobid'], 
        help="List of parsers to use for extraction. Default is ['html', 'nougat', 'pymupdf', 'pypdf', 'marker', 'grobid']."
    )
    parser.add_argument('-s', '--store_dir', type=Path, default=Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/database/pagewise/sub_tables_raw_AFTER'),
        help="Directory to store the output. Default is '/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/database/pagewise'.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    assemble_pagewise_raw_parser_output(i=args.index, parsers=args.parsers, store_dir=args.store_dir)

# entry point
if __name__ == "__main__":
    main()