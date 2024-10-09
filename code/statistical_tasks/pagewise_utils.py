from pathlib import Path
import json
import re
import pymupdf
import pandas as pd

from concurrent.futures import ProcessPoolExecutor

from bisect import bisect_left
from rapidfuzz import fuzz, process
import diff_match_patch as dmp_module

def assemble_pagewise_raw_parser_output(i:int,
                                        parsers:list[str]=['html','nougat', 'pymupdf', 'pypdf', 'marker', 'grobid'],
store_dir:Path=Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/database/pagewise/sub_tables_raw_AFTER'),
                                        p_pdf_root: Path = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/joint'),
                                        save_flag:bool=True):
    """
    Generates page-wise variant of `parser_output_AFTER_raw.csv` of texts (path, html, ..., nougat)
    - i: core (to identify chunk)
    """

    assert i in {-1,0,1,2,3,4}, "Given 5 Sophia machines at a time. only index 0-4 valid or (-1) to process in full."
    store_dir = Path(store_dir)
    p_pdf_root = Path(p_pdf_root)
    assert store_dir.is_dir(), f"Store_dir does not exist. Invalid path: {store_dir}"
    assert p_pdf_root.is_dir(), f"Root directory for PDFs does not exist. Invalid path: {p_pdf_root}"
    assert len(list(p_pdf_root.rglob("*.pdf")))>0, "Root dir for PDFs exist - but has no `.pdf` files in it`"
    
    # load PDF pagelist chunk (PyMuPDF)
    dict_of_page_lists = get_list_of_pdf_page_lists(i=i, p_pdf_root=p_pdf_root) # DEBUG -> subset to 20 for testing
    
    # assemble 
    data_list_dict = {}
    for parser in parsers:
        data_list_dict[parser] = read_out_parser_output(paths=dict_of_page_lists.keys(), 
                                                        parser=parser,
                                                        root_dir=p_pdf_root)
    
    # loop PDF pagelists (for all parsers)
    # - assemble DataFrame [path|html|nougat|pymupdf|pypdf|marker|grobid] w/ path in the style of `arxiv/pdf/2207.11282v4.pdf`
    all_paths = get_unique_pdf_paths_from_data_list_dict(data_list_dict)
    
    # - 
    dict_of_page_lists = {p_pdf_root / end_of_path(k):v for k,v in dict_of_page_lists.items()}

    # loop paths
    all_rows = []
    
    # all_paths
    all_paths = [p_pdf_root / p for p in all_paths]

    for p in all_paths:
        # parser
        for parser in parsers:
            # PyMuPDF reference exists
            if p in dict_of_page_lists.keys():
                
                # load parser's full text
                parser_fulltext = get_text_by_path(p, data_list_dict[parser])
    
                # if None
                if parser_fulltext is None:
                    parser_fulltext = ''
                
                # load page list from PyMuPDF
                page_list = dict_of_page_lists[p]
                
                # split
                if len(parser_fulltext) > 0:
                    # LEGACY
                    parsed_page_list = partition_fulltext_by_pagelist(parser_fulltext, page_list)
                    # TODO: swap in new new_partition_fulltext_by_pagelist()
                    # ...
                else:
                    parsed_page_list = {k : '' for k in range(len(page_list))}
    
                # assemble
                # - grab from source (not the secondary parsing)
                if parser=='pymupdf':
                    for page_idx, page_text in enumerate(page_list):
                        row = {'path' : p, 'page' : page_idx, 'text' : page_text, 'parser' : 'pymupdf'}
                        # - append
                        all_rows.append(row)
                else:
                    for page_idx, page_text in parsed_page_list.items():
                        row = {'path' : p, 'page' : page_idx, 'text' : page_text, 'parser' : parser}
                        # - append
                        all_rows.append(row)
            # else
    # PyMUPDF
    df = assemble_dataframe(all_rows, i, store_dir)
    
    # store
    if save_flag:
        df.to_csv(store_dir / f'pagewise_parser_output_raw_AFTER_{i}_5.csv', sep='|', index=None)

    return df

def get_unique_pdf_paths_from_data_list_dict(data_list_dict:dict) -> list[str]:
    """Extracts all all observed (short) paths from the data_list_dict
    """

    # empty
    all_paths = []
    # loop parsers
    for parser in data_list_dict.keys():
        if data_list_dict[parser] is not None:
            all_paths += [data['path'] for data in data_list_dict[parser] if (data is not None) and (data['path'] is not None)]

    # filrer, format to `/arxiv/pdf/3542.123.pdf etc.`
    all_paths = sorted(list(set([end_of_path(p) for p in all_paths])))

    return all_paths

def end_of_path(p: Path | str) -> str:
    """
    Transforms various path objects into the style of `arxiv/pdf/1311.pdf`
    """
    # Ensure the input is a Path object
    path_obj = Path(p)
    
    # Get the last two parent directories and the filename
    # -2 means the second-to-last directory, -1 is the parent, and the filename is at the end
    parts = path_obj.parts
    if len(parts) >= 3:
        return str(Path(*parts[-3:]))  # Return the last two directories and the filename
    else:
        # If the path has fewer than 3 parts, return the full path
        return str(path_obj)


def filter_unique_paths(parser_data_list):
    """
    Deduplicate list w.r.t. `path` (such that longer `text` gets through)

    parser_data_list: list[dict[]]
    """
    # dictionary to store the longest text length for each unique path
    unique_paths = {}
    
    # loop through the list of dictionaries
    for data in parser_data_list:
        if (data is not None) and (data['text'] is not None):
            # - - -
            path = data['path']
            text_length = len(data['text'])
            
            # If the path is not in the dictionary or the current text is longer, update it
            if path not in unique_paths or text_length > len(unique_paths[path]['text']):
                unique_paths[path] = data
    
    # extract the values from unique_paths (which are the dictionaries) into a list
    return list(unique_paths.values())

def read_out_parser_output(parser:str,
                           paths:list[Path],
                           root_dir:Path=Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/')):
    """
    Returns entire output of parser
    - parser[str] : name of the parser (lowercase) (e.g. nougat)
    - paths[list[Path|str]] : list of pdf paths
    - root_dir[Path] : root directory where various parser outputs are stored (usually ../joint_to_{parser}
    """
    
    # - loop 
    root_dir = Path(root_dir)
    assert parser in ['marker', 'nougat', 'grobid', 'pymupdf', 'pypdf', 'html'], "Only one of these parsers is supported."
    assert root_dir.is_dir(), f"Input `root_dir` is invlaid. Does not exist: {root_dir}"
    
    # - open 
    p_parser_root = root_dir / f'joint_to_{parser}/parsed_pdfs/'
    assert p_parser_root.is_dir(), f"Inferred location of jsonls does not exist. Invalid path: {p_parser_root}"

    # - extract
    parser_paths = list(p_parser_root.rglob("*.jsonl"))
    assert len(parser_paths) > 0, f"Path parser_paths=`{parser_paths}` does exist. But does not contain any jsonl!"
    
    # - read data
    parser_data_list = []
    for jsonl_path in parser_paths:
        with open(jsonl_path, 'r') as file:
            for line in file:
                parser_data_list.append(json.loads(line))
    
    # - sort by the 'path' key (for efficient searching)
    parser_data_list.sort(key=lambda x: end_of_path(x['path']))
    
    # subset indices
    parser_data_list = [f for f in parser_data_list if '/.ipynb_checkpoints/' not in f['path']]
    
    # filter (check if in pdf `paths`)
    parser_data_list = [f for f in parser_data_list if any(f['path'].endswith(str(short_path)) for short_path in paths)]
    
    # de-duplicate entries w.r.t `path` (leave longest `text` in)
    if parser_data_list is not None and (len(parser_data_list) > 0):
        parser_data_list = filter_unique_paths(parser_data_list)
    else:
        parser_data_list = None

    return parser_data_list

def get_text_by_path(pdf_abs_path, parser_data_list, debug:bool=False):
    """
    - pdf_path:str Relative path to PDF source (e.g. arxiv/pdf/2408.03658v1.pdf)
    """
    pdf_path_end = end_of_path(pdf_abs_path)

    if parser_data_list is None:
        return []
    
    # List of all paths for binary search
    paths = [end_of_path(item['path']) for item in parser_data_list]

    # Perform binary search to find the index of the ref_path
    idx = bisect_left(paths, pdf_path_end)

    # Check if the found index is valid and matches the ref_path
    if idx < len(parser_data_list) and end_of_path(parser_data_list[idx]['path']) == pdf_path_end:
        # Copy the text from the found dictionary
        parser_text = parser_data_list[idx]['text']
        return parser_text
    else:
        if debug:
            print(f"Cannot find ... {pdf_path_end}")
        # Handle case where ref_path is not found
        return None



def partition_fulltext_by_pagelist(fulltext:str, page_list:list[str]):
    """
    CURRENT VERSION: partition_fulltext_by_pagelist_SAFECOPY()
    
    Splits singular fulltext string (from one parser) by page strings (from other parser).
    
    Combines fuzzy matching and speculative split (from text lengths)
    
    - fulltext[str]        : Parsed full text from one parser  
    - page_list[list[str]] : Parsed text pages from another parser
    """

    # rename
    S = fulltext
    s_list = page_list
    # Dictionary to store the result with page index as key and extracted text as value
    page_splits = {}
    current_position = 0

    # infer approximate lengths for each page based on the length of s_list items
    approx_len_list = [len(s) for s in s_list]

    
    # loop pages
    for i, s in enumerate(s_list):
        # get the initial guess for the page length
        approx_len = approx_len_list[i]
        guess_position = current_position + approx_len
        
        # find the actual split point by adjusting around the guess
        best_split = S[current_position:guess_position]
        
        # adjust the split point dynamically based on matching the beginning of the page
        match_start = best_split.find(s[:min(100, len(s))])  # first 100 (800 unaffect) chars for quick fuzzy match
        
        if match_start != -1:
            # Adjust current position based on the best match
            best_split = S[current_position + match_start: current_position + match_start + len(s)]
        
        # store the matched chunk in the dictionary with index i as the key
        page_splits[i] = best_split
        
        # dynamically adjust the current position and the next approximate length
        current_position += len(best_split)
        if i + 1 < len(approx_len_list):
            # Adjust the next approximate length based on how well the match performed
            next_len_diff = len(best_split) - approx_len_list[i]
            approx_len_list[i + 1] += next_len_diff
    
    return page_splits

def process_single_pdf(pdf_abs_path):
    """Helper function to process a single PDF."""
    page_list = []
    
    try:
        # Open the PDF document
        with pymupdf.open(str(pdf_abs_path)) as pdf_document:
            # Loop over all pages in the document
            for page_number in range(pdf_document.page_count):
                page = pdf_document.load_page(page_number)
    
                # extraction modes: `text` or `blocks`
                blocks_flag = False # `text` appears slightly better than `blocks`

                # extract text as blocks from the current page
                if blocks_flag:
                    page_text = page.get_text("blocks")
                else:
                    page_text = page.get_text("text")
    
                # Clean the page text
                if blocks_flag:
                    clean_page = "\n".join([b[4] for b in page_text if len(b[4]) > 30])
                else:
                    clean_page = page_text
    
                # Append the text of the current page to the list
                page_list.append(clean_page)
    except Exception as e:
        print(f"Error processing {pdf_abs_path}: {str(e)}")
    
    return page_list

def assemble_dataframe(all_rows:list[dict], i:int, store_dir:Path) -> pd.DataFrame:
    '''
    Compile/transform rows into DataFrame
    '''

    # compile
    df = pd.DataFrame(all_rows)

    # assemble DataFrame
    df_pivoted = df.pivot_table(index=['path', 'page'], columns='parser', values='text', aggfunc='first').reset_index()
    # - rename
    df_pivoted.columns.name = None  # Remove the 'parser' name from columns
    
    # delete duplicates
    df_pivoted['non_nan_count'] = df_pivoted.drop(columns=['path', 'page']).notna().sum(axis=1)
    # -sort by 'path', 'page' and the 'non_nan_count'
    df_pivoted = df_pivoted.sort_values(by=['path', 'page', 'non_nan_count'], ascending=[True, True, False])
    df_cleaned = df_pivoted.drop_duplicates(subset=['path', 'page'], keep='first')
    # -drop the helper column as it's no longer needed
    df_cleaned = df_cleaned.drop(columns=['non_nan_count'])
    # fill NaN
    df_cleaned = df_cleaned.fillna('')
    
    # -> store
    return df_cleaned

def get_list_of_pdf_page_lists(i: int, 
                               p_pdf_root: Path = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/joint'), 
                               n_max_core: int = 5):
    """
    Looks up all available PDFs - returns them page-by-page (via PyMuPDF) in parallel.

    - i [int]: Index of the chunk (of the PDFs) being processed
    - p_pdf_root [Path]: Directory in which PDFs reside (in subdirectories)
    - n_max_core: Maximum number of machines available to split up the work
    """

    # look up PDFs
    assert i in [-1] + list(range(n_max_core)), f"Given `n_max_core`={n_max_core}, pick i in [0,1,...,{n_max_core-1}]"
    p_pdf_root = Path(p_pdf_root)
    assert p_pdf_root.is_dir(), f"`p_pdf_root` is invalid: {p_pdf_root}"
    pdf_paths = list(p_pdf_root.rglob("*.pdf"))
    assert len(pdf_paths) > 0, "`p_pdf_root` exists but no PDF is in there."
    
    # subset indices
    if i!=-1:
        n_chunk = len(pdf_paths) // n_max_core
        i_start, i_end = i * n_chunk, (i + 1) * n_chunk
        
        # subset to (1/n_max_core)-th of all available PDFs
        pdf_abs_paths = pdf_paths[i_start:i_end]
    else:
        pdf_abs_paths = pdf_paths

    # DEBUG
    #pdf_abs_paths = pdf_abs_paths[:100] # XXX
    # = = = = = = 
    
    # parallel processing of PDFs
    with ProcessPoolExecutor(max_workers=8) as executor:
        # submit all PDF processing tasks to the pool
        results = list(executor.map(process_single_pdf, pdf_abs_paths))
    
    # Map the results to the corresponding paths
    pdf_text_dict = dict()
    for pdf_path, text in zip(pdf_abs_paths, results):
        pdf_text_dict[str(pdf_path)] = text  # Store path as key and text as value

    return pdf_text_dict