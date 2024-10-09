from pathlib import Path
import pandas as pd
from typing import List
import os
import re
import sys
import shutil
import subprocess

def get_parsl_run_dir_paths_for_parser_timer() -> List[Path]:
    """
    Retrieve run dir paths for all parsers
    """

    # First reservation's output paths (CONSTANTS)
    p_1 = Path('/lus/eagle/projects/argonne_tpc/hippekp/adaparse-scaling')
    p_2 = Path('/lus/eagle/projects/argonne_tpc/hippekp/adaparse_prod_runs/adaparse')
    p_3 = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/scaling_experiment_1/run_dirs')

    all_run_dirs = [] 
    
    # p_1 (hippekp)
    for subDir in ['pypdf-prod', 'marker-prod', 'nougat-prod', 'pymupdf-prod']:
        curr_path = p_1 / subDir
        # append if exists
        if curr_path.is_dir():
            runs_across_nodes = [curr_path / run_dir for run_dir in os.listdir(curr_path) if run_dir!='submit']
            # append
            all_run_dirs += runs_across_nodes
    
    # p_2 (hippekp)
    for subDir in ['nodes_16', 'nodes_32', 'nodes_1', 'nodes_128', 'nodes_64', 'nodes_2', 'nodes_4', 'nodes_8']:
        curr_path = p_2 / subDir
        # append if exists
        if curr_path.is_dir():
            all_run_dirs += [curr_path]
    
    # p_3 (carlo)
    for subDir in ['adafast_nodes_128_prod', 'tesseract_nodes_64_prod', 'tesseract_nodes_16_prod', 'tesseract_nodes_2_prod', 'tesseract_nodes_4_prod', 'adafast_nodes_64_prod', 'adafast_nodes_128', 'adafast_nodes_32_prod', 
                   'tesseract_nodes_128_prod', 'tesseract_nodes_32_prod', 'adafast_nodes_16_prod', 'tesseract_nodes_1_prod', 'adafast_nodes_8_prod', 'tesseract_nodes_8_prod']:
        curr_path = p_3 / subDir
        # append if exists
        if curr_path.is_dir():
            all_run_dirs += [curr_path]
    
    # all run dirs
    print(f'Found {len(all_run_dirs)} paths')
    
    return all_run_dirs


# Function to extract parser and node information from a given path
def infer_parser_node(path_str):
    path = Path(path_str)

    # constants
    parsers = {'nougat', 'marker', 'tesseract', 'pymupdf', 'pypdf', 'grobid', 'adafast', 'adaparse'}
    nodes = {1, 2, 4, 8, 16, 32, 64, 128}
    
    # Convert nodes to string for easier regex matching
    nodes_str = '|'.join(map(str, sorted(nodes, reverse=True)))  # Ensure larger numbers match first
    
    # Check path name and its parent for a parser match
    parser_match = next((p for p in parsers if p in path.name or p in path.parent.name), None)
    
    # Use regex to extract node numbers
    node_match = re.search(rf'({nodes_str})', path.name)
    
    if parser_match and node_match:
        return f'{parser_match}_{node_match.group(1)}'
    else:
        return None

def infer_all_run_dir_filenames(all_run_dirs:list[Path]) -> list[str]:
    """
    Given the list of run dirs, returns appropriate file stems (that identify #nodes and parser)
    """

    

    # infer
    inferred_file_names = [infer_parser_node(run_dir) for run_dir in all_run_dirs]

    # print status
    for i,inf_file_name in enumerate(inferred_file_names):
        if inf_file_name is None:
            print(f'Could not infer: {all_run_dirs[i]}')
    
    return inferred_file_names