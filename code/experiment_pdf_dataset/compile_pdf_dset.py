from utils import in_notebook, recursive_lookup_pdf_paths, PDFPaths
from transfer_file_utils import copy_file, parallel_copy_files
import random
import argparse
import zipfile
import yaml
from pathlib import Path

def main():
    # get PDFs
    P = PDFPaths()

    # sample
    parser = argparse.ArgumentParser(description="Sample and copy PDF files.")
    parser.add_argument('-n', '--n_sample', type=int, default=5000, help='Maximum number of PDFs sampled per sub-category')
    parser.add_argument('-dst','--dst_root', type=str, required=True, help='Destination directory (root) for domain directories holding the copied PDFs')

    # parse args
    args = parser.parse_args()
    n_sample = args.n_sample
    dst_root = Path(args.dst_root)

    # create dst_root
    if not dst_root.exists():
        dst_root.mkdir(parents=True, exist_ok=True)

    # assemble PDFs to be transferred
    sampled_pdf_path_dict = {}
    for j,k in enumerate(P.data.keys()):
        random.seed(347*j)
        random.shuffle(P.data[k])
        # add
        if len(P.data[k]) > 0:
            sampled_pdf_path_dict[k] = P.data[k][:n_sample]
    
    # create directories
    for k in sampled_pdf_path_dict:
        # create
        directory_path = dst_root / k
        directory_path.mkdir(parents=True, exist_ok=True)

    # run in parallel
    parallel_copy_files(sampled_pdf_path_dict, dst_root)

if __name__=='__main__':
    main()