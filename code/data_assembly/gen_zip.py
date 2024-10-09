import multiprocessing
import argparse
import os
import time
from pathlib import Path
import zipfile
import json
import numpy as np

def process_batch(args):
    i, batch_file_paths, batch_size, p_dst = args
    zip_filename = (
        f"bs{str(batch_size).zfill(4)}"
        f"id{str(i).zfill(3)}.zip"
    )

    if i==0:
        t0 = time.time()
    
    output_zip_path = p_dst / zip_filename
    print(f'Writing {output_zip_path}...')
    zip_files(batch_file_paths, output_zip_path)
    print(f'Done writing {output_zip_path}')

    if i==0:
        print(f'dur: {(time.time()-t0):.2f}s')
        
    return (str(output_zip_path), [str(p) for p in batch_file_paths])

def zip_files(file_paths, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for file_path in file_paths:
            # Add file to ZIP archive with its basename
            zipf.write(file_path, arcname=Path(file_path).name)

def main():
    parser = argparse.ArgumentParser(description='Zip PDFs into batches.')
    parser.add_argument('--batch_size', '-bs', type=int, required=True, help='Batch size for zipping PDFs')
    args = parser.parse_args()

    batch_size = args.batch_size
    t0 = time.time()

    # List of allowed batch sizes
    batch_sizes = [16, 64, 256]
    assert batch_size in batch_sizes, f"batch_size must be one of {batch_sizes}"

    # Source & destination paths
    p_pdf = Path('/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/strong_scaling/data/pdf/')
    p_zip = Path('/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/strong_scaling/data/zip/')

    # For logging
    zip_dict = {}

    # Ensure destination directory exists
    p_zip.mkdir(parents=True, exist_ok=True)

    # Your list of PDF file paths
    pdf_file_paths = list(p_pdf.glob('*.pdf'))

    # Destination directory for this batch size
    p_dst = p_zip / f'b{batch_size}'
    p_dst.mkdir(parents=True, exist_ok=True)

    # Split list into list of lists where each list has length `batch_size`
    list_of_lists = [
        pdf_file_paths[i:i + batch_size]
        for i in range(0, len(pdf_file_paths), batch_size)
    ]

    # (no duplicates yet)
    # Prepare arguments for multiprocessing
    args_list = [(i, batch_file_paths, batch_size, p_dst) for i, batch_file_paths in enumerate(list_of_lists)]

    # Create a pool of worker processes
    with multiprocessing.Pool(12) as pool:
        results = pool.map(process_batch, args_list)

    # Update the zip_dict
    zip_dict = dict(results)

    # Store zip_dict as f'zip_dict_{batch_size}.json'
    zip_dict_path = p_zip / f'zip_dict_{batch_size}.json'
    with open(zip_dict_path, 'w') as f:
        json.dump(zip_dict, f, indent=4)
    print(f'Zip dictionary saved to {zip_dict_path}')

if __name__ == '__main__':
    main()
