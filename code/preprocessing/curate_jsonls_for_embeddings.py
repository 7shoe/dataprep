import json
import math
import os
from pathlib import Path
import argparse

def main(src_dir, dst_dir, parsers, sizes, chunk_size, store_flag):
    # set Paths
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    
    # Create destination directory if store_flag is True
    if store_flag:
        os.makedirs(dst_dir, exist_ok=True)
    else:
        print('Nothing will be stored by this script! Confirm `dst_dir` is suitable and add `--store_flag`')
    
    # 1st loop: parsers + ['html']
    for parser in parsers:
        src_json_dir = src_dir / f'joint_to_{parser}/parsed_pdfs' 
        assert src_json_dir.is_dir(), f"Invalid path. Does not exist: {src_json_dir}"

        # lookup json files
        jsonl_files = [src_json_dir / f for f in os.listdir(src_json_dir) if f.endswith('.jsonl')]
        
        # 2nd loop: maximum number of characters
        for size in sizes:
            content_list = []
            jsonl_file_path = dst_dir / f'{parser}_{size}.jsonl'
            
            # Open jsonl
            for jsonl_file in jsonl_files:
                with open(jsonl_file, 'r') as doc:
                    for line in doc:
                        data = json.loads(line)
                        # Extract relevant information
                        pdf_path = data['path']
                        full_text = '' if data['text'] is None else data['text'][:size]
                        loc_dict = {'path': pdf_path, 'text': full_text}
                        
                        abstract_text = None
                        first_page_text = None
                        if 'metadata' in data.keys():
                            if 'abstract' in data['metadata'].keys():
                                abstract_text = data['metadata']['abstract']
                            if 'first_page' in data['metadata'].keys():
                                first_page_text = data['metadata']['first_page']
                        
                        if abstract_text is not None:
                            loc_dict['abstract'] = abstract_text
                        if first_page_text is not None:
                            loc_dict['firstpage'] = first_page_text
                        
                        content_list.append(loc_dict)
            
            # Split content_list into chunks and store in separate files if necessary
            if store_flag:
                if chunk_size != -1 and len(content_list) > chunk_size:
                    # Calculate the number of chunks needed
                    number_of_chunks = math.ceil(len(content_list) / chunk_size)
                    
                    for chunk_index in range(number_of_chunks):
                        # Determine the start and end indices of the current chunk
                        start_index = chunk_index * chunk_size
                        end_index = min(start_index + chunk_size, len(content_list))
                        
                        # Sublist for the current chunk
                        chunk = content_list[start_index:end_index]
                        
                        # Create the filename for this chunk
                        dst_filename = f"{Path(jsonl_file_path).stem}_{chunk_index + 1}-{number_of_chunks}{Path(jsonl_file_path).suffix}"
                        dst_filepath = Path(jsonl_file_path).parent / dst_filename
                        
                        # Write the current chunk to the jsonl file
                        print(f'Storing ... {dst_filepath}')
                        with open(dst_filepath, 'w') as jsonl_file:
                            for item in chunk:
                                jsonl_file.write(json.dumps(item) + '\n')
                else:
                    # If no need to split, just store the entire content_list in one jsonl
                    print(f'Storing ... {jsonl_file_path}')
                    with open(jsonl_file_path, 'w') as jsonl_file:
                        for item in content_list:
                            jsonl_file.write(json.dumps(item) + '\n')
                
                content_list = None

# entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and store JSONL files.")
    
    # Arguments
    parser.add_argument("--src_dir", type=str, default='/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/',
                        help="Source dir with JSONL files")
    parser.add_argument("--dst_dir", type=str, default='/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/embeddings/jsonls',
                        help="Destination directory to store the new JSONL files")
    parser.add_argument("--parsers", nargs='+', default=['html', 'marker', 'nougat', 'pymupdf', 'pypdf', 'grobid'],
                        help="List of parsers to process")
    parser.add_argument("--sizes", nargs='+', type=int, default=[1600, 3200, 12800],
                        help="List of sizes for the text truncation")
    parser.add_argument("--chunk_size", type=int, default=-1, help="Max elements stored within a single JSONL")
    parser.add_argument("--store_flag", action='store_true',
                        help="If set, store the JSONL files. If not set, just print filenames.")
    
    args = parser.parse_args()
    
    # Run main with the parsed arguments
    main(args.src_dir, args.dst_dir, args.parsers, args.sizes, args.chunk_size, args.store_flag)