import json
import os
from pathlib import Path
import argparse

def main(src_dir, dst_dir, parsers, sizes, store_flag):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    
    # Create destination directory if store_flag is True
    if store_flag:
        os.makedirs(dst_dir, exist_ok=True)
    
    # 1st loop: parsers + ['html']
    for parser in parsers:
        src_json_dir = src_dir / f'joint_to_{parser}/parsed_pdfs' 
        assert src_json_dir.is_dir(), f"Invalid path. Does not exist: {src_json_dir}"

        # lookup json files
        jsonl_files = [src_json_dir / f for f in os.listdir(src_json_dir) if f.endswith('.jsonl')]
        
        # 2nd loop: maximum number of characters
        for size in sizes:
            jsonl_file_path = dst_dir / f'{parser}_{size}.jsonl'
            
            # Print the filename to be created or that would be created
            if store_flag:
                print(f'Storing: {jsonl_file_path}')
            else:
                print(f'Would store: {jsonl_file_path}')
            
            content_list = []
            
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
            
            # Store as one jsonl if store_flag is True
            if store_flag:
                with open(jsonl_file_path, 'w') as jsonl_file:
                    for item in content_list:
                        jsonl_file.write(json.dumps(item) + '\n')
                content_list = None

# entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and store JSONL files.")
    
    # Arguments
    parser.add_argument("--src_dir", type=str, default='/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/',
                        help="Source directory containing the JSONL files")
    parser.add_argument("--dst_dir", type=str, default='/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/embeddings/jsonls',
                        help="Destination directory to store the new JSONL files")
    parser.add_argument("--parsers", nargs='+', default=['html', 'marker', 'nougat', 'pymupdf', 'pypdf', 'grobid'],
                        help="List of parsers to process")
    parser.add_argument("--sizes", nargs='+', type=int, default=[1600, 3200, 6400, 9600, 12800],
                        help="List of sizes for the text truncation")
    parser.add_argument("--store_flag", action='store_true',
                        help="If set, store the JSONL files. If not set, just print filenames.")
    
    args = parser.parse_args()
    
    # Run main with the parsed arguments
    main(args.src_dir, args.dst_dir, args.parsers, args.sizes, args.store_flag)