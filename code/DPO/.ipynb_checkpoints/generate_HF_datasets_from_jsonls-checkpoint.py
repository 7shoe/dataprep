import argparse
from data_utils import generate_HF_dataset
from pathlib import Path

def main(args):
    # read args
    for parser_name in args.parsers:
        generate_HF_dataset(parser=parser_name,
                            src_jsonl_dir=args.src_jsonl_dir,
                            dst_hf_dir=args.dst_hf_dir,
                            store_flag=args.store_flag)

if __name__ == '__main__':
    # REDO
    # parser
    parser = argparse.ArgumentParser(description='Generate Huggingface datasets from JSONL files (as they are written by `pypdfwf`')

    # Define arguments
    parser.add_argument('--parsers', type=str, nargs='+', required=False, default=['nougat', 'pymupdf', 'marker', 'pypdf', 'grobid'],
                        help='List of parser names to process (default: %(default)s)')
    parser.add_argument('--store_flag', type=bool, required=False, default=True,
                        help='Flag to store the generated dataset (default: %(default)s)')
    parser.add_argument('--src_jsonl_dir', type=Path, required=False, default=Path('/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/'),
                        help='Source directory for JSONL files (default: %(default)s)')
    parser.add_argument('--dst_hf_dir', type=Path, required=False, default=Path('/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/embeddings/emb_by_model/'),
                        help='Destination directory for Hugging Face datasets (default: %(default)s)')

    # Parse arguments
    args = parser.parse_args()

    # Run main function
    main(args)
