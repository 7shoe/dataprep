import argparse
from pathlib import Path
from utils import ConvertParserOutput

def main(file_name):
    # instance
    preproc = ConvertParserOutput(jsonl_root=Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt'),
                                  store_path=Path('/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/database'),
                                  parser_name_list=['html', 'pypdf'])

    # create and store
    preproc.create_text_database(file_name=file_name, 
                                 overwrite=True)

if __name__ == '__main__':
    # Argument
    parser = argparse.ArgumentParser(description="Process parser output and create a text database.")
    parser.add_argument('--file_name', type=str, required=True, help='Name of the CSV file into which the output is stored.')
    args = parser.parse_args()

    # validate arg
    assert isinstance(args.file_name, str) and args.file_name.endswith('.csv'), "Input `file_name` must be a `.csv`"
    
    # run
    main(file_name=args.file_name)