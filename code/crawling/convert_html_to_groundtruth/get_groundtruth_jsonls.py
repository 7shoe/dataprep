from pathlib import Path
import os
import random
import json
import uuid
from tqdm import tqdm
import argparse

from scrapers.arxiv_scraper import TextFromArXiVHTML
from scrapers.biorxiv_scraper import TextFromBioRXiVHTML
from scrapers.medrxiv_scraper import TextFromMedRXiVHTML
from scrapers.mdpi_scraper import TextFromMdpiHTML
from scrapers.bmc_scraper import TextFromBmcHTML
from scrapers.nature_scraper import TextFromNatureHTML

def main(args):
    p_data = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/joint')
    assert p_data.is_dir(), f"`p_data` path is invalid: {p_data}"
    p_store_path = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/joint_to_html/parsed_pdfs')
    assert p_store_path.is_dir(), f"`p_store_path` path is invalid: {p_store_path}"

    # Dictionary of scraper classes
    scraper_classes = {
        'arxiv': TextFromArXiVHTML,
        'biorxiv': TextFromBioRXiVHTML,
        'medrxiv': TextFromMedRXiVHTML,
        'mdpi': TextFromMdpiHTML,
        'bmc': TextFromBmcHTML,
        'nature': TextFromNatureHTML,
    }

    # check validity of input
    assert args.journal in scraper_classes.keys(), f"`--journal` must be one of these: {scraper_classes.keys()} but is {args.journal} instead"

    # Allow argparse argument `-j, --journal` (str) that is either `arxiv`, `biorxiv`, etc. and only. Subset dict to that `scraper_classes`
    scraper_classes = {args.journal: scraper_classes[args.journal]}

    # Loop through journals
    for journal in scraper_classes.keys():
        html_source_dir = p_data / journal / 'html'
        file_list = [html_source_dir / f for f in os.listdir(html_source_dir) if f.endswith('.html')]
        
        # scrape
        out_dict_list = []
        for html_file_path in tqdm(file_list, desc=f'Processing {journal}'):
            try:
                journal_scraper = scraper_classes[journal](html_file_path)
                out_dict = journal_scraper.get_metadata_parser_style()
                out_dict_list.append(out_dict)
            except:
                print(f'Error in `html_file_path`={html_file_path}... Skip document.')
                pass
        
        # store the results in a JSONL file
        output_filename = str(uuid.uuid4()) + ".jsonl"
        output_file_path = p_store_path / output_filename

        # write out_dict_list to the JSONL file
        with open(output_file_path, 'w') as f:
            for out_dict in out_dict_list:
                f.write(json.dumps(out_dict) + '\n')

        # debug
        print(f'For journal {journal}.\nSaved {len(out_dict_list)} records to {output_file_path}')

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Process HTML files from specific journal.')
    parser.add_argument('-j', '--journal', type=str, required=True,
                        help='Specify the journal to process. Choices are: arxiv, biorxiv, medrxiv, mdpi, bmc, nature.')
    args = parser.parse_args()

    # run main
    main(args)