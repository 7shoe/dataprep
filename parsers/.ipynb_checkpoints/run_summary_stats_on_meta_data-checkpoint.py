from pypdf import PdfReader
import re
import random
from pathlib import Path
import os
import json

def get_pdf_paths(n:int=500):
    ROOT_DIR = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/joint/')
    pdf_all_list = []
    
    # loop
    for sub_dir in ['arxiv', 'biorxiv', 'medrxiv', 'mdpi', 'nature', 'bmc']:
        pdf_path = ROOT_DIR / sub_dir / 'pdf'
        pdf_list = [pdf_path / f for f in os.listdir(pdf_path)]
        random.shuffle(pdf_list)
        pdf_all_list += pdf_list[:n]
    
    return pdf_all_list

def extract_doi_info(input_str: str) -> str:
    # Regular expression to match 'doi:' followed by any non-whitespace characters
    match = re.search(r'doi:\s*(\S+)', input_str)
    
    if match:
        return match.group(1)  # Return only the part after 'doi:'
    else:
        return ''

def test_pypdf(pdf_all_list):
    d = dict()
    for sub_dir in ['arxiv', 'biorxiv', 'medrxiv', 'mdpi', 'nature', 'bmc']:
        d[sub_dir] = {'title' : 0, 'authors': 0, 'createdate' : 0, 'keywords' : 0, 'doi' : 0, 'abstract' : 0}

    i_failure = 0
    for pdf_path in pdf_all_list:
        # read-in PDF
        try:
            reader = PdfReader(pdf_path)

            # all_tetx
            full_text = ''
            for page in reader.pages:
                full_text += page.extract_text(extraction_mode="layout")
    
            # first_page
            first_page = reader.pages[0] if len(reader.pages[0]) > 0 else ''
            meta = reader.metadata
            abstract_threshold = 200
            
            # meta data
            title = meta.get('/Title', '')
            authors = meta.get('/Author', '')
            createdate = meta.get('/CreationDate', '')
            keywords = meta.get('/Keywords', '')
            doi = meta.get('/doi', '') if meta.get('/doi', '')!='' else extract_doi_info(meta.get('/Subject', ''))  # Use .get() to handle the missing DOI key
            producer = meta.get('/Producer', '')
            format = meta.get('/Format', '')  # Not included for pypdf, so we set it directly
            abstract = meta.get('/Subject', '') if len(meta.get('/Subject', '')) > abstract_threshold else ''
    
            # update dict
            d[pdf_path.parent.parent.name]['title'] += (1 if title!='' else 0)
            d[pdf_path.parent.parent.name]['authors'] += (1 if authors!='' else 0)
            d[pdf_path.parent.parent.name]['createdate'] += (1 if createdate!='' else 0)
            d[pdf_path.parent.parent.name]['keywords'] += (1 if keywords!='' else 0)
            d[pdf_path.parent.parent.name]['doi'] += (1 if doi!='' else 0)
            d[pdf_path.parent.parent.name]['abstract'] += (1 if abstract!='' else 0)
            # elements
            d[pdf_path.parent.parent.name]['page_counts'] += [len(reader.pages)]
            d[pdf_path.parent.parent.name]['title_list'] += [title]
            d[pdf_path.parent.parent.name]['keywords_list'] += [keywords]
            d[pdf_path.parent.parent.name]['abstract_list'] +=[abstract]
            
        except:
            print(f'Skip: {pdf_path}')
            i_failure+=1
            continue

    # store number of PDFs
    d['n'] = len(pdf_all_list) - i_failure

    return d

def main():
    rnd_path_list = get_pdf_paths(200)
    d = test_pypdf(rnd_path_list)  

    # store
    filename = "summary_extract_1200.json"
    
    # Open the file in write mode and save the dictionary
    with open(filename, 'w') as file:
        json.dump(d, file, indent=4)
    
    print(f"Dictionary successfully saved to {filename}")

if __name__=='__main__':
    main()