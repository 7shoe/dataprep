import arxiv

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options

import socket
import getpass
import os
import random
import re
import yaml
import time
import pandas as pd
from pathlib import Path

class ArXiV_MVP:
    machine_id_dict = {
            'lambda0'  : 0,
            'lambda1'  : 1,
            'lambda2'  : 2,
            'lambda4'  : 3,
            'lambda11' : 4, # lambda 11,
            'lambda12' : 5, # lambda 11,
    }
    def __init__(self, machine_name:str, modulo_term=None):
        # set modulo term by hand
        if modulo_term is None:
            self.set_machine_id()
        else:
            self.machine_specific_modulo_term = modulo_term
       
        # get search words
        if self.machine_specific_modulo_term==0:
            self.load_keywords_from_categories()
        else:
            self.load_keywords_from_categories(modulo=self.machine_specific_modulo_term)
        # shuffle
        random.shuffle(self.search_words)

        # set download
        if machine_name=='mac':
            self.download_dir = Path('/Users/carlo/Projects/dataprep/data/arxiv')
        elif 'lambda' in socket.gethostname():
            self.download_dir = Path('/homes/csiebenschuh/Projects/dataprep/data/arxiv')
        else:
            self.download_dir = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/raw_data/arxiv') # Polaris
        assert self.download_dir.is_dir(), f"Initializing `ArXiV_MVP` failed as {self.download_dir} does not exist"

    def __repr__(self,):
        return f"Modulo {self.machine_specific_modulo_term} w/ {len(self.search_words)} search words: {self.search_words[:5]} ..."
        

    def set_machine_id(self, ) -> None:
        host_name = socket.gethostname()
        host_name_id = re.search(r'\d+', host_name).group()
        machine_specific_modulo_term = self.machine_id_dict[host_name_id] if (host_name_id in self.machine_id_dict) else 0
        self.machine_specific_modulo_term = machine_specific_modulo_term

        pass

    def load_keywords_from_categories(self, categories=None, file_path='./config/search_words.yaml', modulo=None) -> None:
        '''
        Load keywords from config YAML according to categories
        modulo:int : Only every `modulo'th term is included from the search word
        '''
        file_path = Path(file_path)
        assert file_path.is_file(), "File doesn`t exist contaiing keywords."

        # load
        with open(file_path, 'r') as file:
            keywords_dict = yaml.safe_load(file)
    
        # categories
        if categories is None or len(categories)==0:
            categories = [
                "Physics",
                "Computer Science",
                "Mathematics",
                "Quantitative Biology",
                "Quantitative Finance",
                "Statistics",
                "Interdisciplinary"
            ]
    
        # filter keywords
        selected_keywords = []
        for category in categories:
            if category in keywords_dict:
                selected_keywords.extend(keywords_dict[category])

        # Filter according to machine (allows stratified scraping across machiens)
        if modulo is not None:
            selected_keywords = [key_w for j,key_w in enumerate(selected_keywords) if (j % len(self.machine_id_dict))==modulo]

        random.shuffle(selected_keywords)
        
        self.search_words = selected_keywords

        pass
        
    def get_arxiv_articles_with_html(self,
                                     query, 
                                     max_results:int=200, 
                                     crawl_delay:int=6, 
                                     download:bool=False, 
                                     sep_symbols:str='|'):
        """
        Download ArXiV PDFs and HTML files
        """
    
        # check directory
        if download:
            assert len(sep_symbols), "TypeError: `delimiter` must be a 1-character string"
            
            # relevant dirs
            download_dir = Path(self.download_dir)
            pdf_path = download_dir / 'pdf'
            html_path = download_dir / 'html'
            csv_path = download_dir / 'csv'
    
            assert download_dir.is_dir(), "`download_dir` invalid directory path"
            
            # create sub-dirs
            pdf_path.mkdir(parents=True, exist_ok=True)
            html_path.mkdir(parents=True, exist_ok=True)
            csv_path.mkdir(parents=True, exist_ok=True)
            
            assert pdf_path.is_dir(), "`pdf_path` invalid directory path"
            assert csv_path.is_dir(), "`csv_path` invalid directory path"
    
            # driver options 
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            driver = webdriver.Chrome(options=options)
        
        # conduct search 
        client = arxiv.Client(delay_seconds=5.0)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        articles_with_html = []
    
        # Iterate
        for result in client.results(search):
            # sleep
            time.sleep(crawl_delay)
            
            if any(link.title == "pdf" for link in result.links):
                # check if HTML-compatible
                html_url = result.entry_id.replace('/abs/', '/html/')
    
                # meta information -> CSV
                df = pd.DataFrame({
                    'id': result.entry_id,
                    'title': result.title,
                    'pdf_url': result.entry_id.replace('abs', 'pdf') + '.pdf',
                    'html_url': result.entry_id.replace('/abs/', '/html/'),
                    'summary' : result.summary,
                    'comment' : result.comment,
                    'date_published' : result.published,
                    'date_updated' : result.updated,
                    'doi' : result.doi,
                    'prim_cat' : result.primary_category,
                    'categories' : result.journal_ref,
                    'journal_ref' : result.journal_ref,
                }, index=[0])
    
                # store data locally
                if download:
                    # file_stem
                    file_stem = result.entry_id.split('/')[-1]
                    # PDF
                    time.sleep(3.0)
                    result.download_pdf(dirpath=pdf_path, filename=(file_stem + '.pdf'))
                    
                    # CSV
                    df.to_csv(csv_path / (file_stem + '.csv'), index=False, sep=sep_symbols)
    
                    # HTML
                    try:
                        # website content
                        html_url = result.entry_id.replace('/abs/', '/html/')
                        driver.get(html_url)
        
                        # retrieve HTML
                        html_content = driver.page_source
        
                        with open(str(html_path / (file_stem + '.html')), 'w', encoding='utf-8') as file:
                            file.write(html_content)
                        
                    except Exception as e:
                        print(f"Error storing {str(html_path / '.html')}, error {e}")
    
        if download:
            driver.quit()

        # shuffle order
        random.shuffle(articles_with_html)
        
        return articles_with_html
