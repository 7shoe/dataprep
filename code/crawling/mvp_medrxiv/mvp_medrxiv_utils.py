import requests
import json
import time
import os
import random
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import socket

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from concurrent.futures import ProcessPoolExecutor

class MedRXiV_Meta_Creator:
    def __init__(self, 
                 medrxiv_src_html_path:Path = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/merged_raw_data/medrxiv/html'),
                 meta_dst_csv_dir:Path = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/meta_tables/medrxiv_meta.csv')):

        # check paths
        meta_dst_csv_dir = Path(meta_dst_csv_dir)
        assert meta_dst_csv_dir.parent.is_dir(), f"`meta_dst_csv_dir` parent directory does not exist: Invalid {meta_dir.parent}"
        self.meta_dst_csv_dir = meta_dst_csv_dir

        medrxiv_src_html_path = Path(medrxiv_src_html_path)
        assert medrxiv_src_html_path.is_dir(), f"`medrxiv_src_html_path` parent directory does not exist: Invalid {meta_dst_csv_dir}"
        self.medrxiv_src_html_path = medrxiv_src_html_path
        
        pass

    def get_abstract(self, driver):
        """
        Extracts abstract from MedRXiv HTML website
        """
        # Extract the abstract
        try:
            abstract_element = driver.find_element(By.CSS_SELECTOR, "div.section.abstract")
            abstract_text = abstract_element.text.replace('|', '')
            return abstract_text
        except Exception as e:
            return ""
    
    def get_title(self, driver):
        """
        Extracts title from MedRXiv HTML website
        """
        # Extract the title
        try:
            title_element = driver.find_element(By.CSS_SELECTOR, "h1.highwire-cite-title")
            title_text = title_element.text.replace('|', '')
            return title_text
        except Exception as e:
            return ""
    
    def get_published_date(self, driver):
        """
        Extracts the published date from a MedRXiv HTML website
        """
        try:
            # Find the element that contains the published date
            date_element = driver.find_element(By.CSS_SELECTOR, "div.panel-pane.pane-custom.pane-1 .pane-content")
            
            # Get the text from the element
            date_text = date_element.text
            
            # Extract the date by removing "Posted" and any surrounding whitespace
            date_text = date_text.replace('Posted', '').strip().split('.')[0]
    
            # process 
            date_obj = datetime.strptime(date_text.strip(), "%B %d, %Y")
        
            # Convert the datetime object to the desired format
            formatted_date = date_obj.strftime("%d-%m-%Y")
            
            # Return the cleaned-up date text
            return formatted_date
        except Exception as e:
            return ""
    
    def get_doi(self, driver):
        """
        Extracts the DOI from a MedRXiv HTML website
        """
        try:
            # Find the element that contains the DOI
            doi_element = driver.find_element(By.CSS_SELECTOR, "span.highwire-cite-metadata-doi.highwire-cite-metadata")
            
            # Get the text from the element
            doi_text = doi_element.text.strip()
    
            # format
            doi_text = doi_text.split('doi: ')[1]
            
            # Return the DOI text
            return doi_text
        except Exception as e:
            return ""
    
    def store_medrxiv_meta(self, i:int=-1, n:int=-1):
        '''Compile metadata in a sequential fashion using a single WebDriver instance.
        i : i-th (out of 10 slices) from
        '''
        
        # Path to MedRXiv metadata
        html_file_paths = [self.medrxiv_src_html_path / f for f in os.listdir(self.medrxiv_src_html_path) if f.endswith('.html')]
        
        # Limit the number of files processed for debugging purposes
        if n > 0:
            html_file_paths = html_file_paths[:round(n)]

        # subset 10-th of list
        if i > 0:
            one_fifth_index = len(html_file_paths) // 3
            sub_list = html_file_paths[one_fifth_index * (i):one_fifth_index * (i+1)]
            html_file_paths = sub_list
        
        meta_list = []
        
        # Initialize the WebDriver
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        driver = webdriver.Chrome(options=options)
        
        try:
            # Process each HTML file sequentially
            for html_path in html_file_paths:
                # Load the HTML file
                driver.get(f"file://{html_path}")
                html_content = driver.page_source
                doc_len = len(html_content)
            
                # Extract the metadata
                title_scraped = self.get_title(driver)
                abstract_scraped = self.get_abstract(driver)
                date_scraped = self.get_published_date(driver)
                doi_scraped = self.get_doi(driver)
                
                # Paths
                p_html = html_path
                p_pdf = str(html_path).replace('/html/', '/pdf/').replace('.html', '.pdf')
                p_csv = str(html_path).replace('/html/', '/csv/').replace('.html', '.csv')
                
                # Create a dictionary of metadata
                tmp_dict = {
                    'p_html': p_html,
                    'p_pdf': p_pdf,
                    'p_csv': p_csv,
                    'title_scraped': title_scraped,
                    'abstract_scraped': abstract_scraped,
                    'date_scraped': date_scraped,
                    'doi_scraped': doi_scraped,
                    'len': doc_len
                }
                
                # Only append if both PDF and CSV files exist
                if Path(p_pdf).is_file() and Path(p_csv).is_file():
                    meta_list.append(tmp_dict)
        
        finally:
            # Ensure the driver is properly closed
            driver.quit()
        
        # Convert list to DataFrame
        df_meta = pd.DataFrame(meta_list)
    
        # Store the DataFrame to a CSV file
        if not(Path(self.meta_dst_csv_dir).is_file()):
            df_meta.to_csv(self.meta_dst_csv_dir, sep='|', index=False)
        else:
            df_meta.to_csv(self.meta_dst_csv_dir, sep='|', mode='a', header=False, index=False)
        
        return df_meta


class MedRXiV_MVP:
    def __init__(self, 
                 i_start:int=0, 
                 i_delta:int=8000, 
                 crawl_delay:int=7.0):
        '''
        Init
        '''
        # store path
        if 'lambda' in socket.gethostname():
            self.download_dir = Path('/homes/csiebenschuh/Projects/dataprep/data/medarxiv')
        else:
            self.download_dir = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/raw_data/medrxiv') # Polaris
        assert self.download_dir.is_dir(), f"Initializing `ArXiV_MVP` failed as {self.download_dir} does not exist"

        self.crawl_delay = crawl_delay
        df = pd.read_csv('./registry/medrxiv_database.csv', sep='|')
        
        # subset
        df_sub = df.iloc[i_start:i_start+i_delta]
        self.df_sub = df_sub.sample(frac=1).reset_index(drop=True)

    def get_arxiv_articles_with_html(self,):
        '''
        Attempt to download PDFs and HTML files
        '''

        # setup directories if needed
        download_dir = Path(self.download_dir)
        pdf_path = download_dir / 'pdf'
        html_path = download_dir / 'html'
        csv_path = download_dir / 'csv'

        assert download_dir.is_dir(), "`download_dir` invalid directory path"

         # driver options 
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        driver = webdriver.Chrome(options=options)

        # loop entries
        for _,row in self.df_sub.iterrows():
            doi = row['doi']
            file_stem = doi.replace('/', '_')

            # HTML / PDF
            html_url = 'https://www.medrxiv.org/content/' + row['doi'] + '.full'
            pdf_url = 'https://www.medrxiv.org/content/' + row['doi'] + '.full.pdf'

            # Check if HTML URL exists
            #print(html_url)

            # HTML
            driver.get(html_url)
            html_content = driver.page_source

            # wait
            time.sleep(self.crawl_delay)
            
            # PDF
            pdf_response = requests.get(pdf_url)
            if pdf_response.status_code == 200:
                # Save HTML content to file
                with open(str(html_path / (file_stem + '.html')), 'w', encoding='utf-8') as file:
                    file.write(html_content)

                # Save PDF content to file
                with open(f"{pdf_path}/{file_stem}.pdf", 'wb') as pdf_file:
                    pdf_file.write(pdf_response.content)

                # Meta 
                row.to_csv(f"{csv_path}/{file_stem}.csv", sep='|')

                # wait again
                time.sleep(self.crawl_delay)
                
            else:
                print(f'nothing written, {pdf_response.status_code}')

        if _ > 5:
            return None
        pass



class MedRXiV_Spider:
    
    glob_start = datetime(2015, 1, 1)
    glob_end =datetime(2024, 8, 1)
    df_path = Path('/home/siebenschuh/Projects/dataprep/code/mvp_medrxiv/registry/medrxiv_crawl.csv')
    medrxiv_stem_url = 'https://api.medrxiv.org/details/medrxiv/'
    
    def __init__(self, n:int=10_000, crawl_delay:float=7.0):
        self.n = n
        self.rounds = self.n // 100
        self.crawl_delay = crawl_delay

        # status
        print(f"{self.rounds} rounds à 100 hits: ~{(7 * self.rounds // 60)} min.")
        
    def random_date(self, 
                    start:datetime = glob_start, 
                    end:datetime = glob_end):
        """
        This function will return a random datetime between two datetime objects.
        """
        delta = end - start
        
        # offset
        offset_days = random.randrange(delta.days)
        offset_seconds = random.randrange(24*60*60)
        rnd_start = start + timedelta(days=offset_days, seconds=offset_seconds)
        
        # duration
        dur_seconds = random.randrange(14*24*3600)
        rnd_end = rnd_start + timedelta(days=0, seconds=dur_seconds)
        
        return rnd_start.strftime('%Y-%m-%d'), rnd_end.strftime('%Y-%m-%d')
    
    def crawl_medrxiv_metadata(self,):
        """Crawls MedRXiV via API
        """
        df_path = Path(self.df_path)
        assert df_path.parent.is_dir(), "Parent directory must exist for pd frame."
    
        # iterate
        for i in range(self.rounds):
            # Generate a random date
            req_url = self.medrxiv_stem_url + '/'.join(self.random_date(self.glob_start, self.glob_end)) + '/' + f'{random.randint(1, 1000)}'
            
            try:
                # load data
                out = requests.get(req_url)
                meta_dict_list = out.json()['collection']
                df = pd.DataFrame(meta_dict_list)
        
                # store
                if df_path.is_file():
                    df.to_csv(df_path, mode='a', sep='|', index=False, header=False)
                else:
                    df.to_csv(df_path, mode='w', sep='|', index=False)
        
            except Exception as e:
                print(f'Error in iteration {i} ... {e}')
                pass