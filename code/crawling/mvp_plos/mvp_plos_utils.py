import pandas as pd
from plos_urls import plos_urls
from pathlib import Path
import time
import random
import socket
import requests
import json
import pandas as pd
from datetime import datetime, timedelta

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options

class PLOS_MVP:
    def __init__(self, 
                 i_start:int=0, 
                 i_delta:int=2000, 
                 crawl_delay:int=8.0):
        '''
        Init
        '''
        # store path
        if 'lambda' in socket.gethostname():
            self.download_dir = Path('/homes/csiebenschuh/Projects/dataprep/data/plos')
        else:
            self.download_dir = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/raw_data/plos') # Polaris
        assert self.download_dir.is_dir(), f"Initializing `ArXiV_MVP` failed as {self.download_dir} does not exist"

        self.crawl_delay = crawl_delay
        df = pd.read_csv('./registry/plos_database.csv', sep='|')
        
        # subset
        df_sub = df.iloc[i_start:i_start+i_delta]
        self.df_sub = df_sub.sample(frac=1).reset_index(drop=True)
        
        # shuffle order
        #self.df_sub = self.df_sub.sample(frac=1).reset_index(drop=True)


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
            # REDO
            # = = = = = = 
            # Wait for re-route to redicreted URL
            # Derive PDF URL
            # Download BOTH
            doi = str(row['html_url']).split('/')[-1]
            file_stem = doi.replace('.', '_')

            # HTML
            html_url_init = row['html_url']
            driver.get(html_url_init)

            # wait for re-route/page built-up
            time.sleep(3.0)

            # content
            html_content = driver.page_source

            # HTML / PDF
            html_url = str(driver.current_url)

            # read out URL
            pdf_url = html_url.replace('/article?id', '/article/file?id') + '&type=printable'

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
                time.sleep(random.uniform(0.2, 1.5))
                
            else:
                print(f'nothing written, {pdf_response.status_code}')
        
        pass

class PLOS_Spyder:
    def __init__(self, url_list:list[str|Path]):
        # Initialize the Chrome driver
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=options)
        self.urls = url_list
        self.articles_urls = []

        #random.shuffle(self.urls)

    def scroll_down(self, pixels):
        ActionChains(self.driver).scroll_by_amount(0, pixels).perform()
    
    def scrape_one_url(self, url) -> None:
        # Open the target URL
        self.driver.get(url)

        main_article_url, sec_main_article_elements, related_article_elements = None, None, None

        # Get the main article URL
        found_something = False
        for k in range(20):
            try:
                # main
                main_article_elements = self.driver.find_elements(By.CSS_SELECTOR, "article a")
                main_article_urls = [element.get_attribute('href') for element in main_article_elements]
                
                # secondary 
                sec_main_article_elements = self.driver.find_elements(By.CSS_SELECTOR, "p.search-results-doi a")
                sec_main_article_elements = [element.get_attribute('href') for element in sec_main_article_elements]
        
                # Get the related article URLs (more specific selector)
                related_article_elements = self.driver.find_elements(By.CSS_SELECTOR, "ol.related li.related-article a")
                related_article_urls = [element.get_attribute('href') for element in related_article_elements]
    
                # specific
                specific_article_elements = self.driver.find_elements(By.CSS_SELECTOR, "p.article-info a")
                related_article_urls = [element.get_attribute('href') for element in specific_article_elements]
                
                # merge lists
                if main_article_url is not None:
                    found_something = True
                    self.articles_urls += main_article_urls
                    main_article_url = None
                else:
                    print('None')
                if sec_main_article_elements is not None:
                    found_something = True
                    self.articles_urls += sec_main_article_elements
                    sec_main_article_elements = None
                else:
                    print('None')
                if related_article_urls is not None:
                    found_something = True
                    self.articles_urls += related_article_urls
                    related_article_urls = None
                else:
                    print('None')
    
                if specific_article_urls is not None:
                    found_something = True
                    self.articles_urls += specific_article_urls
                    specific_article_urls = None
                else:
                    print('None')
            except:
                pass

            # scroll down
            self.scroll_down(pixels=500)
            time.sleep(0.25)
        
        # Close the driver
        self.driver.quit()

        return found_something

    def scrape_all(self,) -> None:
        for url_loc in self.urls:
            time.sleep(3.)
            try:
                out = self.scrape_one_url(url=url_loc)
                print(f'url: {url_loc}, found something: {out}')
            except Exception as e:
                print('Skip')
                print(f'url: {url_loc}, SKIPPED due to error {e}')
        pass