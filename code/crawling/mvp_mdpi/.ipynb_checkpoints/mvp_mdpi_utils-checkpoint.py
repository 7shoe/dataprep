from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

from mdpi_urls import mdpi_urls 

import socket
import random
import time
import pandas as pd
from pathlib import Path
import requests

class MDPI_MVP:
    def __init__(self, 
                 i_start:int=0, 
                 i_delta:int=8000, 
                 crawl_delay:int=8.0):
        '''
        Init
        '''
        # store path
        if 'lambda' in socket.gethostname():
            self.download_dir = Path('/homes/csiebenschuh/Projects/dataprep/data/mdpi')
        else:
            self.download_dir = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/raw_data/mdpi') # Polaris
        assert self.download_dir.is_dir(), f"Initializing `ArXiV_MVP` failed as {self.download_dir} does not exist"

        self.crawl_delay = crawl_delay
        df = pd.read_csv('./registry/mdpi_database.csv', sep='|')
        
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
            doi = str(row['html_url']).split('www.mdpi.com/')[-1].replace('/', '.')  # df['html_url'][0].split('www.mdpi.com/')[-1].replace('/', '.')
            file_stem = doi.replace('/', '_')

            # HTML / PDF
            html_url = row['html_url']
            pdf_url = row['pdf_url']

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
                time.sleep(random.uniform(0.2, 1.5))
                
            else:
                print(f'nothing written, {pdf_response.status_code}')
        
        pass


class MDPI_Spyder:
    def __init__(self,
                 mdpi_urls:list[str] = mdpi_urls,
                 i_start:int=0, 
                 i_delta:int=8000, 
                 crawl_delay:int=10.0,
                 df_path = Path('./registry/mdpi_database.csv')):
        
        # store path
        if 'lambda' in socket.gethostname():
            self.download_dir = Path('/homes/csiebenschuh/Projects/dataprep/data/mdpi')
        else:
            self.download_dir = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/raw_data/mdpi') # Polaris
        assert self.download_dir.is_dir(), f"Initializing `ArXiV_MVP` failed as {self.download_dir} does not exist"
        
        self.mdpi_urls = mdpi_urls
        random.shuffle(self.mdpi_urls)
        assert len(self.mdpi_urls) > 0, "`mdpi_urls` has 0 length"
        
        self.crawl_delay = crawl_delay
        self.df_path = df_path
        assert self.df_path.parent.is_dir(), "Check if parent directory exists"

    def scroll_to_end(driver, pause_time=random.uniform(0.3,1.0)):
        '''Scroll to the end of the page to extract all list results
        '''
        last_height = driver.execute_script("return document.body.scrollHeight")
    
        while True:
            # Scroll down to the bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
            # Wait for new content to load
            time.sleep(pause_time)
    
            # Calculate new scroll height and compare with last scroll height
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        pass

    def scroll_down(self, pixels):
        ActionChains(self.driver).scroll_by_amount(0, pixels).perform()
        
    def create_registry(self,):
        # check inputs
        
        # init the Chrome driver
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=options)
        
        # iterate
        for _, url in enumerate(self.mdpi_urls):
            wait_time = self.crawl_delay + random.uniform(0.0, 1.0)
            time.sleep(wait_time)

            try:

                # scroll through entire list
                self.driver.get(url)
                
                # Wait for the articles to be present
                wait = WebDriverWait(self.driver, 1)
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.generic-item.article-item")))
    
                # scroll to end
                articles = []
                for k in range(10):
                    self.scroll_down(pixels=1000)
                    time.sleep(0.25)
                
                    # Find all articles
                    articles += self.driver.find_elements(By.CSS_SELECTOR, "div.generic-item.article-item")
                
                # Extract the desired information
                data = []
                for article in articles:
                    try:
                        link_element = article.find_element(By.CSS_SELECTOR, "a.title-link")
                        title = link_element.text
                        link = link_element.get_attribute('href')
                        
                        authors = article.find_element(By.CSS_SELECTOR, "div.authors").text
                        journal = article.find_element(By.CSS_SELECTOR, "div.color-grey-dark em").text
            
                        year = article.find_element(By.CSS_SELECTOR, "div.color-grey-dark b").text
            
                        #date = article.find_element(By.CSS_SELECTOR, "div.color-grey-dark a").text
                        
                        data.append({
                            'title': title,
                            'html_url': link,
                            'pdf_url': link + f'/pdf?version={round(1723074000 + random.randint(-10000, 10000))}',
                            'authors': authors,
                            'journal': journal,
                            'year' : year,
                        })

                        df = pd.DataFrame(data)
                        
                        # unique
                        df = df.drop_duplicates(subset='title')
                        
                        # store
                        if self.df_path.is_file():
                            df.to_csv(self.df_path, mode='a', sep='|', index=False, header=False)
                        else:
                            df.to_csv(self.df_path, mode='w', sep='|', index=False)
                    except:
                        pass
            except:
                pass
        pass