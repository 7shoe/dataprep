import pandas as pd
from bmc_urls import bmc_urls
from pathlib import Path
import time
import random
import requests
import socket

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException


class BMC_Spyder:
    def __init__(self, url_list):
        # Initialize the Chrome driver
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=options)
        self.urls = url_list
        self.url_substrings = []
        self.url_journal = []
        
    def scroll_down(self, pixels):
        ActionChains(self.driver).scroll_by_amount(0, pixels).perform()

    def extract_substring(self, url):
        # Open the target URL
        self.driver.get(url)

        for k in range(10):  # Iteratively scroll and scrape to ensure all articles are loaded
            # Find all relevant anchor tags for Full Text and PDF
            article_elements = self.driver.find_elements(By.CSS_SELECTOR, "ul.c-listing__view-options a")

            # Iterate over each article element and extract the href attribute
            for article in article_elements:
                try:
                    href = article.get_attribute('href')
                    # Extract the substring starting with "10." and include everything up to the next "/"
                    # This assumes the structure always follows "10.xxxxx/yyyyyy"
                    if "10." in href:
                        substring = href.split('/')[-2] + '/' + href.split('/')[-1]
                        if not(substring.endswith('.pdf')):
                            self.url_substrings.append(substring)
                            self.url_journal.append(url)
                
                except Exception as e:
                    print(f"Error scraping article: {e}")

            # Scroll down to load more articles
            self.scroll_down(pixels=2500)
            time.sleep(0.15)

    def scrape_all(self) -> None:
        for url_loc in self.urls:
            time.sleep(2.)  # Adding a delay between requests
            try:
                self.extract_substring(url=url_loc)
            except Exception as e:
                print('Skip')
                print(f'url: {url_loc}, SKIPPED due to error {e}')
        pass

    def close(self):
        # Close the driver when done
        self.driver.quit()

    def get_pdf_url(self, page_url):
        """
        BMC's diverse HTML URLs make inference on respective PDF paths impossible: Scrape them manually to complete the database in registry
        """
        try:
            # Set up WebDriver options
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')  # Run in headless mode (no browser UI)
            driver = webdriver.Chrome(options=options)
    
            # Navigate to the page
            driver.get(page_url)
            
            # Wait for the page to load completely
            time.sleep(3)  # You might need to adjust this depending on the page load time
            
            try:
                # Try to find the "Download PDF" button by its text or class (adjust if necessary)
                pdf_button = driver.find_element(By.LINK_TEXT, "Download PDF")
                pdf_url = pdf_button.get_attribute('href')
            except NoSuchElementException:
                print(f'-no such element in {page_url}')
                # Handle the case where the button is not found
                print("Could not find the 'Download PDF' button.")
                pdf_url = None
    
            # Close the WebDriver session
            driver.quit()

            # Debug
            print('Worked')
            return pdf_url
    
        except Exception as e:
            print(f"Loading the page took too long.: Actual error: {e}, in url {page_url}")
            driver.quit()
            return None
    
        except Exception as e:
            print(f"An error occurred: {e}")
            driver.quit()
            return None

    def complete_database_by_augmenting_pdf_urls(self, n:int=-1, crawl_delay:int=3.0):
        """Load article URLs, lookup resp. PDF urls and store them jointly in registry DB
        """
        # source:
        df = pd.read_csv('./registry/bmc_html_only_database.csv', sep='|', on_bad_lines='skip')
        # destination
        store_df_path = Path('./registry/bmc_database.csv')
        df_already_there = pd.read_csv(store_df_path, sep='|', on_bad_lines='skip')
        html_urls_already_in = set(df_already_there['html_url'])
        
        # shuffle 
        shuffled_df = df.sample(frac=1).reset_index(drop=True)

        if n!=-1 and n>0:
            shuffled_df = shuffled_df.iloc[:n]

        print('len(df)', len(df))
        
        # assemble PDF paths
        pdf_url_list = []
        html_url_list = []
        for i,url_loc in enumerate(df['html_url']):
            # check if url already scraped -> skip
            if url_loc in html_urls_already_in:
                continue
            # 
            time.sleep(crawl_delay)
            try:
                pdf_url = self.get_pdf_url(url_loc)
            except Exception as e:
                pdf_url = None
                print(e)
            pdf_url_list.append(pdf_url)
            html_url_list.append(url_loc)

            # store periodically
            if (i%5==0) or i==(len(df)-1):
                df_new = pd.DataFrame({'html_url' : html_url_list, 'pdf_url' : pdf_url_list})
                if store_df_path.is_file():
                    df_new.to_csv(store_df_path, mode='a', header=False, sep='|', index=None)
                else:
                    df_new.to_csv(store_df_path, sep='|', index=None)

        pass

class BMC_MVP:
    def __init__(self, 
                 i_start:int=0, 
                 i_delta:int=300, 
                 crawl_delay:int=5.0):
        '''
        Init
        '''
        # store path
        if 'lambda' in socket.gethostname():
            self.download_dir = Path('/homes/csiebenschuh/Projects/dataprep/data/bmc')
        else:
            self.download_dir = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/raw_data/bmc') # Polaris
        assert self.download_dir.is_dir(), f"Initializing `BMC_MVP` failed as {self.download_dir} does not exist"

        self.crawl_delay = crawl_delay
        df = pd.read_csv('./registry/bmc_database.csv', sep='|', on_bad_lines='skip')
        
        # subset
        #df_sub = df.iloc[i_start*i_delta:i_start*(i_delta+1)]
        self.df_sub = df.sample(frac=1).reset_index(drop=True)
        
        # shuffle order
        #self.df_sub = self.df_sub.sample(frac=1).reset_index(drop=True)

        print('len: ', len(self.df_sub))


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
        options.add_argument('--disable-gpu')  # Generally helpful in headless mode
        options.add_argument('--no-sandbox')  # Can help with performance on Linux
        options.add_argument('--disable-dev-shm-usage')  # Reduces RAM usage
        driver = webdriver.Chrome(options=options)

        # loop entries
        for _,row in self.df_sub.iterrows():
            print(f"Iteration: {_}")
            doi = '_'.join(row['html_url'].split('/')[-2:])
            file_stem = doi

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

   
class Other:
    def clean_urls(urls):
        """Cleans up the bmc_database_not_clean that has errorneously attached article identifier at the end preventing scraping HTML text & PDF diownload URLs
        """
        cleaned_urls = []
        for url in urls:
            # Check if the URL contains the pattern with page and 10.xxxx
            match = re.search(r'(PubDate&page=\d+)(/10\.\d+)', url)
            if match:
                # If match, keep the entire URL up to that match
                cleaned_url = url[:match.start(2)]
            else:
                # If no match, keep the URL as is
                cleaned_url = url
            cleaned_urls.append(cleaned_url)
        return cleaned_urls