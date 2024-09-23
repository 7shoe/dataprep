import pandas as pd
from nature_urls import nature_urls
from pathlib import Path
import time
import random
import requests
import socket

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException

class Nature_Spyder2:
    def __init__(self, url_list):
        # Initialize the Chrome driver
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=options)
        self.urls = url_list
        self.html_urls= []
        
    def scroll_down(self, pixels):
        ActionChains(self.driver).scroll_by_amount(0, pixels).perform()

    def extract_substring(self, url):
        # Open the target URL
        self.driver.get(url)

        for k in range(10):  # Iteratively scroll and scrape to ensure all articles are loaded
            # Find all relevant anchor tags for Full Text and PDF
            article_elements = self.driver.find_elements(By.CSS_SELECTOR, "a.c-card__link.u-link-inherit")

            # Iterate over each article element and extract the href attribute
            for article in article_elements:
                try:
                    # Extract the URL from the href attribute
                    article_url = article.get_attribute('href')
                    if article_url not in self.html_urls:
                        self.html_urls.append(article_url)
                
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
        Nature's diverse HTML URLs make inference on respective PDF paths impossible: Scrape them manually to complete the database in registry
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

class Nature_Spyder:
    def __init__(self, url_list:list[str|Path]):
        # Initialize the Chrome driver
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(options=options)
        self.urls = url_list
        self.articles_urls = []
        
        self.title_list = []
        self.article_url_list = []
        self.description_list = []
        self.authors_list = []
        self.publication_date_list = []
        self.access_type_list = []

        #random.shuffle(self.urls)

    def scroll_down(self, pixels):
        ActionChains(self.driver).scroll_by_amount(0, pixels).perform()
    
    def scrape_articles_info(self, url):
        # Open the target URL
        self.driver.get(url)

        # Find all article elements on the page
        article_elements = self.driver.find_elements(By.CSS_SELECTOR, "article.u-full-height")

        title_list = []
        article_url_list = []
        description_list = []
        authors_list = []
        publication_date_list = []
        access_type_list = []
        
        # Iterate over each article element and extract information
        for article in article_elements:
            try:
                # Extract the title
                title_element = article.find_element(By.CSS_SELECTOR, "h3.c-card__title a")
                title = title_element.text
                title_list.append(title)
                
                # Extract the article URL
                article_url = title_element.get_attribute('href')
                article_url_list.append(article_url)
                
                # Extract the description/summary
                description_element = article.find_element(By.CSS_SELECTOR, "div.c-card__summary p")
                description = description_element.text
                description_list.append(description)

                # Extract the authors
                author_elements = article.find_elements(By.CSS_SELECTOR, "ul.c-author-list li span[itemprop='name']")
                authors = [author.text for author in author_elements]
                authors_list.append(authors)

                # Extract the publication date
                date_element = article.find_element(By.CSS_SELECTOR, "time.c-meta__item")
                publication_date = date_element.get_attribute('datetime')
                publication_date_list.append(publication_date)

                # Extract the access type
                access_element = article.find_element(By.CSS_SELECTOR, "span.u-color-open-access")
                access_type = access_element.text
                access_type_list.append(access_type)

                # Print the scraped information
                print(f"Title: {title}")
                #print(f"URL: {article_url}")
                #print(f"Description: {description}")
                #print(f"Authors: {', '.join(authors)}")
                #print(f"Publication Date: {publication_date}")
                #print(f"Access Type: {access_type}")
                #print("=" * 40)
            except Exception as e:
                print(f"Error scraping article: {e}")

        # Close the driver
        self.driver.quit()

        return title_list, article_url_list, description_list, authors_list, publication_date_list, access_type_list

    def scrape_all(self,) -> None:
        for url_loc in self.urls:
            time.sleep(12.)
            try:
                title_list, article_url_list, description_list, authors_list, publication_date_list, access_type_list = self.scrape_articles_info(url=url_loc)
                self.title_list += title_list
                self.article_url_list += article_url_list
                self.description_list += description_list
                self.authors_list += authors_list
                self.publication_date_list += publication_date_list
                self.access_type_list += access_type_list

            except Exception as e:
                print('Skip')
                print(f'url: {url_loc}, SKIPPED due to error {e}')
        pass

class Nature_MVP:
    def __init__(self, 
                 i_start:int=0, 
                 i_delta:int=100, 
                 crawl_delay:int=8.0):
        '''
        Init
        '''
        # store path
        if 'lambda' in socket.gethostname():
            self.download_dir = Path('/homes/csiebenschuh/Projects/dataprep/data/nature')
        else:
            self.download_dir = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/raw_data/nature') # Polaris
        assert self.download_dir.is_dir(), f"Initializing `ArXiV_MVP` failed as {self.download_dir} does not exist"

        self.crawl_delay = crawl_delay
        df = pd.read_csv('./registry/nature_html_only_database.csv', sep='|', on_bad_lines='skip')
        
        # subset
        df_sub = df.iloc[i_start*i_delta:i_start*(i_delta+1)]
        self.df_sub = df_sub.sample(frac=1).reset_index(drop=True)
        
        # shuffle order
        self.df_sub = self.df_sub.sample(frac=1).reset_index(drop=True)

        # debug
        print(f"Init complete... From index {i_start} to {i_start+i_delta}, total of {len(self.df_sub)} entries.")


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
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(options=options)

        # loop entries
        for _,row in self.df_sub.iterrows():
            print(f"Iteration: {_}")
            doi = '_'.join(row['html_url'].split('/')[-1])
            file_stem = doi

            # HTML / PDF
            html_url = row['html_url']

            # HTML
            time.sleep(random.uniform(0.95, 1.2))
            driver.get(html_url)
            html_content = driver.page_source

            # get pdf url
            pdf_url = self.get_pdf_url(driver, html_url)
            if pdf_url is None:
                print(f'Skip URL: {html_url}')
                continue

            # wait
            time.sleep(self.crawl_delay)
            
            # PDF
            print('Before PDF request')
            pdf_response = requests.get(pdf_url)
            print('After PDF request')
            if pdf_response.status_code == 200:
                print('Worked!')
                # Save HTML content to file
                with open(str(html_path / (file_stem + '.html')), 'w', encoding='utf-8') as file:
                    file.write(html_content)

                # Save PDF content to file
                with open(f"{pdf_path}/{file_stem}.pdf", 'wb') as pdf_file:
                    pdf_file.write(pdf_response.content)

                # Meta 
                row.to_csv(f"{csv_path}/{file_stem}.csv", sep='|')
                
            else:
                print(f'nothing written, {pdf_response.status_code}')
        
        pass

    def get_pdf_url(self, driver, page_url):
        """
        BMC's diverse HTML URLs make inference on respective PDF paths impossible: Scrape them manually to complete the database in registry
        """
        try:
            # Wait for the page to load completely
            time.sleep(random.uniform(0.3, 0.6))  # You might need to adjust this depending on the page load time
            
            # Try to find the "Download PDF" button by its text or class (adjust if necessary)
            pdf_button = driver.find_element(By.LINK_TEXT, "Download PDF")
            pdf_url = pdf_button.get_attribute('href')
            
        except Exception as e:
            print(f"An error occurred: {e}")
            pdf_url = None
            
        return pdf_url

    def complete_database_by_augmenting_pdf_urls(self, ):
        """Load article URLs, lookup resp. PDF urls and store them jointly in registry DB
        """
        df = pd.read_csv('./registry/nature_html_only_database.csv', sep='|', on_bad_lines='skip')

        # shuffle 
        shuffled_df = df.sample(frac=1).reset_index(drop=True)

        print('len(df)', len(df))
        # assemble PDF paths
        pdf_url_list = []
        html_url_list = []
        for i,url_loc in enumerate(df['html_url']):
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
                if Path('./registry/nature_database.csv').is_file():
                    df_new.to_csv('./registry/nature_database.csv', mode='a', header=False, sep='|', index=None)
                else:
                    df_new.to_csv('./registry/nature_database.csv', sep='|', index=None)

        pass
