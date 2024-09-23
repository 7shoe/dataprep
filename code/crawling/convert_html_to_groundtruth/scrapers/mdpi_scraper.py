from scrapers.base_scraper import TextFromHTML
from datetime import datetime
from pathlib import Path
import pandas as pd
import time
from datetime import datetime

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

class TextFromMdpiHTML(TextFromHTML):
    def __init__(self, 
                 html_file_path:Path, 
                 max_wait_time:float = None):
        if max_wait_time is None:
            super().__init__(html_file_path)
        else:
            super().__init__(html_file_path, max_wait_time)

        # set csv path
        csv_file_path = Path(str(html_file_path).replace('/html/', '/csv/').replace('.html', '.csv'))
        self.csv_file_path = csv_file_path if csv_file_path.is_file() else None
        
        # load meta CSV file
        if self.csv_file_path is not None:
            df = pd.read_csv(self.csv_file_path, sep='|').transpose()

            # non-empty meta dataframe
            if len(df) > 1:
                self.csv_data_dict = {k:v for k,v in zip(df.iloc[0], df.iloc[1])}
        else:
            self.csv_data_dict = {}
        
        # scrape content & close
        self.scrape_and_quit()
        
    def get_title(self) -> None:
        """Retrieve the title of the document."""

        # meta data CSV
        if self.csv_data_dict is not None:
            self.title = self.csv_data_dict.get('title', None)

        # scrape frm HTML
        if self.title is None:
            try:
                # wait for it to appear
                title_element = WebDriverWait(self.driver, self.max_wait_time).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "h1.title.hypothesis_container"))
                )
                title = title_element.get_attribute("innerText").strip()
            except Exception as e:
                title = None
    
            self.title = title
        
        pass
        
    def get_authors(self, ) -> None:
        """Retrieve the authors of the document as a list of strings."""
        
        # meta data CSV
        if self.csv_data_dict is not None:
            authors = self.csv_data_dict.get('authors', None)
            if authors is not None:
                self.authors = self.post_process_authors([author.strip() for author in authors.split(';')])

        if self.authors is None:
            authors = []
            
            try:
                container = WebDriverWait(self.driver, self.max_wait_time).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.art-authors.hypothesis_container"))
                )
                
                # find all div elements with class 'profile-card-drop' within the container
                author_elements = container.find_elements(By.CSS_SELECTOR, "div.profile-card-drop")
                
                # loop through each author element and extract the text
                authors = [author.text.strip() for author in author_elements]
            
            except Exception as e:
                # Handle any exceptions (like timeout) by setting authors to an empty list
                print(f'Exception e: {e}')
                authors = []
        
            # Store the authors list in the object
            self.authors = self.post_process_authors(authors)

        pass

    def __date_scraped_from_medrxiv_page__(self, date_str:str) -> str:
        """
        Parse dates from text
        """
        if 'Posted' in date_str:
            try:
                # extract the date part from the text
                date_str = date_str.replace("Posted", "").replace(".", "").strip()
                date_obj = datetime.strptime(date_str, "%B %d, %Y")
                formatted_date = date_obj.strftime("%d-%m-%Y")
            except:
                formatted_date = None

        return formatted_date

        
    def __convert_to_dd_mm_yyyy__(self, date_str: str) -> str:
        """
        Format ArXiV-specific dates. If conversion not clear return None to eschew confusion
        """
        if date_str is None:
            return None
        try:
            # Parse the date string into a datetime object
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Convert the datetime object into the desired format
            return dt.strftime('%d-%m-%Y')
        except ValueError:
            return None
            
    def get_creationdate(self) -> None:
        """Retrieve the creation date of the document."""

        # meta data CSV
        if self.csv_data_dict is not None:
            self.creationdate = self.__convert_to_dd_mm_yyyy__(self.csv_data_dict.get('data', None))

        # scrape frm HTML
        if self.creationdate is None:
            try:
                # Wait for the pubhistory container to be present
                pubhistory_container = WebDriverWait(self.driver, self.max_wait_time).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.pubhistory"))
                )
                
                # Find the <span> tag that contains the "Published:" text and extract its text
                published_span = pubhistory_container.find_elements(By.TAG_NAME, "span")
                
                # Loop through each span to find the one containing "Published:"
                published_date = None
                for span in published_span:
                    if "Published:" in span.text:
                        date_str = span.text.replace("Published: ", "")
                        date_obj = datetime.strptime(date_str, '%d %B %Y')
                        formatted_date = date_obj.strftime('%d-%m-%Y')
                        self.creationdate = formatted_date
            
            except Exception:
                pass
        
        pass

    def get_keywords(self) -> None:
        """
        No keywords explicit to ArXiV
        """
        try:
            # Wait for the keywords container to be present
            keywords_container = WebDriverWait(self.driver, self.max_wait_time).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#html-keywords"))
            )
            
            # Find all <a> tags within the keywords container and extract their text
            keywords_elements = keywords_container.find_elements(By.TAG_NAME, "a")
            keywords = [keyword.text.strip() for keyword in keywords_elements]
        
        except Exception as e:
            print("Element not found or not accessible:", e)

        
        pass

    def get_doi(self) -> None:
        """
        MDPI DOI (from meta data csv or website)
        """

        # meta data CSV
        if self.csv_data_dict is not None:
            self.doi = self.csv_data_dict.get('doi', None)

        # from web
        try:
            doi_element = WebDriverWait(self.driver, self.max_wait_time).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.bib-identity"))
            )
            doi_text = doi_element.find_element(By.TAG_NAME, "a").get_attribute("href")
            self.doi = doi_text.split('doi.org/')[-1]
        except:
            pass
            
        pass

    def get_producer(self) -> None:
        # Implement ArXiV-specific logic to extract the producer information
        self.producer = None
        pass

    def get_document_text(self, ) -> None:
        """Retrieve and concatenate text from all relevant sections of the document."""
    
        # Initialize an empty list to store the section texts
        sections_text = []
    
        try:
            # Wait for the main content area to be present
            WebDriverWait(self.driver, self.max_wait_time).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.html-body section"))
            )
        
            # Find all div elements with class 'section' and id starting with 'sec-'
            sections = self.driver.find_elements(By.CSS_SELECTOR, 'div.html-body section')
        
            # Loop through each section and extract the text
            section_list = []
            for section in sections:
                section_content = section.find_element(By.CLASS_NAME, 'html-p').text.strip()
                # append
                section_list.append(section_content)

            # concatenate
            document_text = " ".join(section_list)
            self.document_text = document_text.replace('\n', ' ').replace('\t', ' ')
        
        except Exception as e:
            print(f'EEEE e : {e}')
            pass

        pass

    def get_abstract(self) -> None:
        """Retrieve the abstract of the document."""
        
        # csv meta data
        if self.csv_data_dict is not None:
            self.abstract = self.csv_data_dict.get('abstract', None)
        
        # scrape it from the HTML
        if self.abstract is None:
            try:
                WebDriverWait(self.driver, self.max_wait_time).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.html-p"))
                )
                
                abstract_element = self.driver.find_element(By.CSS_SELECTOR, 'div.html-p')
                self.abstract = abstract_element.text.strip()
                
            except Exception as e:
                print(f'eeee .. {e}')
                pass
        pass

    def get_valid_html_website(self) -> bool:
        """
        Check if the website has valid HTML content (i.e. if HTML version is indeed available) by detecting a specific error message.
        """
        self.valid_content = True
