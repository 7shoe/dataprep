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

class TextFromMedRXiVHTML(TextFromHTML):
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
            #print('self.csv_file_path exists')
            df = pd.read_csv(self.csv_file_path, sep='|').transpose()

            # non-empty meta dataframe
            if len(df) > 1:
                #print('len(df) larger than 1 exists')
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
                    EC.presence_of_element_located((By.XPATH, '//*[(@id = "page-title")]'))
                )
                title = title_element.text
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
                self.authors = [author.strip() for author in authors.split(';')]

        if self.authors is None:
            authors = []
            
            try:
                # Wait for the author elements to be present
                WebDriverWait(self.driver, self.max_wait_time).until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, "ltx_personname"))
                )
                
                # Find all elements with the class 'ltx_personname'
                author_elements = self.driver.find_elements(By.CLASS_NAME, "ltx_personname")
                
                for element in author_elements:
                    # Get the text content of each author element
                    authors_text = element.text
                    
                    # Handle cases where multiple names might be separated by <br> tags
                    split_authors = authors_text.split("\n")  # Splitting by newline which is how <br> tags are often handled
                    
                    # Add each individual author name to the list
                    authors.extend([author.strip() for author in split_authors if author.strip()])
            
            except Exception as e:
                # Handle any exceptions (like timeout) by setting authors to an empty list
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
                # Wait for the specific element to be present
                date_element = WebDriverWait(self.driver, self.max_wait_time).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.panel-pane.pane-custom.pane-1 > div.pane-content"))
                )
        
                # find
                date_text = date_element.get_attribute("innerText").strip()
                
                # extract the text content from the element
                self.creationdate = self.__date_scraped_from_medrxiv_page__(date_text)
            except Exception as e:
                pprint(f'Exception : {e}')
        
        pass

    def get_keywords(self) -> None:
        """
        No keywords explicit to ArXiV
        """
        self.keywords = []
        
        pass

    def get_doi(self) -> None:
        """
        MedRXiV allows 3-way extraction (meta data csv, website or by inference from URL/lcoal HTML file path)
        """

        # meta data CSV
        if self.csv_data_dict is not None:
            self.doi = self.csv_data_dict.get('doi', None)

        # from web
        try:
            doi_element = WebDriverWait(self.driver, self.max_wait_time).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "span.highwire-cite-metadata-doi.highwire-cite-metadata"))
            )
            doi_text = doi_element.get_attribute("innerText").strip()
            self.doi = doi_text.split('doi.org/')[-1]
        except:
            # infer from filepath
            self.doi = '/'.join(str(self.html_file_path).split('/')[-1].split('_')).replace('.html', '')
            
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
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.highwire-markup"))
            )
        
            # Find all div elements with class 'section' and id starting with 'sec-'
            sections = self.driver.find_elements(By.CSS_SELECTOR, "div.section[id^='sec-']")
        
            # Loop through each section and extract the text
            for section in sections:
                section_id = section.get_attribute("id")
                sections_text.append(section.text)

            # concatenate
            document_text = " ".join(sections_text)
            self.document_text = document_text.replace('\n', ' ').replace('\t', ' ')
        
        except:
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
                # Wait for the ltx_abstract div to appear
                abstract_section = WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#abstract-1"))
                )
                
                # Find all child elements of the abstract section that are not <h2>
                child_elements = abstract_section.find_elements(By.XPATH, "./*[not(self::h2)]")
    
                # Extract and concatenate text from these child elements
                abstract_text = " ".join([element.text for element in child_elements])

                # delete escape characters
                abstract_text = self.clean_html_text_adaptive(abstract_text)
                
                self.abstract = abstract_text if abstract_text else None
                
            except Exception:
                pass
        pass


    def get_unsupported_packages(self) -> list[str]:
        """
        ArXiV-specific helper function that scrapes unsuppoted libraries that were used in the source .tex.
        Might indicate that the HTML text is not properly displayed.

        Important: Assumes that the ArXiV HTML was scraped when the `unsupported packages` 
        warning box popped up (did not happen for the first two batches). Unsupported packes by default empty.
        """
        try:
            # Look for the `ul` element with the warning about unsupported packages
            ul_element = WebDriverWait(self.driver, self.max_wait_time).until(
                lambda driver: driver.find_element(By.CSS_SELECTOR, "ul[aria-label='Unsupported packages used in this paper']"),
                message="Unsupported packages warning not found."
            )
            
            # Extract the text from each `li` item within the `ul`
            li_elements = ul_element.find_elements(By.TAG_NAME, "li")
            self.unsupported_packages = [li.text for li in li_elements]
        
        except:
            # If the element is not found, just return an empty list
            self.unsupported_packages = []
        
        return self.unsupported_packages

    def get_valid_html_website(self) -> bool:
        """
        Check if the website has valid HTML content (i.e. if HTML version is indeed available) by detecting a specific error message.
        """
        try:
            # Look for the paragraph element that indicates HTML is not available
            error_message_element = WebDriverWait(self.driver, self.max_wait_time).until(
                EC.presence_of_element_located((By.XPATH, "//p[text()='HTML is not available for the source.']"))
            )
            # If the element is found, return False indicating invalid HTML
            self.valid_content = False
        
        except:
            # If the element is not found, the HTML is valid, return True
            self.valid_content = True
