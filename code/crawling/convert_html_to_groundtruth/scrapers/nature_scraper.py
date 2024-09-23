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

class TextFromNatureHTML(TextFromHTML):
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

        # Nature-specific sections not considered `document text`
        self.skip_sections = {"References", "Availability of data and materials", "Abbreviations", "Acknowledgements",
                              "Funding", "Author information", "About this article", "Rights and permissions",
                              "Ethics declarations", "Additional information", "Supplementary Information"}
        
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
    
        # scrape from HTML
        if self.title is None:
            try:
                # wait for the title element to appear
                title_element = WebDriverWait(self.driver, self.max_wait_time).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "h1.c-article-title"))
                )
                title = title_element.get_attribute("innerText").strip()
            except Exception as e:
                title = None
        
            self.title = title
    
        pass
        
    def get_authors(self, ) -> None:
        """Retrieve the authors of the document as a list of strings."""
        

        if self.authors is None:
            authors = []
            
            try:
                # wait for the authors list to appear
                authors_list_element = WebDriverWait(self.driver, self.max_wait_time).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "ul.c-article-author-list"))
                )
                
                # find all author name elements within the list
                author_elements = authors_list_element.find_elements(By.CSS_SELECTOR, "a[data-test='author-name']")
                
                # extract the text for each author
                for author_element in author_elements:
                    author_name = author_element.get_attribute("innerText").strip()
                    authors.append(author_name)
                
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

        # scrape frm HTML
        if self.creationdate is None:
            try:
                # Wait for the time element that contains the publication date to appear
                date_element = WebDriverWait(self.driver, self.max_wait_time).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "li.c-article-identifiers__item time"))
                )
                
                # Extract the text for the date
                publication_date = date_element.get_attribute("innerText").strip()
                
                date_str = publication_date.replace("Published: ", "")
                date_obj = datetime.strptime(date_str, '%d %B %Y')
                formatted_date = date_obj.strftime('%d-%m-%Y')
                self.creationdate = formatted_date
            
            except Exception as e:
                print(f"Error in get_creationdate {e}")
                pass
        
        pass

    def get_keywords(self) -> None:
        """
        No keywords explicit to BioMedicalCentral (BMC)
        """
    
        pass

    def get_doi(self) -> None:
        """
        BMC DOI (website-only)
        """

        # from web
        try:
            # wait for the span element that contains the DOI to appear
            doi_parent_element = WebDriverWait(self.driver, self.max_wait_time).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "li.c-bibliographic-information__list-item--full-width"))
            )
            
            # locate the span within the parent element
            doi_element = doi_parent_element.find_element(By.CSS_SELECTOR, "span.c-bibliographic-information__value")
        
            
            # Extract the text for the DOI
            doi_text = doi_element.get_attribute("innerText").strip()
            self.doi = doi_text.split('doi.org/')[-1]
            
        except:
            pass
            
        pass

    def get_producer(self) -> None:
        """
        No producer on BioMedicalCentral (BMC)
        """
        pass

    def get_document_text(self) -> None:
        """
        Retrieve and concatenate the content after 'Conclusion' from dynamically generated sections like #Sec{X}-content.
        """
    
        # Initialize section_text variable
        self.section_text = ""
    
        try:
            # Wait for the main article content to appear
            WebDriverWait(self.driver, self.max_wait_time).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#content > main > article > div.c-article-body > div.main-content"))
            )
            
            # Find all divs that match the pattern #Sec{X}-content
            content_divs = self.driver.find_elements(By.CSS_SELECTOR, "div[id^='Sec'][id$='-content']")
            
            all_texts = []
            end_of_main_text_reached = False
    
            # Iterate over each content div
            for content_div in content_divs:
                # Get the section header (if any) to determine the title
                header = content_div.find_element(By.XPATH, "preceding-sibling::h2[1]")
                section_title = header.get_attribute("innerText").strip() if header else "Unknown"
    
                # SKIP
                if section_title in self.skip_sections:
                    continue
                
                # END
                if section_title in {"References"}:
                    end_of_main_text_reached = True
                
                # Extract text from all <p> elements within the content div
                paragraphs = content_div.find_elements(By.CSS_SELECTOR, "p")
                section_text = " ".join([p.get_attribute("innerText").strip() for p in paragraphs])
                all_texts.append(section_text)
    
                # If we just processed the 'Conclusion' section, break the loop
                if end_of_main_text_reached:
                    break
    
            # Combine all sections' texts into one string
            self.document_text = "\n\n".join(all_texts).replace('\n', ' ').replace('\t', ' ').strip()
    
        except Exception as e:
            print(f'Error retrieving document_text: {e}')
            pass

    def get_abstract(self) -> None:
        """Retrieve the abstract of the document."""
        
        try:
            abstract_section = WebDriverWait(self.driver, self.max_wait_time).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'section[aria-labelledby="Abs1"]'))
            )
            
            content_div = abstract_section.find_element(By.CSS_SELECTOR, "div[id$='-content']")
            
            # Extract text from all <p> elements within the content div
            paragraphs = content_div.find_elements(By.CSS_SELECTOR, "p")
            abstract_text = " ".join([p.get_attribute("innerText").strip() for p in paragraphs])
            
            # Set the abstract text as an attribute
            self.abstract = abstract_text.replace('\n', ' ').replace('\t', ' ').strip()
    
        except Exception as e:
            print(f"An error occurred while extracting the abstract: {e}")

        pass

    def get_valid_html_website(self) -> bool:
        """
        Correction Articles -> not exactly articles
        """
        self.valid_html_content = True
        if self.title.startswith('Correction to:'):
            self.valid_html_content = False

        
