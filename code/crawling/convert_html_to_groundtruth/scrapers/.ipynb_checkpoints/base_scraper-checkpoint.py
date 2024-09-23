from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import time
import re

from bs4 import BeautifulSoup
import html

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

class TextFromHTML(ABC):
    def __init__(self, 
                 html_file_path:Path,
                 max_wait_time:float=0.2):

        # meta
        self.html_file_path = Path(html_file_path)
        self.max_wait_time = max_wait_time if max_wait_time > 0 else 0.0
        self.unsupported_packages = []
        self.skip_sections = None

        # attributes
        self.csv_data_dict = None
        
        self.document_text = None
        self.title = None
        self.authors = None
        self.creationdate = None
        self.keywords = None
        self.doi = None
        self.producer = None
        self.format = None
        self.first_page = None
        self.abstract = None

        # check
        assert self.html_file_path.is_file(), f"`html_file_path` is invalid file path. No file in: {self.html_file_path}"

        # infer pdf file path
        self.pdf_file_path = Path(str(self.html_file_path).replace('/html/', '/pdf/').replace('.html', '.pdf'))
        assert self.pdf_file_path.is_file(), f"Inferred `pdf_file_path` (from `html_file_path`) is invalid file path. No file in: {self.pdf_file_path}"

        # driver options
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--headless")

        # initialize driver for scraping
        self.driver = webdriver.Chrome(options=chrome_options)
        
        # load page from disk
        self.driver.get(f"file://{self.html_file_path}")

    def scrape_and_quit(self,):
        # scrape
        self.scrape()
        
        # close
        self.driver.quit()
        

    @abstractmethod
    def get_title(self) -> None:
        """Retrieve the title of the document."""
        pass

    @abstractmethod
    def get_authors(self) -> None:
        """Retrieve the list of authors."""
        pass

    @abstractmethod
    def get_creationdate(self) -> None:
        """Retrieve the creation date of the document."""
        pass

    @abstractmethod
    def get_keywords(self) -> None:
        """Retrieve the list of keywords."""
        pass

    @abstractmethod
    def get_doi(self) -> None:
        """Retrieve the DOI (Digital Object Identifier) of the document."""
        pass

    @abstractmethod
    def get_producer(self) -> None:
        """Retrieve the producer information (e.g., publisher or institution)."""
        pass

    @abstractmethod
    def get_document_text(self) -> None:
        """Retrieve the main text of the document."""
        pass

    def get_format(self) -> None:
        """Retrieve the format of the document (e.g., PDF, HTML)."""
        pass

    def get_first_page(self) -> None:
        """Retrieve the first page of the document."""
        pass

    def get_abstract(self) -> None:
        """Retrieve the abstract of the document."""
        self.abstract = "Abstract content"  # Placeholder logic
        pass

    def get_unsupported_packages(self) -> None:
        """Error messages on the HTML of any kind tha might inform 
        how useful the scraped text is (e.g. ArXiV: unsuppoted text packages)
        """
        self.unsupported_packages = []
        pass

    def get_valid_html_website(self) -> bool:
        """
        Does the HTML content appear valid? Depends on the journal
        """
        self.valid_html_content = True
        pass

    def __clean_html_text__(self, html_content: str) -> str:
        """
        Cleans the provided HTML content by removing escape characters, converting HTML entities,
        and handling special characters without introducing unwanted formatting.
        """
        # Parse HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
    
        # Extract text from the parsed HTML
        text = soup.get_text()
    
        # Convert HTML entities to their corresponding characters
        text = html.unescape(text)
    
        # Remove escape characters like \n, \t, etc.
        text = text.replace('\n', '').replace('\t', '').replace('\r', '')
    
        # Remove any leading or trailing whitespace
        text = text.strip()
    
        return text

    def clean_html_text_adaptive(self, text: str) -> str:
        """
        Cleans the provided HTML content by removing escape characters, converting HTML entities,
        and handling special characters without introducing unwanted formatting. If the input
        doesn't require formatting, it returns the original text.
        """
        # no text
        if text is None:
            return text
        
        # Check if the string contains any HTML tags
        if bool(re.search(r'<[^>]+>', text)):
            try:
                # Attempt to clean the text using the clean_html_text function
                return self.__clean_html_text__(text)
            except Exception as e:
                # If an error occurs, return the original text
                return text.replace('\n', '').replace('\t', '')
        else:
            # If no HTML tags are found, return the original text
            return text.replace('\n', '').replace('\t', '').replace('\r', '')

    def post_process_authors(self, authors: list[str] | str) -> list[str]:
        """
        Post-process the list of author names by applying the following transformations:
        - Split names containing ' and ', ' & ', or commas into separate list elements.
        - Remove symbols like '&', '‡', '§', '†', '*' entirely.
        - Remove digit(s) at the end of a string that are preceded by a whitespace or ','.
        - Remove digit(s) at the beginning of a string that follow a whitespace.
        - Remove ',' altogether in each str.
        
        :param authors: A list of author names or a single string of author names.
        :return: A cleaned list of author names.
        """
    
        # If a single string is passed, convert it to a list
        if isinstance(authors, str):
            authors = [authors]
        
        processed_authors = []
        for author in authors:
            # Split on ' and ', ' & ', or commas followed by non-digits (to avoid splitting initials)
            split_authors = re.split(r'\s+and\s+|\s+&\s+|,\s*(?!\d)', author)
            
            for individual_author in split_authors:
                # Remove symbols like '&', '‡', '§', '†', '*'
                individual_author = re.sub(r'[&‡§†*]', '', individual_author)
                
                # Remove digits at the end of a string preceded by a whitespace or ','
                individual_author = re.sub(r'[\s,]\d+$', '', individual_author)
                
                # Remove digits at the beginning of a string that follow a whitespace
                individual_author = re.sub(r'(?<=\s)\d+', '', individual_author)
                
                # Remove any remaining commas
                individual_author = individual_author.replace(',', '')
                
                # Remove any leading or trailing whitespace
                individual_author = individual_author.strip()

                # Remove all digits from the string
                individual_author = re.sub(r'\d+', '', individual_author)

                # delete leading `by`
                if individual_author.startswith('by '):
                    individual_author = individual_author[3:] 
                
                # Add the cleaned author to the list if it's not empty
                if individual_author:
                    processed_authors.append(individual_author)
        
        return processed_authors
    
    def scrape(self) -> dict:
        """Perform the scraping and return the results in a dictionary."""
        
        # get attributes
        self.get_title()
        self.get_authors()
        self.get_creationdate()
        self.get_keywords()
        self.get_doi()
        self.get_producer()
        self.get_format()
        self.get_first_page()
        self.get_abstract()
        self.get_document_text()

        # get meta information
        self.get_unsupported_packages()
        self.get_valid_html_website()         

        # Return a dictionary with the desired structure
        return {
            "text_groundtruth": self.document_text,
            "path": self.html_file_path,
            "metadata": {
                "title": self.title,
                "authors": self.authors,
                "creationdate": self.creationdate,
                "keywords": self.keywords,
                "doi": self.doi,
                "producer": self.producer,
                "format": self.format,
                "first_page": self.first_page,
                "abstract": self.abstract,
            },
            "extrametadata" : {
                "unsupported_packages" : self.unsupported_packages,
                "csv_data_dict" : self.csv_data_dict
            }
        }
    def get_metadata_parser_style(self,) -> dict:
        """
        Returns a dictionary `extactly` in the format of each line of the jsonls that store the 
        parser output in ./nougat/parsed_pdfs/ etc.
        `text`, `path`, `metadata` where the `metadata` holds `title etc.`
        """

        output_dict = {
            "text": self.document_text,
            "path": str(self.pdf_file_path),
            "metadata": {
                "title": self.title,
                "authors": self.authors,
                "creationdate": self.creationdate,
                "keywords": self.keywords,
                "doi": self.doi,
                "producer": self.producer,
                "format": self.format,
                "first_page": self.first_page,
                "abstract": self.abstract,
            }
        }

        return output_dict

