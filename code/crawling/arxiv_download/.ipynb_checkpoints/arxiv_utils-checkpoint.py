import arxiv
import re
import time
import yaml
from pathlib import Path
from tqdm import tqdm

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

class ArXiV_HTML_Parser:
    def __init__(self, wait_time:float=0.1, 
                 max_results:int=200, 
                 categories=None, 
                 keywords_file_path:Path=Path('./config/search_words.yaml')):
        self.urls = []
        self.wait_time = wait_time
        self.driver = None
        self.page_content = {}
        self.keywords_file_path = Path(keywords_file_path)
        self.max_results = max_results
        self.categories = categories

        # load search words
        self.load_keywords_from_categories(categories=self.categories, file_path=self.keywords_file_path)
        
    def init_webdriver(self):
        if not self.driver:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            self.driver = webdriver.Chrome(options=options)

    def load_keywords_from_categories(self, categories=None, file_path='./config/search_words.yaml'):
        '''
        Load keywords from config YAML according to categories
        '''
        assert self.keywords_file_path.is_file(), "File doesn`t exist contaiing keywords."

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
        
        self.search_words = selected_keywords

    def get_arxiv_articles_with_html(self, query, max_results=100):
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        articles_with_html = []

        time.sleep(3)
        
        for result in client.results(search):
            if any(link.title == "pdf" for link in result.links):
                # check if HTML-compatible
                html_url = result.entry_id.replace('/abs/', '/html/')
                time.sleep(1)
                response = requests.get(html_url)
                # load
                if response.status_code == 200:
                    articles_with_html.append({
                        'id': result.entry_id,
                        'title': result.title,
                        'pdf_url': result.entry_id.replace('abs', 'pdf') + '.pdf',
                        'html_url': result.entry_id.replace('/abs/', '/html/'),
                        'authors' : [aut.name for aut in result.authors],
                        'summary' : result.summary,
                        'date_published' : result.published,
                        'date_updated' : result.updated,
                        'doi' : result.doi,
                        'prim_cat' : result.primary_category,
                        'categories' : result.journal_ref,
                        'journal_ref' : result.journal_ref,
                    })

        # shuffle
        random.shuffle(articles_with_html)
        
        return articles_with_html

    def __len__(self,):
        return len(self.urls)

    def find_papers(self,):
        """
        Search ArXiV for papers according to search word list
        """
        print(f'... Search ArXiV for ≤{self.max_results} papers for each of the {len(self.search_words)} search words of {len(self.categories)} categories...')
        
        for search_word in tqdm(self.search_words, desc="Search words..."):
            self.get_urls(search_word)
        
    def get_urls(self, query, n:int=-1):
        self.urls += self.get_arxiv_articles_with_html(query, max_results=self.max_results if n<0 else round(n))
            

    def load_page_content(self, url):
        self.init_webdriver()
        if url not in self.page_content:
            self.driver.get(url)
            self.page_content[url] = self.driver.page_source

    def download_plain_text_from_html(self, url):
        self.load_page_content(url)
        soup = BeautifulSoup(self.page_content[url], 'html.parser')
        text = soup.get_text()
        return text

    def beautify_text(self, text):
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        replacements = {
            r'\\xa0': ' ',
            r'\\u2062': '',
            r'\\n': '\n',
            r'\\t': '\t',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        text = re.sub(r'\\', '', text)
        
        return text

    def extract_title(self, url):
        self.load_page_content(url)
        soup = BeautifulSoup(self.page_content[url], 'html.parser')
        title_element = soup.find('h1', class_='ltx_title ltx_title_document')
        if title_element:
            return title_element.get_text(strip=True)
        else:
            return ''

    def extract_authors(self, url):
        self.load_page_content(url)
        soup = BeautifulSoup(self.page_content[url], 'html.parser')
        authors_div = soup.find('div', class_='ltx_authors')
        authors = []
        if authors_div:
            author_spans = authors_div.find_all('span', class_='ltx_personname')
            for span in author_spans:
                authors.append(span.get_text(strip=True))
        return authors

    def extract_date_and_domain(self, url, sec_to_timeout=3):
        self.load_page_content(url)
        soup = BeautifulSoup(self.page_content[url], 'html.parser')
        watermark_div = soup.find(id="watermark-tr")
        if watermark_div:
            text = watermark_div.text.strip()
            match = re.search(r'arXiv:\d{4}\.\d{5}(?:v\d)? \[(.*?)\] (\d{2} \w{3} \d{4})', text)
            if match:
                domain = match.group(1)
                date = match.group(2)
            else:
                domain = ''
                date = ''
        else:
            domain = ''
            date = ''
        return {'date': date, 'domain': domain}

    def extract_abstract(self, url, sec_to_timeout=3):
        self.load_page_content(url)
        soup = BeautifulSoup(self.page_content[url], 'html.parser')
        abstract_div = soup.find(id="abstract")
        if abstract_div:
            abstract_text = abstract_div.text.strip()
        else:
            abstract_text = None
        return abstract_text

    def extract_author_emails(self, url, sec_to_timeout=5):
        self.load_page_content(url)
        soup = BeautifulSoup(self.page_content[url], 'html.parser')
        authors_div = soup.find('div', class_='ltx_authors')
        
        authors = []
        if authors_div:
            author_elements = authors_div.find_all('span', class_='ltx_personname')
            authors = [author.text.strip() for author in author_elements]

        email_elements = soup.find_all('span', class_='ltx_contact ltx_role_email')
        emails = [email.text.strip() for email in email_elements]

        author_emails = {author: email for author, email in zip(authors, emails)}
        return author_emails

    def post_process_emails(self, author_emails):
        processed_emails = {}
        unmatched_count = 1

        for names, emails in author_emails.items():
            name_list = [name.strip() for name in names.split(',')]
            email_list = [email.strip() for email in emails.split(',')]

            name_map = {name.split()[-1].lower(): name for name in name_list}

            for email in email_list:
                email_user = email.split('@')[0].lower()
                matched = False

                for key, full_name in name_map.items():
                    if key in email_user:
                        processed_emails[full_name] = email
                        matched = True
                        break

                if not matched:
                    processed_emails[f'unmatched_name_{unmatched_count}'] = email
                    unmatched_count += 1

            for name in name_list:
                if name not in processed_emails:
                    processed_emails[name] = ''

        return processed_emails

    def clean_text(self, text):
        text = re.sub(r'\\[A-Za-z]+[*]?', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.replace('†', '')
        text = text.replace('&', '')
        return text

    def extract_emails_ltx_contact(self, url, sec_to_timeout=10):
        '''Extract author emails and institutions from an ArXiv paper page'''
        # Initialize the WebDriver
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run headless browser
        driver = webdriver.Chrome(options=options)

        # debug
        print('url: ', url)
        
        # Load the page
        driver.get(url)
    
        author_emails = {}
        author_institutions = {}
        unknown_author_counter = 1
        
        try:
            # Wait until the ltx_authors element is present
            WebDriverWait(driver, sec_to_timeout).until(
                EC.presence_of_element_located((By.CLASS_NAME, "ltx_authors"))
            )
    
            authors = driver.find_elements(By.CLASS_NAME, "ltx_role_author")
    
            for author in authors:
                # Extracting the author's name
                name_elem = author.find_element(By.CLASS_NAME, "ltx_personname")
                name = self.clean_text(name_elem.text.strip())
    
                # Extracting the email address
                email_elem = author.find_elements(By.CLASS_NAME, "ltx_contact.ltx_role_email")
                email = email_elem[0].text.strip() if email_elem else ''
    
                # Cleaning the name if it has unwanted characters
                name = re.sub(r'[*\d]', '', name).strip()
    
                # Handling cases with multiple authors in one element
                if ' and ' in name or ', ' in name:
                    multiple_names = re.split(r' and |, ', name)
                    for n in multiple_names:
                        n = self.clean_text(n)
                        if not n:
                            n = f'unknown_author_{unknown_author_counter}'
                            unknown_author_counter += 1
                        author_emails[n.strip()] = email
                else:
                    name = self.clean_text(name)
                    if not name:
                        name = f'unknown_author_{unknown_author_counter}'
                        unknown_author_counter += 1
                    author_emails[name] = email
    
                # Extracting the institution
                institution_elems = author.find_elements(By.CLASS_NAME, "ltx_contact.ltx_role_affiliation")
                if institution_elems:
                    for inst_elem in institution_elems:
                        institution = self.clean_text(inst_elem.text.strip())
                        if institution not in author_institutions:
                            author_institutions[institution] = []
                        author_institutions[institution].append(name)
    
            for institution in author_institutions:
                author_institutions[institution] = list(set(author_institutions[institution]))

        except:
            author_emails = {}
            author_institutions = {'' : []}
    
        finally:
            driver.quit()
    
        return {'emails': author_emails, 'institutions': author_institutions}

    def extract_all(self, wait_time:float=-1):
        if wait_time > 0:
            self.wait_time = wait_time
        results = []
        for article in self.urls:
            time.sleep(self.wait_time)
            url = article['html_url']
            self.load_page_content(url)

            # scrape components
            emails_and_institutions = self.extract_emails_ltx_contact(url=article['html_url'])
            #title = self.extract_title(url)
            date_and_domain = self.extract_date_and_domain(url)
            #abstract = self.extract_abstract(url)
            

            # scrapped metadata
            result = {
                'title_scraped': '', # title,
                'date_scraped': date_and_domain['date'],
                'domain_scraped': date_and_domain['domain'],
                'abstract_scraped': '', # abstract,
                'emails': emails_and_institutions['emails'],
                'institutions': emails_and_institutions['institutions']
            }
            
            # add ArXiV-provided metadata
            result.update(article)
            # append
            results.append(result)
        return results

    def close(self):
        if self.driver:
            self.driver.quit()
            self.driver = None