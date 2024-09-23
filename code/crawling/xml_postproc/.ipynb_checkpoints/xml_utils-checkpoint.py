from pathlib import Path
import os
from lxml import etree
import xml.etree.ElementTree as ET
from datetime import datetime
import json
import uuid

class XMLSciDocParser:
    def __init__(self, 
                 path:Path=Path('/eagle/projects/argonne_tpc/ogokdemir/GrobidParse/'),
                 save_dir:Path=Path('/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/joint_to_grobid/parsed_pdfs'),
                 modified_root_dir:Path=Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/joint'),
                 n:int=-1):
        '''
        XML reads out .tei.xml files (not the PDFs) and those are stored at a different location
        --> need to modify the path to allow matching with other parsed texts later in `./preprocessing/get_tables.py
        '''
        
        self.save_dir = Path(save_dir)
        self.modified_root_dir = Path(modified_root_dir)
        assert self.save_dir.is_dir(), f"Directory to which jsonls are saved does not exist. Invalid path: {self.save_dir}"
        
        # walk all subdirectories, store each xml's path in list
        self.xml_file_list = list(path.rglob("*.xml"))

        # restrict
        if n > 0 and n < len(self.xml_file_list):
            print(f"Only n={n} files ...")
            self.xml_file_list = self.xml_file_list[:round(n)]

        # STATUS
        print(f"Found {len(self.xml_file_list)} XML files to parse...")

        pass

    def _parse_arxiv_(self, p):
        """
        Parse a PDF from ArXiV
        """
        p = Path(p)
        assert p.is_file() and p.suffix=='.xml', "Assume its an XML"
    
        # Load and parse the XML file
        tree = ET.parse(p)
        root = tree.getroot()
        
        # full text
        full_text = ''
        for elem in root.iter():
            if elem.tag.endswith('p') or elem.tag.endswith('head'):
                if elem.text is not None:
                    full_text += elem.text 
        
        # title
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        title = root.find(".//tei:titleStmt/tei:title", ns)
        title_text = title.text if title.text is not None else ''
        
        # authors
        authors = []
        bibl_struct = root.find(".//tei:sourceDesc/tei:biblStruct", ns)
        if bibl_struct is not None:
            for author in bibl_struct.findall(".//tei:author", ns):
                # elem
                forename = author.find(".//tei:forename", ns)
                surname = author.find(".//tei:surname", ns)
                # strs
                if forename is not None and forename.text is not None:
                    forename_text = forename.text
                else:
                    forename_text = ''
                if surname is not None and surname.text is not None:
                    surname_text = surname.text
                else:
                    surname_text = ''
                full_name = f"{forename_text} {surname_text}" if forename_text!='' else surname_text
                if full_name.strip():
                    authors.append(full_name)
        
        # date 
        date_element = root.find(".//tei:publicationStmt/tei:date", ns)
        
        # Extract the publication year
        date_text = ''
        if date_element is not None:
            when = date_element.get('when')
            # try
            try:
                date_obj = datetime.strptime(when, '%Y-%m-%d')
                date_text = date_obj.strftime('%d-%m-%Y')
            except:
                date_text = ''
        
        # abstract
        abstract_element = root.find(".//tei:profileDesc/tei:abstract//tei:p", ns)
        abstract_text = abstract_element.text if abstract_element is not None else ''
    
        # metadata
        meta_output = {'title' : title_text, 'authors' : authors, 'creationdate' : date_text, 'abstract' : abstract_text}

        # derive modified path (necessary to match .tei.xml with actual source .pdf)
        p_modified = (self.modified_root_dir / p.parent / 'pdf') / p.name.replace('.grobid.tei.xml', '.pdf')
        
        # output dictionary
        output = {'text' : full_text, 'path' : str(p_modified), 'metadata' : meta_output}
    
        return output

    def _parse_Xrxiv_(self, p):
        """
        Parse a PDF from BioRXiv or MedRXiv or BMC
        """
        p = Path(p)
        assert p.is_file() and p.suffix=='.xml', "Assume its an XML"
    
        # Load and parse the XML file
        tree = ET.parse(p)
        root = tree.getroot()
        
        # full text
        full_text = ''
        for elem in root.iter():
            if elem.tag.endswith('p') or elem.tag.endswith('head'):
                if elem.text is not None:
                    full_text += elem.text 
        
        # title
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        title = root.find(".//tei:titleStmt/tei:title", ns)
        title_text = title.text if title.text is not None else ''
        
        # authors
        authors = []
        bibl_struct = root.find(".//tei:sourceDesc/tei:biblStruct", ns)
        authors = []
        if bibl_struct is not None:
            for author in bibl_struct.findall(".//tei:author", ns):
                # elem
                forename = author.find(".//tei:forename", ns)
                surname = author.find(".//tei:surname", ns)
                # strs
                if forename is not None and forename.text is not None:
                    forename_text = forename.text
                else:
                    forename_text = ''
                if surname is not None and surname.text is not None:
                    surname_text = surname.text
                else:
                    surname_text = ''
                full_name = f"{forename_text} {surname_text}" if forename_text!='' else surname_text
                if full_name.strip():
                    authors.append(full_name)
        
        # date 
        date_element = root.find(".//tei:publicationStmt/tei:date", ns)
        
        # Extract the publication year
        if date_element is not None:
            when = date_element.get('when')
            # try
            try:
                date_obj = datetime.strptime(when, '%Y-%m-%d')
                date_text = date_obj.strftime('%d-%m-%Y')
            except:
                date_text = ''
        date_text = ''
        
        # Find the abstract element
        abstract_div = root.find(".//tei:profileDesc/tei:abstract//tei:p", ns)
        
        # Extract the abstract text, concatenating parts and ignoring citation refs
        abstract_text = ''
        if abstract_div is not None:
            for elem in abstract_div.iter():
                if elem.tag.endswith('p'):
                    abstract_text += elem.text or ''
                elif elem.tag.endswith('ref'):
                    # Optionally, include the citation mark or ignore it
                    abstract_text += elem.tail or ''
                else:
                    if elem.tail:
                        abstract_text += elem.tail
        
        # Strip any leading/trailing whitespace
        abstract_text = abstract_text.strip()
    
        # DOI
        # Use XPath to find the DOI element
        doi_element = root.find(".//tei:idno[@type='DOI']", {'tei': 'http://www.tei-c.org/ns/1.0'})
        
        # Extract the DOI text
        doi = doi_element[0] if doi_element else ''
    
        # metadata
        meta_output = {'title' : title_text, 'authors' : authors, 'creationdate' : date_text, 'abstract' : abstract_text, 'doi' : doi}

        # modify past 
        new_parent = p.parent.stem.split('_')[0]
        new_path = p.parent.parent / new_parent / p.name

        # derive modified path (necessary to match .tei.xml with actual source .pdf)
        p_modified = (self.modified_root_dir / p.parent / 'pdf') / p.name.replace('.grobid.tei.xml', '.pdf')

        # output dictionary
        output = {'text' : full_text, 'path' : str(p_modified), 'metadata' : meta_output}
    
        return output

    def _parse_nature_(self, p):
        """
        Parse a PDF from Nature
        """
        p = Path(p)
        assert p.is_file() and p.suffix=='.xml', "Assume its an XML"
    
        # Load and parse the XML file
        tree = ET.parse(p)
        root = tree.getroot()
        
        # full text
        full_text = ''
        for elem in root.iter():
            if elem.tag.endswith('p') or elem.tag.endswith('head'):
                if elem.text is not None:
                    full_text += elem.text 
        
        # title
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        title = root.find(".//tei:titleStmt/tei:title", ns)
        title_text = title.text if title.text is not None else ''
        
        # authors
        authors = []
        bibl_struct = root.find(".//tei:sourceDesc/tei:biblStruct", ns)
        authors = []
        if bibl_struct is not None:
            for author in bibl_struct.findall(".//tei:author", ns):
                # elem
                forename = author.find(".//tei:forename", ns)
                surname = author.find(".//tei:surname", ns)
                # strs
                if forename is not None and forename.text is not None:
                    forename_text = forename.text
                else:
                    forename_text = ''
                if surname is not None and surname.text is not None:
                    surname_text = surname.text
                else:
                    surname_text = ''
                full_name = f"{forename_text} {surname_text}" if forename_text!='' else surname_text
                if full_name.strip():
                    authors.append(full_name)
        
        # date 
        date_element = root.find(".//tei:publicationStmt/tei:date", ns)
    
        # Extract the publication year
        if date_element is not None:
            when = date_element.get('when')
            # try
            try:
                date_obj = datetime.strptime(when, '%Y-%m-%d')
                date_text = date_obj.strftime('%d-%m-%Y')
            except:
                date_text = ''
        date_text = ''
        
        # Find the abstract element
        abstract_div = root.find(".//tei:abstract//tei:div", ns)
        
        # Extract the abstract text, concatenating parts and ignoring citation refs
        abstract_text = ''
        if abstract_div is not None:
            for elem in abstract_div.iter():
                if elem.tag.endswith('p'):
                    abstract_text += elem.text or ''
                elif elem.tag.endswith('ref'):
                    # Optionally, include the citation mark or ignore it
                    abstract_text += elem.tail or ''
                else:
                    if elem.tail:
                        abstract_text += elem.tail
        
        # Strip any leading/trailing whitespace
        abstract_text = abstract_text.strip()
    
        # DOI
        # Use XPath to find the DOI element
        doi_element = root.find(".//tei:idno[@type='DOI']", {'tei': 'http://www.tei-c.org/ns/1.0'})
        
        # Extract the DOI text
        doi = doi_element[0] if doi_element else ''
    
        # metadata
        meta_output = {'title' : title_text, 'authors' : authors, 'creationdate' : date_text, 'abstract' : abstract_text, 'doi' : doi}
    
        # derive modified path (necessary to match .tei.xml with actual source .pdf)
        p_modified = (self.modified_root_dir / p.parent / 'pdf') / p.name.replace('.grobid.tei.xml', '.pdf')

        # output dictionary
        output = {'text' : full_text, 'path' : str(p_modified), 'metadata' : meta_output}
    
        return output
        
    def parse_to_jsonl(self,):
        """
        Parse each XML with the respective method
        """

        json_list = []
        # loop
        for xml_path in self.xml_file_list:
            try:
                # parse accordingly
                if '/arxiv' in str(xml_path):
                    output = self._parse_arxiv_(xml_path)
                elif ('/medrxiv' in str(xml_path)) or ('/biorxiv' in str(xml_path)) or ('/bmc' in str(xml_path)):
                    output = self._parse_Xrxiv_(xml_path)
                elif ('/nature' in str(xml_path)) or ('/mdpi' in str(xml_path)):
                    output = self._parse_nature_(xml_path)
                else:
                    pass
                # append
                json_list.append(output)
            except Exception as e:
                print(f"Error with .../{xml_path.parent.name}/{xml_path.name}")
                print(e)
                break

        # store
        self.json_list = json_list

        # DEBUG
        print('... Completed postprocessing of all XML files.')

    def save_jsonl_files(self, ):
        """
        Store dictionaries into JSONLs
        """
        # Ensure the save directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Split the json_list into 5 parts
        split_size = len(self.json_list) // 5
        json_chunks = [self.json_list[i:i + split_size] for i in range(0, len(self.json_list), split_size)]

        # If there are any remaining elements, add them to the last chunk
        if len(json_chunks) > 5:
            json_chunks[-2].extend(json_chunks[-1])
            json_chunks = json_chunks[:-1]

        # Write each chunk to a separate JSONL file
        for chunk in json_chunks:
            filename = f"{uuid.uuid4()}.jsonl"
            file_path = self.save_dir / filename
            with file_path.open('w', encoding='utf-8') as f:
                for item in chunk:
                    f.write(json.dumps(item) + '\n')
        pass
        