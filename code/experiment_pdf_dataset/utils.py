import sys
import yaml
from pathlib import Path
import random
import os
from pathlib import Path

# Function to detect if running in a Jupyter notebook
def in_notebook():
    '''
    Adaptive tqdm import
    '''
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def recursive_lookup_pdf_paths(p:Path, max_len:int=50_000) -> list[Path]:
    '''Find paths to PDF files in directory p
    '''
    p = Path(p)
    assert p.is_dir(), "Path `p` is invalid"
    
    # ArXiV
    pdf_paths = []

    # PDFs in root
    pdf_paths = [p / f for f in os.listdir(p) if f.endswith('.pdf')]
    if len(pdf_paths) > 0:
        return pdf_paths
    
    
    # No PDFs in root: iterate through each subdirectory and list all PDF files
    for j,subdirectory in enumerate(p.iterdir()):
        if subdirectory.is_dir():
            if j > 10 and (len(pdf_paths)==0 or len(pdf_paths) >= max_len):
                print(f"{len(pdf_paths)} pdfs found. Skip.")
                return pdf_paths
            for k,pdf_file in enumerate(subdirectory.glob('*.pdf')):
                pdf_paths.append(pdf_file)
                if k > 1000:
                    continue
                if len(pdf_paths) >= max_len:
                    return pdf_paths
    
    return pdf_paths




class PDFPaths():
    '''Returns Paths to pdfs
    '''

    def __init__(self, p:Path='/home/siebenschuh/Projects/dataprep/data_paths/paths.yaml', n_max_pdfs:int=10_000):
        self.yaml_path = Path(p)
        with open(self.yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        self.meta = data
        self.data = {dset : [] for dset in data.keys()}
        self.n_max_pdfs = n_max_pdfs
        self.find_pdfs()

        pass

    def parse_Xiv(self, p_dir) -> list[Path]:
        '''
        BioRxiv-specific parser due to weird directory structure
        '''
        pdf_paths = []
        for i, sub_dir in enumerate(p_dir.iterdir()):
            if i > 100:
                continue
            if sub_dir.is_dir():
                for j, sub_sub_dir in enumerate(sub_dir.iterdir()):
                    if j > 100:
                        continue
                    if sub_sub_dir.is_dir():
                        content_dir = sub_sub_dir / 'content'
                        if content_dir.is_dir():
                            pdf_paths.extend(recursive_lookup_pdf_paths(content_dir, max_len=50_000 - len(pdf_paths)))
                            if len(pdf_paths) >= 50_000:
                                return pdf_paths
        
        return pdf_paths

    def parse_ASM(self, p_dir) -> list[Path]:
        '''
        ASM-specific parser due to weird directory structure
        '''
        pdf_paths = []
        for i, sub_dir in enumerate(p_dir.iterdir()):
            if sub_dir.is_dir():
                for j, sub_sub_dir in enumerate(sub_dir.iterdir()):
                    if sub_sub_dir.is_dir():
                        pdf_paths.extend(recursive_lookup_pdf_paths(sub_sub_dir, max_len=50_000 - len(pdf_paths)))
                        if len(pdf_paths) >= 50_000:
                            return pdf_paths
        
        return pdf_paths

    def find_pdfs(self, ):
        '''
        Searches paths for PDFs or ZIPs (depending on dataset)
        '''
        for dset in self.data.keys():
            # skip certain datasets
            p_dir = Path(self.meta[dset])

            # Exclude the following dsets
            if '/ASM' in str(p_dir):
                self.data[dset] = self.parse_ASM(p_dir)
            # BioRxiV
            elif (('Biorxiv' in str(p_dir)) or ('Medrxiv' in str(p_dir)) or ('ASM' in str(p_dir))):
                self.data[dset] = self.parse_Xiv(p_dir)
            # Remainder
            elif p_dir.is_dir():
                found_pdfs = recursive_lookup_pdf_paths(p_dir, self.n_max_pdfs)
                self.data[dset] = found_pdfs
            else:
                pass

        pass

    def __repr__(self,):
        '''Print'''
        return str({k : len(self.data[k]) for k in self.data.keys()})

class ZIPPaths():
    '''
    Returns Paths to ZIPs (that are to be unzipped). Precursor for PDFPaths that consumes `p_dst`
    Use:
    Z = ZIPPaths()
    Z.unzip(`asm`, p_dst=...)
    # Later
    PDFPaths(p_dst)
    '''

    def __init__(self, p:Path='/home/siebenschuh/Projects/dataprep/data_paths/paths.yaml', n_max_zips:int=20_000):
        self.yaml_path = Path(p)
        with open(self.yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        self.meta = data
        self.data = {dset : [] for dset in data.keys()}
        self.n_max_zips = n_max_zips
        self.find_zips()

        pass

    def find_zips(self, ):
        '''
        Searches paths for ZIPs (depending on dataset)
        '''
        for dset in self.data.keys():
            # skip certain datasets
            p_dir = Path(self.meta[dset])

            # Exclude the following dsets
            if 'ASM' in str(p_dir):
                random.seed(55)
                zip_files = [p_dir / f for f in os.listdir(p_dir) if f.endswith('.zip')]
                random.shuffle(zip_files)
                self.data[dset] =  zip_files[:self.n_max_zips]

        pass

    def _unzip_to_(self, p_src: Path, p_dst: Path):
        '''Unzip the source zip file to the destination directory'''
        p_src = Path(p_src)
        p_dst = Path(p_dst)
    
        assert p_src.is_file(), f"Source path `{p_src}` is invalid"
        assert p_dst.is_dir(), f"Destination path `{p_dst}` is invalid"
    
        with zipfile.ZipFile(p_src, 'r') as zip_ref:
            zip_ref.extractall(p_dst)
        print(f"Unzipped `{p_src}` to `{p_dst}`")

    def unzip(self, dset:str, p_dst: Path):
        '''
        Unzip list of files
        '''
        p_dst = Path(p_dst)
        assert p_dst.is_dir(), "p_dst must be valid directory path"

        if len(Z.data[dset]) > 0:
            for zip_path_loc in Z.data[dset]:
                self._unzip_to_(zip_path_loc, p_dst)

        print('Done')
        pass
        

    def __repr__(self,):
        '''Print'''
        return str({k : len(self.data[k]) for k in self.data.keys()})