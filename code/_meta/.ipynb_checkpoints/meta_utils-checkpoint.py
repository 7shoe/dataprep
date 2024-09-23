from pathlib import Path
import pandas as pd
import json
import os

class Terminator_of_already_parsed_Files:
    def __init__(self, 
                 subdir_name:str,
                 p = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/merged_pdf_only_data_to_nougat/parsed_pdfs'),
                 parsed_root:Path = Path('/home/siebenschuh/Projects/dataprep/code/_meta/already_parsed'), 
                 target_dir_root:Path = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/additional_merged_raw_data')):

        print('subdir_name: ', subdir_name)
        assert subdir_name in ['arxiv', 'biorxiv', 'bioarxiv', 'medarxiv', 'medrxiv', 'mdpi', 'plos', 'bmc', 'nature'], ""
        
        self.p = Path(p)
        self.subdir_name = subdir_name
        self.parsed_root = Path(parsed_root)
        self.target_dir_root=Path(target_dir_root) #=Path('/homes/csiebenschuh/Projects/dataprep/data_copy/')
        
    def load_first_set_pdfs(self,):
        """
        Run on Polaris, only
        Initially, ~18k PDFs from ArXiV, MedRXiv, BioRXiv, MDPI were parsed. These are their source paths.
        """
        assert self.p.is_dir(), ""
        
        # path to which nougat parsed
        parsed_jsons = [self.p / f for f in os.listdir(self.p)]
        
        # list
        pdf_list = []
        for jsonl_file in parsed_jsons:
            # open the JSONL file
            with open(jsonl_file, 'r') as file:
                for line in file:
                    json_obj = json.loads(line)
                    # PDF path
                    pdf_list.append(json_obj['path'])
        
        pdf_list = list(set(pdf_list))
    
        return pdf_list
        
    def create_meta(self,):
        """
        Run on Polaris, only
        Create .txt files that hold file IDs (=stem, i.e. name w/o suffix) that are already present in nougat output
        """
        pdf_list = self.load_first_set_pdfs()
        
        # loop
        for subdir_name in ['mdpi', 'arxiv', 'biorxiv', 'medrxiv']:
            subdir_list = []
            
            # check
            os.makedirs(self.output_dir, exist_ok=True)
            
            # loop
            for f in pdf_list:
                if f.startswith(f'/homes/csiebenschuh/Projects/dataprep/meta/already_parsed/{subdir_name}/'):
                    subdir_list.append(Path(f).name.split('.pdf')[0])
            # store
            output_file = os.path.join(self.output_dir, f"{subdir_name}.txt")
            
            # Write the list to the file
            with open(output_file, 'w') as f:
                for item in subdir_list:
                    f.write(f"{item}\n")
        
        pass
    
    def load_id_list(self, subdir_name: str) -> list[str]:
        """Load IDs/stems
        """
        file_path = Path(f'./already_parsed/{subdir_name}.txt')
        assert file_path.is_file(), f"File does not exist: {file_path}"
        
        with open(file_path, 'r') as f:  # Open in read mode
            l = f.readlines()
            
        # Optionally strip newlines from each line
        l = [line.strip() for line in l]
        
        return l
    
    def delete_already_parsed_files(self,):
        """
        Deletion on lambda (before tar.zip and file transfer to polaris)
        Go through csv/pdf/html subdirectories and delete what is already parsed
        """

        # check paths
        assert self.target_dir_root.is_dir(), "<-"
        
        # load list
        already_parsed_id_list = self.load_id_list(self.subdir_name)
        assert len(already_parsed_id_list) > 0, "Length is 0. Nothing to parse."
        
        # target directory
        target_dir = self.target_dir_root / self.subdir_name
        # debug
        print(target_dir)
        
        assert target_dir.is_dir(), "Target directory does not exist"
        parsed_file = self.parsed_root / f"{self.subdir_name}.txt"
        assert parsed_file.is_file(), "File holding the parsed PDF IDs (for the specific dataset) does not exist"
        
        for subDirectory in ['csv', 'pdf', 'html']:
            target_file_dir = target_dir / subDirectory
            
            # collect existing files
            target_file_paths = [target_file_dir / f for f in os.listdir(target_file_dir)]
        
            # Identify & delete
            for target_file in target_file_paths:
                if target_file.stem in already_parsed_id_list:
                    #print(target_file)
                    os.remove(target_file)
        pass
