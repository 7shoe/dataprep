import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import jiwer

import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
import random

class ConvertParserOutput:
    def __init__(self, 
                 jsonl_root:Path = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt'),
                 store_path:Path=Path('./database'),
                 parser_name_list:list[str] = ['nougat', 'html', 'marker', 'pymupdf', 'pypdf', 'grobid', 'tesseract']):

        self.jsonl_root = jsonl_root
        self.store_path = Path(store_path)
        self.parser_name_list = parser_name_list
        
        assert self.jsonl_root.is_dir(), "Root directory for jsonl does not exist."
        assert self.store_path.is_dir(), f"Invalid `store_path`. {store_path} does not exist"
        assert len(self.parser_name_list) > 0, "Length of `parser_name_list` cannot be 0."

        pass

    def __transform_path__(self, full_path:str) -> str:
        """
        Transform paths to standardized format
        """

        full_path = Path(full_path)
        
        # get rid of duplicates
        parent_parent = full_path.parents[1].name 
        parent = full_path.parent.name             
        filename = full_path.name
        
        # combine them to get the desired result
        result = Path(parent_parent) / parent / filename

        return str(result)

    def __remove_surrogates__(self, text):
            '''Format column so df can be stored
            '''
            if isinstance(text, str):
                # This regular expression matches surrogate pairs
                return re.sub(r'[\ud800-\udfff]', '', text)
            return text
        

    def create_text_database(self, 
                             file_name:str, 
                             overwrite:bool=True,
                             fill_na_with_empty_str:bool=True):
        """
        Looks-up all jsonl files (across the list of parsers) to extract `path` and `text` and merge them into a DB
        """
        
        # store dir
        store_file_path = self.store_path / file_name
        if store_file_path.is_file() and not(overwrite):
            print(f"Cannot create CSV. `store_file_path`={store_file_path} already exist. And `overwrite` flag is set to {overwrite}")
            pass
        
        # def creat_text_database (rows: pdf files, column parser)
        all_dict = {}

        # DEBUG
        print('self.parser_name_list : ', self.parser_name_list)
        
        # loop each parser
        for parser_name in self.parser_name_list:
            # 
            p_parser_output = self.jsonl_root / f'joint_to_{parser_name}/parsed_pdfs'
            if not(p_parser_output.is_dir()):
                print(f"Skip {parser_name}. Directory `p_parser_output`={p_parser_output} not found")
            
            # grab HTML
            jsonl_files = [p_parser_output / f for f in os.listdir(p_parser_output) if f.endswith('.jsonl')]
            
            # each parser
            pdf_path_list = []
            pdf_text_list = []
        
            # status
            print(f'Parser: {parser_name} w/ {len(jsonl_files)} jsonl files.')
        
            # jsonl
            for jsonl_file in jsonl_files:
                # open
                with open(jsonl_file, 'r') as f:
                    for _,line in enumerate(f):
                        data = json.loads(line)
                        
                        # extract path / text
                        pdf_path = str(data['path'])

                        # only extract valid text
                        if data['text'] is not None:
                            decoded_text = data['text'].replace('|', '')

                            # non-duplicate
                            if pdf_path not in pdf_path_list:
                                # append 
                                pdf_path_list.append(pdf_path)
                                pdf_text_list.append(decoded_text)
        
            # append to to dict
            all_dict[parser_name] = {'path' : pdf_path_list, 'text' : pdf_text_list}

        # index : available PDFs (sorted by parent, filename)
        pdf_paths  = [item for sublist in [all_dict[k]['path'] for k in all_dict.keys()] for item in sublist]
        index_set  = set(pdf_paths)
        index_list = [idx for idx in index_set if 'ipynb_checkpoints' not in idx]
        index_list = list(index_set)
        sorted_index_list = sorted(index_list, key=lambda p: (Path(p).parent.parent, Path(p).name))
        
        # convert to str
        sorted_index_list = [self.__transform_path__(idx) for idx in sorted_index_list]
        
        # setup DataFrame
        df = pd.DataFrame(index=sorted_index_list, columns=self.parser_name_list)

        print("HERRE >>>> ")
        
        # Iterate over parser_name_list
        for parser_name in self.parser_name_list:
            # skip non-existing entries
            if parser_name not in all_dict:
                continue
            
            # extract the corresponding dictionary from all_dict
            paths = all_dict[parser_name]['path']
            texts = all_dict[parser_name]['text']

            # kill duplicates
            unique_paths, indices = np.unique(paths, return_index=True)
            unique_texts = [texts[i] for i in indices]

            # path
            unique_paths = [self.__transform_path__(p) for p in unique_paths]
            
            # create a temporary DataFrame with paths as the index and texts as the data
            temp_df = pd.DataFrame(unique_texts, index=unique_paths, columns=[parser_name])

            # kill duplicates
            temp_df = temp_df[~temp_df.index.duplicated(keep='first')]
            
            # Update the main DataFrame df with the data from temp_df
            df.update(temp_df)

        # Optionally, fill any remaining NaN values with a default value (e.g., empty string)
        if fill_na_with_empty_str:
            df.fillna('', inplace=True)

        # filter (according to path)
        df['path'] = list(df.index)
        df.reset_index(drop=True, inplace=True)
        # - remove duplicates of same PDFs
        df['nan_count'] = df.isna().sum(axis=1)
        df_sorted = df.sort_values(by=['path', 'nan_count'])
        df_unique = df_sorted.drop_duplicates(subset='path', keep='first')
        df_unique = df_unique[~df_unique['path'].str.contains('.ipynb_checkpoints')]
        # - reset index
        df_unique = df_unique.set_index('path')
        df_unique = df_unique.drop(columns=['nan_count'])
        # - filter out rows for which groundtruth `html` text is not NaN
        df_unique = df_unique[~df_unique['html'].isna()]

        # print
        print('Before unique')
        
        # re-assign
        df = pd.DataFrame(df_unique)

        # surrogate removal function to all str columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].apply(self.__remove_surrogates__)
        
        # store
        print(f"STORE!!! {store_file_path}")
        df.to_csv(store_file_path, sep='|')

        pass