import pandas as pd
import numpy as np
from pathlib import Path
import random

from matplotlib import pyplot as plt

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import Levenshtein
from rapidfuzz import fuzz

import string
import re

from multiprocessing import Pool, cpu_count

class TextScoreTable:
    def __init__(self,
                 db_src_filename:str,
                 db_dst_filename:str,
                 chunk_size:int=-1,
                 chunk_index:int=-1,
                 max_char:int=-1,
                 len_df_raw:int=284_470,
                 overwrite_flag:bool = False,
                 root_dir:Path=Path('/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/database/pagewise')) -> None:

        # validate
        root_dir = Path(root_dir)
        assert root_dir.is_dir(), f"Path`root_dir` does not exist or is not directory. Invalid directory: {root_dir}"
        
        # paths
        self.db_src_path = Path(root_dir) / db_src_filename
        self.db_dst_path = Path(root_dir) / db_dst_filename
        self.overwrite_flag = overwrite_flag
        self.chunk_index = chunk_index
        self.chunk_size = chunk_size
        self.max_char = max_char

        # compute length
        assert self.db_src_path.is_file(), f"Source CSV path invalid. No such path: {self.db_src_path}"
        if (len_df_raw is None) or (len_df_raw==-1):
            print('No `len_df_raw` provided. Check length of raw table manually...')
            df_raw_0 = pd.read_csv(self.db_src_path, sep='|', usecols=[0])
            len_df_raw = len(df_raw_0)
            print(f'... {len_df_raw}')
        
        # raw data
        if not(self.overwrite_flag):
            assert not(self.db_dst_path.is_file()), f"Destination Path invalid. File already exists at path: {self.db_dst_path}"

        # subset df into chunks
        if self.chunk_index != -1 and self.chunk_size != -1:
            assert self.chunk_index >= 0, f"`chunk_index` should be non-negative (or -1) but is {self.chunk_index}"
            assert self.chunk_size > 0, f"`chunk_size` should be positive (or -1) but is {self.chunk_size}"

            # insert chunk_index into filename
            db_dst_filename = Path(db_dst_filename).stem + f'_{self.chunk_index}-{len_df_raw // self.chunk_size}' + Path(db_dst_filename).suffix
            self.db_dst_path = Path(root_dir) / db_dst_filename

            # identify target start/end indices
            i_start, i_end = self.chunk_index * self.chunk_size, (self.chunk_index + 1) * self.chunk_size
            if i_start > len_df_raw:
                raise ValueError(f'SET BY HAND!! Chunk_index to big! i_start={i_start} > {len_df_raw}=len_df_raw. chunk_index<={len_df_raw // self.chunk_size}!')
            # load df
            try:
                df_raw = pd.read_csv(self.db_src_path, sep='|', skiprows=range(1, i_start), nrows=i_end-i_start)
            except Exception as e:
                print(f"Tried reading a subset of the raw_dataframe (via) skiprows. Encontered error e: \n{e}")

            # DEBUG
            print(f'Loaded subset of df_raw w/ df_raw: {len(df_raw)} rows since i_start/i_end: {i_start}/{i_end}.')
            print(f'Output file path will be: {self.db_dst_path}')
        else:
            # load entire df
            df_raw = pd.read_csv(self.db_src_path, sep='|')

            # DEBUG
            print(f'Loaded entire df_raw w/ len(df_raw): {len(df_raw)}.')

        # process raw table
        # - fill NaNs properly for parser output (NaN output = ``)
        for col in df_raw.columns:
            if df_raw[col].dtype == 'object' and col != 'path' and col != 'page':
                df_raw[col] = df_raw[col].fillna('')
        # - reduce number of chars per element (statistically undesirable but required for tractability)
        if self.max_char!=-1:
            assert self.max_char > 0, "Max. number of chars considered `max_char` should be positive or -1 (inactive)."
            for col in df_raw.columns:
                if df_raw[col].dtype == 'object' and col != 'path' and col != 'page':
                    df_raw[col] = df_raw[col].apply(lambda x: x[:round(max_char)] if isinstance(x, str) else x)
        
        # drop NA rows
        df_proc = df_raw.dropna()

        # status
        print(f'NaN filtering of \n{len(df_raw)} rows ... {len(df_proc)} remaining after NaN removal.')

        # assign
        self.df = df_proc

        pass 

    # Function to calculate BLEU score between two texts
    def calculate_bleu(self,
                       reference:str,
                       hypothesis:str) -> float:
        """
        Compute BLEU score
        """
        reference_tokens = nltk.word_tokenize(reference)
        hypothesis_tokens = nltk.word_tokenize(hypothesis)
        
        # Use SmoothingFunction to handle cases with zero counts in higher-order n-grams
        smoothing_function = SmoothingFunction().method1
        
        return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing_function)

    # Function to calculate BLEU score between two texts
    def calculate_meteor(self, reference: str, hypothesis: str) -> float:
        """
        Compute METEOR score
        """
        reference_tokens = nltk.word_tokenize(reference)
        hypothesis_tokens = nltk.word_tokenize(hypothesis)

        # METEOR expects a list of reference sentences
        reference_tokens_list = [reference_tokens]
        
        return meteor_score(reference_tokens_list, hypothesis_tokens)
    
    def calculate_rouge(self,
                        reference:str, 
                        hypothesis:str) -> float:
        """
        Compute ROUGE score
        """
        scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        
        return scores['rouge2'].fmeasure
    
    def calculate_car(self,
                      reference:str, 
                      hypothesis:str) -> float:
        """
        Compute character accuracy rate (CAR)
        """
        
        return fuzz.ratio(reference, hypothesis) / 100.
    
    def remove_latex(self,
                     text:str) -> str:
        """
        Remove LaTeX formatting from a string
        """
        # Remove LaTeX commands (e.g., \textbf{...}, \emph{...}, etc.)
        text = re.sub(r'\\[a-zA-Z]+\{(.*?)\}', r'\1', text)
        
        # Remove inline math (e.g., $...$)
        text = re.sub(r'\$(.*?)\$', r'\1', text)
        
        # Remove display math (e.g., \[...\] or $$...$$)
        text = re.sub(r'\$\$(.*?)\$\$', r'\1', text)
        text = re.sub(r'\\\[(.*?)\\\]', r'\1', text)
        
        # Remove other LaTeX-specific characters (e.g., \, \%, etc.)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # Remove braces and any content between them
        text = re.sub(r'\{|\}', '', text)
        
        return text
    
    
    def normalize(self,
                  x:str, 
                  remove_latex_flag:bool=True) -> str:
        """
        Normalize the text
        """
    
        # const
        REMOVE_PUNCT = str.maketrans("", "", string.punctuation)
    
        # remove latex
        if remove_latex_flag:
            x = self.remove_latex(x)
    
        # remove escape characters
        x = x.translate(REMOVE_PUNCT)
        x = re.sub(r"\s+", " ", x)
        x = x.lower()
        x = x.strip()
        
        return x

    def _compute_metrics_for_row(self, row, normalized):
        """Helper function to compute metrics for a single row"""
        result = {}
        # DEBUG
        #print('\n\nrow : ', row, '\n\n')
        parsers = ['nougat', 'pymupdf', 'grobid', 'pypdf', 'marker']

        # resilient parallelization requires try except
        try:
            # Normalize text
            if normalized:
                for column in ['html'] + parsers:
                    result[f'{column}_norm'] = self.normalize(row[column])
    
            # Calculate BLEU scores - raw and normalized
            for parser in parsers:
                result[f'bleu_{parser}'] = self.calculate_bleu(row['html'], row[parser])
                result[f'rouge_{parser}'] = self.calculate_rouge(row['html'], row[parser])
                #result[f'meteor_{parser}'] = self.calculate_meteor(row['html'], row[parser]) # WAY TO SLOW
                result[f'car_{parser}'] = self.calculate_car(row['html'], row[parser])
                
                if normalized:
                    result[f'bleu_{parser}_norm'] = self.calculate_bleu(result['html_norm'], result[f'{parser}_norm'])
                    result[f'rouge_{parser}_norm'] = self.calculate_rouge(result['html_norm'], result[f'{parser}_norm'])
                    #result[f'meteor_{parser}_norm'] = self.calculate_meteor(result['html_norm'], result[f'{parser}_norm']) # TOO SLOW
                    result[f'car_{parser}_norm'] = self.calculate_car(result['html_norm'], result[f'{parser}_norm'])
    
            return result
        except Exception as err:
            print(f'Exception in ...(_compute_metrics_for_row). Error e is: \n{err}')
            return None

    def compute_metrics(self, normalized: bool = True) -> None:
        """
        Processes the table in parallel
        """
        # Determine the number of workers based on available CPU cores
        num_workers = 8  # Adjust as needed 8 appears* solid

        # Parallelize the computation across rows
        with Pool(num_workers) as pool:
            results = pool.starmap(self._compute_metrics_for_row, [(row, normalized) for _, row in self.df.iterrows()])

        # Merge the results back into the DataFrame
        for i, result in enumerate(results):
            if result is not None:
                for key, value in result.items():
                    self.df.loc[i, key] = value

        # DEBUG
        print(f'Columns : {self.df.columns}')

        # Assign the DataFrame with new scores
        self.df_score = self.df

        pass

    def save_table(self,) -> None:
        """Store processed table
        """
        
        # store table
        self.df_score.to_csv(self.db_dst_path, sep='|')

        pass 

    def load_table(self,) -> None:
        """Store processed table
        """
        assert self.db_dst_path.is_file(), ""
        # store table
        self.df_score = pd.read_csv(self.db_dst_path, sep='|')

        pass