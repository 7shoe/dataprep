import argparse
import pandas as pd

import pandas as pd
import dask.dataframe as dd

from dask import delayed, compute
from dask.distributed import Client

import re
import sys
import csv
import time
import string
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.translate import bleu_score
from rapidfuzz import fuzz
from rouge_score import rouge_scorer

# __init__
def _normalize_(x: str, remove_latex_flag: bool = True) -> str:
    REMOVE_PUNCT = str.maketrans("", "", string.punctuation)

    if remove_latex_flag:
        x = _remove_latex_(x)

    x = x.translate(REMOVE_PUNCT)
    x = re.sub(r"\s+", " ", x)
    x = x.lower()
    x = x.strip()

    return x

def _extract_latex_(text):
    if pd.isna(text):
        return []

    latex_pattern = re.compile(r'(\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)|\\begin\{.*?\}.*?\\end\{.*?\})', re.DOTALL)

    latex_expressions = latex_pattern.findall(text)

    stripped_text = latex_pattern.sub('', text).strip()

    return latex_expressions

def _remove_latex_(text: str) -> str:
    if pd.isna(text):
        return ''
    try:
        text = re.sub(r'\\[a-zA-Z]+\{(.*?)\}', r'\1', text)
        text = re.sub(r'\$(.*?)\$', r'\1', text)
        text = re.sub(r'\$\$(.*?)\$\$', r'\1', text)
        text = re.sub(r'\\\[(.*?)\\\]', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'\{|\}', '', text)
    except:
        text = ''

    return text

# tokenize()
def safe_word_tokenize(text):
    if pd.isna(text):
        return []
    return word_tokenize(text)

def parallel_tokenize(series):
    """
    Tokenize a Dask Series in parallel.
    """
    return series.map(safe_word_tokenize, meta=('token', 'object'))

def extract_parts(token_list):
    """
    Extract sublist (of tokens) post tokenization of the text column.
    """
    length = len(token_list)
    first_10 = token_list[:max(1, length // 10)]
    mid_start = length // 2 - max(1, length // 20)
    mid_end = length // 2 + max(1, length // 20)
    middle_10 = token_list[mid_start:mid_end]
    last_10 = token_list[-max(1, length // 10):]
    
    return first_10, middle_10, last_10

# compute_metrics
# - BLEU
def compute_bleu(row, reference_col, candidate_col):
    return bleu_score.sentence_bleu(
        [row[reference_col]], 
        row[candidate_col], 
        smoothing_function=bleu_score.SmoothingFunction().method1
    )

def parallel_bleu(df, reference_col, candidate_col):
    return df.apply(lambda row: compute_bleu(row, reference_col, candidate_col), axis=1, meta=('x', 'f8'))

# - CAR (Character Accuracy Rate)
def compute_approx_car(row, reference_col, candidate_col):
    '''
    Complement of character error rate (CER); hence: character accuracy rate (CAR)
    '''
    similarity = fuzz.ratio(row[reference_col], row[candidate_col])
    return similarity / 100.0

# Compute CAR scores in parallel for the entire dataframe
def parallel_car(df, reference_col, candidate_col):
    return df.apply(lambda row: compute_approx_car(row, reference_col, candidate_col), axis=1, meta=('car_score', 'f8'))

# - ROUGE-1
def compute_rouge1(row, reference_col, candidate_col, scorer):
    reference_text = str(row[reference_col]) if pd.notnull(row[reference_col]) else ""
    candidate_text = str(row[candidate_col]) if pd.notnull(row[candidate_col]) else ""
    score = scorer.score(reference_text, candidate_text)
    return score['rougeL'].fmeasure

def parallel_rouge1(df, reference_col, candidate_col, scorer):
    return df.apply(lambda row: compute_rouge1(row, reference_col, candidate_col, scorer), axis=1, meta=('x', 'f8'))

class DaskResponseTable:
    def __init__(self, db_src_filename: str, db_dst_filename: str, chunk_index: int = -1,
                 chunk_size: int = -1, num_cores: int = 100, overwrite_flag: bool = False,
                 parser_columns: list[str] = ['pymupdf', 'nougat', 'grobid'],
                 root_dir: Path = Path('/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/database')) -> None:
        """
        Initialize the DaskResponseTable with the specified parameters.
        """
        max_int = sys.maxsize
        while True:
            try:
                csv.field_size_limit(max_int)
                break
            except OverflowError:
                max_int = int(max_int / 10)

        root_dir = Path(root_dir)
        assert root_dir.is_dir(), f"Path `root_dir` does not exist or is not a directory. Invalid directory: {root_dir}"

        self.db_src_path = Path(root_dir) / db_src_filename
        self.db_dst_path = Path(root_dir) / db_dst_filename
        self.overwrite_flag = overwrite_flag
        self.parser_columns = parser_columns
        self.chunk_index = chunk_index
        self.chunk_size = chunk_size
        self.num_cores = num_cores

        assert self.db_src_path.is_file(), f"Source CSV path invalid. No such path: {self.db_src_path}"
        if not self.overwrite_flag:
            assert not self.db_dst_path.is_file(), f"Destination Path invalid. File already exists at path: {self.db_dst_path}"

        self.df = dd.read_csv(self.db_src_path, sep='|', sample=5000000, sample_rows=10, on_bad_lines='skip', engine='python')

        for col in self.df.columns:
            if self.df[col].dtype == 'object' and col not in {'html', 'path'}:
                self.df[col] = self.df[col].fillna('')

        df_proc = self.df.dropna()

        if self.chunk_index != -1 and self.chunk_size != -1:
            assert self.chunk_index >= 0, f"`chunk_index` should be non-negative (or -1) but is {self.chunk_index}"
            assert self.chunk_size > 0, f"`chunk_size` should be positive (or -1) but is {self.chunk_size}"

            db_dst_filename = Path(db_dst_filename).stem + f'_{self.chunk_index}-{len(self.df) // self.chunk_size}' + Path(db_dst_filename).suffix
            self.db_dst_path = Path(root_dir) / db_dst_filename

            i_start, i_end = self.chunk_index * self.chunk_size, min((self.chunk_index + 1) * self.chunk_size, len(self.df))
            if i_start >= len(self.df):
                raise ValueError(f'i_start index exceeds length of Dataframe: i_start={i_start} for len(df_proc)={len(df_proc)}')
            self.df = self.df.loc[i_start:i_end]

            print(f'len(df_proc): {len(self.df)}, i_start/i_end: ', i_start, i_end)

        print(f'DF loaded, nrows : {len(self.df)} after removing NANs & subsetting')

        for parser_col in ['html'] + self.parser_columns:
            self.df[f'{parser_col}_norm'] = self.df.apply(lambda row: _normalize_(row[parser_col]), axis=1, meta=('norm', 'object'))

        for parser_col in ['html'] + self.parser_columns:
            self.df[f'{parser_col}_latex'] = self.df.apply(lambda row: _extract_latex_(row[parser_col]), axis=1, meta=('latex', 'object'))

        self.client = Client(n_workers=self.num_cores)

    def tokenize(self,):
        """
        Tokenize html/parser text columns in the DataFrame.
        """
        for parser_col in ['html'] + self.parser_columns:
            print(f'Tokenizing {parser_col} ... ')
            self.df[f'{parser_col}_token'] = parallel_tokenize(self.df[parser_col])
            self.df[f'{parser_col}_norm_token'] = parallel_tokenize(self.df[f'{parser_col}_norm'])
            print('... completed!\n')

        def extract_and_assign(series, col_prefix):
            length = series.map(len, meta=('length', 'int'))
            self.df[f'{col_prefix}Beg_token'] = series.map(lambda x: x[:max(1, len(x) // 10)], meta=('token', 'object'))
            self.df[f'{col_prefix}Mid_token'] = series.map(lambda x: x[len(x) // 2 - max(1, len(x) // 20):len(x) // 2 + max(1, len(x) // 20)], meta=('token', 'object'))
            self.df[f'{col_prefix}End_token'] = series.map(lambda x: x[-max(1, len(x) // 10):], meta=('token', 'object'))

        for parser_col in ['html'] + self.parser_columns:
            extract_and_assign(self.df[f'{parser_col}_token'], parser_col)
            extract_and_assign(self.df[f'{parser_col}_norm_token'], f'{parser_col}_norm')

        pass

    def compute_metrics(self):
        """
        Compute metrics BLEU, CAR, and ROUGE etc.
        """
        for parser_col in self.parser_columns:
            print(f'Computing BLEU for {parser_col} ... ')
            self.df[f'{parser_col}_bleu'] = parallel_bleu(self.df, f'{parser_col}_token', 'html_token')
            self.df[f'{parser_col}_norm_bleu'] = parallel_bleu(self.df, f'{parser_col}_norm_token', 'html_norm_token')
            print(f'... completed!\n')

        for parser_col in self.parser_columns:
            print(f'Computing CAR for {parser_col} ... ')
            self.df[f'{parser_col}_car'] = parallel_car(self.df, f'{parser_col}_token', 'html_token')
            self.df[f'{parser_col}_norm_car'] = parallel_car(self.df, f'{parser_col}_norm_token', 'html_norm_token')
            print(f'... completed!\n')

        if False:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            for parser_col in self.parser_columns:
                print(f'Computing ROUGE for {parser_col} ... ')
                self.df[f'{parser_col}_rougeL'] = parallel_rouge1(self.df, f'{parser_col}', 'html', scorer)
                self.df[f'{parser_col}_norm_rougeL'] = parallel_rouge1(self.df, f'{parser_col}_norm', 'html_norm', scorer)
                print(f'... completed!\n')

        pass

    def save(self):
        """
        Save the computed metrics DataFrame to a CSV file with '|' as the separator.
        """
        try:
            self.df.to_csv(self.db_dst_path, sep='|', index=False, single_file=True)
        except:
            self.df.to_csv(self.db_dst_path, sep='|', index=False)
        print(f"Metrics saved to {self.db_dst_path}.")
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="Dask Response Table processing")

    parser.add_argument('--db_src_filename', type=str, default='parser_output_raw.csv', help="Source CSV file name")
    parser.add_argument('--db_dst_filename', type=str, default='parser_output_with_metricsNEW.csv', help="Destination CSV file name")
    parser.add_argument('--chunk_index', type=int, default=0, help="Index of the chunk to process")
    parser.add_argument('--chunk_size', type=int, default=4000, help="Size of the chunk to process")
    parser.add_argument('--num_cores', type=int, default=8, help="Number of cores to use for Dask")
    parser.add_argument('--overwrite_flag', action='store_true', help="Flag to overwrite existing output file")
    parser.add_argument('--parser_columns', nargs='+', default=['pymupdf', 'nougat', 'grobid', 'pypdf', 'marker'], help="Columns to process")
    parser.add_argument('--root_dir', type=str, default='/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/database', help="Root directory path")

    return parser.parse_args()

def main():
    args = parse_args()

    t = DaskResponseTable(
        db_src_filename=args.db_src_filename,
        db_dst_filename=args.db_dst_filename,
        chunk_index=args.chunk_index,
        chunk_size=args.chunk_size,
        num_cores=args.num_cores,
        overwrite_flag=args.overwrite_flag,
        parser_columns=args.parser_columns,
        root_dir=Path(args.root_dir)
    )

    t.tokenize()
    t.compute_metrics()
    t.save()

if __name__ == '__main__':
    main()

def main():
    t.tokenize()
    t.compute_metrics()
    t.save()