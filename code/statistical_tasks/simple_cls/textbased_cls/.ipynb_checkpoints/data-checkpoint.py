from pathlib import Path
import pandas as pd
import sys
import os
import time
import numpy as np
import inspect

import numpy as np
import pandas as pd
from pathlib import Path

import fasttext
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

# source of data_utils
sys.path.append(os.path.join('/home/siebenschuh/Projects/dataprep/code/DPO'))

from data_utils import compile_DatasetFrames

def find_variable_name(df):
    '''
    Returns name of dataframe
    '''
    for name, value in inspect.currentframe().f_back.f_locals.items():
        if value is df:
            return name
    return None


def load_data(parser:str = 'pymupdf'):
    """
    Returns 3 dataframes (df_train, df_test, df_val)
    """
    # path constants
    p_embeddings_root_dir = Path('/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/embeddings/emb_by_model')
    p_response_csv_path = Path('/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/database/parser_metrics_without_text_output.csv')
    normalized = False
    predefined_split = True
    p_split_yaml_path = Path('/home/siebenschuh/Projects/dataprep/code/DPO/meta_split/pymupdf.yaml')
    
    # compile the dataset frames (train/val/test) using `compile_DatasetFrames`
    df_train, df_test, df_val = compile_DatasetFrames(
        p_embeddings=p_embeddings_root_dir,
        p_response=p_response_csv_path,
        parser=parser,
        normalized=normalized,
        predefined_split=predefined_split,
        p_split_yaml_path=p_split_yaml_path
    )
    
    return df_train, df_test, df_val


def process_data(df_train: pd.DataFrame,
                 df_test: pd.DataFrame,
                 df_val: pd.DataFrame,
                 mode: str,
                 score: str,
                 n_max_chars: int,
                 parsers: list[str],
                 max_features: int = 2500,
                 tmp_dir: str = "./tmp"):
    """
    Processes a dataframe `df_` into the format (X,y)
    - mode:str Type of processing
        - fasttext: Represent text using fastText embeddings
        - countvectorizer: Represent text using CountVectorizer
        - llm: Represent text using LLM embeddings from Hugging Face
    """
    # Validate arguments
    assert mode in {'countvectorizer', 'fasttext', 'llm'}, "Must be one of those"
    assert score in {'bleu', 'rouge', 'car'}, "Only these metrics are supported."
    assert max_features > 0, "Maximum number of features must be positive."
    assert n_max_chars > 0, "Maximum number of characters considered `n_max_chars` should be positive."
    allowed_parsers = {'grobid', 'marker', 'nougat', 'pymupdf', 'pypdf'}
    assert set(parsers).issubset(allowed_parsers), f"`parsers` list must be subset of `allowed_parsers`={allowed_parsers}"

    # Check columns
    for parser in parsers:
        for df in [df_train, df_test, df_val]:
            assert f"{score}_{parser}" in df.columns, f"Column `{score}_{parser}` not found in df `{df.name}`."

    # Raw X
    X_train = df_train['text'].str[:n_max_chars].str.replace('\n', ' ')
    X_val = df_val['text'].str[:n_max_chars].str.replace('\n', ' ')
    X_test = df_test['text'].str[:n_max_chars].str.replace('\n', ' ')

    # Process X
    if mode == 'countvectorizer':
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)
        X_test_vec = vectorizer.transform(X_test)
    elif mode == 'fasttext':
        # Create temporary directory if it doesn't exist
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)

        # Save the training data to a file
        train_file_path = Path(tmp_dir) / "train.txt"
        with open(train_file_path, "w") as f:
            f.write("\n".join(X_train.tolist()))
        
        # Load fastText model
        ft_model = fasttext.train_unsupervised(str(train_file_path), model='skipgram')
        
        # Convert text to vectors
        def text_to_vector(text_series):
            return np.vstack([ft_model.get_sentence_vector(text) for text in text_series])
        
        X_train_vec = text_to_vector(X_train)
        X_val_vec = text_to_vector(X_val)
        X_test_vec = text_to_vector(X_test)

    elif mode == 'llm':
        # Load the LLM model from Hugging Face
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Convert text to vectors
        def text_to_vector_llm(text_series):
            return np.vstack([model.encode(text) for text in text_series])

        X_train_vec = text_to_vector_llm(X_train)
        X_val_vec = text_to_vector_llm(X_val)
        X_test_vec = text_to_vector_llm(X_test)

    # Target columns
    target_columns = [f'{score}_{parser}' for parser in parsers]
    y_train = df_train[target_columns]
    y_val = df_val[target_columns]
    y_test = df_test[target_columns]

    print('(X_train_vec, y_train), (X_val_vec, y_val), (X_test_vec, y_test)')
    
    return [(X_train_vec, y_train), (X_val_vec, y_val), (X_test_vec, y_test)]











