from datasets import load_from_disk, concatenate_datasets, Dataset
import os
from pathlib import Path
import pandas as pd
import yaml
import json

def clean_invalid_chars(df):
    """
    Helper function that re-formats string columns in a Pandas DataFrame to allow proper storage as HF Dataset
    """
    # Iterate through all string columns and clean invalid characters
    for col in df.select_dtypes(include=[object]):
        df[col] = df[col].apply(lambda x: x.encode('utf-8', 'replace').decode('utf-8') if isinstance(x, str) else x)
    return df

def load_predefined_split(df:pd.DataFrame, p_split_yaml_path:Path):
    """
    Retrieves sampled train/val/test split from YAML, splits `df` accordingly

    - df[pd.DataFrame] : 
    - p_split_yaml_path [Path] : Path to YAML that stores the split (keys `train`, `val`, `test` with list of paths as values)
    
    """
    
    # convert
    p_split_yaml_path = Path(p_split_yaml_path)

    # check
    assert p_split_yaml_path.is_file(), f"Not fiel with path {p_split_yaml_path}"
    assert str(p_split_yaml_path).endswith('.yaml'), f"File exists but is not YAML: {p_split_yaml_path}"
    assert 'path' in df.columns, "`path` must be one of `df`s columns but it's not present."
    
    # Load the YAML file
    with open(p_split_yaml_path, "r") as file:
        split_dict = yaml.safe_load(file)

    # subset
    df_train = df[df['path'].isin(split_dict['train'])]
    df_val = df[df['path'].isin(split_dict['val'])]
    df_test = df[df['path'].isin(split_dict['test'])]

    return df_train, df_val, df_test

def normalize_column(dataset, feature_name):
    """Unify authors
    """
    # Ensure the 'authors' column is a list of strings
    if isinstance(dataset.features[feature_name], Value):
        dataset = dataset.map(lambda example: {feature_name: [example[feature_name]]})
    return dataset


def compile_DatasetFrames(p_embeddings:Path,
                          p_response:Path,
                          parser:str,
                          normalized:bool,
                          predefined_split:bool,
                          p_split_yaml_path:Path):
    """
    Reads in embeddings (among other features) of the huggingface dataset directories and the responses (BLEU, ROUGE etc.) and merges the two and samples.
    """

    # convert
    p_embeddings = Path(p_embeddings)
    p_response = Path(p_response)
    
    assert Path(p_embeddings).is_dir(), "`p_embeddings` must be path to a directory" 
    assert len(os.listdir(p_embeddings)), "`p_embeddings` must contain directories (which themselves contain pyarrrow/huggingface datasets)" 
    assert p_response.is_file(), "File must exist"
    if not(predefined_split):
        assert str(p_response).endswith('.csv'), "`p_response` exists but file path is not to that of a CSV!"
        assert 0<=f_train<=1, "Frequency must be in [0.0, 1.0]"
    
    # Target columns
    target_columns = []
    for acc in ['bleu', 'rouge', 'car']:
        for pars in ['nougat', 'marker', 'pymupdf', 'grobid', 'pypdf']:
            target_columns.append(f"{acc}_{pars}" + ("_norm" if normalized else ""))

    # load 
    df_resp = pd.read_csv(p_response, sep='|')
    
    # - debug
    dataset_dir = p_embeddings / parser
    assert Path(dataset_dir).is_dir(), f"Inferred HF dataset path does not exist: {p_embeddings / parser}"
    
    # load all
    dataset = load_from_disk(str(dataset_dir))
    
    # -> pd.DataFrame
    df_emb = dataset.to_pandas()

    # fill NaNs of this particular parser
    df_emb['text'] = df_emb['text'].fillna('')
    
    # unify `path` representation
    df_emb['path'] = df_emb['path'].str.split('/').apply(lambda x: '/'.join(x[-3:]))

    # LEGACY: kill duplicates
    # separate rows with valid text (non-NaN) and those with NaN in 'text'
    #valid_text_df = df_emb.dropna(subset=['text'])
    #invalid_text_df = df_emb[df_emb['text'].isna()]
    #valid_text_df = valid_text_df.drop_duplicates(subset='path')
    #invalid_text_df = invalid_text_df[~invalid_text_df['path'].isin(valid_text_df['path'])]
    #df_emb = pd.concat([valid_text_df, invalid_text_df])
    # New: pagewise and regular

    # Check if 'page' column exists in the DataFrame
    if 'page' in df_emb.columns:
        # drop duplicates where both 'path' and 'page' are identical
        valid_text_df = df_emb.dropna(subset=['text'])
        invalid_text_df = df_emb[df_emb['text'].isna()]
        # drop
        valid_text_df = valid_text_df.drop_duplicates(subset=['path', 'page'])
        # retain invalid text rows where the 'path' and 'page' combination does not exist in valid_text_df
        invalid_text_df = invalid_text_df[~invalid_text_df[['path', 'page']].apply(tuple, axis=1).isin(valid_text_df[['path', 'page']].apply(tuple, axis=1))]
        # concatenate valid and invalid text DataFrames
        df_emb = pd.concat([valid_text_df, invalid_text_df])
    else:
        # if 'page' is NOT present, proceed with killing duplicates only based on 'path'xt'
        valid_text_df = df_emb.dropna(subset=['text'])
        invalid_text_df = df_emb[df_emb['text'].isna()]
        # drop duplicates based only on 'path'
        valid_text_df = valid_text_df.drop_duplicates(subset='path')
        # retain invalid text rows where the 'path' does not exist in valid_text_df
        invalid_text_df = invalid_text_df[~invalid_text_df['path'].isin(valid_text_df['path'])]
        # Concatenate valid and invalid text DataFrames
        df_emb = pd.concat([valid_text_df, invalid_text_df])
    
    # Optionally sort the final DataFrame based on 'path' and 'page' if 'page' exists
    if 'page' in df_emb.columns:
        df_emb = df_emb.sort_values(by=['path', 'page'], ascending=[True, True])
    else:
        df_emb = df_emb.sort_values(by=['path'], ascending=True)
    
    # merge the pandas DataFrame `df_resp` onto `df_dset` on 'path'
    df = pd.merge(left=df_emb, right=df_resp, on='path', how='left')

    
    # DataFrame
    df_all = df.dropna(subset=target_columns)

    # Add classes: 
    # `journal_cls` : source journal
    # define the mapping of strings to numerical values
    mapping = {'arxiv': 0, 'biorxiv': 1, 'medrxiv': 2, 'mdpi': 3, 'nature': 4, 'bmc': 5}
    # - create the 'journal_cls' column by extracting the third-to-last part of the 'path' and mapping it to numerical values
    df_all = df_all.assign(journal_cls=df_all['path'].str.split('/').str[-3].map(mapping))

    # `best_bleu_cls` best parser class
    parser_mapping = {'nougat': 0, 'marker': 1, 'pymupdf': 2, 'grobid': 3, 'pypdf': 4}
    # parse matrics
    for acc_metric in {'bleu', 'rouge', 'car'}:
        for normVal in {True, False}:
            norm_str = '_norm' if normVal else ''
            parser_idx = -2 if normVal else -1
            
            # name column
            max_column_strings = df_all[[f'{acc_metric}_nougat{norm_str}', f'{acc_metric}_marker{norm_str}', f'{acc_metric}_pymupdf{norm_str}', f'{acc_metric}_grobid{norm_str}', f'{acc_metric}_pypdf{norm_str}']].fillna(-float('inf')).idxmax(axis=1).str.split('_').str[parser_idx]
            
            # actual column
            df_all[f'best_{acc_metric}{norm_str}_cls'] = max_column_strings.map(parser_mapping)
    
    # Subset into `train`, `val`, `test` sets
    # - Pre-defined (retrieve from YAML file usually in `./meta_split`)
    if predefined_split:
        print('\nLoad pre-defined split...\n')
        df_train, df_val, df_test = load_predefined_split(df=df_all, p_split_yaml_path=p_split_yaml_path)
    # - Split manually
    else:
        print('\nGenerate manual split...\n')
        # - val/test
        df_test_and_val_1 = df_all[df_all['path'].str.contains('bmc/') | df_all['path'].str.contains('nature/')]
        
        # - train: exclude `nature`/`bmc`
        df_filtered = df_all.drop(df_test_and_val_1.index)
        # - stratified sampling
        df_filtered['prefix'] = df_filtered['path'].str.split('/').str[0]
        # - split
        df_all_2 = df_filtered.groupby('prefix', group_keys=False, as_index=False).apply(lambda x: x.sample(frac=f_train, random_state=seed_val))
        df_all_1 = df_filtered.drop(df_all_2.index)
        
        # re-merge
        df_test_and_val_1_and_2 = pd.concat([df_all_1, df_test_and_val_1], axis=0, ignore_index=True)
        # -train
        df_train = df_all_2
        # - shuffle
        df_test_and_val_1_and_2 = df_test_and_val_1_and_2.sample(frac=1, random_state=42)
        # - val
        df_val = df_test_and_val_1_and_2.loc[0:len(df_test_and_val_1_and_2) // 2,:]
        # - test
        df_test = df_test_and_val_1_and_2.drop(df_val.index)
        
        # remove overlaps explicitely
        df_val = df_val[~df_val['path'].isin(df_train['path'])]
        df_test = df_test[~df_test['path'].isin(df_train['path'])]
        df_test = df_test[~df_test['path'].isin(df_val['path'])]
    
    # Check overlap
    # - compute overlap
    train_val_overlap = pd.Series(list(set(df_train['path']).intersection(set(df_val['path']))))
    train_test_overlap = pd.Series(list(set(df_train['path']).intersection(set(df_test['path']))))
    val_test_overlap = pd.Series(list(set(df_val['path']).intersection(set(df_test['path']))))
    # - print the overlaps (if any)
    print(f"Train-Val Overlap: {len(train_val_overlap)}")
    print(f"Train-Test Overlap: {len(train_test_overlap)}")
    print(f"Val-Test Overlap: {len(val_test_overlap)}")

    print("df_train, df_test, df_val")
    
    return df_train, df_test, df_val

def generate_HF_dataset(parser:str,
                        src_jsonl_dir:Path,
                        dst_hf_dir:Path,
                        store_flag:bool):
    '''
    Creates Huggingface Dataset from jsonls that came from parsed PDFs (regardless of parser)

    - src_jsonl_dir[Path|str]: Directory in which parser=specific subdirectories `.../joint_to_{parser}/parsed_pdfs` reside
    - dst_hf_dir[Path|str]: Directory to which you store Huggingface datasets (chosen to be identifiable for subsequence `run_regression.py` routine) 
    - save_flag[bool]: Indicate if you want to store (& potentially overwrite)
    '''

    # convert
    src_jsonl_dir = Path(src_jsonl_dir)
    dst_hf_dir = Path(dst_hf_dir)
    
    # function reading in
    assert src_jsonl_dir.is_dir(), "Source directory path in which the parser subidrectories (with jsonls doo not exist)."
    assert (src_jsonl_dir / f'joint_to_{parser}/parsed_pdfs').is_dir(), f"Source root directory DOES exist. Regardless, there is no subdirectory {(src_jsonl_dir / f'joint_to_{parser}/parsed_pdfs')}"
    assert dst_hf_dir.is_dir(), "Destination for huggingface dataset objects does not exist"
    assert parser in {'pymupdf', 'pypdf', 'nougat', 'marker', 'grobid'}, "Input must be one of the parsers."
    
    # (create) dir
    p_dest = Path(dst_hf_dir / parser)
    p_dest.mkdir(parents=True, exist_ok=True)
    
    # jsonl source path
    p_json = Path(src_jsonl_dir / f'joint_to_{parser}/parsed_pdfs')
    jsonl_files = [p_json / f for f in os.listdir(p_json) if f.endswith('.jsonl')]
    
    # rows
    rows = []
    
    # read-in all data entries
    for jsonl_f in jsonl_files:
        with open(jsonl_f, 'r') as f:
            for line in f:
                # load
                r=json.loads(line)
                
                # append
                rows.append(r)
    
    # store
    df = pd.DataFrame(rows)

    if 'metadata' in df.columns:
        # merge `metadata` as sep columns
        metadata_df = pd.json_normalize(df['metadata'])
        df_combined = df.drop(columns=['metadata']).join(metadata_df)
        df_combined.fillna(value=pd.NA, inplace=True)
        
        # rename
        df = df_combined.copy()

    # clean
    df_clean = clean_invalid_chars(df)
    
    # -> HF Dataset
    dset = Dataset.from_pandas(df_clean)
    
    # store
    if store_flag:
        dset.save_to_disk(str(p_dest))
        print(f"Parser: `{parser}` ... Stored {len(dset)} datapoints at {str(p_dest)}.")

    pass