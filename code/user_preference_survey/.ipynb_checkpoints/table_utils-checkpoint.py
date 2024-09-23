import pandas as pd
from pathlib import Path
import random
import yaml
import itertools
import numpy as np
import pymupdf
from PIL import Image
import io

from sampling_utils import sample_k_pairs

def extract_page_as_image(pdf_path, page_idx, dst_store_dir:Path=Path('./tmp_imgs')):
    """
    Extracts page `page_idx` from PDF in `pdf_path` and stores it as image
    """
    
    pdf_path = Path(pdf_path)
    assert pdf_path.is_file(), "pdf_path is not path to a file"
    assert pdf_path.suffix=='.pdf', f"`pdf_path` exists but is not PDF since suffix: {pdf_path.suffix}"
    dst_store_dir = Path(dst_store_dir)
    assert dst_store_dir.is_dir(), "Destination dir to store image into does not exist."
    
    # open the PDF
    pdf_document = pymupdf.open(pdf_path)
    
    # select the page (page_idx is zero-based)
    page = pdf_document.load_page(page_idx)
    
    # convert the page to a pixmap (image)
    pix = page.get_pixmap()
    
    # convert the pixmap to an image file (e.g., PNG)
    image_path = f"page_{page_idx + 1}.png"
    pix.save(image_path)
    
    # close the PDF document
    pdf_document.close()
    
    return image_path

# inputs
def append_IMAGE_and_CHOICE_IDs(DF:pd.DataFrame, n_user_groups:int = 12):
    """
    Given a Dataframe of pre-selected choice tuples, derives the 
    - IMAGE_ID (to uniquely identify the page's image) and 
    - CHOICE_ID (o uniquely identify all involves assets for a choice - image, pdf, )
    """
    
    required_columns = ['path', 'page', 'subset']
    for req_col in required_columns:
        assert req_col in DF.columns, "..."
    
    # Image ID
    DF['IMAGE_ID'] = DF['path'].map(pdf_path_id_dict).astype(str) + DF['page'].astype(str).str.zfill(3) + DF['subset'].map(subset_id_dict).astype(str)
    
    # Choice ID
    unique_paths = set(DF['path'])
    img_to_user_mapping = {pdf_file_path : (file_idx % n_user_groups) for file_idx,pdf_file_path in enumerate(unique_paths)}
    DF['user_group'] = DF['path'].map(img_to_user_mapping).astype(str).str.zfill(2)
    DF['occurence_id'] = DF.groupby('user_group').cumcount()
    
    # cc
    DF['CHOICE_ID'] = DF['IMAGE_ID'] + DF['path'].map(img_to_user_mapping).astype(str).str.zfill(2) + DF['parser_left'].map(parser_id_dict).astype(str) + DF['parser_right'].map(parser_id_dict).astype(str)

    return DF

def get_columns_to_be_extracted_for_choice(parser_left:str, parser_right:str):
    """
    Given two parser names (left & right), respectively, returns column names that are to be extracted from df
    """

    columns = ['path', 'page', 'subset', 'group', 'mean_bleu_score', 'pdf_id', parser_left, parser_right]

    # add 
    for parser in [parser_left, parser_right]:
        columns += [f'bleu_{parser}', f'rouge_{parser}', f'bleu_{parser}']

    return columns

# inputs
def append_IMAGE_and_CHOICE_IDs(DF:pd.DataFrame, n_user_groups:int = 12):
    """
    Given a Dataframe of pre-selected choice tuples, derives the 
    - IMAGE_ID (to uniquely identify the page's image) and 
    - CHOICE_ID (o uniquely identify all involves assets for a choice - image, pdf, )
    """

    # load mappings
    pdf_path_id_dict, parser_id_dict, subset_id_dict, options_per_img_id = load_id_dictionaries()
    
    required_columns = ['path', 'page', 'subset']
    for req_col in required_columns:
        assert req_col in DF.columns, "..."
    
    # Image ID
    DF['IMAGE_ID'] = DF['path'].map(pdf_path_id_dict).astype(str) + DF['page'].astype(str).str.zfill(3) + DF['subset'].map(subset_id_dict).astype(str)
    
    # Choice ID
    unique_paths = set(DF['path'])
    img_to_user_mapping = {pdf_file_path : (file_idx % n_user_groups) for file_idx,pdf_file_path in enumerate(unique_paths)}
    DF['user_group'] = DF['path'].map(img_to_user_mapping).astype(str).str.zfill(2)
    DF['occurence_id'] = DF.groupby('user_group').cumcount()
    
    # cc
    DF['CHOICE_ID'] = DF['IMAGE_ID'] + DF['path'].map(img_to_user_mapping).astype(str).str.zfill(2) + DF['parser_left'].map(parser_id_dict).astype(str) + DF['parser_right'].map(parser_id_dict).astype(str)

    return DF

def load_id_dictionaries(path_of_pdf_path_id_dict = Path('./dicts/pdf_path_id_dict.yaml'),
                         path_of_parser_id_dict = Path('./dicts/parser_id_dict.yaml'),
                         path_of_subset_id_dict = Path('./dicts/subset_id_dict.yaml')):
    """
    Loads dicts that uniquely map
    - pdf_path (e.g. arxiv/pdf/1848.90v1.pdf) -> pdf_path_id
    - parser (e.g. `grobid`) -> parser_id
    - 
    """
    
    # - pdf_path id
    with open(path_of_pdf_path_id_dict, 'r') as file:
        pdf_path_id_dict = yaml.safe_load(file)
    
    # - parser id
    with open(path_of_parser_id_dict, 'r') as file:
        parser_id_dict = yaml.safe_load(file)
    
    # - subset id
    with open(path_of_subset_id_dict, 'r') as file:
        subset_id_dict = yaml.safe_load(file)

    # so small, can be local
    options_per_img_id = {'A' : 0, 'B' : 1}
    
    return pdf_path_id_dict, parser_id_dict, subset_id_dict, options_per_img_id

def get_frames_of_choices_raw(p_df = Path('/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/database/pagewise/pagewise_parser_output_proc.csv'),
                              p_split = Path('/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/split_official/pymupdf.yaml')):
    """
    Loads pagewise dataframe an returns 3 dataframes (train, test, val) according to previous split WITH sampled parser choices

    TODO:
    - `df_test` should be loaded statically (not sampled) as it now entails `tesseract` and will entail `gpt4` - not in the original input
       --> see ./augment_test_tables for these routines.
    
    """

    # load metadata split
    with open(str(p_split), 'r') as file:
        meta_split = yaml.safe_load(file)
    reversed_dict = {item: key for key, value in meta_split.items() for item in value}
    
    # load text & metrics
    df = pd.read_csv(p_df, sep='|')
    
    # assign column `train`, `test`, `val`
    df_meta = pd.DataFrame({'path' : reversed_dict.keys(), 'subset' : reversed_dict.values()})
    
    # merge `subsets` into df
    df = pd.merge(df, df_meta[['path', 'subset']], on='path', how='left')
    df['subset'] = df['subset'].fillna('-')
    
    # parser output
    text_columns = ['html', 'grobid', 'marker', 'nougat', 'pymupdf', 'pypdf']
    score_columns = [f'bleu_{txt_col}' for txt_col in text_columns if txt_col!='html']
    meta_columns = ['path', 'subset', 'page']
    # - columns
    columns = meta_columns + text_columns + score_columns
    
    # df_data
    df_data = df[columns]
    
    # - assign faux-score column
    df_data = df_data.assign(bleu_html=[1.0] * len(df_data))
    
    # subset: only pages (1-4)
    df = df[df['page'].isin([0, 1])]
    
    # assign group
    df['group'] = np.random.choice(['A', 'B'], size=len(df), p=[3./8, 5./8])
    
    # add mean bleu
    bleu_columns = df.filter(like='bleu_').columns
    bleu_columns = [col for col in df.filter(like='bleu_').columns if (not(col.endswith('_norm')) and ('pymupdf' not in col) and ('pypdf' not in col) and ('grobid' not in col))]
    df['mean_bleu_score'] = df[bleu_columns].mean(axis=1)
    # - filter by quality
    df = df[df['mean_bleu_score'] > 0.6]
    
    # ID (every PDF gets a 4-digit ID)
    df = df.assign(pdf_id=range(100_000, 100_000+len(df)))
    
    # sample pages 
    df_train = df[df['subset']=='train'].sample(1500, random_state=45)
    df_test = df[df['subset']=='test'].sample(1000, random_state=45) # TO be chnaged
    df_val = df[df['subset']=='val'].sample(450, random_state=45) 
    
    return df_train, df_test, df_val

def get_sampled_choices(df:pd.DataFrame,
                        mode:str,
                        perf_sens_parsers:list[str]=['pypdf', 'grobid'],
                        p_root=Path('/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/joint'),
                        n_user_groups:int=12):
    """
    Args:
    - mode `train`, `val`, `test` (slightly influences sampling)
    - p_root : Root directories where PDFs (identified by relative paths in `path` reside)
    Given a sub-sampled pandas Dataframe: 
    - filters according to quality of parsed text to get meaningful annotation (excl. `-` - text parsings)
    - 
    
    """
    assert mode in {'train', 'test', 'val'}, "One of these"

    # init dictionaries
    pdf_path_id_dict, parser_id_dict, subset_id_dict, options_per_img_id = load_id_dictionaries()
    

    # group A : 5 samples
    train_5_tuples = [sample_k_pairs(5, mode) for _ in range(10 * len(df))]
    train_3_tuples = [sample_k_pairs(3, mode) for _ in range(10 * len(df))]
        
    # init
    i, j = 0,0

    # DEBUG
    print(f'len(df)={len(df)}, len(train_5_tuples): {len(train_5_tuples)}, len(train_3_tuples)={len(train_3_tuples)}')

    # append
    df_row_list = []
    
    # loop df and choice_tuples
    while ((i+j) < len(train_3_tuples)) and (i < len(df)):
        # 
        quality_check_passed = True
        
        # extract PDF's meta
        pdf_path = p_root / df.iloc[i]['path']
        group = df.iloc[i]['group']
        page_idx = int(df.iloc[i]['page'])
    
        # current tuple
        cur_tuples = train_5_tuples[i+j] if group=='A' else train_3_tuples[i+j]
    
        # identify current parsers involved in bin. choices
        unique_cur_parsers = [item for tup in cur_tuples for item in tup]
        unique_cur_parsers = list(set(unique_cur_parsers))
    
        # run quality check: any parser where `-` would be shown?
        for sens_parser in perf_sens_parsers:
            if sens_parser in unique_cur_parsers:
                if (str(df.iloc[i][sens_parser])=='-'):
                    quality_check_passed = False
        
        # passed quality check:
        if quality_check_passed:
            # - extract & store images (once)
            if pdf_path.is_file():
                doc = pymupdf.open(pdf_path) # open the PDF
                pixmap = doc[page_idx].get_pixmap(dpi=100)
                page_img = pixmap.tobytes()
                image = Image.open(io.BytesIO(page_img))
                # store
            else:
                print(f'Skip. PDF path invalid: {pdf_path}')
                i+=1
                continue
            
            # - compile choices (add image every time)
            for cur_tuple in cur_tuples:
                (parser_left, parser_right) = cur_tuple
    
                # store data
                df_row = df.copy().iloc[i][get_columns_to_be_extracted_for_choice(parser_left, parser_right)]
                # append
                df_row['parser_left'] = parser_left
                df_row['parser_right'] = parser_right
                df_row['image_path'] = '...'
    
                # new columns names:
                df_row.index = [col.replace(parser_left, 'left').replace(parser_right, 'right') for col in df_row.index]
                df_row = df_row.to_frame().T
                
                # append
                df_row_list.append(df_row)
    
            # - update to new df row
            j+=1 # next choice
            i+=1 # next row
        else:
            #print(f'... failed. unique_cur_parsers=, {unique_cur_parsers}')
            j+=1 # look at new choice
            # skip this data row altogether with probability: 25% -> good trade-off
            if random.random() < 1./3:
                i+=1

    # combine
    DF = pd.concat(df_row_list)

    # add IMAGE_ID & CHOICE ID
    DF = append_IMAGE_and_CHOICE_IDs(DF, n_user_groups)
    
    return DF