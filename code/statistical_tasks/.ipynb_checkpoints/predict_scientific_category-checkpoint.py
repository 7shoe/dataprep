import pandas as pd
import os
import torch
from pathlib import Path
from datasets import Dataset
from transformers import pipeline
import argparse
import math

def classify_batch(batch, classifier, labels, sub_labels):
    # Classify the titles into the main scientific category
    category_classifications = classifier(batch['title'], candidate_labels=labels, multi_label=False)
    top_categories = [classification['labels'][0] for classification in category_classifications]
    category_scores = [classification['scores'][0] for classification in category_classifications]
    
    # Classify the titles into the scientific sub-category
    subcategory_classifications = classifier(batch['title'], candidate_labels=sub_labels, multi_label=False)
    top_subcategories = [classification['labels'][0] for classification in subcategory_classifications]
    subcategory_scores = [classification['scores'][0] for classification in subcategory_classifications]
    
    return {
        'predicted_category': top_categories,
        'category_confidence': category_scores,
        'predicted_subcategory': top_subcategories,
        'subcategory_confidence': subcategory_scores
    }

def main(chunk_index, model_name):
    # Read the CSV file
    p = Path('/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/database/meta_raw_table.csv')
    df = pd.read_csv(p, sep='|')

    # DEBUG
    #df = df.loc[0:250,]
    #print('!!! \n This is running on a limited subset of DATA...DEBUG MODE !!!')

    # fill NaN
    df['prim_cat'] = df['prim_cat'].fillna('Unknown')
    df['journal_ref'] = df['journal_ref'].fillna('Unknown')
    df['journal_name'] = df['path'].str.split('/').str[0]
    df['journal_name'] = df['journal_name'].fillna('Unknown')
    
    # Get titles without primary category
    title_list = list(df[(~df['title'].isna())]['title'])
    path_list = list(df[(~df['pdf_path'].isna())]['pdf_path'])
    pdf_path_list = list(df[(~df['pdf_path'].isna())]['pdf_path'])
    journal_name_list = list(df[(~df['journal_name'].isna())]['journal_name'])
    journal_ref_list = list(df[(~df['journal_ref'].isna())]['journal_ref'])
    prim_cat_list = list(df[(~df['prim_cat'].isna())]['prim_cat'])
    
    # Determine the size of each chunk
    num_chunks = 5
    chunk_size = math.ceil(len(title_list) / num_chunks)
    
    # Split the title list into chunks and select the required chunk
    start_index = chunk_index * chunk_size
    end_index = start_index + chunk_size
    
    # content to be extracted...
    title_chunk = title_list[start_index:end_index]
    path_chunk = path_list[start_index:end_index]
    primCat_chunk = prim_cat_list[start_index:end_index]
    pdfName_chunk = pdf_path_list[start_index:end_index]
    journal_name_chunk = journal_name_list[start_index:end_index]
    journal_ref_chunk = journal_ref_list[start_index:end_index]
    
    # merge
    context_chunk = [f"Title of the paper: {title_text}. Journal name in which it was published: {journal_text}. Journal ref. list {journal_ref} Primary category: {primCat_text}" for (title_text, journal_text, journal_ref, primCat_text) in zip(title_chunk, journal_name_chunk, journal_ref_chunk, primCat_chunk)]
    
    # Set the Hugging Face cache directory
    os.environ['HF_HOME'] = '/eagle/projects/argonne_tpc/siebenschuh/HF_home'
    os.environ['TRANSFORMERS_CACHE'] = '/eagle/projects/argonne_tpc/siebenschuh/HF_cache'
    
    # Determine the device (GPU or CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the zero-shot classification pipeline
    # - load model (model_name)
    classifier = pipeline('zero-shot-classification', model=model_name, device=device)
    
    # Scientific categories
    labels = ["Biology", "Chemistry", "Physics", "Mathematics", "Computer Science", "Medicine", "Engineering", "Economics"]
    
    # Scientific sub-categories
    sub_labels = [
        "Astronomy", "Biotechnology", "Cancer Biology", "Cell Biology", "Climatology", "Evolutionary Biology",
        "Genetics", "Immunology", "Neuroscience", "Psychiatry", "Public Health", "Bioinformatics", "Biochemistry",
        "Bioengineering", "Biophysics", "Epidemiology", "Cardiovascular Medicine", "Oncology", "Neurology",
        "Mechanical Engineering", "Electrical Engineering", "Civil Engineering", "Chemical Engineering",
        "Environmental Engineering", "Industrial Engineering", "Aerospace Engineering", "Materials Science and Engineering",
        "Computer Engineering", "Architecture", "Analytical Chemistry", "Organic Chemistry", "Inorganic Chemistry",
        "Physical Chemistry", "Polymer Chemistry", "Microeconomics", "Macroeconomics", "Econometrics", "Development Economics",
        "Behavioral Economics", "Financial Economics", "Labor Economics", "Health Economics", "International Economics",
        "Environmental Economics", "Algebra", "Calculus", "Statistics", "Probability", "Geometry", "Topology",
        "Number Theory", "Mathematical Logic", "Discrete Mathematics", "Applied Mathematics", "Classical Mechanics",
        "Quantum Mechanics", "Thermodynamics", "Electromagnetism", "Optics", "Acoustics", "Nuclear Physics",
        "Condensed Matter Physics", "Particle Physics", "Astrophysics"
    ]
    
    # convert the title chunk to a Dataset object
    dataset = Dataset.from_dict({"title": context_chunk})
    
    # apply the classification function to the dataset in batches
    classified_dataset = dataset.map(lambda batch: classify_batch(batch, classifier, labels, sub_labels), batched=True)
    
    # convert the result back to a DataFrame
    classified_titles = classified_dataset.to_dict()
    df_pred = pd.DataFrame(classified_titles)
    df_pred['path'] = path_chunk

    # convert model name
    model_name_str = model_name.replace('/', '').replace('.', '').replace('_', '')
    
    # save the DataFrame to a CSV file
    output_path = Path(f'/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/database/predicted_categories/df_pred_{model_name_str}_{chunk_index}_CONTENT.csv')
    df_pred.to_csv(output_path, index=False)
    print(f'Successfully saved predictions to {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify scientific titles into categories and subcategories.")
    parser.add_argument('--chunk_index', type=int, required=True, help="Index of the chunk to process (0, 1, 2, 3, 4).")
    parser.add_argument('--model_index', type=int, required=True, choices=[0, 1, 2, 3, 4], help="Integer index of (5) Huggingface model retrieving  {institution}/{model_id}")

    # CONST MODEL LIST
    #model_list = ['typeform/distilbert-base-uncased-mnli', 'valhalla/distilbart-mnli-12-3', 'cross-encoder/nli-deberta-v3-large', 
    #              'facebook/bart-large-mnli', 'cross-encoder/nli-deberta-v3-large']
    # NEW
    model_list = ['allenai/scibert_scivocab_uncased', 
                  'google/flan-t5-xxl', 
                  'google/bigbird-roberta-large', 
                  'cross-encoder/nli-deberta-v3-large',
                  'facebook/bart-large-mnli']
    
    args = parser.parse_args()
    chunk_index = args.chunk_index
    model_index = args.model_index
    model_name = model_list[args.model_index]
    
    if chunk_index not in {0, 1, 2, 3, 4}:
        raise ValueError("chunk_index must be one of {0, 1, 2, 3, 4}")

    # entry
    main(chunk_index, model_name)