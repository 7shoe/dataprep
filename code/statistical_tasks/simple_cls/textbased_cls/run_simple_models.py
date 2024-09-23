from pathlib import Path
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import argparse
import random

from data import load_data, process_data
from model import train_model
from validate import evaluate

def validate_subset(input_list, valid_values, name):
    if not set(input_list).issubset(set(valid_values)):
        raise ValueError(f"Invalid {name}. Allowed values are: {valid_values}")

def main():
    # argparse
    parser = argparse.ArgumentParser(description="Run experiments with different parsers, modes, and models")
    parser.add_argument('--parsers', nargs='+', required=True, help="List of parsers to use, valid options: pymupdf, nougat, marker")
    parser.add_argument('--modes', nargs='+', required=True, help="List of modes to use, valid options: countvectorizer, fasttext, llm")
    parser.add_argument('--scores', nargs='+', required=True, help="List of scores to use, valid options: rouge, bleu, car")
    parser.add_argument('--models', nargs='+', required=True, help="List of models to use, valid options: ridge, lasso, svm, xgb, knn")
    parser.add_argument('--frame_name', required=True, help="Name of the CSV file to store results. Must end with `.csv`")
    
    args = parser.parse_args()

    # Validate the input lists
    valid_parsers = {'pymupdf', 'nougat', 'marker'}
    valid_modes = {'countvectorizer', 'fasttext', 'llm'}
    valid_scores = {'rouge', 'bleu', 'car'}
    valid_models = {'ridge', 'lasso', 'svm', 'xgb', 'knn'}
    
    validate_subset(args.parsers, valid_parsers, "parsers")
    validate_subset(args.modes, valid_modes, "modes")
    validate_subset(args.scores, valid_scores, "scores")
    validate_subset(args.models, valid_models, "models")
    
    # Validate frame_name
    frame_name = args.frame_name
    
    parsers = args.parsers
    modes = args.modes
    scores = args.scores
    models = args.models

    # constants
    tasks = ['reg', 'cls']
    p_store_reg = Path(f'./performance/{frame_name}_reg.csv')
    p_store_cls = Path(f'./performance/{frame_name}_cls.csv')
    
    # data
    all_df_metrics = []
    for parser in parsers:
        # get raw data frames
        df_train, df_test, df_val = load_data(parser=parser)
    
        # DEBUG <- leave this in for a moment
        #df_train = df_train.loc[:450, :]
    
        # subset
        for score in scores:
            for mode in modes:
                # process data
                data_list = process_data(df_train, df_test, df_val, n_max_chars=3200, max_features=1000, score=score, mode=mode, parsers=parsers)
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_list
                
                # - keep score list (for cls task to recoup BLEU regret)
                y_score_list = [data_list[i][1] for i in range(len(data_list))]
                
                # tasks
                # - derive cls task
                y_train_cls = np.array(y_train).argmax(1).reshape(-1, 1)
                y_val_cls = np.array(y_val).argmax(1).reshape(-1, 1)
                y_test_cls = np.array(y_test).argmax(1).reshape(-1, 1)
                
                # - recombine
                data_list_cls = (X_train, y_train_cls), (X_val, y_val_cls), (X_test, y_test_cls)
                
                # - models
                for model in models:
                    for task in tasks:
                        # meta
                        info = {'mode': mode, 'model': model, 'score': score, 'parser': parser, 'task': task}
                        
                        try:
                            if task == 'cls':
                                # train
                                trained_model = train_model(model, X_train, y_train_cls)
                                # evaluate
                                out = evaluate(trained_model, data_list_cls, y_score_list, info, parsers)
                            else:
                                # train
                                trained_model = train_model(model, X_train, y_train)
                                # evaluate
                                out = evaluate(trained_model, data_list, y_score_list, info, parsers)
                            
                            # append
                            all_df_metrics.append(out)
                            
                            # store or append results to CSV
                            p_store = p_store_reg if task=='reg' else p_store_cls
                            # - sep by task
                            if p_store.exists():
                                out.to_csv(p_store, mode='a', header=False, sep='|', index=False)
                            else:
                                out.to_csv(p_store, sep='|', index=False)
                                
                        except Exception as e:
                            print(f'Error for {info}. The error is {e}')

if __name__ == "__main__":
    main()