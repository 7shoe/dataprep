import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import is_classifier, is_regressor

def inverse_rel_regret(y_gt, y_pred):
    """
    Computes the inverse relative regret:
    Inv. rel regret = BLEU(of choice) / best BLEU 
    """
    # convert
    y_gt = np.array(y_gt)
    y_pred = np.array(y_pred)
    
    # argmax
    pred_choice = y_pred.argmax(axis=1)
    act_score = y_gt[np.arange(len(y_gt)), pred_choice]
    max_values = np.array(y_gt).max(axis=1)

    # div by 0
    max_values = np.where(max_values == 0, -1, max_values)
    
    # Calculate rel_regret, setting to 1 where max_values is 0
    inv_rel_regret = np.mean(np.where(max_values < 0, 1, act_score / max_values))
    
    return float(np.mean(inv_rel_regret))

def inverse_rel_regret_from_cls(y_gt_cls, y_pred_cls, y_gt_score):
    """
    Computes the inverse relative regret - a little more tricky from class-label
    (as it requires looking up the respec. BLEU score value)
    Inv. rel regret = BLEU(of choice) / best BLEU 
    """
    
    # convert
    y_gt_cls = np.array(y_gt_cls)
    y_pred_cls = np.array(y_pred_cls)
    y_gt_score = np.array(y_gt_score)
    
    # chosen vs. best score
    act_score  = y_gt_score[np.arange(len(y_gt_score)), y_pred_cls]
    max_values = y_gt_score[np.arange(len(y_gt_score)), y_gt_cls]

    # div by 0
    max_values = np.where(max_values == 0, -1, max_values)
    
    # Calculate rel_regret, setting to 1 where max_values is 0
    inv_rel_regret = np.mean(np.where(max_values < 0, 1, act_score / max_values))
    
    return float(np.mean(inv_rel_regret))

def goodput(y_gt, y_pred):
    """
    Goodput (1 if best was chosen 0 otherwise)
    """

    # test
    y_gt = y_gt.to_numpy()
    
    # DEBUG
    #print('type(y_gt)  : ', type(y_gt))
    #print('type(y_pred): ', type(y_pred))
    #print('y_gt.shape  : ', y_gt.shape)
    #print('y_pred.shape: ', y_pred.shape)
    #print('(y_gt)  : ', (y_gt))
    #print('(y_pred): ', (y_pred))
    
    # convert
    y_gt = np.array(y_gt)
    y_pred = np.array(y_pred)
    
    # compare choices
    #pred_choice = y_pred.argmax(axis=1)
    #act_score = y_gt[np.arange(len(y_gt)), pred_choice]
    #max_values = np.array(y_gt).max(axis=1)
    
    return float(np.mean(y_pred.argmax(axis=1) == y_gt.argmax(axis=1)))

def evaluate(trained_model, data_list:list[tuple], y_score_list:list[pd.DataFrame], info, parsers):
    """
    Given a trained model and data, evaluates the canoncial measures on it:
    - regression R^2, Root-MSE, Root-MAE, relative inv. regret (i.e. relative score through choice compared to best), goodput=accuracy
    - classification: accuracy, precision, recall, rir
    """
    
    # unravel (output format of process_data() )
    (X_train,y_train), (X_val,y_val), (X_test,y_test) = data_list
    
    # predict & evaluate
    X_vecs  = [X_train, X_val, X_test]
    y_vecs  = [y_train, y_val, y_test]
    subsets = ['train', 'val', 'test']

    metrics_list = []
    for (X_vec, y_gt, y_score, subset) in zip(X_vecs, y_vecs, y_score_list, subsets):
        # predict
        y_pred = trained_model.predict(X_vec)
    
        # regression
        if is_regressor(trained_model):
            # regression scores
            # - multivariate
            rmse = np.sqrt(mean_squared_error(y_gt, y_pred, multioutput='raw_values'))
            rmse_dict = {f"rmse_{p}" : score for p, score in zip(parsers, rmse)}
            rmae = np.sqrt(mean_absolute_error(y_gt, y_pred, multioutput='raw_values'))
            rmae_dict = {f"rmae_{p}" : score for p, score in zip(parsers, rmae)}
            r2 = r2_score(y_gt, y_pred, multioutput='raw_values')
            r2_dict = {f"r2_{p}" : score for p, score in zip(parsers, r2)}
            # - aggregate
            r2_agg = r2_score(y_gt, y_pred, multioutput='uniform_average')
            rmse_agg = np.sqrt(mean_squared_error(y_gt, y_pred, multioutput='uniform_average'))
            rmae_agg = np.sqrt(mean_absolute_error(y_gt, y_pred, multioutput='uniform_average'))
            # own metrics
            rel_inv_reg = inverse_rel_regret(y_gt=y_gt, y_pred=y_pred)
            gp = goodput(y_gt, y_pred) # same as rel. number of correct picks
            # store
            metrics = {'subset' : subset, 
                       'r2' : r2_agg, 
                       'rmse' : rmse_agg, 
                       'rmae' : rmae_agg, 
                       'rir' : rel_inv_reg, 
                       'acc' : gp, 
                       **r2_dict, 
                       **rmse_dict, 
                       **rmae_dict}
            
        # classification
        else:
            # transform to cls-format
            y_gt = np.array(y_gt).argmax(1).reshape(len(y_gt), -1)
            y_train = np.array(y_train).argmax(1).reshape(len(y_train), -1)
            y_test = np.array(y_test).argmax(1).reshape(len(y_test), -1)
            y_val = np.array(y_val).argmax(1).reshape(len(y_val), -1)
            
            # classification scores
            print("Before acc computation ...")
            accuracy = accuracy_score(y_gt, y_pred)
            print("... after accuracy_score()")
            
            precision = precision_score(y_gt, y_pred, average='macro', zero_division=1)
            recall = recall_score(y_gt, y_pred, average='macro', zero_division=1)
            f1 = f1_score(y_gt, y_pred, average='macro', zero_division=1)
            
            # store
            metrics = {'subset' : subset, 
                       'acc' : accuracy, 
                       'prec' : precision, 
                       'rec' : recall, 
                       'rir' : inverse_rel_regret_from_cls(y_gt_cls=y_gt, y_pred_cls=y_pred, y_gt_score=y_score)}
        
        # include meta information (data,model)
        all_metrics = {**info, **metrics, "n" : len(y_pred)}
        # append
        metrics_list.append(all_metrics)

    # metrics dataframe
    df_metrics = pd.DataFrame(metrics_list)
    
    return df_metrics