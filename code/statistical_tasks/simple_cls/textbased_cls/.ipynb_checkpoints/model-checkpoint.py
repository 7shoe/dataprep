import numpy as np
from scipy.stats import uniform, randint
import pandas as pd

from sklearn.linear_model import Ridge, ElasticNet, Lasso, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR, SVC

def get_model_parameters(model:str):
    """
    Model paramters for sklearn
    """
    # remove task type suffix
    model = model.replace('_reg', '').replace('_cls','')
    assert model in {'ridge', 'lasso', 'svm', 'xgb', 'knn'}, "Model must be one of"

    # seed scipy
    np.random.seed(177)
    
    # define parameter grids for each model
    ridge_param_dist = {
        'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]  # Adjust alpha values based on your data
    }
    lasso_param_dist = {
        'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
    }
    svm_param_dist = {
        'C': uniform(0.1, 10), 
        'epsilon': uniform(0.01, 1), 
        'kernel': ['linear', 'poly', 'rbf']
    }
    br_param_dist = {
        'n_estimators': 1,
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(1, 10),
        'subsample': uniform(0.5, 1.0)
    }

    # model para dict
    model_para = {'ridge' : ridge_param_dist, 'lasso' : lasso_param_dist, 
                  'svm' : svm_param_dist, 'xgb' : br_param_dist}

    return model_para[model]
    
# models
def train_model(model:str, 
                X_train,
                y_train,
                n_iter:int=10,
                cv:int=5,
                n_estimators:int=50,
                random_state:int=678):
    """
    """
    assert model in {'ridge', 'lasso', 'svm', 'xgb', 'knn'}, "Only these models supported"

    # identify stat task
    if (y_train.max().max() <= 1.0) and (len(np.unique(np.array(y_train))) > 10):
        # regression
        task_type = 'reg'
    else:
        task_type = 'cls'
    # identify model type
    model += f"_{task_type}"
    
    # search parameters
    search_params = {
        'n_iter': n_iter,
        'cv': cv,
        'scoring': ('neg_mean_squared_error' if task_type=='reg' else 'accuracy'),
        'random_state': random_state
    }

    if model=='ridge_reg':
        # Ridge regression (l2)
        search = RandomizedSearchCV(Ridge(), get_model_parameters(model), **search_params)
        model = MultiOutputRegressor(search)
    elif model=='ridge_cls':
        # Ridge classifier
        search = LogisticRegression(penalty='l2', solver='saga')
        model = MultiOutputClassifier(search)
    elif model=='lasso_reg':
        # LASSO (l1)
        search = RandomizedSearchCV(Lasso(), get_model_parameters(model), **search_params)
        model = MultiOutputRegressor(search)
    elif model=='lasso_cls':
        # LASSO classifier (using Logistic Regression as LASSO classification)
        search = LogisticRegression(penalty='l1', solver='saga')
        model = MultiOutputClassifier(search)
    elif model=='svm_reg':
        # Support Vector Machine
        search = RandomizedSearchCV(SVR(), get_model_parameters(model), **search_params)
        model = MultiOutputRegressor(svr_search)
    elif model=='svm_cls':
        # Support Vector Machine classifier
        search = SVC()
        model = MultiOutputClassifier(search)
    elif model=='xgb_reg':
        xgb_para = get_model_parameters(model)
        xgb_para['n_estimators'] = n_estimators # overwrite
        # XGBoost Regression
        search = RandomizedSearchCV(GradientBoostingRegressor(), xgb_para, **search_params)
        model = MultiOutputRegressor(search)
    elif model=='xgb_cls':
        # XGBoost classifier
        search = GradientBoostingClassifier()
        model = MultiOutputClassifier(search)
    elif model=='knn_reg':
        # kNN-Regression
        search = RandomizedSearchCV(KNeighborsRegressor(), get_model_parameters(model), **search_params)
        model = MultiOutputRegressor(search)
    elif model=='knn_cls':
        # k-Nearest Neighbors classifier
        search = KNeighborsClassifier()
        model = MultiOutputClassifier(search)
    else:
        raise NotImplementedError("Unknown model.")

    # train
    model.fit(X_train, y_train)

    return model