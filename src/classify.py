import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, average_precision_score

def svc_classifier(X_train, y_train, X_test, y_test):
    """
    Pipeline for SVM classifier with parameter selection by CV.

    Parameters:
        X_train - training set
        y_train - training labels
        X_test  - test set
        y_test  - test labels
    
    Returns:
        fitted SVC model,
        dictionary with metrics for the model (accuracy on train/test, auc roc and pr on test)
    """
    svc = LinearSVC(dual=False)
    reg_Cs = np.logspace(-5, 1, 20)
    linear_svc = GridSearchCV(svc, {"C": reg_Cs}, cv=10)    # chooses best by score estimate
    model = linear_svc.fit(X_train, y_train)

    best_model_svc = linear_svc.best_estimator_
    train_score = best_model_svc.score(X_train, y_train)
    test_score = best_model_svc.score(X_test, y_test)
    
    b_pred_svc = best_model_svc.decision_function(X_test)
    auc_roc_svc_ = roc_auc_score(y_test, b_pred_svc)
    auc_pr_svc_ = average_precision_score(y_test, b_pred_svc)
    
    return best_model_svc, {
        'train_acc': train_score, 'test_acc': test_score, 
        'auc_roc_test': auc_roc_svc_, 'auc_pr_test': auc_pr_svc_
    }

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from itertools import product


def decision_tree_classifier(
    X_train, y_train, X_test, y_test, 
    depth_grid=range(3, 16), samples_leaf_grid=range(1, 5), random_forest=False
):
    """
    Pipeline for DT/RF classifier for different parameters.

    Parameters:
        X_train - training set
        y_train - training labels
        X_test  - test set
        y_test  - test labels
        depth_grid - set of values for max_depth parameter in DT/RF
        samples_leaf_grid - set of values for min_samples_leaf parameter in DT/RF
        random_forest - if True, train a Random Forest classifier, 
                        if False, train a Decision Tree classifier
    
    Returns:
        fitted models for all parameters,
        dictionary with accuracy values on train/test for each model
    """

    models = {}
    accuracy = {part: np.zeros((len(depth_grid), len(samples_leaf_grid))) for part in ['train', 'test']}

    for i, depth in tqdm(enumerate(depth_grid), total=len(depth_grid), leave=False):
        for j, samples_leaf in enumerate(samples_leaf_grid):
            if random_forest:
                model = RandomForestClassifier(
                    max_depth = depth, 
                    min_samples_leaf = samples_leaf
                ).fit(X_train, y_train)
            else:
                model = DecisionTreeClassifier(
                    max_depth = depth, 
                    min_samples_leaf = samples_leaf
                ).fit(X_train, y_train)
            pred_train = model.predict(X_train)
            pred = model.predict(X_test)
            accuracy['train'][i, j] = accuracy_score(y_train, pred_train) 
            accuracy['test'][i, j] = accuracy_score(y_test, pred)
            models[(depth, samples_leaf)] = model
    for part in accuracy:
        accuracy[part] = pd.DataFrame(accuracy[part])
        accuracy[part].columns = samples_leaf_grid
        accuracy[part].index = depth_grid
    return models, accuracy

import glob
np.random.seed(2024)
bot_names = ['lstm', 'balaboba', 'gpt2', 'mGPT']
def get_train_test_datasets(lang="RU", part="word", bot_subset=("gpt2", "balaboba")):
    """
    Helper function to get train/test data.
    """
    def read_subdf(filename, text_type):
        a = pd.read_csv(filename)
        a[a.columns[-2]] = a[a.columns[-2]].astype(int)
        a['text_type'] = text_type
        return a
    
    lit_train = read_subdf(f"features/Train_{lang}_lit_{part}_features.csv", "lit")
    
    ## Crop for all cases except for dataset
    lit_train_idx = np.random.choice(len(lit_train), 600)
    lit_train = lit_train.iloc[lit_train_idx].reset_index(drop=True)    
    
    lit_test = read_subdf(f"features/Test_{lang}_lit_{part}_features.csv", "lit")
    
    bot_train = []
    bot_test = []
    for text_type in bot_names:
        fn = glob.glob(f"features/*_{lang}_{text_type}_{part}_features.csv")[0]
        bot_df = read_subdf(fn, text_type)
        if len(bot_df) > 300:
            idx = np.random.choice(len(bot_df), 300)
            bot_df = bot_df.iloc[idx].reset_index(drop=True)    
        if text_type in bot_subset:
            bot_train.append(bot_df)
        else:
            bot_test.append(bot_df)
    return (
        pd.concat([lit_train] + bot_train).reset_index(drop=True),
        pd.concat([lit_test] + bot_test).reset_index(drop=True)
    )

def pipeline_clf(bot_subset, method='svc'):
    """
    Full pipeline.

    Parameter:
        bot_subset - names of two bots to use during training 
                     (models are tested on texts of remaining bots)
        method     - "svc"/"dt"/"rf", classifier model to use
    
    Returns:
        results (see `svc_classifier` and `decision_tree_classifier`) for English and Russian data
        on word/bigram/trigram levels.
    """
    results = {'EN': {}, 'RU': {}}
    for lang in ['EN', 'RU']:
        for part in ['word', 'bigram', 'trigram']:
            try:
                train_df, test_df = get_train_test_datasets(lang, part, bot_subset)
                # print("\tTrain text types:", train_df.text_type.unique())
                # print("\tTest text types:", test_df.text_type.unique())
                # print("\n")
                if method == 'svc':
                    res = svc_classifier(
                        train_df[train_df.columns[:-2]], train_df[train_df.columns[-1]] == 'lit', 
                        test_df[test_df.columns[:-2]], test_df[test_df.columns[-1]] == 'lit'
                    )
                elif method == 'dt':
                    res = decision_tree_classifier(
                        train_df[train_df.columns[:-2]], train_df[train_df.columns[-1]] == 'lit', 
                        test_df[test_df.columns[:-2]], test_df[test_df.columns[-1]] == 'lit'
                    )
                elif method == 'rf':
                    res = decision_tree_classifier(
                        train_df[train_df.columns[:-2]], train_df[train_df.columns[-1]] == 'lit', 
                        test_df[test_df.columns[:-2]], test_df[test_df.columns[-1]] == 'lit',
                        random_forest=True
                    )
                results[lang][part] = res
                print(lang, part)
                if method == 'svc':
                    print(res[1])
                else:
                    print(res[1]['train'].values.max(), res[1]['test'].values.max())
                print('*' * 10)
            except:
                continue
    return results


