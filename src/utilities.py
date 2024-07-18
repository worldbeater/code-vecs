from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

from pprint import pprint
from warnings import catch_warnings, simplefilter

import numpy as np
import csv
import os

def read_csv(fname):
    with open(fname, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for entry, in reader:
            yield entry

def read_dataset_folder(folder):
    dataset = []
    files = sorted(file for file in os.listdir(folder) if 'task' in file)
    for i, file in enumerate(files):
        for code in list(read_csv(f'{folder}/{file}')):
            dataset.append([code, i])
    return dataset

def score(grid_search, scorers):
    required = ['fit_time', 'score_time']
    output = dict()
    for scorer in [f'test_' + s for s in scorers] + required:
        means = grid_search.cv_results_[f'mean_{scorer}']
        stds = grid_search.cv_results_[f'std_{scorer}']
        mean = np.max(means)
        std = np.mean(stds)
        key = scorer if scorer in required else scorer[5:]
        output[key] = (round(mean, 3), round(std, 3))
    return dict(sorted(output.items()))

def find(x, y, estimator, params, verbose=2):
    keys = ['precision_weighted', 'recall_weighted', 'f1_weighted',
            'precision_macro', 'recall_macro', 'f1_macro',
            'accuracy']
    grid_search = GridSearchCV(
        scoring=dict((key, key) for key in keys),
        estimator=estimator(),
        param_grid=params,
        refit='f1_macro',
        verbose=verbose,
        cv=5)
    with catch_warnings():
        simplefilter('ignore')
        grid_search.fit(x, y)
    print()
    pprint(grid_search.best_params_)
    pprint(score(grid_search, keys))
    return grid_search

def statistics(dataset, classifier, parameters, *limits):
    keys, scores = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], []
    for count in range(*limits):
        _, X, Y = dataset(count=count)
        X, Y = shuffle(X, Y, random_state=0)
        model = find(X, Y, classifier, parameters)
        scores.append(score(model, keys))
    pprint(scores, width=400)
    return scores

def describe(dataset):
    H, X, Y = dataset(count=100)
    print(f'Components: {len(X[0])}.')
    print(f'Non-zero components: {len(np.where(X[0] != 0)[0])}.')
    print(f'Zero components: {len(np.where(X[0] == 0)[0])}.')
