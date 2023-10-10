# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 07:49:46 2022

@author: ys8mz
"""


import numpy as np
import pandas as pd
from scipy import stats
import pickle
from collections import Counter
from sklearn.metrics import precision_recall_curve, roc_auc_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from scipy.stats.mstats import gmean
import math
import datetime as dt
import matplotlib
font = {'size': 24}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import seaborn as sns
import re


df_new = pd.read_stata("../degree_completion_1/full_data_truncated_enlarged_new.dta")
predictors = pickle.load(open("../degree_completion_1/predictors_rf2.p", "rb"))


l1 = ['coll_lvl_cred_earn', 'prop_comp_pre', 'cum_gpa_pre']
for p in l1:
    v = np.min(df_new[p]) - 1
    df_new.loc[:, p] = df_new.apply(lambda x: v if x['enrolled_pre'] != 1 else x[p], axis=1)
l2 = ['admrate', 'gradrate', 'satvr25', 'satvr75', 'satmt25', 'satmt75', 'satwr25', 'satwr75', 'nsc_coll_type_1', 'nsc_coll_type_2', 'nsc_coll_type_3', 'nsc_coll_type_4', 'nsc_coll_type_5', 'nsc_coll_type_6', 'nsc_coll_type_7', 'nsc_coll_type_8']
for p in l2:
    v = np.min(df_new[p]) - 1
    df_new.loc[:, p] = df_new.apply(lambda x: v if x['enrolled_nsc'] != 1 else x[p], axis=1)
l3 = ['degree_seeking', 'term_cred_att', 'term_gpa', 'prop_comp', 'withdrawn_prop_comp', 'lvl2_prop_comp', 'dev_prop_comp', 'repeat']
for t in ['su', 'fa', 'sp']:
    for i in range(1,7):
        for pp in l3:
            p = pp + "_" + t + str(i)
            v = np.min(df_new[p]) - 1
            df_new.loc[:, p] = df_new.apply(lambda x: v if x['enrolled_' + t + str(i)] != 1 else x[p], axis=1)
l4 = ['enrl_intensity_nsc']
for t in ['su', 'fa', 'sp']:
    for i in range(1,7):
        for pp in l4:
            p = pp + "_" + t + str(i)
            v = np.min(df_new[p]) - 1
            df_new.loc[:, p] = df_new.apply(lambda x: v if x['enrolled_nsc_' + t + str(i)] != 1 else x[p], axis=1)
l5 = ['grants', 'sub_loans', 'unsub_loans', 'others']
for i in range(1,7):
    for pp in l5:
        p = pp + "_yr" + str(i)
        v = np.min(df_new[p]) - 1
        df_new.loc[:, p] = df_new.apply(lambda x: v if x['enrolled_yr' + str(i)] != 1 else x[p], axis=1)
to_drop = ['enrolled_pre', 'enrolled_nsc'] + ['enrolled_' + t + str(i) for t in ['su', 'fa', 'sp'] for i in range(1,7)] + ['enrolled_nsc_' + t + str(i) for t in ['su', 'fa', 'sp'] for i in range(1,7)]
predictors = [p for p in predictors if p not in set(to_drop)]
print(len(predictors))

train_df = df_new[df_new.valid == 0]
test_df = df_new[df_new.valid == 1]

np.random.seed(4321)
afam_indices = np.where(train_df.afam == 1)[0]
ratio = len(afam_indices) / (train_df.shape[0] - len(afam_indices))
train_df_new_afam = train_df[train_df.afam == 1]
train_df_new_nonafam = train_df[train_df.afam == 0]
race = pd.Series(["mi"] * train_df_new_nonafam.shape[0])
for r in ['white','hisp','asian','other']:
    race.iloc[np.where(np.array(train_df_new_nonafam[r]) == 1)[0]] = r
train_df_new_nonafam.loc[:,'race_column'] = list(race)
_, train_df_new_nonafam_sample = train_test_split(train_df_new_nonafam, test_size = ratio, random_state = 4321,
                                                  stratify = train_df_new_nonafam.grad_6years.astype(str) + "_" + train_df_new_nonafam.race_column)
train_df_new_nonafam_sample = train_df_new_nonafam_sample.drop(['race_column'], axis=1)
train_df_old = train_df.loc[:,:]
train_df = pd.concat([train_df_new_afam, train_df_new_nonafam_sample])
train_df.to_stata("train_df_downsampled.dta", write_index=False)


# Fit the RF model
def create_cv_folds(train, n_fold = 5):
    folds = []
    k_fold = StratifiedKFold(n_splits = n_fold, random_state = 12345, shuffle=True)
    for train_indices, test_indices in k_fold.split(train, train.grad_6years):
        train_part = train.iloc[train_indices,:]
        test_part = train.iloc[test_indices,:]
        X_1 = train_part.loc[:,predictors]
        y_1 = train_part.grad_6years
        X_2 = test_part.loc[:,predictors]
        y_2 = test_part.grad_6years
        folds.append([(X_1.copy(),y_1.copy()),(X_2.copy(),y_2.copy())])
    return folds

five_folds = create_cv_folds(train_df)


def cross_validation_RF(rf_model, folds):
    auc_by_fold = []
    for f in folds:
        X_1 = f[0][0]
        y_1 = f[0][1]
        X_2 = f[1][0]
        y_2 = f[1][1]
        rf_model.fit(X_1,y_1)
        y_2_pred = rf_model.predict_proba(X_2)[:,1]
        auc_by_fold.append(roc_auc_score(y_2,y_2_pred))
    return round(np.mean(auc_by_fold),4)

def calc_cw(y):
    # Calculate the weight of each letter grade to be used in the modeling fitting procedure: the weight is inversely proportional to the square root of the frequency of the letter grade in the training sample
    cw = Counter(y)
    class_weight = {k:np.sqrt(cw.most_common()[0][-1]/v, dtype=np.float32) for k,v in cw.items()}
    return class_weight # The output is a dictionary mapping letter grade to the corresponding weight


print("Start time: {}".format(dt.datetime.now()))


### Using grid search to find the optimal maximum tree depth
auc_by_d=[]
running_auc = -math.inf
for d in range(2,36):
    rf = RandomForestClassifier(n_estimators=200, criterion="entropy", 
                                max_depth=d,
                                random_state=0, n_jobs=-1, max_features="auto",
                                class_weight = calc_cw(train_df.grad_6years))
    auc = cross_validation_RF(rf, five_folds)
    auc_by_d.append(auc)
    print("Max_depth =", d)
    print("Mean CV AUC:", auc)
    if auc - running_auc >= 0.001:
        running_auc = auc
    else:
        optimal_d = d - 1
        break
else:
    optimal_d = d
print("Optimal max_depth = {}".format(optimal_d))



### Using grid search to find the optimal number of estimators (trees)
auc_by_n = []
running_auc = -math.inf
for n in range(100,320,20):
    rf = RandomForestClassifier(n_estimators=n, criterion="entropy", 
                                max_depth=optimal_d,
                                random_state=0, n_jobs=-1, max_features="auto",
                                class_weight = calc_cw(train_df.grad_6years))
    auc = cross_validation_RF(rf, five_folds)
    auc_by_n.append(auc)
    print("Number of Trees =", n)
    print("Mean CV AUC:", auc)
    if auc - running_auc >= 0.0001:
        running_auc = auc
    else:
        optimal_n = n - 20
        break
else:
    optimal_n = n
print("Optimal number of estimators = {}".format(optimal_n))



### Using grid search to find the optimal maximum number of features (trees)
auc_by_nf = []
max_nf = int(np.floor(2*np.sqrt(len(predictors))))
running_auc = -math.inf
for nf in range(2,max_nf+1):
    rf = RandomForestClassifier(n_estimators=optimal_n, criterion="entropy", 
                                max_depth=optimal_d,
                                random_state=0, n_jobs=-1, max_features=nf,
                                class_weight = calc_cw(train_df.grad_6years))
    auc = cross_validation_RF(rf, five_folds)
    auc_by_nf.append(auc)
    print("Max_features =", nf)
    print("Mean CV AUC:", auc)
    if auc - running_auc >= 0.0005:
        running_auc = auc
    else:
        optimal_nf = nf - 1
        break
else:
    optimal_nf = nf
print("Optimal maximum number of features = {}".format(optimal_nf))