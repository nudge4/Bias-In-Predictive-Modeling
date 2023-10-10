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
from sklearn.model_selection import StratifiedKFold
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



# Load the training/validation sample
gender_df = pd.read_stata("../bias_analyses/race.dta").loc[:,['vccsid', 'male', 'white', 'afam', 'hisp', 'asian', 'other']]
firstgen_df = pd.read_stata("../bias_analyses/firstgen.dta")
pell_df = pd.read_stata("../bias_analyses/pell.dta")
zip_df = pd.read_stata("../bias_analyses/zip.dta")
train_df = pd.read_stata("../bias_analyses/train_df.dta")
train_df = train_df.merge(gender_df, how='inner', on=['vccsid']).merge(firstgen_df, how='inner', on=['vccsid']).merge(pell_df, how='inner', on=['vccsid', 'strm']).merge(zip_df, how='inner', on=['vccsid', 'strm'])
test_df = pd.read_stata("../bias_analyses/test_df.dta")
test_df = test_df.merge(gender_df, how='inner', on=['vccsid']).merge(firstgen_df, how='inner', on=['vccsid']).merge(pell_df, how='inner', on=['vccsid', 'strm']).merge(zip_df, how='inner', on=['vccsid', 'strm'])
original_test_grade = np.array(test_df.grade)
original_train_grade = np.array(train_df.grade)
train_df.loc[:,'grade'] = train_df.grade.apply(lambda x: x in {'A','B','C'}).astype(int)
test_df.loc[:,'grade'] = test_df.grade.apply(lambda x: x in {'A','B','C'}).astype(int)
print(train_df.shape,test_df.shape)
df = pd.concat([train_df, test_df])
predictors = list(pickle.load(open("../bias_analyses/predictors.p", "rb"))) + ['male', 'pell_ever_0', 'pell_ever_1', 'pell_target_0', 'pell_target_1', 'firstgen_0', 'firstgen_1', 'distance', 'median_income_households', 'perc_below_pov']
predictors = [p for p in predictors if p not in {'afam','other','white', 'asian', 'hisp'}]
l3 = ['cum_cred_earn', 'pct_stopped', 'pct_withdrawn', 'pct_incomplete', 'pct_dev', 'prop_comp_sd', 'overall_prop_comp', 'ever_dual']
for p in l3:
    v = np.min(df[p]) - 1
    df.loc[:, p] = df.apply(lambda x: v if x['first_ind'] == 1 else x[p], axis=1)
l4 = [p for p in predictors if re.search("^[A-Z]{3}_[A-Z]{3}$", p)]
for p in l4:
    v = np.min(df[p+"_grade"]) - 1
    df.loc[:, p + "_grade"] = df.apply(lambda x: v if x[p] != 1 else x[p+"_grade"], axis=1)
l5 = ['has_repeat_grade', 'has_prereq_grade', 'has_avg_g_concurrent', 'has_term_gpa_1', 'has_term_gpa_2', 'has_past_avg_grade']
for p in l5:
    v = np.min(df[p.replace("has_", "")]) - 1
    df.loc[:, p.replace("has_", "")] = df.apply(lambda x: v if x[p] != 1 else x[p.replace("has_", "")], axis=1)
predictors = [p for p in predictors if p not in set(l4+l5)]
print(len(predictors))
train_df = df[df.strm < 2192]
test_df = df[df.strm >= 2192]


def calc_cw(y):
    # Calculate the weight of each letter grade to be used in the modeling fitting procedure: the weight is inversely proportional to the square root of the frequency of the letter grade in the training sample
    cw = Counter(y)
    class_weight = {k:np.sqrt(cw.most_common()[0][-1]/v, dtype=np.float32) for k,v in cw.items()}
    return class_weight # The output is a dictionary mapping letter grade to the corresponding weight


optimal_d = 34
optimal_n = 260
optimal_nf = 15
rf = RandomForestClassifier(n_estimators=optimal_n, criterion="entropy",
                            max_depth=optimal_d,
                            random_state=0, n_jobs=-1, max_features=optimal_nf,
                            class_weight = calc_cw(train_df.grade))
rf.fit(train_df.loc[:,predictors], train_df.grade)


xx = np.array(predictors)[np.argsort(rf.feature_importances_)[::-1]]
yy = rf.feature_importances_[np.argsort(rf.feature_importances_)[::-1]]
pd.DataFrame({'predictor':xx, 'fi':yy}).to_csv("fi_base.csv", index=False)



optimal_d = 32
optimal_n = 240
optimal_nf = 15
rf = RandomForestClassifier(n_estimators=optimal_n, criterion="entropy",
                            max_depth=optimal_d,
                            random_state=0, n_jobs=-1, max_features=optimal_nf,
                            class_weight = calc_cw(train_df.grade))
rf.fit(train_df[train_df.white==1].loc[:,predictors], train_df[train_df.white==1].grade)


xx = np.array(predictors)[np.argsort(rf.feature_importances_)[::-1]]
yy = rf.feature_importances_[np.argsort(rf.feature_importances_)[::-1]]
pd.DataFrame({'predictor':xx, 'fi':yy}).to_csv("fi_white.csv", index=False)



optimal_d = 29
optimal_n = 260
optimal_nf = 13
rf = RandomForestClassifier(n_estimators=optimal_n, criterion="entropy",
                            max_depth=optimal_d,
                            random_state=0, n_jobs=-1, max_features=optimal_nf,
                            class_weight = calc_cw(train_df.grade))
rf.fit(train_df[train_df.afam==1].loc[:,predictors], train_df[train_df.afam==1].grade)


xx = np.array(predictors)[np.argsort(rf.feature_importances_)[::-1]]
yy = rf.feature_importances_[np.argsort(rf.feature_importances_)[::-1]]
pd.DataFrame({'predictor':xx, 'fi':yy}).to_csv("fi_afam.csv", index=False)


