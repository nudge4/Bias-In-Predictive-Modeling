import numpy as np
import pandas as pd
from scipy import stats
import pickle
from collections import Counter
from sklearn.metrics import precision_recall_curve, roc_auc_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
import statsmodels.api as sm
from statsmodels.tools import add_constant
from scipy.stats.mstats import gmean
import matplotlib
font = {'size': 24}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


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
predictors = list(pickle.load(open("../bias_analyses/predictors.p", "rb"))) + ['male', 'pell_ever_0', 'pell_ever_1', 'pell_target_0', 'pell_target_1', 'firstgen_0', 'firstgen_1', 'distance', 'median_income_households', 'perc_below_pov']


# Number of unique students in the sample
print(len(np.unique(train_df.vccsid)), len(np.unique(test_df.vccsid)))
print(len(np.unique(train_df.course)), len(np.unique(test_df.course)))
# Number of unique college x course observations in the sample
print(train_df.loc[:,['college', 'course']].drop_duplicates().shape[0],
      test_df.loc[:,['college', 'course']].drop_duplicates().shape[0])
# Total number of unique college x course observations in the entire sample (training + test)
print(pd.concat([train_df.loc[:,['college', 'course']], test_df.loc[:,['college', 'course']]]).drop_duplicates().shape[0])
print(len(np.union1d(np.unique(train_df.course), np.unique(test_df.course))))


# Fit the Logit model
print(dt.datetime.now())
lr = sm.Logit(np.array(train_df.grade), add_constant(train_df.loc[:,predictors], prepend=True)).fit()
print(dt.datetime.now())


# Coefficients and predicted scores
y_test_pred_lr = list(lr.predict(add_constant(test_df.loc[:,predictors], prepend=True)))
y_train_pred_lr = list(lr.predict(add_constant(train_df.loc[:,predictors], prepend=True)))
pickle.dump(y_test_pred_lr, open("y_test_pred_lr.p", "wb"))
pickle.dump(list(test_df.grade), open("y_test.p", "wb"))
pickle.dump(y_train_pred_lr, open("y_train_pred_lr.p", "wb"))
pickle.dump(list(train_df.grade), open("y_train.p", "wb"))
print("Logistic Regression:")
print("Validation AUC = {}".format(round(roc_auc_score(np.array(test_df.grade), y_test_pred_lr),4)))
print("Training AUC = {}".format(round(roc_auc_score(np.array(train_df.grade), y_train_pred_lr),4)))
lr_summary = pd.DataFrame({'coef': lr.params, 'std_err': lr.bse, 'p-values': lr.pvalues}, index=lr.params.index).loc[:,['coef','std_err','p-values']]
lr_summary.to_csv("lr_summary.csv", index=True)


white_indices = np.where(np.array(test_df.white == 1) & np.array(test_df.num_of_prior_terms == 0))[0]
afam_indices = np.where(np.array(test_df.afam == 1) & np.array(test_df.num_of_prior_terms == 0))[0]
print("white first-term AUC = {}".format(round(roc_auc_score(np.array(test_df.grade)[white_indices], np.array(y_test_pred_lr)[white_indices]),4)))
print("afam first-term AUC = {}".format(round(roc_auc_score(np.array(test_df.grade)[afam_indices], np.array(y_test_pred_lr)[afam_indices]),4)))

white_indices = np.where(np.array(test_df.white == 1) & np.array(test_df.num_of_prior_terms > 0))[0]
afam_indices = np.where(np.array(test_df.afam == 1) & np.array(test_df.num_of_prior_terms > 0))[0]
print("white non-first-term AUC = {}".format(round(roc_auc_score(np.array(test_df.grade)[white_indices], np.array(y_test_pred_lr)[white_indices]),4)))
print("afam non-first-term AUC = {}".format(round(roc_auc_score(np.array(test_df.grade)[afam_indices], np.array(y_test_pred_lr)[afam_indices]),4)))


'''
def find_optimal_threshold(p,r,t):
    to_drop = np.union1d(np.where(pd.isnull(p[:-1]) == True)[0], np.where(pd.isnull(r[:-1]) == True)[0])
    to_drop = np.union1d(to_drop, np.where(pd.isnull(t) == True)[0])
    to_keep = np.setdiff1d(np.array(list(range(len(p)-1))), to_drop)
    p,r,t = p[to_keep],r[to_keep],t[to_keep]
    to_keep_2 = np.intersect1d(np.where(t < 0.9)[0], np.where(t > 0.1)[0])
    p,r,t = p[to_keep_2],r[to_keep_2],t[to_keep_2]
    f1 = 2*p*r/(p+r)
    best_t = t[np.argmax(f1)]
    return best_t

def cross_validation_lr(train):
    threshold_list = []
    auc_list = []
    k_fold =  StratifiedKFold(n_splits = 10, random_state = 12345, shuffle=True)
    for train_indices, test_indices in k_fold.split(train, train.grade):
        train_part = train.iloc[train_indices,:]
        test_part = train.iloc[test_indices,:]
        X_1 = train_part.loc[:,predictors]
        y_1 = train_part.grade
        X_2 = test_part.loc[:,predictors]
        y_2 = test_part.grade
        model = sm.Logit(y_1, add_constant(X_1,prepend=True)).fit()
        p,r,t = precision_recall_curve(y_2, 1-np.array(model.predict(add_constant(X_2, prepend=True))), pos_label = 0)
        auc = roc_auc_score(y_2, model.predict(add_constant(X_2, prepend=True)))
        threshold_list.append(1 - find_optimal_threshold(p,r,t))
        auc_list.append(auc)
    print(threshold_list)
    print(np.mean(auc_list), np.std(auc_list, ddof=1))
    return gmean(threshold_list)
'''

def two_side_z_test(n1, p1, n2, p2):
    z = (p2-p1)/np.sqrt(p2*(1-p2)/(n2-1)+p1*(1-p1)/(n1-1))
    return 2*(1-stats.norm.cdf(np.abs(z)))

def two_side_z_test_2(n1, p1, v1, n2, p2, v2):
    z = (p2-p1)/np.sqrt(v2/n2+v1/n1)
    return 2*(1-stats.norm.cdf(np.abs(z)))


sr = sum(train_df.grade)/train_df.shape[0]
n = int(train_df.shape[0] - train_df.shape[0] * sr)
best_threshold = sorted(y_train_pred_lr)[n-1]
# best_threshold = cross_validation_lr(train_df)
print("Best threshold = {}".format(best_threshold))
bas = balanced_accuracy_score(np.array(test_df.grade), np.where(np.array(y_test_pred_lr) > best_threshold, 1, 0))
print("Balanced accuracy score = {}".format(bas))
bas0 = balanced_accuracy_score(np.array(test_df.grade)[white_indices], np.where(np.array(y_test_pred_lr)[white_indices] > best_threshold, 1, 0))
print("white first-term BAS = {}".format(bas0))
bas1 = balanced_accuracy_score(np.array(test_df.grade)[afam_indices], np.where(np.array(y_test_pred_lr)[afam_indices] > best_threshold, 1, 0))
print("afam first-term BAS = {}".format(bas1))
print("p-value for white vs. afam first-term BAS: {}".format(two_side_z_test(len(white_indices),bas0,len(afam_indices),bas1)))

def create_confusion_matrix(y_test_pred, threshold, fname):
    cm_arr = confusion_matrix(np.array(test_df.grade), np.where(y_test_pred > threshold, 1, 0))
    cm_df = pd.DataFrame(cm_arr, columns=['Pred_DFW','Pred_ABC'], index=['Actual_DFW', 'Actual_ABC'])
    cm_df.loc[:,''] = cm_df.sum(axis=1)
    cm_df.loc['',:] = cm_df.sum(axis=0)
    print(cm_df)
    print("")
    p1 = cm_df.iloc[1,1]/cm_df.iloc[2,1]
    r1 = cm_df.iloc[1,1]/cm_df.iloc[1,2]
    p0 = cm_df.iloc[0,0]/cm_df.iloc[2,0]
    r0 = cm_df.iloc[0,0]/cm_df.iloc[0,2]    
    print("F1 score for A/B/C = {}".format(round(2*p1*r1/(p1+r1),4)))
    print("F1 score for D/F/W = {}".format(round(2*p0*r0/(p0+r0),4))) 
    cm_df.to_csv(fname + ".csv")
    y_test_pred_bin = np.where(y_test_pred > best_threshold, 1, 0)
    cm_dict = {}
    cm_dict['Pred_DFW'] = Counter(original_test_grade[np.where(y_test_pred_bin==0)[0]])
    cm_dict['Pred_ABC'] = Counter(original_test_grade[np.where(y_test_pred_bin==1)[0]])
    new_cm = pd.DataFrame.from_dict(cm_dict, orient='index').T.loc[['W','F','D','C','B','A'],['Pred_DFW','Pred_ABC']]
    new_cm.index = ["Actual_"+e for e in new_cm.index]
    new_cm.loc[:,''] = new_cm.sum(axis=1)
    new_cm.loc['',:] = new_cm.sum(axis=0)
    new_cm.to_csv(fname + "_6x2.csv")
    return round(p1,4),round(r1,4),round(p0,4),round(r0,4),round(2*p1*r1/(p1+r1),4),round(2*p0*r0/(p0+r0),4)


pr_rf = create_confusion_matrix(np.array(y_test_pred_lr), best_threshold, "LR_1_cm")
print(pr_rf)


success_rate_1 = []
for r in ['overall', 'white', 'afam', 'hisp', 'asian', 'other']:
    if r == 'overall':
        y_arr = np.array(y_test_pred_lr)
        y_actual = np.array(test_df.grade)
    else:
        y_arr = np.array(y_test_pred_lr)[np.where(np.array(test_df[r]) == 1)[0]]
        y_actual = np.array(test_df.grade)[np.where(np.array(test_df[r]) == 1)[0]]
    success_rate_1.append((r, len(y_arr), np.mean(y_actual), np.mean(y_arr > best_threshold)))
    fig = plt.figure(figsize=(16,11)) 
    ax = fig.add_subplot(1, 1, 1)
    pd.DataFrame({r:y_arr}).hist(r, bins = np.linspace(0,1,26), density=True, color='orange', figsize=(16,11), ax=ax)
    for p,q in zip(np.percentile(y_arr, q = [10,25,50,75,90]), [10,25,50,75,90]):
        ax.axvline(x=p, color='g', linestyle='dashed', linewidth=2)
        ax.text(p-0.02,ax.get_ylim()[1]*1.005,"{}%".format(q),fontsize=12)
    ax.set_xlabel("Predicted Score")
    ax.set_ylabel("Density")
    plt.savefig(r +"_1.png")
n1 = success_rate_1[1][1]
p1 = success_rate_1[1][2]
asr_pval = []
for i in range(2, len(success_rate_1)):
    subgp = success_rate_1[i][0]
    n2 = success_rate_1[i][1]
    p2 = success_rate_1[i][2]
    asr_pval.append((subgp, two_side_z_test(n1,p1,n2,p2)))
n1 = success_rate_1[1][1]
p1 = success_rate_1[1][3]
psr_pval = []
for i in range(2, len(success_rate_1)):
    subgp = success_rate_1[i][0]
    n2 = success_rate_1[i][1]
    p2 = success_rate_1[i][3]
    psr_pval.append((subgp, two_side_z_test(n1,p1,n2,p2)))
pd.DataFrame(success_rate_1, columns=['subgroup', 'N', 'actual_success_rate', 'pred_success_rate'])\
.merge(pd.DataFrame(asr_pval, columns=['subgroup', 'p-value_for_actual_success_rate']), on=['subgroup'], how='left')\
.merge(pd.DataFrame(psr_pval, columns=['subgroup', 'p-value_for_pred_success_rate']), on=['subgroup'], how='left')\
.loc[:,['subgroup', 'N', 'actual_success_rate', 'p-value_for_actual_success_rate', 'pred_success_rate', 'p-value_for_pred_success_rate']]\
.round(4).to_csv("success_rate_1.csv", index=False)


race_column = []
for i in range(test_df.shape[0]):
    if test_df.white.iloc[i] == 1:
        race_column.append("white")
    elif test_df.afam.iloc[i] == 1:
        race_column.append("afam")
    elif test_df.hisp.iloc[i] == 1:
        race_column.append("hisp")
    elif test_df.asian.iloc[i] == 1:
        race_column.append("asian")
    elif test_df.other.iloc[i] == 1:
        race_column.append("other")
    else:
        race_column.append("mi")
race_column = np.array(race_column)
pred_y = np.array(y_test_pred_lr)
test_y = np.array(test_df.grade)
pred_y = pred_y[race_column != "mi"]
test_y = test_y[race_column != "mi"]
race_column = race_column[race_column != "mi"]
print(len(race_column), len(pred_y), len(test_y))


def ba_auc(subgp):
    l2 = pred_y
    l3 = test_y
    indices = np.where(np.array(race_column) == subgp)[0]
    l2_sub = l2[indices]
    l3_sub = l3[indices]
    return roc_auc_score(l3_sub, l2_sub)

results_auc = []
for gp in ['white', 'afam', 'hisp', 'asian', 'other']:
    results_auc.append((gp, ba_auc(gp)))
results_auc_df = pd.DataFrame(results_auc, columns = ['subgroup', 'c-statistic']).round(4)
results_auc_df.to_csv("bias_cstat.csv", index=False)
pred_score_by_race = pd.DataFrame({'race_column': race_column, 'pred_y': pred_y, 'test_y': test_y})
pred_score_by_race.to_csv("pred_score_by_race.csv", index=False)


def ba_0(subgp):
    l2 = pred_y
    l3 = test_y
    indices = np.where(np.array(race_column) == subgp)[0]
    l2_sub = l2[indices]
    l3_sub = l3[indices]
    N1 = sum(l2_sub <= best_threshold)
    N2 = sum(l3_sub == 0)
    a1 = 1-np.mean(l3_sub[l2_sub <= best_threshold]) #precision_0
    a2 = np.mean(l2_sub[l3_sub == 0] > best_threshold) #false positive rate
    return subgp, N1, a1, N2, a2

results_0 = []
for gp in ['white', 'afam', 'hisp', 'asian', 'other']:
    results_0.append(ba_0(gp))
    

def ba_1(subgp):
    l2 = pred_y
    l3 = test_y
    indices = np.where(np.array(race_column) == subgp)[0]
    l2_sub = l2[indices]
    l3_sub = l3[indices]
    N = sum(l3_sub == 0)
    a = np.mean(l2_sub[l3_sub == 1] > best_threshold) #true positive rate
    return subgp, N, a

results_1 = []
for gp in ['white', 'afam', 'hisp', 'asian', 'other']:
    results_1.append(ba_1(gp))
    
    
def ba_bas(subgp):
    l2 = pred_y
    l3 = test_y
    indices = np.where(np.array(race_column) == subgp)[0]
    l2_sub = l2[indices]
    l3_sub = l3[indices]
    N = len(indices)
    a = balanced_accuracy_score(l3_sub, np.where(l2_sub > best_threshold, 1, 0))
    return subgp, N, a

results_bas = []
for gp in ['white', 'afam', 'hisp', 'asian', 'other']:
    results_bas.append(ba_bas(gp))
    

def ba_pred_score(subgp):
    l2 = pred_y
    l3 = test_y
    indices = np.where(np.array(race_column) == subgp)[0]
    l2_sub = l2[indices]
    l3_sub = l3[indices]
    sr = np.mean(l3_sub)
    N = len(indices)
    a = np.mean(l2_sub) - sr
    return subgp, N, a, np.var(l2_sub)  

results_pred_score = []
for gp in ['white', 'afam', 'hisp', 'asian', 'other']:
    results_pred_score.append(ba_pred_score(gp))
    

race_column_train = []
for i in range(train_df.shape[0]):
    if train_df.white.iloc[i] == 1:
        race_column_train.append("white")
    elif train_df.afam.iloc[i] == 1:
        race_column_train.append("afam")
    elif train_df.hisp.iloc[i] == 1:
        race_column_train.append("hisp")
    elif train_df.asian.iloc[i] == 1:
        race_column_train.append("asian")
    elif train_df.other.iloc[i] == 1:
        race_column_train.append("other")
    else:
        race_column_train.append("mi")
race_column_train = np.array(race_column_train)
pred_y_train = np.array(y_train_pred_lr)
test_y_train = np.array(train_df.grade)
pred_y_train = pred_y_train[race_column_train != "mi"]
test_y_train = test_y_train[race_column_train != "mi"]
race_column_train = race_column_train[race_column_train != "mi"]
print(len(race_column_train), len(pred_y_train), len(test_y_train))


def ba_pred_score_train(subgp):
    l2 = pred_y_train
    l3 = test_y_train
    indices = np.where(np.array(race_column_train) == subgp)[0]
    l2_sub = l2[indices]
    l3_sub = l3[indices]
    sr = np.mean(l3_sub)
    N = len(indices)
    a = np.mean(l2_sub) - sr
    return subgp, N, a, np.var(l2_sub)  

results_pred_score_train = []
for gp in ['white', 'afam', 'hisp', 'asian', 'other']:
    results_pred_score_train.append(ba_pred_score_train(gp))
    
    
n1 = results_1[0][1]
p1 = results_1[0][2]
tpr_pval = []
for i in range(1, len(results_1)):
    subgp = results_1[i][0]
    n2 = results_1[i][1]
    p2 = results_1[i][2]
    tpr_pval.append((subgp, two_side_z_test(n1,p1,n2,p2)))
        
n1 = results_0[0][1]
p1 = results_0[0][2]
precision0_pval = []
for i in range(1, len(results_0)):
    subgp = results_0[i][0]
    n2 = results_0[i][1]
    p2 = results_0[i][2]
    precision0_pval.append((subgp, two_side_z_test(n1,p1,n2,p2)))
        
n1 = results_0[0][3]
p1 = results_0[0][4]
fpr_pval = []
for i in range(1, len(results_0)):
    subgp = results_0[i][0]
    n2 = results_0[i][3]
    p2 = results_0[i][4]
    fpr_pval.append((subgp, two_side_z_test(n1,p1,n2,p2)))
    
n1 = results_bas[0][1]
p1 = results_bas[0][2]
bas_pval = []
for i in range(1, len(results_bas)):
    subgp = results_bas[i][0]
    n2 = results_bas[i][1]
    p2 = results_bas[i][2]
    bas_pval.append((subgp, two_side_z_test(n1,p1,n2,p2)))
    
n1 = results_pred_score[0][1]
p1 = results_pred_score[0][2]
v1 = results_pred_score[0][3]
pred_score_pval = []
for i in range(1, len(results_pred_score)):
    subgp = results_pred_score[i][0]
    n2 = results_pred_score[i][1]
    p2 = results_pred_score[i][2]
    v2 = results_pred_score[i][3]
    pred_score_pval.append((subgp, two_side_z_test_2(n1,p1,v1,n2,p2,v2)))
    
n1 = results_pred_score_train[0][1]
p1 = results_pred_score_train[0][2]
v1 = results_pred_score_train[0][3]
pred_score_pval_train = []
for i in range(1, len(results_pred_score_train)):
    subgp = results_pred_score_train[i][0]
    n2 = results_pred_score_train[i][1]
    p2 = results_pred_score_train[i][2]
    v2 = results_pred_score_train[i][3]
    pred_score_pval_train.append((subgp, two_side_z_test_2(n1,p1,v1,n2,p2,v2)))


for t in results_pred_score:
    print(np.sqrt(t[-1]))
for t in results_pred_score_train:
    print(np.sqrt(t[-1]))


results_tpr_df = pd.DataFrame(results_1, columns=['subgroup', 'N', 'true_positive_rate']).drop(['N'], axis=1).\
merge(pd.DataFrame(tpr_pval, columns = ['subgroup', 'p-value']), on=['subgroup'], how='left').round(4)
results_tpr_df.to_csv("bias_tpr.csv", index=False)
results_fpr_df = pd.DataFrame(results_0, columns=['subgroup', 'N1', 'precision0', 'N2', 'false_positive_rate']).drop(['N1','precision0','N2'], axis=1).\
merge(pd.DataFrame(fpr_pval, columns = ['subgroup', 'p-value']), on=['subgroup'], how='left').round(4)
results_fpr_df.to_csv("bias_fpr.csv", index=False)
results_precision0_df = pd.DataFrame(results_0, columns=['subgroup', 'N1', 'precision0', 'N2', 'false_positive_rate']).drop(['N1','N2','false_positive_rate'], axis=1).\
merge(pd.DataFrame(precision0_pval, columns = ['subgroup', 'p-value']), on=['subgroup'], how='left').round(4)
results_precision0_df.to_csv("bias_precision0.csv", index=False)
results_bas_df = pd.DataFrame(results_bas, columns=['subgroup', 'N', 'balanced_accuracy_score']).drop(['N'], axis=1).\
merge(pd.DataFrame(bas_pval, columns = ['subgroup', 'p-value']), on=['subgroup'], how='left').round(4)
results_bas_df.to_csv("bias_bas.csv", index=False)
results_pred_score_df = pd.DataFrame(results_pred_score, columns=['subgroup', 'N', 'mean_pred_score', 'var_pred_score']).drop(['N', 'var_pred_score'], axis=1).\
merge(pd.DataFrame(pred_score_pval, columns = ['subgroup', 'p-value']), on=['subgroup'], how='left').round(4)
results_pred_score_df.to_csv("bias_pred_score.csv", index=False)
results_pred_score_train_df = pd.DataFrame(results_pred_score_train, columns=['subgroup', 'N', 'mean_pred_score', 'var_pred_score']).drop(['N', 'var_pred_score'], axis=1).\
merge(pd.DataFrame(pred_score_pval_train, columns = ['subgroup', 'p-value']), on=['subgroup'], how='left').round(4)
results_pred_score_train_df.to_csv("bias_pred_score_train.csv", index=False)


new_pred_real = pd.DataFrame({'pred_y': pred_y,'real_y': test_y, 'race': race_column})
new_pred_real.loc[:,'pred_y_binned'] = pd.cut(new_pred_real.pred_y, bins=[0] + list(np.percentile(new_pred_real.pred_y, np.arange(2,100,2))) + [1])
new_pred_real.loc[:,'pred_y_binned_2'] = pd.cut(new_pred_real.pred_y, bins=[0] + list(np.percentile(new_pred_real.pred_y, np.arange(10,100,10))) + [1])
pct_dict = {e:(10*indx+5) for indx, e in enumerate(sorted(np.unique(new_pred_real.pred_y_binned_2)))}


for r in ['afam', 'hisp', 'asian', 'other']:
    print(r)
    new_sub = new_pred_real.copy()[new_pred_real.race.apply(lambda x: x in ['white', r])]
    new_sub = new_sub.groupby(['pred_y_binned', 'race']).agg({'real_y':'mean'}).reset_index()
    new_sub.loc[:,r] = new_sub.race.apply(lambda x: 1 if x == r else 0)
    new_sub = new_sub.sort_values([r, 'pred_y_binned'])
    print(new_sub.shape[0])
    new_sub.loc[:,'pred_score_percentile'] = list(np.linspace(1,99,50))*2
    new_sub = new_sub.rename(columns={'real_y':'share_of_actual_ABC'}).drop(['pred_y_binned'], axis=1)

    sns.set_style(style = "darkgrid")
    fig, ax = plt.subplots(1,1, figsize=(24,16.5))
    sns.scatterplot(x="pred_score_percentile", y="share_of_actual_ABC", hue='race', hue_order = ['white', r],
                    data=new_sub,
                    palette = ['C0','C2'], marker="x", ax=ax, s=150, alpha=0.7, linewidth = 3)
    ax.legend(fontsize='x-large', title_fontsize='40', markerscale=3)
    plt.xticks(np.linspace(0,100, 11),fontsize=18)
    plt.yticks(np.linspace(0,1,11),fontsize=18)
    
    new_sub = new_pred_real.copy()[new_pred_real.race.apply(lambda x: x in ['white', r])]
    new_sub.loc[:,'pred_score_percentile_new'] = new_sub.pred_y_binned_2.apply(lambda x: pct_dict[x])
    np.random.seed(4321)
    sns.lineplot(data=new_sub, x="pred_score_percentile_new", y="real_y", hue='race', hue_order = ['white', r],
                 err_style="bars", err_kws = {'capsize': 8, 'elinewidth':3, 'capthick':2}, 
                 ci=95, ax=ax, linewidth = 6,
                 palette = ['C0','C2'], legend=False,
                 marker=".", markersize=36)
    plt.xlabel("pred_score_percentile", fontsize=24)
    plt.ylabel("share_of_actual_ABC", fontsize=24)
    
    plt.savefig(r +"_predictive_parity.png")


new_sub_list = []
for r in ['white', 'afam', 'hisp', 'asian', 'other']:
    new_sub = new_pred_real.copy()[new_pred_real.race == r]
    new_sub.loc[:,'pred_score_percentile_new'] = new_sub.pred_y_binned_2.apply(lambda x: pct_dict[x])
    new_sub = new_sub[new_sub.pred_score_percentile_new.apply(lambda x: x in {15,55,85})]
    new_sub = new_sub.groupby(['pred_score_percentile_new']).agg({'real_y': ['count', 'mean']}).reset_index()
    new_sub = new_sub[new_sub[('real_y', 'count')] != 0]
    new_sub.columns = ['percentile', 'N', 'mean']
    new_sub.loc[:,'race'] = r
    new_sub_list.append(new_sub.copy())
new_sub_all = pd.concat(new_sub_list)


for p in [15,55,85]:
    sub = new_sub_all[new_sub_all.percentile == p].copy()
    p_val_list = [np.nan]
    n1 = sub.N.iloc[0]
    p1 = sub['mean'].iloc[0]
    for i in range(1,sub.shape[0]):
        n2 = sub.N.iloc[i]
        p2 = sub['mean'].iloc[i]
        p_val_list.append(two_side_z_test(n1,p1,n2,p2))
    sub.loc[:,'p-value'] = p_val_list
    sub = sub.loc[:,['race', 'N', 'mean','p-value']].rename(columns={'mean':'share_of_actual_ABC'}).round(4)
    sub.to_csv("predictive_parity_pct_{}.csv".format(p))