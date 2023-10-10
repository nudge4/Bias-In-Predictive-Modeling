# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:06:20 2021

@author: ys8mz
"""


import pickle
import pandas as pd
import numpy as np
import multiprocessing
import datetime as dt

fpath = "C:/Users/ys8mz/Box Sync/Clickstream/data/bias"


def create_cdict(df):
    cdict = {}
    for j in range(df.shape[0]):
        course = df.course.iloc[j]
        credit = df.credit.iloc[j]
        est_grade = df.est_grade.iloc[j]
        try:
            cdict[course].append((credit,est_grade))
        except KeyError:
            cdict[course] = [(credit,est_grade)]
    return cdict.copy()

def calc_avg_grade(t):
    tot_cred = 0
    tot_grade_point = 0
    for p in t:
        tot_cred += p[0]
        tot_grade_point += p[0]*p[1]
    return tot_grade_point/tot_cred


class Prereq(object):

    def __init__(self, indx, vccs_prerequisite_dict):
        self.indx = indx
        self.target_courses = pd.read_stata("C:\\Users\\ys8mz\\Box Sync\\Clickstream\\data\\bias\\target_courses\\part{}.dta".format(self.indx))
        self.prior_courses = pd.read_stata("C:\\Users\\ys8mz\\Box Sync\\Clickstream\\data\\bias\\prior_courses\\part{}.dta".format(self.indx))
        self.vccs_prerequisite_dict = vccs_prerequisite_dict
        
    def do(self):
        results = []
        for i in range(self.target_courses.shape[0]):
            vccsid = self.target_courses.vccsid.iloc[i]
            strm = self.target_courses.strm.iloc[i]
            college = self.target_courses.college.iloc[i]
            section = self.target_courses.section.iloc[i]
            course = self.target_courses.course.iloc[i]
            prior_sub = self.prior_courses[self.prior_courses.vccsid == vccsid]
            prior_sub = prior_sub[prior_sub.strm < strm]
            if prior_sub.shape[0] == 0:
                results.append((vccsid, strm, college, course, section, 0, 0))
            else:
                if course not in self.vccs_prerequisite_dict:
                    results.append((vccsid, strm, college, course, section, 0, 0))
                    continue
                else:
                    prerequisite_l = []
                    for l in self.vccs_prerequisite_dict[course]:
                        prerequisite_l.extend(l)
                    cdict = create_cdict(prior_sub)
                    clist = list(cdict.keys())
                    common_l = np.intersect1d(prerequisite_l, clist)
                    if len(common_l) == 0:
                        results.append((vccsid, strm, college, course, section, 0, 0))
                    else:
                        new_t = []
                        for c in common_l:
                            new_t.extend(cdict[c])
                        results.append((vccsid, strm, college, course, section, 1, calc_avg_grade(new_t)))
        pickle.dump(results, open("C:\\Users\\ys8mz\\Box Sync\\Clickstream\\data\\bias\\prereq\\part{}.p".format(self.indx), "wb"))


        
def prereq_wrapper(q):
    s = Prereq(q[0], q[1])
    s.do()
    
    
if __name__ == '__main__':
    print("Beginning Time:", dt.datetime.now())
    d = pickle.load(open("C:/Users/ys8mz/Box Sync/GAA Transfer/data/intermediate_files/ys8mz/degree_audit_analyses/vccs_prerequisite_dict_2021.p", "rb"))
    query_list = [(i, d) for i in range(500)]
    pool = multiprocessing.Pool(10) # Use parallel processing to speed up this script, as the trendline predictors of each student x term can be constructed independently
    pool.map(prereq_wrapper, query_list)
    print("End Time:", dt.datetime.now())