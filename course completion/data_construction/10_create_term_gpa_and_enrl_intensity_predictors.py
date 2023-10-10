import pickle
from sklearn.linear_model import LinearRegression
import numpy as np
import multiprocessing
import datetime as dt

fpath = "C:/Users/ys8mz/Box Sync/Clickstream/data/bias"


class Term_GPA_and_Credits_Processor(object):
    # The purpose of this class object is to calculate the GPA/Enrollment_Intensity of each student x term based on the prior term GPA and term credits earned data prior to this term

    def __init__(self, ftype, indx, ol):
        self.ftype = ftype # it's either "gpa" or "enrl_intensity"
        self.indx = indx
        self.ol = ol # outer list, a list of tuples
        self.val_dict = {}
        
    def find_slope(self):
        for t in self.ol: # Loop through the list of prior term GPA/Enrollment_Intensity values corresponding to different student x term
            vccsid = t[0][:-4]
            strm = int(t[0][-4:])
            l = t[1] # The list of term GPA or term intensity values sorted in chronological order
            if len(l) < 2: # If there's only one prior term, the slope will be 0
                self.val_dict[(vccsid,strm)] = 0
            elif len(l) == 2: # If there're precisely two prior terms, the slope will simply be the difference between the previous term and the term before the previous term
                self.val_dict[(vccsid,strm)] = l[1] - l[0]
            else: # If there're more than two prior terms, assume the time point of the current term is t, we'll loop through t_0 = 1,2,3,...,t-3, and calculate the slope of the list that contains the data points from t_0 to t-1 using simple linear regression, and choose the slope value which has the highest R^2 score as the final value for the predictor
                slope_l = []
                rsq_l = []
                for i in range(len(l)-2):
                    x = np.array([list(range(i,len(l)))]).T
                    y = l[i:]
                    reg = LinearRegression()
                    reg.fit(x,y)
                    slope_l.append(reg.coef_[0])
                    rsq_l.append(reg.score(x,y)) # reg.score returns the R^2 of linear regression
                    self.val_dict[(vccsid,strm)] = slope_l[np.argmax(rsq_l)]
    
    def save(self):
        # The output of parallel processing will be individual pickled dictionary files corresponding to each student x term: they'll be merged in later steps
        pickle.dump(self.val_dict, open(fpath + "/term_{0}_trend_values/{1}.p".format(self.ftype, self.indx), "wb"))
        
def term_gpa_and_credits_processor_wrapper(q):
    # Wrapper for the individual processors which is necessary for parallel processing using the multiprocessing library
    tgcp = Term_GPA_and_Credits_Processor(q[0], q[1], q[2])
    tgcp.find_slope()
    tgcp.save()
    
    
if __name__ == '__main__':
    print("Beginning Time:", dt.datetime.now())
    # First create GPA trend predictors
    results_1 = pickle.load(open(fpath + "/results_1.p", "rb")) # The input file is created by the script "find_prior_terms_gpa_and_enrl_intensity.ipynb"
    query_list = [results_1[i:(i+int(1e4))] for i in range(0,len(results_1),int(1e4))] # Parallelization: each processor will process 10,000 instances (student x term)
    query_list = [("gpa", indx, q) for indx,q in enumerate(query_list)] # The 3rd element of the tuple q is itself a list of tuples containing the pairs of student x (list of prior gpa/enrl_intensity values)
    pool = multiprocessing.Pool(multiprocessing.cpu_count()) # Use parallel processing to speed up this script, as the trendline predictors of each student x term can be constructed independently
    pool.map(term_gpa_and_credits_processor_wrapper, query_list)
    # Next create enrollment intensity trend predictors
    results_2 = pickle.load(open(fpath + "/results_2.p", "rb")) # The input file is created by the script "find_prior_terms_gpa_and_enrl_intensity.ipynb"
    query_list = [results_2[i:(i+int(1e4))] for i in range(0,len(results_2),int(1e4))] # Parallelization: each processor will process 10,000 instances (student x term)
    query_list = [("enrl_intensity", indx, q) for indx,q in enumerate(query_list)]
    pool = multiprocessing.Pool(multiprocessing.cpu_count() // 3)
    pool.map(term_gpa_and_credits_processor_wrapper, query_list)
    print("End Time:", dt.datetime.now())