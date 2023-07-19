from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC

from scipy.optimize import curve_fit

from imblearn.ensemble import EasyEnsembleClassifier

import numpy as np
import pandas as pd

# Local imports
from tools_and_calib_metrics import cap_proba, adjust_posterior_prob_to_new_prior, get_inc_poly, get_inc_poly_bounded


class Classifier(): 
    
    def __init__(self):        
        self.X_train = None
        self.ml_pipeline = None
        self.classes_ = np.array([0, 1])
    
    
    def set_training_data(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    
    def fit(self):
        """ Fit the algorithm with the train_set """
        
        if self.X_train is None:
            raise Exception("The training set is missing --> call first: set_training_data()")
            
        fs_procent_ratio=0.8 ## % of variables to keep
        fs_nb_var = int(len(self.X_train.columns)*fs_procent_ratio)
        print(f" Feature selection: keep {fs_nb_var} variables from {len(self.X_train.columns)}")
        
        self.ml_pipeline = make_pipeline(StandardScaler(),
                                         SelectKBest(mutual_info_classif, k=fs_nb_var), 
                                         EasyEnsembleClassifier(n_estimators = 120,
                                                                estimator = RandomForestClassifier(), 
                                                                sampling_strategy = 'all',
                                                                n_jobs = 30))        
        self.ml_pipeline.fit(self.X_train, self.y_train) 

    
    def fit_SVM(self):
        """ Fit the algorithm with the train_set """
        
        if self.X_train is None:
            raise Exception("The training set is missing --> call first: set_training_data()")
            
        fs_procent_ratio=0.8 ## % of variables to keep
        fs_nb_var = int(len(self.X_train.columns)*fs_procent_ratio)
        print(f" Feature selection: keep {fs_nb_var} variables from {len(self.X_train.columns)}")
        
        self.ml_pipeline = make_pipeline(StandardScaler(),
                                         SelectKBest(mutual_info_classif, k=fs_nb_var), 
                                         EasyEnsembleClassifier(n_estimators = 120,
                                                                estimator = SVC(probability=True), 
                                                                sampling_strategy = 'all',
                                                                n_jobs = 30))        
        self.ml_pipeline.fit(self.X_train, self.y_train) 


    def fit_SVM_without_reb(self):
        """ Fit and Calibrate the algorithm with the training and calibration set """

        if self.X_train is None:
            raise Exception("The training set is missing --> call first: set_training_data()")
        

        # Training
        fs_procent_ratio=0.8 ## % of variables to keep
        fs_nb_var = int(len(self.X_train.columns)*fs_procent_ratio)
        print(f" Feature selection: keep {fs_nb_var} variables from {len(self.X_train.columns)}")
        
        self.ml_pipeline = make_pipeline(StandardScaler(),
                                            SelectKBest(mutual_info_classif, k=fs_nb_var), 
                                            SVC(probability=True))        
        self.ml_pipeline.fit(self.X_train, self.y_train)

        
    def predict_proba(self, X): 
        """ Return the prediction for X """
        
        if self.ml_pipeline is None:
            raise Exception("It is needed to call fit() before predict_proba()")

        res = pd.DataFrame(self.ml_pipeline.predict_proba(X)[:,1], index=X.index)
        
        return res
    
    
    def predict(self, X): 
        """ Return the prediction class for X """
        
        if self.ml_pipeline is None:
            raise Exception("It is needed to call fit() before predict_proba()")

        res = pd.DataFrame(self.ml_pipeline.predict(X), index=X.index)
        
        return res  

    

class ClassifierCalibrated(Classifier):

    def __init__(self):
        super().__init__()
        self.X_val_for_calibration = None
        self.method = "isotonic"
        self.estimated_test_prior = None


    def set_method_to_calibrate(self, method):
        """ Set the calibration method: 
        - 'adjust_posterior' for adjusting the posterior probability only
        - 'isotonic' for isotonic regression
        - 'adjust_posterior_and_isotonic' for adjusting the posterior probability before isotonic regression 
        - 'sigmoid' for Platt Scalling
        - 'adjust_posterior_and_sigmoid' for adjusting the posterior probability before Platt Scalling
        - 'exponential' for exponential regression ('exponential_bounded' or 'exponential_min1' for adding constraints)
        - 'polynomial_pos' for polynomial regression with a degree of 10 and positive coefficient only ('polynomial_pos_bounded' or 'polynomial_pos_min1' for adding constraints)
        - 'polynomial_inc' for polynomial non-decreasing regression with a degree of 10 ('polynomial_inc_bounded' for adding constraints)
        """

        self.method = method


    def fit_RF_and_let_val_set(self, calibration_size):
        """ Fit and Calibrate the algorithm with the training and calibration set """

        if self.X_train is None:
            raise Exception("The training set is missing --> call first: set_training_data()")
        
        self.X_train_for_calibration, self.X_val_for_calibration, self.y_train_for_calibration, self.y_val_for_calibration = train_test_split(self.X_train, self.y_train, test_size=calibration_size, random_state=42, stratify=self.y_train)

        # Training
        fs_procent_ratio=0.8 ## % of variables to keep
        fs_nb_var = int(len(self.X_train_for_calibration.columns)*fs_procent_ratio)
        print(f" Feature selection: keep {fs_nb_var} variables from {len(self.X_train_for_calibration.columns)}")
        
        self.ml_pipeline = make_pipeline(StandardScaler(),
                                            SelectKBest(mutual_info_classif, k=fs_nb_var), 
                                            EasyEnsembleClassifier(n_estimators = 120,
                                                                estimator = RandomForestClassifier(), 
                                                                sampling_strategy = 'all',
                                                                n_jobs = 30))        
        self.ml_pipeline.fit(self.X_train_for_calibration, self.y_train_for_calibration)


    def fit_SVM_and_let_val_set(self, calibration_size):
        """ Fit and Calibrate the algorithm with the training and calibration set """

        if self.X_train is None:
            raise Exception("The training set is missing --> call first: set_training_data()")
        
        self.X_train_for_calibration, self.X_val_for_calibration, self.y_train_for_calibration, self.y_val_for_calibration = train_test_split(self.X_train, self.y_train, test_size=calibration_size, random_state=42, stratify=self.y_train)

        # Training
        fs_procent_ratio=0.8 ## % of variables to keep
        fs_nb_var = int(len(self.X_train_for_calibration.columns)*fs_procent_ratio)
        print(f" Feature selection: keep {fs_nb_var} variables from {len(self.X_train_for_calibration.columns)}")
        
        self.ml_pipeline = make_pipeline(StandardScaler(),
                                            SelectKBest(mutual_info_classif, k=fs_nb_var), 
                                            EasyEnsembleClassifier(n_estimators = 120,
                                                                estimator = SVC(probability=True), 
                                                                sampling_strategy = 'all',
                                                                n_jobs = 30))        
        self.ml_pipeline.fit(self.X_train_for_calibration, self.y_train_for_calibration)
        
        
    def predict_proba_calibrated(self, X, proba_uncalibrated, preds_uncalibrated_val):
        """ Return the calibrated prediction for X """

        if self.ml_pipeline is None or self.X_train_for_calibration is None:
            raise Exception("It is needed to call fit_and_calibrate() before predict_proba_calibrated()")

        if self.method == "not_calibrated":
            proba_calibrated = proba_uncalibrated
            
        elif self.method == "adjust_posterior_to_source":
            N_pos = len(self.y_train_for_calibration[self.y_train_for_calibration == 1])
            N_neg = len(self.y_train_for_calibration[self.y_train_for_calibration == 0])
            self.beta = N_pos/N_neg # probability of selecting a negative instance with undersampling. Since in EasyEnsemble we have 50-50 and N_neg_after_undersampling = beta * N_neg
            
            proba_calibrated = adjust_posterior_prob_to_new_prior(self.beta, proba_uncalibrated)
        
        elif self.method == "isotonic":
            self.calibration_method = IsotonicRegression(out_of_bounds = 'clip')
            self.calibration_method.fit(preds_uncalibrated_val, self.y_val_for_calibration)
            
            proba_calibrated = pd.DataFrame(self.calibration_method.predict(proba_uncalibrated), index=X.index)

        elif self.method == "sigmoid":
            self.calibration_method = LogisticRegression(penalty="none")
            self.calibration_method.fit(preds_uncalibrated_val, self.y_val_for_calibration)
            
            proba_calibrated = pd.DataFrame(self.calibration_method.predict_proba(proba_uncalibrated)[:, 1], index=X.index)            
            
        elif self.method == "exponential":
            def func(t, a, b):
                return a * (np.exp(b * t) - 1)
            
            popt, pcov = curve_fit(func, preds_uncalibrated_val[0].to_numpy(), self.y_val_for_calibration.to_numpy(), maxfev=10000000, bounds=(0, np.inf))
            a = popt[0]
            b = popt[1]

            proba_calibrated = (a * (np.exp(b * proba_uncalibrated) - 1)).applymap(cap_proba)
            
        elif self.method == "exponential_bounded":
            popt, pcov = curve_fit(lambda t, a: (np.exp(a * t) - 1) / (np.exp(a) - 1), preds_uncalibrated_val[0].to_numpy(), self.y_val_for_calibration.to_numpy(), maxfev=10000000)
            a = popt[0]

            proba_calibrated = ((np.exp(a * proba_uncalibrated) - 1) / (np.exp(a) - 1)).applymap(cap_proba)
            
        elif self.method == "exponential_min1":
            def func(t, a, b):
                return [min(a * (np.exp(b * t_bis) - 1), 1) for t_bis in t]
            
            frac = 2
            frac -= 1
            weights = self.y_val_for_calibration * frac + 1
            popt, pcov = curve_fit(func, preds_uncalibrated_val[0].to_numpy(), self.y_val_for_calibration.to_numpy(), maxfev=10000000, bounds=(0, np.inf))
            a = popt[0]
            b = popt[1]

            proba_calibrated = (a * (np.exp(b * proba_uncalibrated) - 1)).applymap(cap_proba)
            
        elif self.method == "polynomial_pos":
            deg = 10
            poly = PolynomialFeatures(deg, include_bias=False)
            preds_uncalibrated_val_poly = poly.fit_transform(preds_uncalibrated_val[0].to_numpy().reshape(-1, 1))
            self.calibration_method = LinearRegression(positive=True, fit_intercept=False)
            self.calibration_method.fit(preds_uncalibrated_val_poly, self.y_val_for_calibration)
            proba_uncalibrated_poly = poly.transform(proba_uncalibrated)
            
            proba_calibrated = pd.DataFrame(self.calibration_method.predict(proba_uncalibrated_poly), index=X.index).applymap(cap_proba) 
        
        elif self.method == "polynomial_pos_bounded":
            def func_poly(t, b, c, d, e, f, g, h, i , j, k):
                vand = np.vander(t, 10+1, increasing=True)[:, 1:]
                p = np.array([b, c, d, e, f, g, h, i, j, k])
                t_output = (vand @ p) / p.sum()
                
                return t_output
            
            popt, pcov = curve_fit(func_poly, preds_uncalibrated_val[0].to_numpy(), self.y_val_for_calibration.to_numpy(), maxfev=10000000, bounds=(0, np.inf), p0=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
            V = np.vander(proba_uncalibrated[0].to_numpy(), 10+1, increasing=True)[:, 1:]
            
            proba_calibrated = pd.DataFrame(V @ popt / popt.sum(), index=X.index).applymap(cap_proba)
            
        elif self.method == "polynomial_pos_min1":            
            def func_poly(t, b, c, d, e, f, g, h, i , j, k):
                vand = np.vander(t, 10+1, increasing=True)[:, 1:]
                p = np.array([b, c, d, e, f, g, h, i, j, k])
                t_output = vand @ p
                
                return [min(t_output_in, 1) for t_output_in in t_output]
            
            popt, pcov = curve_fit(func_poly, preds_uncalibrated_val[0].to_numpy(), self.y_val_for_calibration.to_numpy(), maxfev=10000000, bounds=(0, np.inf), p0=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
            V = np.vander(proba_uncalibrated[0].to_numpy(), 10+1, increasing=True)[:, 1:]
            
            proba_calibrated = pd.DataFrame(V @ popt, index=X.index).applymap(cap_proba)
                                 
        elif self.method == "polynomial_inc":
            deg = 10
            P = get_inc_poly(preds_uncalibrated_val[0].to_numpy(), self.y_val_for_calibration, deg)
            V = np.vander(proba_uncalibrated[0].to_numpy(), deg+1, increasing=True)[:, 1:]
            
            proba_calibrated = pd.DataFrame(V @ P, index=X.index).applymap(cap_proba)
            
        elif self.method == "polynomial_inc_bounded":
            deg = 10
            P = get_inc_poly_bounded(preds_uncalibrated_val[0].to_numpy(), self.y_val_for_calibration, deg)
            V = np.vander(proba_uncalibrated[0].to_numpy(), deg+1, increasing=True)[:, 1:]
            
            proba_calibrated = pd.DataFrame(V @ P, index=X.index)

        elif self.method == "adjust_posterior_to_source_and_isotonic":
            N_pos = len(self.y_train_for_calibration[self.y_train_for_calibration == 1])
            N_neg = len(self.y_train_for_calibration[self.y_train_for_calibration == 0])
            self.beta = N_pos/N_neg # probability of selecting a negative instance with undersampling. Since in EasyEnsemble we have 50-50 and N_neg_after_undersampling = beta * N_neg
            preds_adjusted_val = adjust_posterior_prob_to_new_prior(self.beta, preds_uncalibrated_val)
            self.calibration_method = IsotonicRegression(out_of_bounds = 'clip')
            self.calibration_method.fit(preds_adjusted_val, self.y_val_for_calibration)
            
            proba_adj = adjust_posterior_prob_to_new_prior(self.beta, proba_uncalibrated)
            proba_calibrated = pd.DataFrame(self.calibration_method.predict(proba_adj), index=X.index)

        elif self.method == "adjust_posterior_to_source_and_sigmoid":
            N_pos = len(self.y_train_for_calibration[self.y_train_for_calibration == 1])
            N_neg = len(self.y_train_for_calibration[self.y_train_for_calibration == 0])
            self.beta = N_pos/N_neg # probability of selecting a negative instance with undersampling. Since in EasyEnsemble we have 50-50 and N_neg_after_undersampling = beta * N_neg
            
            preds_adjusted_val = adjust_posterior_prob_to_new_prior(self.beta, preds_uncalibrated_val)

            # Find a linear combination between adjusted and non-adjusted proba in order to have a reliability diagram that is the most similar as possible to a sigmoid
            all_loss = {}
            for lambdaa in np.linspace(0, 1, 21):
                calibration_method = LogisticRegression(penalty="none")
                calibration_method.fit(preds_adjusted_val*lambdaa + preds_uncalibrated_val*(1-lambdaa), self.y_val_for_calibration)
                
                loss = mean_squared_error(self.y_val_for_calibration, calibration_method.predict_proba(np.array(preds_adjusted_val*lambdaa + preds_uncalibrated_val*(1-lambdaa)).reshape(-1, 1))[:, 1])
                all_loss[lambdaa] = loss

            self.lambdaa = min(all_loss.keys(), key= lambda key:all_loss[key])
            
            proba_adj = adjust_posterior_prob_to_new_prior(self.beta, proba_uncalibrated)
            self.calibration_method = LogisticRegression(penalty="none")
            self.calibration_method.fit(preds_adjusted_val*self.lambdaa + preds_uncalibrated_val*(1-self.lambdaa), self.y_val_for_calibration)            

            proba_calibrated = pd.DataFrame(self.calibration_method.predict_proba(proba_adj*self.lambdaa + proba_uncalibrated*(1-self.lambdaa))[:, 1], index=X.index)


        return proba_calibrated