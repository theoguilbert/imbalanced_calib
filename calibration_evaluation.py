import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.stats as st
import random
import os

from model import ClassifierCalibrated
from tools import return_calibration_metrics_top


def evaluate_calibration_for_all_metrics_topk_giving_one_dataset(X, y, method_list=['isotonic'], test_size=0.2, val_size_for_calibration=0.4, number_of_different_training=5, nb_bins=15):
    '''Take a dataset with features and targeted values and return a dictionary with errors for each metric.
    - X: Features
    - y: Target
    - method: Take the following argument as a list of str : 
        - 'adjust_posterior' for adjusting the posterior probability only
        - 'isotonic' for isotonic regression
        - 'adjust_posterior_and_isotonic' for adjusting the posterior probability before isotonic regression 
        - 'sigmoid' for Platt Scalling
        - 'adjust_posterior_and_sigmoid' for adjusting the posterior probability before Platt Scalling
        - 'exponential' for exponential regression
        - 'polynomial_pos' for polynomial regression with a degree of 10 and positive coefficient only
        - 'polynomial_inc' for polynomial non-decreasing regression with a degree of 10
    - test_size: Test_size to test our calibration measures
    - val_size_for_calibration: Validation set size to train our post processing algorithm for calibrate the model
    - number_of_different_training: Number of time to repeat the same procedure in order to make a statistical analysis
    - nb_bins: Number of bins to compute the different calibration errors
    '''

    # metrics = ["ECE", "ECE_eq_weight", "ECE_L2", "ECE_L2_eq_weight", "MCE", "ACE", "LCE", "NLL", "BS"]
    metrics = ["ECE", "ECE_topk", "ECE_topk_perc", "ECE_weight_adj_violent", "ECE_weight_adj_soft", "ECE_weight_adj_semi", "ECE_weight_entropy", "ACE", "ACE_topk", "ACE_topk_perc"]
    calib_errors_all_metric = {}
    calib_errors_all_metric_without_calib = {}
    for metric in metrics:
        calib_errors_all_metric[metric] = {}
        calib_errors_all_metric_without_calib[metric] = {}


    for i in range(1, number_of_different_training+1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

        model = ClassifierCalibrated()
        model.set_training_data(X_train, y_train)
        model.fit_RF_and_let_val_set(calibration_size=val_size_for_calibration)
        pred_proba_uncalibrated = model.predict_proba(X_test)
        preds_uncalibrated_val = model.predict_proba(model.X_val_for_calibration)

        for method in method_list: 
            model.set_method_to_calibrate(method)
            
            pred_proba = model.predict_proba_calibrated(X_test, pred_proba_uncalibrated, preds_uncalibrated_val)
            calibration_measures_test = return_calibration_metrics_top(y_test, pred_proba, nb_bins, pred_proba_uncalibrated)

            for metric_num, metric in enumerate(metrics):
                if metric not in ["ECE", "ECE_weight_adj_violent", "ECE_weight_adj_soft", "ECE_weight_adj_semi", "ECE_weight_entropy", "ACE"]:
                    if method not in calib_errors_all_metric[metric]:
                        calibration_measures_test[metric_num].columns = [i] # i = 1 here / to initialize for the first iteration
                        calib_errors_all_metric[metric][method] = calibration_measures_test[metric_num]
                    else:
                        calib_errors_all_metric[metric][method][i] = calibration_measures_test[metric_num]["error"]
                        
                else:
                    if method not in calib_errors_all_metric[metric]:
                        calib_errors_all_metric[metric][method] = pd.Series(calibration_measures_test[metric_num], index=[i]) # i = 1 here / to initialize for the first iteration
                    else:
                        calib_errors_all_metric[metric][method][i] = calibration_measures_test[metric_num]

        print(f"Calibration for the training number {i} is finished")


    return calib_errors_all_metric
        
    
def save_and_plot_graphs(calib_errors_all_metric, method_selec, method_selec_plot, name_dataset, X, y, folder_experiment):

    for method in method_selec:
        data = calib_errors_all_metric["ECE_topk"][method].loc[:int(y.shape[0] * 0.3 * 0.2), :]
        data_to_plot = data.mean(axis=1)
        data_std = data.std(axis=1, ddof=1) / np.sqrt(data.shape[0])
        confidence_interval = st.t.interval(confidence= 0.90, df= len(data)-1, loc= data_to_plot.values, scale= data_std.values)
        
        if method in method_selec_plot:
            data_to_plot.plot(label = method)
            plt.fill_between(data_to_plot.index, confidence_interval[0], confidence_interval[1], alpha=0.1)
        
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title("ECE@k")
    plt.xlabel("k")
    plt.ylabel("ECE")
    if not os.path.exists("%s/%s" % (folder_experiment, name_dataset)):
            os.makedirs("%s/%s" % (folder_experiment, name_dataset))
    plt.savefig("%s/%s/ECE_topk_until_20perc_fig.png" % (folder_experiment, name_dataset), bbox_inches="tight")
    plt.close()
    # plt.show()

    for method in method_selec:
        data = calib_errors_all_metric["ECE_topk_perc"][method]
        data_to_plot = data.mean(axis=1)
        data_std = data.std(axis=1, ddof=1) / np.sqrt(data.shape[0])
        confidence_interval = st.t.interval(confidence= 0.90, df= len(data)-1, loc= data_to_plot.values, scale= data_std.values)
        
        if method in method_selec_plot:
            data_to_plot.plot(label = method)
            plt.fill_between(data_to_plot.index, confidence_interval[0], confidence_interval[1], alpha=0.1)
        
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title("ECE@%k")
    plt.xlabel("%k")
    plt.ylabel("ECE")
    if not os.path.exists("%s/%s" % (folder_experiment, name_dataset)):
            os.makedirs("%s/%s" % (folder_experiment, name_dataset))
    plt.savefig("%s/%s/ECE_topk_perc_fig.png" % (folder_experiment, name_dataset), bbox_inches="tight")
    plt.close()
    # plt.show()

    for method in method_selec:
        data = calib_errors_all_metric["ACE_topk"][method].loc[:int(y.shape[0] * 0.3 * 0.2), :]
        data_to_plot = data.mean(axis=1)
        data_std = data.std(axis=1, ddof=1) / np.sqrt(data.shape[0])
        confidence_interval = st.t.interval(confidence= 0.90, df= len(data)-1, loc= data_to_plot.values, scale= data_std.values)
        
        if method in method_selec_plot:
            data_to_plot.plot(label = method)
            plt.fill_between(data_to_plot.index, confidence_interval[0], confidence_interval[1], alpha=0.1)        
        
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title("ACE@k")
    plt.xlabel("k")
    plt.ylabel("ACE")
    if not os.path.exists("%s/%s" % (folder_experiment, name_dataset)):
            os.makedirs("%s/%s" % (folder_experiment, name_dataset))
    plt.savefig("%s/%s/ACE_topk_until_20perc_fig.png" % (folder_experiment, name_dataset), bbox_inches="tight")
    plt.close()
    # plt.show()

    for method in method_selec:
        data = calib_errors_all_metric["ACE_topk_perc"][method]
        data_to_plot = data.mean(axis=1)
        data_std = data.std(axis=1, ddof=1) / np.sqrt(data.shape[0])
        confidence_interval = st.t.interval(confidence= 0.90, df= len(data)-1, loc= data_to_plot.values, scale= data_std.values)
        
        if method in method_selec_plot:
            data_to_plot.plot(label = method)
            plt.fill_between(data_to_plot.index, confidence_interval[0], confidence_interval[1], alpha=0.1)
        
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title("ACE@%k")
    plt.xlabel("%k")
    plt.ylabel("ACE")
    plt.close()
    # plt.show()               
    
    
def get_avg_ece_for_all(calib_errors_all_metric_all_datasets, list_calibration_method):
    name_dataset_list = calib_errors_all_metric_all_datasets.keys()
    
    avg_ece_for_all_mean = pd.DataFrame(index=list_calibration_method, columns=name_dataset_list)
    avg_ece_for_all_std = pd.DataFrame(index=list_calibration_method, columns=name_dataset_list)

    for name_dataset in name_dataset_list:
        for method in list_calibration_method:
            error_dataset_method = calib_errors_all_metric_all_datasets[name_dataset]["ECE_topk"][method].mean(axis=0)
        
            avg_ece_for_all_mean.loc[method, name_dataset] = error_dataset_method.mean(axis=0)
            avg_ece_for_all_std.loc[method, name_dataset] = error_dataset_method.std(ddof = 1) / np.sqrt(error_dataset_method.shape[0])
        
    
    return avg_ece_for_all_mean, avg_ece_for_all_std


def get_ece_topk_perc_for_all(calib_errors_all_metric_all_datasets, list_calibration_method):
    name_dataset_list = calib_errors_all_metric_all_datasets.keys()
    
    ece_topk_for_all_mean = {}
    ece_topk_for_all_std = {}
    
    for name_dataset in name_dataset_list:
        for method in list_calibration_method:
            
            for threshold in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                if threshold not in ece_topk_for_all_mean:
                    ece_topk_for_all_mean[threshold] = pd.DataFrame()
                    ece_topk_for_all_std[threshold] = pd.DataFrame()
                    
                error_dataset_method = calib_errors_all_metric_all_datasets[name_dataset]["ECE_topk_perc"][method].loc[threshold]
        
                ece_topk_for_all_mean[threshold].loc[method, name_dataset] = error_dataset_method.mean()
                ece_topk_for_all_std[threshold].loc[method, name_dataset] = error_dataset_method.std(ddof = 1) / np.sqrt(error_dataset_method.shape[0])
                
    return ece_topk_for_all_mean, ece_topk_for_all_std


def get_metric_for_all(calib_errors_all_metric_all_datasets, list_calibration_method, name_metric):
    name_dataset_list = calib_errors_all_metric_all_datasets.keys()
    
    metric_for_all_mean = pd.DataFrame(index=list_calibration_method, columns=name_dataset_list)
    metric_for_all_std = pd.DataFrame(index=list_calibration_method, columns=name_dataset_list)

    for name_dataset in name_dataset_list:
        for method in list_calibration_method:
            error_dataset_method = calib_errors_all_metric_all_datasets[name_dataset][name_metric][method]
            metric_for_all_mean.loc[method, name_dataset] = error_dataset_method.mean()
            metric_for_all_std.loc[method, name_dataset] = error_dataset_method.std(ddof = 1) / np.sqrt(error_dataset_method.shape[0])
            
    return metric_for_all_mean, metric_for_all_std


def rank_method(error_metric_mean, columns_interest, threshold=None):
    columns_interest_bis = columns_interest.copy()
    columns_interest_bis.remove("not_calibrated")
    
    if type(error_metric_mean) == dict:
        if threshold is None:
            print("You need to provide a threshold for this metric")
        error_metric_mean = error_metric_mean[threshold]
    
    mean_dataset = pd.DataFrame([error_metric_mean.T[columns_interest_bis].mean(axis=1).values for i in range(len(columns_interest_bis))], index=columns_interest_bis, columns = error_metric_mean.columns).T
    std_dataset = pd.DataFrame([error_metric_mean.T[columns_interest_bis].std(axis=1).values for i in range(len(columns_interest_bis))], index=columns_interest_bis, columns = error_metric_mean.columns).T
    z_score_dataset = (error_metric_mean.T[columns_interest_bis] - mean_dataset) / std_dataset
    rank_dataset = error_metric_mean.T[columns_interest].rank(axis=1)
    
    return rank_dataset, z_score_dataset