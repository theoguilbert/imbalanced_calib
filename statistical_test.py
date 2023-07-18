import numpy as np
import pandas as pd
import scipy.stats as st
import scikit_posthocs as sp

def get_error_all_iterations(calib_errors_all_metric_all_datasets, columns_interest, name_metric, threshold=None):
    name_dataset = calib_errors_all_metric_all_datasets.keys()
    wilc_test = pd.DataFrame(columns=name_dataset, index=columns_interest)
    error_all_iterations = {}

    for dataset in name_dataset:
        if name_metric == "Avg_ECE_top_k":            
            error = calib_errors_all_metric_all_datasets[dataset]["ECE_topk"]
            for method in wilc_test.index:
                if dataset not in error_all_iterations:
                    error_all_iterations[dataset] = pd.DataFrame()         
                error_all_iterations[dataset][method] = error[method].mean(axis=0).values
                           
        else:
            error = calib_errors_all_metric_all_datasets[dataset][name_metric]
            for method in wilc_test.index:
                if dataset not in error_all_iterations:
                    error_all_iterations[dataset] = pd.DataFrame()
                    
                if name_metric == "ECE_topk" or name_metric == "ECE_topk_perc":
                    if threshold is None:
                        print("You need to provide a threshold for this metric")
                    error_all_iterations[dataset][method] = error[method].loc[threshold].values
                    
                else:
                    error_all_iterations[dataset][method] = error[method]
                
    return error_all_iterations

def wilcoxon_test(calib_errors_all_metric_all_datasets, columns_interest, name_metric, threshold=None):
    name_dataset = calib_errors_all_metric_all_datasets.keys()
    wilc_test = pd.DataFrame(columns=name_dataset, index=columns_interest)
    error_all_iterations = get_error_all_iterations(calib_errors_all_metric_all_datasets, columns_interest, name_metric, threshold=threshold)
    
    for dataset in name_dataset:   
        error_mean = error_all_iterations[dataset].mean(axis=0)
        best_method = min(error_mean.index, key= lambda x:error_mean[x])
        error_best_method = error_all_iterations[dataset][best_method]
        for method in wilc_test.index:
            error_method = error_all_iterations[dataset][method]
            if (error_method - error_best_method).abs().sum() != 0:
                wilc_test.loc[method, dataset] = st.wilcoxon(error_method, error_best_method).pvalue
                
    return wilc_test


def regroup_all_iterations_and_datasets(calib_errors_all_metric_all_datasets, number_of_different_training, method_selec_plot):
    n_dataset = len(calib_errors_all_metric_all_datasets.keys())
    dataset_nb = 1
    index_for_df = [i for i in range(1, number_of_different_training * n_dataset + 1)]

    set_calib_errors = {"ECE_top_1_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ECE_top_5_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ECE_top_10_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ECE_top_20_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ECE_top_30_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ECE_top_40_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ECE_top_50_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ECE_top_60_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ECE_top_70_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ECE_top_80_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ECE_top_90_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ECE_top_100_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "Avg_ECE_top_k":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "Classic_ECE":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ACE_top_1_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ACE_top_5_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ACE_top_10_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ACE_top_20_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ACE_top_30_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ACE_top_40_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ACE_top_50_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ACE_top_60_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ACE_top_70_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ACE_top_80_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ACE_top_90_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "ACE_top_100_perc":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "Avg_ACE_top_k":pd.DataFrame(index=index_for_df, columns=method_selec_plot),
                            "Classic_ACE":pd.DataFrame(index=index_for_df, columns=method_selec_plot)}

    for dataset in calib_errors_all_metric_all_datasets.keys():
        calib_errors_all_metric = calib_errors_all_metric_all_datasets[dataset]   
        experiences_nb = list(np.array([i for i in range(1, number_of_different_training+1)]) + (dataset_nb - 1) * number_of_different_training)  

        ECE_topk_perc = calib_errors_all_metric["ECE_topk_perc"]
        for method in ECE_topk_perc.keys():
            for threshold in ECE_topk_perc[method].index:
                name_error = "ECE_top_%s_perc" % (int(threshold * 100))
                set_calib_errors[name_error].loc[experiences_nb, method] = ECE_topk_perc[method].T[threshold].values
                
        ECE_topk = calib_errors_all_metric["ECE_topk"]
        for method in ECE_topk.keys():
            avg_ece_topk = ECE_topk[method].mean(axis=0).values
            set_calib_errors["Avg_ECE_top_k"].loc[experiences_nb, method] = avg_ece_topk
            
        ECE = calib_errors_all_metric["ECE"]
        for method in ECE.keys():
            set_calib_errors["Classic_ECE"].loc[experiences_nb, method] = ECE[method].values
        
                
        ACE_topk_perc = calib_errors_all_metric["ACE_topk_perc"]
        for method in ACE_topk_perc.keys():
            for threshold in ACE_topk_perc[method].index:
                name_error = "ACE_top_%s_perc" % (int(threshold * 100))
                set_calib_errors[name_error].loc[experiences_nb, method] = ACE_topk_perc[method].T[threshold].values
        
        ACE_topk = calib_errors_all_metric["ACE_topk"]
        for method in ACE_topk.keys():
            avg_ace_topk = ACE_topk[method].mean(axis=0).values
            set_calib_errors["Avg_ACE_top_k"].loc[experiences_nb, method] = avg_ace_topk
            
        ACE = calib_errors_all_metric["ACE"]
        for method in ACE.keys():
            set_calib_errors["Classic_ACE"].loc[experiences_nb, method] = ACE[method].values
                
        dataset_nb += 1
        
    return set_calib_errors


def nemenyi_friedman_test(set_calib_errors):
    results_test_stats = {}
    results_test_stats_only_poly_pos = {}
    results_test_stats_only_poly_inc = {}
    results_test_stats_only_exp = {}

    for error_name in set_calib_errors.keys():
        error = set_calib_errors[error_name][["isotonic", "sigmoid", "adjust_posterior_to_source_and_sigmoid", "polynomial_pos", "polynomial_inc", "exponential"]]
        p_value_friedman_test = st.friedmanchisquare(*[list(error[method].values) for method in error.columns if method != "not_calibrated"]).pvalue
        nemenyi_test = sp.posthoc_nemenyi_friedman(error.to_numpy())
        nemenyi_test.columns = error.columns
        nemenyi_test.index = error.columns
        
        results_test_stats[error_name] = {"p_val_friedman":p_value_friedman_test, "p_vals_nemenyi":nemenyi_test}
        
        error = set_calib_errors[error_name][["polynomial_pos", "polynomial_pos_bounded", "polynomial_pos_min1", "polynomial_pos_l2"]]
        p_value_friedman_test = st.friedmanchisquare(*[list(error[method].values) for method in error.columns if method != "not_calibrated"]).pvalue
        nemenyi_test = sp.posthoc_nemenyi_friedman(error.to_numpy())
        nemenyi_test.columns = error.columns
        nemenyi_test.index = error.columns
        
        results_test_stats_only_poly_pos[error_name] = {"p_val_friedman":p_value_friedman_test, "p_vals_nemenyi":nemenyi_test}
        
        error = set_calib_errors[error_name][["polynomial_inc", "polynomial_inc_bounded", "polynomial_inc_l2"]]
        p_value_friedman_test = st.friedmanchisquare(*[list(error[method].values) for method in error.columns if method != "not_calibrated"]).pvalue
        nemenyi_test = sp.posthoc_nemenyi_friedman(error.to_numpy())
        nemenyi_test.columns = error.columns
        nemenyi_test.index = error.columns
        
        results_test_stats_only_poly_inc[error_name] = {"p_val_friedman":p_value_friedman_test, "p_vals_nemenyi":nemenyi_test}
            
        error = set_calib_errors[error_name][["exponential", "exponential_bounded", "exponential_min1"]]
        p_value_friedman_test = st.friedmanchisquare(*[list(error[method].values) for method in error.columns if method != "not_calibrated"]).pvalue
        nemenyi_test = sp.posthoc_nemenyi_friedman(error.to_numpy())
        nemenyi_test.columns = error.columns
        nemenyi_test.index = error.columns
        
        results_test_stats_only_exp[error_name] = {"p_val_friedman":p_value_friedman_test, "p_vals_nemenyi":nemenyi_test}
        
    return results_test_stats, results_test_stats_only_poly_pos, results_test_stats_only_poly_inc, results_test_stats_only_exp