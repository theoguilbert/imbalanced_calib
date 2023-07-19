import numpy as np
import pandas as pd
import cvxpy as cp


def cap_proba(p):
    if p > 1:
        return 1
    elif p < 0:
        return 0
    else:
        return p

def adjust_posterior_prob_to_new_prior(beta, p_s):
    p_adjusted = (beta * p_s) / (beta * p_s - p_s + 1)
    if type(p_adjusted) == pd.DataFrame:
        return p_adjusted.applymap(cap_proba)
    
    return p_adjusted


def return_calibration_metrics(y_test, pred_probas, n_bins, pred_probas_uncalibrated):
    # Classic ECE
    ECE = compute_ECE(y_test, pred_probas, n_bins)
    ECE_not_calibrated = compute_ECE(y_test, pred_probas_uncalibrated, n_bins)
    
    ECE_eq_weight = compute_ECE_eq_weight(y_test, pred_probas, n_bins)
    ECE_eq_weight_not_calibrated = compute_ECE_eq_weight(y_test, pred_probas_uncalibrated, n_bins)

    # ECE with L2 norm
    ECE_L2 = compute_ECE_L2(y_test, pred_probas, n_bins)
    ECE_L2_not_calibrated = compute_ECE_L2(y_test, pred_probas_uncalibrated, n_bins)

    # MCE 
    MCE = compute_ECE_topk(y_test, pred_probas, n_bins, 200)
    MCE_not_calibrated = compute_ECE_topk(y_test, pred_probas_uncalibrated, n_bins, 200)

    # ACE (~ ECE but with equal amount of datas in each bin)
    ACE = compute_ACE(y_test, pred_probas, n_bins)
    ACE_not_calibrated = compute_ACE(y_test, pred_probas_uncalibrated, n_bins)

    return [ECE, ECE_eq_weight, ECE_L2, MCE, ACE], [ECE_not_calibrated, ECE_eq_weight_not_calibrated, ECE_L2_not_calibrated, MCE_not_calibrated, ACE_not_calibrated]


def return_calibration_metrics_top(y_test, pred_probas, n_bins, pred_probas_uncalibrated):
    ECE = compute_ECE(y_test, pred_probas, n_bins)
    
    ECE_topk = pd.DataFrame([[k, compute_ECE_topk(y_test, pred_probas, pred_probas_uncalibrated, n_bins, k)] for k in range(10, int(y_test.shape[0] * 0.2), 10)], columns=["threshold", "error"])
    ECE_topk.set_index("threshold", inplace=True)
    
    ECE_topk_perc = pd.DataFrame([[perc, compute_ECE_topk_perc(y_test, pred_probas, pred_probas_uncalibrated, n_bins, perc)] for perc in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]], columns=["threshold", "error"])
    ECE_topk_perc.set_index("threshold", inplace=True)
    
    ECE_weight_ajd_violent = compute_ECE_weight_adj_violent(y_test, pred_probas, n_bins)
    ECE_weight_ajd_soft = compute_ECE_weight_adj_soft(y_test, pred_probas, n_bins)
    ECE_weight_ajd_semi = compute_ECE_weight_adj_semi(y_test, pred_probas, n_bins)
    ECE_weight_entropy = compute_ECE_weight_entropy(y_test, pred_probas, n_bins)
    
    ACE = compute_ACE(y_test, pred_probas, n_bins)
    
    ACE_topk = pd.DataFrame([[k, compute_ACE_topk(y_test, pred_probas, pred_probas_uncalibrated, n_bins, k)] for k in range(10, int(y_test.shape[0] * 0.2), 10)], columns=["threshold", "error"])
    ACE_topk.set_index("threshold", inplace=True)
    
    ACE_topk_perc = pd.DataFrame([[perc, compute_ACE_topk_perc(y_test, pred_probas, pred_probas_uncalibrated, n_bins, perc)] for perc in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]], columns=["threshold", "error"])
    ACE_topk_perc.set_index("threshold", inplace=True)
    
    return ECE, ECE_topk, ECE_topk_perc, ECE_weight_ajd_violent, ECE_weight_ajd_soft, ECE_weight_ajd_semi, ECE_weight_entropy, ACE, ACE_topk, ACE_topk_perc


def compute_ECE(y_true, pred_probas, n_bins):  
    if pred_probas.shape[0] == 0:
        return np.nan
    else:
        bins = np.linspace(0, 1, n_bins + 1)
        dico_ece_bins = {}
        for bin_nb in range(len(bins)-1):
            bin_start, bin_end = bins[bin_nb], bins[bin_nb + 1]
            if bin_end == 1:
                bin_end = 1.01 # To ensure selecting the proba = 1 in the following line
                
            probas_in_bin, y_corresponding = pred_probas[(pred_probas[0] >= bin_start) * (pred_probas[0] < bin_end)], y_true[(pred_probas[0] >= bin_start) * (pred_probas[0] < bin_end)]
            ece_bin = abs(probas_in_bin[0].mean() - y_corresponding.mean()) * probas_in_bin.shape[0]
            dico_ece_bins[bin_nb] = ece_bin

        return sum(pd.Series(dico_ece_bins.values()).dropna()) / pred_probas.shape[0]
    

def compute_ECE_eq_weight(y_true, pred_probas, n_bins):
    if pred_probas.shape[0] == 0:
        return np.nan
    else:
        bins = np.linspace(0, 1, n_bins + 1)
        dico_ece_bins = {}
        n_bins_filled = 0
        for bin_nb in range(len(bins)-1):
            bin_start, bin_end = bins[bin_nb], bins[bin_nb + 1]
            if bin_end == 1:
                bin_end = 1.01 # To ensure selecting the proba = 1 in the following line
                
            probas_in_bin, y_corresponding = pred_probas[(pred_probas[0] >= bin_start) * (pred_probas[0] < bin_end)], y_true[(pred_probas[0] >= bin_start) * (pred_probas[0] < bin_end)]
            ece_bin = abs(probas_in_bin[0].mean() - y_corresponding.mean())
            dico_ece_bins[bin_nb] = ece_bin
            
            if probas_in_bin.shape[0] != 0:
                    n_bins_filled += 1

        return sum(pd.Series(dico_ece_bins.values()).dropna()) / n_bins_filled
    

def compute_ECE_topk(y_true, pred_probas, pred_probas_uncalibrated, n_bins, k):
    preds_probas_and_values = pd.concat([pred_probas_uncalibrated, pred_probas, y_true], axis=1)
    preds_probas_and_values.columns = ["unc", 0, "y"]
    preds_probas_and_values_topk = preds_probas_and_values.sort_values(by="unc").iloc[-k:, :]
    pred_probas = preds_probas_and_values_topk[0]
    y_true = preds_probas_and_values_topk["y"]
    
    if pred_probas.shape[0] == 0:
        return np.nan
    else:
        n_bins = max(1, int(np.log10(y_true.shape[0])))
        bins = np.linspace(preds_probas_and_values_topk[0].min(), 1, n_bins + 1)
        dico_ece_bins = {}
        for bin_nb in range(len(bins)-1):
            bin_start, bin_end = bins[bin_nb], bins[bin_nb + 1]
            if bin_end == 1:
                bin_end = 1.01 # To ensure selecting the proba = 1 in the following line
                
            probas_in_bin, y_corresponding = pred_probas[(pred_probas >= bin_start) * (pred_probas < bin_end)], y_true[(pred_probas >= bin_start) * (pred_probas < bin_end)]
            ece_bin = abs(probas_in_bin.mean() - y_corresponding.mean()) * probas_in_bin.shape[0]
            dico_ece_bins[bin_nb] = ece_bin

        return sum(pd.Series(dico_ece_bins.values()).dropna()) / pred_probas.shape[0]
    
    
def compute_ECE_topk_perc(y_true, pred_probas, pred_probas_uncalibrated, n_bins, perc):
    k = int(perc * y_true.shape[0])
    preds_probas_and_values = pd.concat([pred_probas_uncalibrated, pred_probas, y_true], axis=1)
    preds_probas_and_values.columns = ["unc", 0, "y"]
    preds_probas_and_values_topk = preds_probas_and_values.sort_values(by="unc").iloc[-k:, :]
    pred_probas = preds_probas_and_values_topk[0]
    y_true = preds_probas_and_values_topk["y"]
    
    if pred_probas.shape[0] == 0:
        return np.nan
    else:
        n_bins = max(1, int(np.log10(y_true.shape[0])))
        bins = np.linspace(preds_probas_and_values_topk[0].min(), 1, n_bins + 1)
        dico_ece_bins = {}
        for bin_nb in range(len(bins)-1):
            bin_start, bin_end = bins[bin_nb], bins[bin_nb + 1]
            if bin_end == 1:
                bin_end = 1.01 # To ensure selecting the proba = 1 in the following line
                
            probas_in_bin, y_corresponding = pred_probas[(pred_probas >= bin_start) * (pred_probas < bin_end)], y_true[(pred_probas >= bin_start) * (pred_probas < bin_end)]
            ece_bin = abs(probas_in_bin.mean() - y_corresponding.mean()) * probas_in_bin.shape[0]
            dico_ece_bins[bin_nb] = ece_bin

        return sum(pd.Series(dico_ece_bins.values()).dropna()) / pred_probas.shape[0]
    
    
def compute_ECE_weight_adj_violent(y_true, pred_probas, n_bins):    
    if pred_probas.shape[0] == 0:
        return np.nan
    else:
        bins = np.linspace(0, 1, n_bins + 1)
        dico_ece_bins = {}
        frac_pos = y_true.sum() / y_true.shape[0]
        sum_weights = 0
        for bin_nb in range(len(bins)-1):
            bin_start, bin_end = bins[bin_nb], bins[bin_nb + 1]
            if bin_end == 1:
                bin_end = 1.01 # To ensure selecting the proba = 1 in the following line
                
            probas_in_bin, y_corresponding = pred_probas[(pred_probas[0] >= bin_start) * (pred_probas[0] < bin_end)], y_true[(pred_probas[0] >= bin_start) * (pred_probas[0] < bin_end)]
            ece_bin = abs(probas_in_bin[0].mean() - y_corresponding.mean()) * (probas_in_bin.shape[0] ** (2*frac_pos))
            sum_weights += (probas_in_bin.shape[0] ** (2*frac_pos))
            dico_ece_bins[bin_nb] = ece_bin

        return sum(pd.Series(dico_ece_bins.values()).dropna()) / sum_weights
    
    
def compute_ECE_weight_adj_soft(y_true, pred_probas, n_bins):    
    if pred_probas.shape[0] == 0:
        return np.nan
    else:
        bins = np.linspace(0, 1, n_bins + 1)
        dico_ece_bins = {}
        frac_pos = y_true.sum() / y_true.shape[0]
        sum_weights = 0
        for bin_nb in range(len(bins)-1):
            bin_start, bin_end = bins[bin_nb], bins[bin_nb + 1]
            if bin_end == 1:
                bin_end = 1.01 # To ensure selecting the proba = 1 in the following line
                
            probas_in_bin, y_corresponding = pred_probas[(pred_probas[0] >= bin_start) * (pred_probas[0] < bin_end)], y_true[(pred_probas[0] >= bin_start) * (pred_probas[0] < bin_end)]
            ece_bin = abs(probas_in_bin[0].mean() - y_corresponding.mean()) * (probas_in_bin.shape[0] ** (0.5 + frac_pos))
            sum_weights += (probas_in_bin.shape[0] ** (0.5 + frac_pos))
            dico_ece_bins[bin_nb] = ece_bin

        return sum(pd.Series(dico_ece_bins.values()).dropna()) / sum_weights
    
    
def compute_ECE_weight_adj_semi(y_true, pred_probas, n_bins):   
    if pred_probas.shape[0] == 0:
        return np.nan
    else:
        bins = np.linspace(0, 1, n_bins + 1)
        dico_ece_bins = {}
        frac_pos = y_true.sum() / y_true.shape[0]
        sum_weights = 0
        for bin_nb in range(len(bins)-1):
            bin_start, bin_end = bins[bin_nb], bins[bin_nb + 1]
            if bin_end == 1:
                bin_end = 1.01 # To ensure selecting the proba = 1 in the following line
                
            probas_in_bin, y_corresponding = pred_probas[(pred_probas[0] >= bin_start) * (pred_probas[0] < bin_end)], y_true[(pred_probas[0] >= bin_start) * (pred_probas[0] < bin_end)]
            ece_bin = abs(probas_in_bin[0].mean() - y_corresponding.mean()) * ((probas_in_bin.shape[0] ** (2*frac_pos)) + (probas_in_bin.shape[0] ** (0.5 + frac_pos))) / 2
            sum_weights += ((probas_in_bin.shape[0] ** (2*frac_pos)) + (probas_in_bin.shape[0] ** (0.5 + frac_pos))) / 2
            dico_ece_bins[bin_nb] = ece_bin

        return sum(pd.Series(dico_ece_bins.values()).dropna()) / sum_weights
    
    
def compute_ECE_weight_entropy(y_true, pred_probas, n_bins):        
        if pred_probas.shape[0] == 0:
            return np.nan
        else:
            bins = np.linspace(0, 1, n_bins + 1)
            dico_ece_bins = {}
            frac_pos = y_true.sum() / y_true.shape[0]
            sum_weights = 0
            for bin_nb in range(len(bins)-1):
                bin_start, bin_end = bins[bin_nb], bins[bin_nb + 1]
                if bin_end == 1:
                    bin_end = 1.01 # To ensure selecting the proba = 1 in the following line
                    
                probas_in_bin, y_corresponding = pred_probas[(pred_probas[0] >= bin_start) * (pred_probas[0] < bin_end)], y_true[(pred_probas[0] >= bin_start) * (pred_probas[0] < bin_end)]
                ece_bin = abs(probas_in_bin[0].mean() - y_corresponding.mean()) * (probas_in_bin.shape[0] ** entropy(frac_pos))
                sum_weights += (probas_in_bin.shape[0] ** entropy(frac_pos))
                dico_ece_bins[bin_nb] = ece_bin

            return sum(pd.Series(dico_ece_bins.values()).dropna()) / sum_weights


def compute_ECE_L2(y_true, pred_probas, n_bins):
    y_true = y_true.copy()[pred_probas[0] > 0.6]
    pred_probas = pred_probas.copy()[pred_probas[0] > 0.6]
    
    if pred_probas.shape[0] == 0:
        return np.nan
    else:
        bins = np.linspace(0, 1, n_bins + 1)
        dico_ece_bins = {}
        for bin_nb in range(len(bins)-1):
            bin_start, bin_end = bins[bin_nb], bins[bin_nb + 1]
            if bin_end == 1:
                bin_end = 1.01 # To ensure selecting the proba = 1 in the following line
                
            probas_in_bin, y_corresponding = pred_probas[(pred_probas[0] >= bin_start) * (pred_probas[0] < bin_end)], y_true[(pred_probas[0] >= bin_start) * (pred_probas[0] < bin_end)]
            ece_bin = (abs(probas_in_bin[0].mean() - y_corresponding.mean()) * probas_in_bin.shape[0]) ** 2
            dico_ece_bins[bin_nb] = ece_bin

        return (sum(pd.Series(dico_ece_bins.values()).dropna()) ** 0.5) / pred_probas.shape[0]


def compute_MCE(y_true, pred_probas, n_bins):
    bins = np.linspace(0, 1, n_bins + 1)
    dico_ece_bins = {}
    for bin_nb in range(len(bins)-1):
        bin_start, bin_end = bins[bin_nb], bins[bin_nb + 1]
        if bin_end == 1:
                bin_end = 1.01 # To ensure selecting the proba = 1 in the following line
                
        probas_in_bin, y_corresponding = pred_probas[(pred_probas[0] >= bin_start) * (pred_probas[0] < bin_end)], y_true[(pred_probas[0] >= bin_start) * (pred_probas[0] < bin_end)]
        ece_bin = abs(probas_in_bin[0].mean() - y_corresponding.mean())
        dico_ece_bins[bin_nb] = ece_bin

    return max(pd.Series(dico_ece_bins.values()).dropna())


def compute_ACE(y_true, pred_probas, n_bins):
    preds_probas_and_values = pd.concat([pred_probas, y_true], axis=1)
    preds_probas_and_values_sorted = preds_probas_and_values.sort_values(by=0)

    n_datas_per_bins = preds_probas_and_values_sorted.shape[0] // n_bins + 1
    ace_bins = []

    for i in range(n_bins):
        probas_in_bin = preds_probas_and_values_sorted.iloc[i : n_datas_per_bins * (1 + i)][0]
        y_corresponding = preds_probas_and_values_sorted.iloc[i : n_datas_per_bins * (1 + i)]["y"]
        ace_bin = abs(probas_in_bin.mean() - y_corresponding.mean())
        ace_bins.append(ace_bin)

    return sum(ace_bins) / n_bins


def compute_ACE_topk(y_true, pred_probas, pred_probas_uncalibrated, n_bins, k):
    preds_probas_and_values = pd.concat([pred_probas_uncalibrated, pred_probas, y_true], axis=1)
    preds_probas_and_values.columns = ["unc", 0, "y"]
    preds_probas_and_values_topk = preds_probas_and_values.sort_values(by="unc").iloc[-k:, :]
    pred_probas = preds_probas_and_values_topk[0]
    y_true = preds_probas_and_values_topk["y"]
    
    if pred_probas.shape[0] == 0:
        return np.nan
    else:
        n_bins = max(1, int(np.log10(y_true.shape[0])))
        n_datas_per_bins = preds_probas_and_values_topk.shape[0] // n_bins + 1
        ace_bins = []
        for i in range(n_bins):
            probas_in_bin = pred_probas.iloc[i : n_datas_per_bins * (1 + i)]
            y_corresponding = y_true.iloc[i : n_datas_per_bins * (1 + i)]
            ace_bin = abs(probas_in_bin.mean() - y_corresponding.mean())
            ace_bins.append(ace_bin)

        return sum(ace_bins) / n_bins


def compute_ACE_topk_perc(y_true, pred_probas, pred_probas_uncalibrated, n_bins, perc):
    k = int(perc * y_true.shape[0])
    preds_probas_and_values = pd.concat([pred_probas_uncalibrated, pred_probas, y_true], axis=1)
    preds_probas_and_values.columns = ["unc", 0, "y"]
    preds_probas_and_values_topk = preds_probas_and_values.sort_values(by="unc").iloc[-k:, :]
    pred_probas = preds_probas_and_values_topk[0]
    y_true = preds_probas_and_values_topk["y"]
    
    if pred_probas.shape[0] == 0:
        return np.nan
    else:
        n_bins = max(1, int(np.log10(y_true.shape[0])))
        n_datas_per_bins = preds_probas_and_values_topk.shape[0] // n_bins + 1
        ace_bins = []
        for i in range(n_bins):
            probas_in_bin = pred_probas.iloc[i : n_datas_per_bins * (1 + i)]
            y_corresponding = y_true.iloc[i : n_datas_per_bins * (1 + i)]
            ace_bin = abs(probas_in_bin.mean() - y_corresponding.mean())
            ace_bins.append(ace_bin)

        return sum(ace_bins) / n_bins
    
    
def entropy(x):
    return -(x*np.log(x) + (1-x)*np.log(1-x)) / np.log(2)


def get_inc_poly(x, y, deg):
    p = cp.Variable(deg)
    V = np.vander(x, deg+1, increasing=True)[:, 1:]
    axis = np.linspace(0, 1, 1000)
    V_axis_1 = np.vander(axis, deg, increasing=True) * np.array([i for i in range(1, deg+1)])
    
    objective = cp.Minimize(cp.sum_squares(V @ p - y))
    constraints = [V_axis_1 @ p >=0]
    
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver="SCS")
    
    return p.value


def get_inc_poly_bounded(x, y, deg):
    p = cp.Variable(deg)
    V = np.vander(x, deg+1, increasing=True)[:, 1:]
    axis = np.linspace(0, 1, 1000)
    V_axis_1 = np.vander(axis, deg+1, increasing=True)[:, 1:]
    V_axis_2 = np.vander(axis, deg, increasing=True) * np.array([i for i in range(1, deg+1)])
    
    objective = cp.Minimize(cp.sum_squares(V @ p - y))
    constraints = [V_axis_1[-1, :] @ p == 1, V_axis_2 @ p >=0]
    
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver="SCS")
    
    return p.value


def write_table_latex(error_metric_mean, error_metric_std, columns_interest, wilc_test, threshold=None):
    if type(error_metric_mean) == dict:
        if threshold is None:
            print("You need to provide a threshold for this metric. Ensure the wilcoxon test has been made with the corresponding threshold also.")
        error_metric_mean = error_metric_mean[threshold]
        error_metric_std = error_metric_std[threshold]
        
    metric_all_for_latex = pd.DataFrame()
    table_mean = error_metric_mean.loc[columns_interest]
    table_std = error_metric_std.loc[columns_interest]

    for dataset in table_mean.columns:
        for method in table_mean.index:
            pvalue = wilc_test.loc[method, dataset]
            if pvalue > 0.05: # means that not significantly different from the best method
                metric_all_for_latex.loc[dataset.replace("%", "\%").replace("_", " "), method] = "\\underline{" + "{:.2f}".format(table_mean.loc[method, dataset]*100) + "}"
            
            elif np.isnan(pvalue): # means that it is the best method
                metric_all_for_latex.loc[dataset.replace("%", "\%").replace("_", " "), method] = "\\underline{\\textbf{" + "{:.2f}".format(table_mean.loc[method, dataset]*100) + "}" + "}"
            
            else:
                metric_all_for_latex.loc[dataset.replace("%", "\%").replace("_", " "), method] = "{:.2f}".format(table_mean.loc[method, dataset]*100)         

    with open('table_latex.tex', 'w') as tf:
        tf.write((metric_all_for_latex).to_latex())