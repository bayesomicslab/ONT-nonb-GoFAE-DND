import os
import pandas as pd
import numpy as np
from utils import *
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random

def windows_2_features_version1(forward_train_100, reverse_train_100, label):
    assert forward_train_100.shape[0] == reverse_train_100.shape[0]
    features_cols = ['min', 'max', 'mean', 'std'] + ['quantile_' + str(i) for i in [0.05, 0.25, 0.50, 0.75, 0.95]]
    all_features = ['forward_' + elem for elem in features_cols] + ['reverse_' + elem for elem in features_cols] + ['label']
    data_df = pd.DataFrame(columns=all_features, index=range(forward_train_100.shape[0]))
    forward_min_vector = np.min(forward_train_100, axis=1)
    forward_max_vector = np.max(forward_train_100, axis=1)
    forward_mean_vector = np.mean(forward_train_100, axis=1)
    forward_std_vector = np.std(forward_train_100, axis=1)
    forward_quantile_5_vector = np.quantile(forward_train_100, 0.05, axis=1)
    forward_quantile_25_vector = np.quantile(forward_train_100, 0.25, axis=1)
    forward_quantile_50_vector = np.quantile(forward_train_100, 0.50, axis=1)
    forward_quantile_75_vector = np.quantile(forward_train_100, 0.75, axis=1)
    forward_quantile_95_vector = np.quantile(forward_train_100, 0.95, axis=1)
    reverse_min_vector = np.min(reverse_train_100, axis=1)
    reverse_max_vector = np.max(reverse_train_100, axis=1)
    reverse_mean_vector = np.mean(reverse_train_100, axis=1)
    reverse_std_vector = np.std(reverse_train_100, axis=1)
    reverse_quantile_5_vector = np.quantile(reverse_train_100, 0.05, axis=1)
    reverse_quantile_25_vector = np.quantile(reverse_train_100, 0.25, axis=1)
    reverse_quantile_50_vector = np.quantile(reverse_train_100, 0.50, axis=1)
    reverse_quantile_75_vector = np.quantile(reverse_train_100, 0.75, axis=1)
    reverse_quantile_95_vector = np.quantile(reverse_train_100, 0.95, axis=1)
    data_df['forward_min'] = forward_min_vector
    data_df['forward_max'] = forward_max_vector
    data_df['forward_mean'] = forward_mean_vector
    data_df['forward_std'] = forward_std_vector
    data_df['forward_quantile_0.05'] = forward_quantile_5_vector
    data_df['forward_quantile_0.25'] = forward_quantile_25_vector
    data_df['forward_quantile_0.5'] = forward_quantile_50_vector
    data_df['forward_quantile_0.75'] = forward_quantile_75_vector
    data_df['forward_quantile_0.95'] = forward_quantile_95_vector
    data_df['reverse_min'] = reverse_min_vector
    data_df['reverse_max'] = reverse_max_vector
    data_df['reverse_mean'] = reverse_mean_vector
    data_df['reverse_std'] = reverse_std_vector
    data_df['reverse_quantile_0.05'] = reverse_quantile_5_vector
    data_df['reverse_quantile_0.25'] = reverse_quantile_25_vector
    data_df['reverse_quantile_0.5'] = reverse_quantile_50_vector
    data_df['reverse_quantile_0.75'] = reverse_quantile_75_vector
    data_df['reverse_quantile_0.95'] = reverse_quantile_95_vector
    data_df['label'] = label
    return data_df


def windows_2_features_version2(vector, label):
    features_cols = ['min', 'max', 'mean', 'std'] + ['quantile_' + str(i) for i in [0.05, 0.25, 0.50, 0.75, 0.95]] + ['label']
    data_df = pd.DataFrame(columns=features_cols, index=range(vector.shape[0]))
    min_vector = np.min(vector, axis=1)
    max_vector = np.max(vector, axis=1)
    mean_vector = np.mean(vector, axis=1)
    std_vector = np.std(vector, axis=1)
    quantile_5_vector = np.quantile(vector, 0.05, axis=1)
    quantile_25_vector = np.quantile(vector, 0.25, axis=1)
    quantile_50_vector = np.quantile(vector, 0.50, axis=1)
    quantile_75_vector = np.quantile(vector, 0.75, axis=1)
    quantile_95_vector = np.quantile(vector, 0.95, axis=1)
    data_df['min'] = min_vector
    data_df['max'] = max_vector
    data_df['mean'] = mean_vector
    data_df['std'] = std_vector
    data_df['quantile_0.05'] = quantile_5_vector
    data_df['quantile_0.25'] = quantile_25_vector
    data_df['quantile_0.5'] = quantile_50_vector
    data_df['quantile_0.75'] = quantile_75_vector
    data_df['quantile_0.95'] = quantile_95_vector
    data_df['label'] = label
    return data_df



def prepare_iolation_forest_data():
    
    # b dna data:
    forward_tr, forward_val, forward_te, reverse_tr, reverse_val, reverse_te = prepare_bdna_dataset()
    bdna_df_tr = windows_2_features_version1(forward_tr, reverse_tr, label=0)
    bdna_df_val = windows_2_features_version1(forward_val, reverse_val, label=0)
    bdna_df_te = windows_2_features_version1(forward_te, reverse_te, label=0)
    
    # non b (center):
    # number of outliers: 4,081,948
    # nonb_center_forward, nonb_center_reverse = prepare_nonb_dataset_center()
    # nonb_df_center = windows_2_features(nonb_center_forward, nonb_center_reverse, label=1)
    
    # non b (sliding):
    # number of outliers: 28,277
    nonb_sliding = prepare_nonb_dataset_sliding(nonb_type='all', min_motif_proportion=0.7)
    forward_cols = ['forward_'+str(i) for i in range(100)]
    reverse_cols = ['reverse_'+str(i) for i in range(100)]
    nonb_sliding_forward = nonb_sliding[forward_cols].to_numpy()
    nonb_sliding_reverse = nonb_sliding[reverse_cols].to_numpy()
    nonb_df_sliding = windows_2_features_version1(nonb_sliding_forward, nonb_sliding_reverse, label=1)
    nonb_sliding_tr, nonb_sliding_val, nonb_sliding_te = separate_df(nonb_df_sliding)
    
    # 1014880 bdna, 22621 outliers
    train_df = bdna_df_tr.append(nonb_sliding_tr).sample(frac=1).reset_index(drop=True)
    
    # 126860 bdna, 2828 outliers
    val_df = bdna_df_val.append(nonb_sliding_val).sample(frac=1).reset_index(drop=True)
    
    # 126860 bdna, 2828 outliers
    test_df = bdna_df_te.append(nonb_sliding_te).sample(frac=1).reset_index(drop=True)
    train_x = train_df.drop(['label'], axis=1)
    train_y = train_df[['label']]
    
    val_x = val_df.drop(['label'], axis=1)
    val_y = val_df[['label']]
    
    test_x = test_df.drop(['label'], axis=1)
    test_y = test_df[['label']]
    
    return train_x, train_y, val_x, val_y, test_x, test_y


def build_isolation_forest_old(train_x, train_y, test_x, test_y, motif_prop):
    measures = ['tn', 'fp', 'fn', 'tp', 'accuracy', 'precision', 'recall', 'f-score', 'fpr', 'fnr']
    results = pd.DataFrame(columns=measures + ['data', 'motif_proportion', 'contamination'], index=[0, 1])

    
    cont_param = round(np.sum(np.array(train_y))/len(train_y), 3)
    isolation_forest_model = IsolationForest(contamination=cont_param).fit(train_x)

    train_y_pred = isolation_forest_model.predict(train_x)
    train_y_pred[train_y_pred == 1] = 0
    train_y_pred[train_y_pred == -1] = 1
    tr_tn, tr_fp, tr_fn, tr_tp = confusion_matrix(train_y, train_y_pred).ravel()
    tr_accuracy, tr_precision, tr_recall, tr_fscore, tr_fpr, tr_fnr = compute_accuracy_metrics(tr_tn, tr_fp, tr_fn, tr_tp)
    results.loc[0, :] = tr_tn, tr_fp, tr_fn, tr_tp, tr_accuracy, tr_precision, tr_recall, tr_fscore, tr_fpr, tr_fnr, 'train', motif_prop, cont_param
    
    test_y_pred = isolation_forest_model.predict(test_x)
    test_y_pred[test_y_pred == 1] = 0
    test_y_pred[test_y_pred == -1] = 1
    te_tn, te_fp, te_fn, te_tp = confusion_matrix(test_y, test_y_pred).ravel()
    te_accuracy, te_precision, te_recall, te_fscore, te_fpr, te_fnr = compute_accuracy_metrics(te_tn, te_fp, te_fn, te_tp)
    results.loc[1, :] = te_tn, te_fp, te_fn, te_tp, te_accuracy, te_precision, te_recall, te_fscore, te_fpr, te_fnr, 'test', motif_prop, cont_param
    return results

def iolation_forest_method():
    # UCHC
    # save_path = '/labs/Aguiar/non_bdna/methods/isolation_forest/results'
    
    # beagle
    save_path = '/home/mah19006/projects/nonBDNA/methods/isolation_forest/results'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    # b dna data:
    forward_tr, forward_val, forward_te, reverse_tr, reverse_val, reverse_te = prepare_bdna_dataset()
    bdna_df_tr = windows_2_features_version1(forward_tr, reverse_tr, label=0)
    bdna_df_val = windows_2_features_version1(forward_val, reverse_val, label=0)
    bdna_df_te = windows_2_features_version1(forward_te, reverse_te, label=0)
    
    # non b (center):
    # number of outliers: 4,081,948
    # nonb_center_forward, nonb_center_reverse = prepare_nonb_dataset_center()
    # nonb_df_center = windows_2_features(nonb_center_forward, nonb_center_reverse, label=1)
    
    # non b (sliding):
    # number of outliers: 28,277
    measures = ['tn', 'fp', 'fn', 'tp', 'accuracy', 'precision', 'recall', 'f-score', 'fpr', 'fnr']
    results_df = pd.DataFrame(columns=measures + ['data', 'motif_proportion', 'contamination'])
    forward_cols = ['forward_'+str(i) for i in range(100)]
    reverse_cols = ['reverse_'+str(i) for i in range(100)]
    non_b_inclusion_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for nonb_inc in non_b_inclusion_list:
        nonb_sliding = prepare_nonb_dataset_sliding(nonb_type='all', min_motif_proportion=nonb_inc)
        nonb_sliding_forward = nonb_sliding[forward_cols].to_numpy()
        nonb_sliding_reverse = nonb_sliding[reverse_cols].to_numpy()
        nonb_df_sliding = windows_2_features_version1(nonb_sliding_forward, nonb_sliding_reverse, label=1)
        nonb_sliding_tr, nonb_sliding_val, nonb_sliding_te = separate_df(nonb_df_sliding)
    
        # (at 0.7) 1014880 bdna, 22621 outliers
        train_df = bdna_df_tr.append(nonb_sliding_tr).sample(frac=1).reset_index(drop=True)
        
        # (at 0.7) 126860 bdna, 2828 outliers
        val_df = bdna_df_val.append(nonb_sliding_val).sample(frac=1).reset_index(drop=True)
        
        # (at 0.7) 126860 bdna, 2828 outliers
        test_df = bdna_df_te.append(nonb_sliding_te).sample(frac=1).reset_index(drop=True)
        
        train_x = train_df.drop(['label'], axis=1)
        train_y = train_df[['label']]
        
        val_x = val_df.drop(['label'], axis=1)
        val_y = val_df[['label']]
        
        test_x = test_df.drop(['label'], axis=1)
        test_y = test_df[['label']]

        train_x = train_x.append(val_x).reset_index(drop=True)
        train_y = train_y.append(val_y).reset_index(drop=True)

        this_results = build_isolation_forest(train_x, train_y, test_x, test_y, nonb_inc)
        results_df = results_df.append(this_results).reset_index(drop=True)
        results_df.to_csv(os.path.join(save_path, 'isolation_forest_results.csv'))


def plot_isolatio_forest_results():
    plot_path = 'Data/methods/isolation_forest/plots/'
    results_path = 'Data/methods/isolation_forest/results/isolation_forest_results.csv'
    df = pd.read_csv(results_path, index_col=0)
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif']
    elements_name = ['A Phased Repeat', 'G Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z DNA']

    motif_proportions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # results_df_dict = {mpor: {data: df[(df['data'] == data) & (df['motif_proportion'] == mpor)].reset_index(drop=True)
    #                           for data in ['train', 'test']} for mpor in motif_proportions}
    # metrics = ['accuracy', 'precision', 'recall', 'f-score']
    metrics = ['precision', 'recall', 'f-score']

    df = df.fillna(0)
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
    # 'tab:olive', 'tab:cyan']
    data_list = ['train', 'test']
    results_df_dict = {data: {metric: list(df.loc[df['data'] == data, metric]) for metric in metrics} for data in data_list}
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    for metric_id in range(len(metrics)):
        metric = metrics[metric_id]
        tr_line = results_df_dict['train'][metric]
        te_line = results_df_dict['test'][metric]
        color = colors[metric_id]
        # ax.plot(motif_proportions, tr_line, color=color, linestyle='--', label=metric + ' - training')
        ax.plot(motif_proportions, te_line, color=color, label=metric + ' - test')
    
        ax.set_xlabel('Motif Proportion in Windows', fontsize=15)
        ax.set_ylabel('Value', fontsize=15)
        ax.legend()
        ax.set_title('Isolation Forest Results', fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, 'isolation_forest.png'))


def iolation_forest_method_center():
    cont_prop = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

    non_b_types = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                   'Short_Tandem_Repeat', 'Z_DNA_Motif']
    # UCHC
    # save_path = '/labs/Aguiar/non_bdna/methods/isolation_forest/results'

    # beagle
    save_path = '/home/mah19006/projects/nonBDNA/methods/isolation_forest/results'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    measures = ['tn', 'fp', 'fn', 'tp', 'accuracy', 'precision', 'recall', 'f-score', 'fpr', 'fnr']
    results_df = pd.DataFrame(columns=measures + ['data', 'motif_proportion', 'contamination', 'non_b_type'])

    # b dna data:
    forward_tr, forward_val, forward_te, reverse_tr, reverse_val, reverse_te = prepare_bdna_dataset()
    bdna_df_tr = windows_2_features_version1(forward_tr, reverse_tr, label=0)
    bdna_df_val = windows_2_features_version1(forward_val, reverse_val, label=0)
    bdna_df_te = windows_2_features_version1(forward_te, reverse_te, label=0)
    bdna_df_tr = bdna_df_tr.append(bdna_df_val).sample(frac=1).reset_index(drop=True)
    
    n_bdna_tr = bdna_df_tr.shape[0]
    n_bdna_te = bdna_df_te.shape[0]

    # non b (center):
    # number of outliers: 4,081,948
    nonb_center_forward, nonb_center_reverse = prepare_nonb_dataset_center()
    nonb_center_forward_np = nonb_center_forward.drop(['label'], axis=1).to_numpy()
    nonb_center_forward_labels = nonb_center_forward[['label']]
    nonb_center_reverse_np = nonb_center_reverse.drop(['label'], axis=1).to_numpy()
    nonb_center_reverse_labels = nonb_center_reverse[['label']]
    # assert nonb_center_forward_labels == nonb_center_reverse_labels
    
    nonb_df_center = windows_2_features_version1(nonb_center_forward_np, nonb_center_reverse_np, label=1)
    nonb_df_center['nonb_type'] = nonb_center_reverse_labels

    # nonb_df_tr, nonb_df_val, nonb_df_te = separate_df(nonb_df_center)

    grouped_df = nonb_df_center.groupby('nonb_type')
    arr_list = [np.split(g, [int(.8 * len(g)), int(.9 * len(g))]) for i, g in grouped_df]

    nonb_df_tr = pd.concat([t[0] for t in arr_list])
    nonb_df_te = pd.concat([t[1] for t in arr_list])
    nonb_df_val = pd.concat([v[2] for v in arr_list])
    
    nonb_df_tr = nonb_df_tr.append(nonb_df_val).sample(frac=1).reset_index(drop=True)

    for nonb in non_b_types:
        this_nonb_tr = nonb_df_tr[nonb_df_tr['nonb_type'] == nonb].reset_index(drop=True)
        this_nonb_te = nonb_df_te[nonb_df_te['nonb_type'] == nonb].reset_index(drop=True)
        this_nonb_tr = this_nonb_tr.drop(['nonb_type'], axis=1)
        this_nonb_te = this_nonb_te.drop(['nonb_type'], axis=1)
        
        for cont in cont_prop:
            print('Computing results for', nonb, cont, '...')
            n_outliers_tr = int((cont*n_bdna_tr)/(1-cont))
            n_outliers_te = int((cont*n_bdna_te)/(1-cont))
            outlier_tr = this_nonb_tr.sample(n=n_outliers_tr).reset_index(drop=True)
            outlier_te = this_nonb_te.sample(n=n_outliers_te).reset_index(drop=True)

            tr_df = bdna_df_tr.append(outlier_tr).reset_index(drop=True)
            te_df = bdna_df_te.append(outlier_te).reset_index(drop=True)

            train_x = tr_df.drop(['label'], axis=1)
            train_y = tr_df[['label']]
        
            test_x = te_df.drop(['label'], axis=1)
            test_y = te_df[['label']]

            this_results = build_isolation_forest(train_x, train_y, test_x, test_y, motif_prop=-1)
            this_results['non_b_type'] = nonb
            results_df = results_df.append(this_results).reset_index(drop=True)
            results_df.to_csv(os.path.join(save_path, 'isolation_forest_results_nonb_types.csv'))

    this_nonb_tr = nonb_df_tr.drop(['nonb_type'], axis=1)
    this_nonb_te = nonb_df_te.drop(['nonb_type'], axis=1)

    for cont in cont_prop:
        print('Computing results for All', cont, '...')
        n_outliers_tr = int((cont*n_bdna_tr)/(1-cont))
        n_outliers_te = int((cont*n_bdna_te)/(1-cont))
        outlier_tr = this_nonb_tr.sample(n=n_outliers_tr).reset_index(drop=True)
        outlier_te = this_nonb_te.sample(n=n_outliers_te).reset_index(drop=True)
        
        tr_df = bdna_df_tr.append(outlier_tr).reset_index(drop=True)
        te_df = bdna_df_te.append(outlier_te).reset_index(drop=True)
        
        train_x = tr_df.drop(['label'], axis=1)
        train_y = tr_df[['label']]
        
        test_x = te_df.drop(['label'], axis=1)
        test_y = te_df[['label']]
        
        this_results = build_isolation_forest(train_x, train_y, test_x, test_y, motif_prop=-1)
        this_results['non_b_type'] = 'All'
        results_df = results_df.append(this_results).reset_index(drop=True)
        results_df.to_csv(os.path.join(save_path, 'isolation_forest_results_nonb_types.csv'))

#
# def iolation_forest_method_center4():
#     cont_prop = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
#
#     non_b_types = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
#                    'Short_Tandem_Repeat', 'Z_DNA_Motif']
#     # UCHC
#     # save_path = '/labs/Aguiar/non_bdna/methods/isolation_forest/results'
#
#     # beagle
#     save_path = '/home/mah19006/projects/nonBDNA/methods/isolation_forest/results'
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#
#     measures = ['tn', 'fp', 'fn', 'tp', 'accuracy', 'precision', 'recall', 'f-score', 'fpr', 'fnr']
#     results_df = pd.DataFrame(columns=measures + ['data', 'motif_proportion', 'contamination', 'non_b_type'])
#
#     # b dna data:
#     forward_tr, forward_val, forward_te, reverse_tr, reverse_val, reverse_val = prepare_bdna_dataset()
#     tr = np. concatenate((forward_tr, reverse_tr), axis=1)
#     val = np. concatenate((forward_val, reverse_val), axis=1)
#     te = np. concatenate((forward_te, reverse_te), axis=1)
#
#
#     # normalize:
#     mu_train = np.mean(tr)
#     std_train = np.std(tr)
#     tr_norm = (tr - mu_train) / std_train
#     val_norm = (val - mu_train) / std_train
#     te_norm = (te - mu_train) / std_train
#
#     # bdna data
#     bdna_df_tr = windows_2_features_version2(tr_norm, label=0)
#     bdna_df_val = windows_2_features_version2(val_norm, label=0)
#     bdna_df_te = windows_2_features_version2(te_norm, label=0)
#
#
#     n_bdna_samples = len(bdna_df_tr) + len(bdna_df_val) + len(bdna_df_te)
#     n_non_b_needed = int(np.floor(n_bdna_samples * 0.01))
#     n_non_b_needed_tr = int(np.floor(n_non_b_needed * 0.8))
#     n_non_b_needed_val = int(np.floor(n_non_b_needed * 0.1))
#
#     # non b (center):
#     # number of outliers: 4,081,948
#     nonb_train_dfs, nonb_test_dfs = prepare_nonb_dataset_center()
#     for nonb in nonb_train_dfs.keys():
#         this_tr_df = nonb_train_dfs[nonb]
#         this_te_df = nonb_test_dfs[nonb]
#         nonb_tr_np = this_tr_df[['forward_'+str(i) for i in range(100)] + ['reverse_' + str(i) for i in range(100)]].to_numpy()
#         nonb_te_np = this_te_df[['forward_'+str(i) for i in range(100)] + ['reverse_' + str(i) for i in range(100)]].to_numpy()
#         nonb_df_tr = windows_2_features_version2(nonb_tr_np, label=1)
#         nonb_df_te = windows_2_features_version2(nonb_te_np, label=1)
#         nonb_df = nonb_df_tr.append(nonb_df_te)
#         nonb_df.sample(n=n_non_b_needed, random_state=22)
#
#
#
#
#     for nonb in non_b_types:
#         this_nonb_tr = nonb_df_tr[nonb_df_tr['nonb_type'] == nonb].reset_index(drop=True)
#         this_nonb_te = nonb_df_te[nonb_df_te['nonb_type'] == nonb].reset_index(drop=True)
#         this_nonb_tr = this_nonb_tr.drop(['nonb_type'], axis=1)
#         this_nonb_te = this_nonb_te.drop(['nonb_type'], axis=1)
#
#         for cont in cont_prop:
#             print('Computing results for', nonb, cont, '...')
#             n_outliers_tr = int((cont*n_bdna_tr)/(1-cont))
#             n_outliers_te = int((cont*n_bdna_te)/(1-cont))
#             outlier_tr = this_nonb_tr.sample(n=n_outliers_tr).reset_index(drop=True)
#             outlier_te = this_nonb_te.sample(n=n_outliers_te).reset_index(drop=True)
#
#             tr_df = bdna_df_tr.append(outlier_tr).reset_index(drop=True)
#             te_df = bdna_df_te.append(outlier_te).reset_index(drop=True)
#
#             train_x = tr_df.drop(['label'], axis=1)
#             train_y = tr_df[['label']]
#
#             test_x = te_df.drop(['label'], axis=1)
#             test_y = te_df[['label']]
#
#             this_results = build_isolation_forest(train_x, train_y, test_x, test_y, motif_prop=-1)
#             this_results['non_b_type'] = nonb
#             results_df = results_df.append(this_results).reset_index(drop=True)
#             results_df.to_csv(os.path.join(save_path, 'isolation_forest_results_nonb_types.csv'))
#
#     this_nonb_tr = nonb_df_tr.drop(['nonb_type'], axis=1)
#     this_nonb_te = nonb_df_te.drop(['nonb_type'], axis=1)
#
#     for cont in cont_prop:
#         print('Computing results for All', cont, '...')
#         n_outliers_tr = int((cont*n_bdna_tr)/(1-cont))
#         n_outliers_te = int((cont*n_bdna_te)/(1-cont))
#         outlier_tr = this_nonb_tr.sample(n=n_outliers_tr).reset_index(drop=True)
#         outlier_te = this_nonb_te.sample(n=n_outliers_te).reset_index(drop=True)
#
#         tr_df = bdna_df_tr.append(outlier_tr).reset_index(drop=True)
#         te_df = bdna_df_te.append(outlier_te).reset_index(drop=True)
#
#         train_x = tr_df.drop(['label'], axis=1)
#         train_y = tr_df[['label']]
#
#         test_x = te_df.drop(['label'], axis=1)
#         test_y = te_df[['label']]
#
#         this_results = build_isolation_forest(train_x, train_y, test_x, test_y, motif_prop=-1)
#         this_results['non_b_type'] = 'All'
#         results_df = results_df.append(this_results).reset_index(drop=True)
#         results_df.to_csv(os.path.join(save_path, 'isolation_forest_results_nonb_types.csv'))
#

def plot_isolatio_forest_results_center():
    plot_path = 'Data/methods/isolation_forest/plots/'
    results_path = 'Data/methods/isolation_forest/results/isolation_forest_results_nonb_types.csv'
    df = pd.read_csv(results_path, index_col=0)
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif', 'All']
    # elements_name = ['A Phased Repeat', 'G Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
    #                  'Short Tandem Repeat', 'Z DNA']
    
    contaminations = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    # results_df_dict = {mpor: {data: df[(df['data'] == data) & (df['motif_proportion'] == mpor)].reset_index(drop=True)
    #                           for data in ['train', 'test']} for mpor in motif_proportions}
    # metrics = ['accuracy', 'precision', 'recall', 'f-score']
    metrics = ['f-score']
    
    df = df.fillna(0)
    
    # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan']
    data_list = ['train', 'test']
    col_counter = 0
    results_df_dict = {data: {elem: {metric: list(df.loc[(df['data'] == data) & (df['non_b_type'] == elem), metric]) for metric in metrics} for elem in elements} for data in data_list}
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    for metric_id in range(len(metrics)):
        metric = metrics[metric_id]
        
        for nonb in elements:
            color = colors[col_counter]
            # print(metric, nonb, col_counter)
            tr_line = results_df_dict['train'][nonb][metric]
            te_line = results_df_dict['test'][nonb][metric]
            # ax.plot(contaminations[0:len(te_line)], tr_line, color=color, linestyle='--', label=nonb + ' - ' + metric + ' - training')
            ax.plot(contaminations[0:len(te_line)], te_line, color=color, label=nonb + ' - ' + metric + ' - test')
            col_counter += 1
        ax.set_xlabel('Percentage of outliers', fontsize=15)
        ax.set_ylabel('Value', fontsize=15)
        ax.legend()
        ax.set_title('Isolation Forest Results', fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, 'isolation_forest_center.png'))


def data4pca():
    save_path = '/labs/Aguiar/non_bdna/clustering'
    path = '/labs/Aguiar/non_bdna/annotations/vae_windows/prepared_windows/original/complete_centered_windows'
    non_b_types = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                   'Short_Tandem_Repeat', 'Z_DNA_Motif']
    for nonb in non_b_types:
        this_path = os.path.join(path, nonb + '_centered.csv')
        df = pd.read_csv(this_path, index_col=0)
        forward_cols = ['forward_'+str(i) for i in range(100)]
        reverse_cols = ['reverse_'+str(i) for i in range(100)]
        nonb_forward = df[forward_cols].to_numpy()
        nonb_reverse = df[reverse_cols].to_numpy()
        feature_df = windows_2_features_version1(nonb_forward, nonb_reverse, label=nonb)
        feature_df.to_csv(os.path.join(save_path, 'features_' + nonb + '.csv'))
    forward_tr, forward_val, forward_te, reverse_tr, reverse_val, reverse_val = prepare_bdna_dataset()
    control_df = windows_2_features_version1(forward_tr, reverse_tr, label='Control')
    control_df.to_csv(os.path.join(save_path, 'features_Control.csv'))
    
    
def plot_n_reads():
    forward_path = 'Data/stats/complemete_forward_read_counts_in_motifs.npy'
    reverse_path = 'Data/stats/complemete_reverse_read_counts_in_motifs.npy'
    fwrd = np.load(forward_path)
    revr = np.load(reverse_path)
    forward = list(fwrd)
    reverse = list(revr)
    
    hist, bins, _ = plt.hist(reverse, bins=20)
    #
    logbins = np.logspace(np.log10(bins[0]+1), np.log10(bins[-1]), len(bins))
    #
    plt.hist(forward, bins=logbins)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('# Reads in motif regions', fontsize=15)  # Add an x-label to the axes.
    plt.ylabel('Count', fontsize=15)  # Add a y-label to the axes.
    # ax.legend()
    
    # plt.title('Total # base pairs in B DNA region: \n2,757,403,425 bp', fontsize=15)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join('Data/stats/n_reads_reverse.png'))


def check():
    non_b_types = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                   'Short_Tandem_Repeat', 'Z_DNA_Motif']
    path = 'D:/UCONN/nonBDNA/Data/clustering'
    for nonb in non_b_types:
        this = os.path.join(path, 'features_' + nonb + '.csv')
        fdf = pd.read_csv(this, index_col=0)
        fmax = np.max(fdf['forward_max'])
        rmax = np.max(fdf['reverse_max'])
        print(nonb, 'max in forward:', fmax, ' - max in revers:', rmax)


def build_isolation_forest(train_x, train_y, test_x, test_y, cont_param):
    measures = ['tn', 'fp', 'fn', 'tp', 'accuracy', 'precision', 'recall', 'f-score', 'fpr', 'fnr', 'fdr']
    results = pd.DataFrame(columns=measures + ['data', 'contamination'], index=[0, 1])
    isolation_forest_model = IsolationForest(contamination=cont_param).fit(train_x)
    train_y_pred = isolation_forest_model.predict(train_x)
    train_y_pred[train_y_pred == 1] = 0
    train_y_pred[train_y_pred == -1] = 1
    tr_tn, tr_fp, tr_fn, tr_tp = confusion_matrix(train_y, train_y_pred).ravel()
    tr_accuracy, tr_precision, tr_recall, tr_fscore, tr_fpr, tr_fnr, tr_fdr = compute_accuracy_metrics(tr_tn, tr_fp, tr_fn, tr_tp)
    results.loc[0, :] = tr_tn, tr_fp, tr_fn, tr_tp, tr_accuracy, tr_precision, tr_recall, tr_fscore, tr_fpr, tr_fnr, tr_fdr, 'train', cont_param
    test_y_pred = isolation_forest_model.predict(test_x)
    test_y_pred[test_y_pred == 1] = 0
    test_y_pred[test_y_pred == -1] = 1
    te_tn, te_fp, te_fn, te_tp = confusion_matrix(test_y, test_y_pred).ravel()
    te_accuracy, te_precision, te_recall, te_fscore, te_fpr, te_fnr, te_fdr = compute_accuracy_metrics(te_tn, te_fp, te_fn, te_tp)
    results.loc[1, :] = te_tn, te_fp, te_fn, te_tp, te_accuracy, te_precision, te_recall, te_fscore, te_fpr, te_fnr, te_fdr, 'test', cont_param
    return results


def isolation_forest_with_cross_val_experimental_data():
    method = 'IF_50'
    non_b_types = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                   'Short_Tandem_Repeat', 'Z_DNA_Motif']
    # non_b_path = '/home/mah19006/projects/nonBDNA/data/prepared_windows_req5/dataset/outliers/centered_windows'
    # bdna_path = '/home/mah19006/projects/nonBDNA/data/prepared_windows_req5/dataset/bdna'
    methods_results_path = '/home/mah19006/projects/nonBDNA/data/methods_results'
    bdna_path = '/home/mah19006/projects/nonBDNA/data/experimental_50'
    non_b_path = '/home/mah19006/projects/nonBDNA/data/experimental_50'
    
    if not os.path.exists(methods_results_path):
        os.mkdir(methods_results_path)
    save_path = os.path.join(methods_results_path, method)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    measures = ['tn', 'fp', 'fn', 'tp', 'accuracy', 'precision', 'recall', 'f-score', 'fpr', 'fnr', 'fdr']
    results = pd.DataFrame(columns=measures + ['data', 'contamination', 'label'])
    final_results_df = pd.DataFrame(columns=['method', 'label', 'alpha', 'tail', 'potential_nonb_counts'],
                                    index=range(7*19*2))
    counter = 0
    global_train, global_val, global_test = load_data(bdna_path, non_b_path)

    for nonb in non_b_types:
        train = global_train[(global_train['label'] == nonb)|(global_train['label'] == 'bdna')].reset_index(drop=True)
        val = global_val[(global_val['label'] == nonb)|(global_val['label'] == 'bdna')].reset_index(drop=True)
        test = global_test[(global_test['label'] == nonb)|(global_test['label'] == 'bdna')].reset_index(drop=True)
        
        train.loc[train['label'] == 'bdna', 'label'] = 0
        train.loc[train['label'] == nonb, 'label'] = 1
        
        val.loc[val['label'] == 'bdna', 'label'] = 0
        val.loc[val['label'] == nonb, 'label'] = 1
        
        test.loc[test['label'] == 'bdna', 'label'] = 0
        test.loc[test['label'] == nonb, 'label'] = 1
        
        convert_dict = {'label': int}
        
        train = train.astype(convert_dict)
        val = val.astype(convert_dict)
        test = test.astype(convert_dict)
        
        train_x = train.drop(['label'], axis=1).to_numpy()
        train_y = train[['label']].to_numpy()
        
        val_x = val.drop(['label'], axis=1).to_numpy()
        val_y = val[['label']].to_numpy()
        
        # test_x = test.drop(['label'], axis=1).to_numpy()
        # test_y = test[['label']].to_numpy()
        
        contaminations = [0.0001, 0.001, 0.01, 0.04, 0.07, 0.1]
        
        for cont in contaminations:
            print(nonb, cont)
            this_results_df = build_isolation_forest(train_x, train_y, val_x, val_y, cont)
            this_results_df['label'] = nonb
            results = results.append(this_results_df).reset_index(drop=True)
            results.to_csv(os.path.join(save_path, method + '_cv.csv'))
        
        # find the best params in validation data
        test_results = results[results['data'] == 'test']
        min_fdr = np.min(test_results['fdr'])
        best_row = test_results[test_results['fdr'] == min_fdr]

        # select the best param:
        best_param = best_row['contamination'].values[0]
        train = train.append(val).reset_index(drop=True)
        train_x = train.drop(['label'], axis=1).to_numpy()
        
        # train with new train (train + val) and the best param
        isolation_forest_model = IsolationForest(contamination=best_param).fit(train_x)
        
        # test only using the test set:
        test_bdna = test[test['label'] == 0].reset_index(drop=True)
        test_bdna_x = test_bdna.drop(['label'], axis=1).to_numpy()
        # test_bdna_y = test_bdna[['label']].to_numpy()
        
        test_nonb = test[test['label'] == 1].reset_index(drop=True)
        test_nonb_x = test_nonb.drop(['label'], axis=1).to_numpy()
        
        null_dist_scores = isolation_forest_model.decision_function(test_bdna_x)
        eval_scores = isolation_forest_model.decision_function(test_nonb_x)
        
        plot_histogram(null_dist_scores, method + '_' + nonb + '_test_bdna_scores', save_path)
        plot_histogram(eval_scores, method + '_' + nonb + '_test_nonb_scores', save_path)
        
        alpha_list = np.arange(0.05, 1, 0.05)
        emp_dist, indices = compute_empirical(null_dist_scores, eval_scores, tail='upper')
        plot_histogram(emp_dist, method + '_' + nonb + '_p_values_upper', save_path)

        for alpha in alpha_list:
            val_emp, _ = FDR_BHP(emp_dist, alpha=alpha)
            final_results_df.loc[counter, :] = method, nonb, alpha, 'upper', val_emp
            counter += 1
            
        emp_dist, indices = compute_empirical(null_dist_scores, eval_scores, tail='lower')
        plot_histogram(emp_dist, method + '_' + nonb + '_p_values_lower', save_path)

        for alpha in alpha_list:
            val_emp, _ = FDR_BHP(emp_dist, alpha=alpha)
            final_results_df.loc[counter, :] = method, nonb, alpha, 'lower', val_emp
            counter += 1

        final_results_df.to_csv(os.path.join(save_path, method + '_final_results.csv'))
        

def isolation_forest_with_cross_val_synthetic_data():
    method = 'IF_50_sim'
    non_b_types = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                   'Short_Tandem_Repeat', 'Z_DNA_Motif']
    # non_b_path = '/mnt/research/aguiarlab/proj/nonBDNA/data/exponential_simulated_data'
    # bdna_path = '/mnt/research/aguiarlab/proj/nonBDNA/data/exponential_simulated_data/'

    bdna_path = '/home/mah19006/projects/nonBDNA/data/sim_50'
    non_b_path = '/home/mah19006/projects/nonBDNA/data/sim_50'
    
    methods_results_path = '/mnt/research/aguiarlab/proj/nonBDNA/data/methods_results'
    if not os.path.exists(methods_results_path):
        os.mkdir(methods_results_path)
    save_path = os.path.join(methods_results_path, method)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    measures = ['tn', 'fp', 'fn', 'tp', 'accuracy', 'precision', 'recall', 'f-score', 'fpr', 'fnr', 'fdr']
    results = pd.DataFrame(columns=measures + ['data', 'contamination', 'label'])
    final_results_df = pd.DataFrame(columns=measures + ['data', 'contamination', 'label', 'method'])
    global_train, global_val, global_test = load_data(bdna_path, non_b_path)
    
    for nonb in non_b_types:
        train = global_train[(global_train['label'] == nonb)|(global_train['label'] == 'bdna')].reset_index(drop=True)
        val = global_val[(global_val['label'] == nonb)|(global_val['label'] == 'bdna')].reset_index(drop=True)
        test = global_test[(global_test['label'] == nonb)|(global_test['label'] == 'bdna')].reset_index(drop=True)
        
        train.loc[train['label'] == 'bdna', 'label'] = 0
        train.loc[train['label'] == nonb, 'label'] = 1
        
        val.loc[val['label'] == 'bdna', 'label'] = 0
        val.loc[val['label'] == nonb, 'label'] = 1
        
        test.loc[test['label'] == 'bdna', 'label'] = 0
        test.loc[test['label'] == nonb, 'label'] = 1
        
        convert_dict = {'label': int}
        
        train = train.astype(convert_dict)
        val = val.astype(convert_dict)
        test = test.astype(convert_dict)
        
        train_x = train.drop(['label'], axis=1).to_numpy()
        train_y = train[['label']].to_numpy()
        
        val_x = val.drop(['label'], axis=1).to_numpy()
        val_y = val[['label']].to_numpy()
        
        test_x = test.drop(['label'], axis=1).to_numpy()
        test_y = test[['label']].to_numpy()
        
        contaminations = [0.0001, 0.001, 0.01, 0.04, 0.07, 0.1]
        
        for cont in contaminations:
            print(nonb, cont)
            this_results_df = build_isolation_forest(train_x, train_y, val_x, val_y, cont)
            this_results_df['label'] = nonb
            results = results.append(this_results_df).reset_index(drop=True)
            results.to_csv(os.path.join(save_path,  method + '_cv.csv'))
        
        # find the best params in validation data
        test_results = results[results['data'] == 'test']
        min_fdr = np.min(test_results['fdr'])
        best_row = test_results[test_results['fdr'] == min_fdr]
        
        
        # select the best param:
        best_param = best_row['contamination'].values[0]

        # train with new train (train + val) and the best param
        train = train.append(val).reset_index(drop=True)
        train_x = train.drop(['label'], axis=1).to_numpy()
        train_y = train[['label']].to_numpy()

        best_results_df = build_isolation_forest(train_x, train_y, test_x, test_y, best_param)
        best_results_df['label'] = nonb
        best_results_df['method'] = method

        final_results_df = final_results_df.append(best_results_df).reset_index(drop=True)
        final_results_df.to_csv(os.path.join(save_path, method + '_final_results_sim.csv'))
    

if __name__ == '__main__':

    isolation_forest_with_cross_val_experimental_data()
    isolation_forest_with_cross_val_synthetic_data()
