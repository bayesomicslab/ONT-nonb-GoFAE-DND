from utils import *
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool


def build_svm(train_x, train_y, test_x, test_y, kern):
    measures = ['tn', 'fp', 'fn', 'tp', 'accuracy', 'precision', 'recall', 'f-score', 'fpr', 'fnr', 'fdr']
    results = pd.DataFrame(columns=measures + ['data', 'kernel'], index=[0, 1])
    # svm_model = OneClassSVM(kernel=kern, verbose=True, max_iter=100).fit(train_x)
    svm_model = OneClassSVM(verbose=True, cache_size=100000, max_iter=100).fit(train_x)
    train_y_pred = svm_model.predict(train_x)
    train_y_pred[train_y_pred == 1] = 0
    train_y_pred[train_y_pred == -1] = 1
    tr_tn, tr_fp, tr_fn, tr_tp = confusion_matrix(train_y, train_y_pred).ravel()
    tr_accuracy, tr_precision, tr_recall, tr_fscore, tr_fpr, tr_fnr, tr_fdr = compute_accuracy_metrics(tr_tn, tr_fp, tr_fn, tr_tp)
    results.loc[0, :] = tr_tn, tr_fp, tr_fn, tr_tp, tr_accuracy, tr_precision, tr_recall, tr_fscore, tr_fpr, tr_fnr, tr_fdr, 'train', kern
    test_y_pred = svm_model.predict(test_x)
    test_y_pred[test_y_pred == 1] = 0
    test_y_pred[test_y_pred == -1] = 1
    te_tn, te_fp, te_fn, te_tp = confusion_matrix(test_y, test_y_pred).ravel()
    te_accuracy, te_precision, te_recall, te_fscore, te_fpr, te_fnr, tr_fdr = compute_accuracy_metrics(te_tn, te_fp, te_fn, te_tp)
    results.loc[1, :] = te_tn, te_fp, te_fn, te_tp, te_accuracy, te_precision, te_recall, te_fscore, te_fpr, te_fnr, tr_fdr, 'test', kern
    return results


def train_svm_nonb_exp(inp):
    
    global_train, global_val, global_test, save_path, nonb, method = inp
    # for nonb in non_b_types:
    
    final_results_df = pd.DataFrame(columns=['method', 'label', 'alpha', 'tail', 'potential_nonb_counts'],
                                    index=range(19*2))
    counter = 0
    
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
    
    # kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
    #
    # for ker in kernel_list:
    #     print(nonb, ker)
    #     this_results_df = build_svm(train_x, train_y, val_x, val_y, ker)
    #     this_results_df['label'] = nonb
    #     results = results.append(this_results_df).reset_index(drop=True)
    #     results.to_csv(os.path.join(save_path, method + '_exp_cv.csv'))
    #
    # # find the best params in validation data
    # test_results = results[results['data'] == 'test']
    # min_fdr = np.min(test_results['fdr'])
    # best_row = test_results[test_results['fdr'] == min_fdr]
    #
    # # select the best param:
    # best_kernel = best_row['kernel'].values[0]
    #
    # train with new train (train + val) and the best param
    train = train.append(val).reset_index(drop=True)
    train_x = train.drop(['label'], axis=1).to_numpy()
    print(nonb, len(train))
    
    svm_model = OneClassSVM(verbose=True, cache_size=10000, max_iter=100).fit(train_x)
    
    # test only using the test set:
    test_bdna = test[test['label'] == 0].reset_index(drop=True)
    test_bdna_x = test_bdna.drop(['label'], axis=1).to_numpy()
    # test_bdna_y = test_bdna[['label']].to_numpy()
    
    test_nonb = test[test['label'] == 1].reset_index(drop=True)
    test_nonb_x = test_nonb.drop(['label'], axis=1).to_numpy()
    
    null_dist_scores = svm_model.decision_function(test_bdna_x)
    eval_scores = svm_model.decision_function(test_nonb_x)
    
    plot_histogram(null_dist_scores, nonb + '_' + method + '_test_bdna_scores', save_path)
    plot_histogram(eval_scores, nonb + '_' + method + '_test_nonb_scores', save_path)
    
    alpha_list = np.arange(0.05, 1, 0.05)
    emp_dist, indices = compute_empirical(null_dist_scores, eval_scores, tail='upper')
    plot_histogram(emp_dist, nonb + '_' + method + '_p_values_upper', save_path)
    
    for alpha in alpha_list:
        val_emp, _ = FDR_BHP(emp_dist, alpha=alpha)
        final_results_df.loc[counter, :] = method, nonb, alpha, 'upper', val_emp
        counter += 1
    
    emp_dist, indices = compute_empirical(null_dist_scores, eval_scores, tail='lower')
    plot_histogram(emp_dist, nonb + '_' + method + '_p_values_lower', save_path)
    
    for alpha in alpha_list:
        val_emp, _ = FDR_BHP(emp_dist, alpha=alpha)
        final_results_df.loc[counter, :] = method, nonb, alpha, 'lower', val_emp
        counter += 1
    
    final_results_df.to_csv(os.path.join(save_path, method + '_' + nonb +'_final_results.csv'))

def train_svm_nonb_sim(inp):
    global_train, global_val, global_test, save_path, nonb, method = inp
    # for nonb in non_b_types:
    
    measures = ['tn', 'fp', 'fn', 'tp', 'accuracy', 'precision', 'recall', 'f-score', 'fpr', 'fnr', 'fdr']
    final_results_df = pd.DataFrame(columns=measures + ['data', 'kernel', 'label', 'method'])
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

    # kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
    #
    # for ker in kernel_list:
    #     print(nonb, ker)
    #     this_results_df = build_svm(train_x, train_y, val_x, val_y, ker)
    #     this_results_df['label'] = nonb
    #     results = results.append(this_results_df).reset_index(drop=True)
    #     results.to_csv(os.path.join(save_path, method + '_cv_sim_data.csv'))
    #
    # # find the best params in validation data
    # test_results = results[results['data'] == 'test']
    # min_fdr = np.min(test_results['fdr'])
    # best_row = test_results[test_results['fdr'] == min_fdr]
    #
    # # select the best param:
    # best_kernel = best_row['kernel'].values[0]

    # train with new train (train + val) and the best param
    train = train.append(val).reset_index(drop=True)
    train_x = train.drop(['label'], axis=1).to_numpy()
    train_y = train[['label']].to_numpy()

    print(nonb, len(train))
    best_results_df = build_svm(train_x, train_y, test_x, test_y, 'linear')
    best_results_df['label'] = nonb
    best_results_df['method'] = method

    final_results_df = final_results_df.append(best_results_df).reset_index(drop=True)
    final_results_df.to_csv(os.path.join(save_path, method + '_' + nonb + '_final_results.csv'))



def one_class_svm_with_cross_val_experimental_data():
    method = 'SVM_50'
    non_b_types = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                   'Short_Tandem_Repeat', 'Z_DNA_Motif']
    # beagle
    # non_b_path = '/home/mah19006/projects/nonBDNA/data/prepared_windows_req5/dataset/outliers/centered_windows'
    # bdna_path = '/home/mah19006/projects/nonBDNA/data/prepared_windows_req5/dataset/bdna'
    bdna_path = '/home/mah19006/projects/nonBDNA/data/experimental_50'
    non_b_path = '/home/mah19006/projects/nonBDNA/data/experimental_50'
    
    # local
    # non_b_path = 'Data/experimental'
    # bdna_path = 'Data/experimental'
    
    # uchc =
    # bdna_path = '/labs/Aguiar/non_bdna/annotations/vae_windows/prepared_windows_req5/dataset/bdna'
    # non_b_path = '/labs/Aguiar/non_bdna/annotations/vae_windows/prepared_windows_req5/dataset/outliers/centered_windows'
    
    
    # beagle
    methods_results_path = '/home/mah19006/projects/nonBDNA/data/methods_results'
    
    #local
    # methods_results_path = 'Data/methods_results'
    
    # UCHC
    # methods_results_path = '/labs/Aguiar/non_bdna/methods'
    
    if not os.path.exists(methods_results_path):
        os.mkdir(methods_results_path)
    save_path = os.path.join(methods_results_path, method)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # measures = ['tn', 'fp', 'fn', 'tp', 'accuracy', 'precision', 'recall', 'f-score', 'fpr', 'fnr', 'fdr']
    # results = pd.DataFrame(columns=measures + ['data', 'kernel', 'label'])

    global_train, global_val, global_test = load_data(bdna_path, non_b_path)
    # global_train, global_val, global_test = load_data_with_downsample_bdna(bdna_path, non_b_path)
    print(len(global_train))

    inputs = [[global_train, global_val, global_test, save_path, nonb, method] for nonb in non_b_types]
    pool = Pool(7)
    pool.map(train_svm_nonb_exp, inputs)
    

def one_class_svm_with_cross_val_synthetic_data():
    method = 'SVM_50_sim'
    non_b_types = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                   'Short_Tandem_Repeat', 'Z_DNA_Motif']
    # beagle
    # non_b_path = '/mnt/research/aguiarlab/proj/nonBDNA/data/exponential_simulated_data'
    # bdna_path = '/mnt/research/aguiarlab/proj/nonBDNA/data/exponential_simulated_data/'

    non_b_path = '/mnt/research/aguiarlab/proj/nonBDNA/data/sim_50'
    bdna_path = '/home/mah19006/projects/nonBDNA/data/sim_50'
    # local
    # non_b_path = 'Data/simulated'
    # bdna_path = 'Data/simulated'

    # # uchc
    # non_b_path = '/labs/Aguiar/non_bdna/annotations/exponential_simulated_data/dataset'
    # bdna_path = '/labs/Aguiar/non_bdna/annotations/exponential_simulated_data/dataset'
    
    # beagle
    methods_results_path = '/mnt/research/aguiarlab/proj/nonBDNA/data/methods_results'
    
    # local
    # methods_results_path = 'Data/methods_results'
    
    # uchc
    # methods_results_path = '/labs/Aguiar/non_bdna/methods'

    if not os.path.exists(methods_results_path):
        os.mkdir(methods_results_path)
    save_path = os.path.join(methods_results_path, method)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # measures = ['tn', 'fp', 'fn', 'tp', 'accuracy', 'precision', 'recall', 'f-score', 'fpr', 'fnr', 'fdr']
    # results = pd.DataFrame(columns=measures + ['data', 'kernel', 'label'])
    #
    # final_results_df = pd.DataFrame(columns=measures + ['data', 'kernel', 'label', 'method'])
    # global_train, global_val, global_test = load_data(bdna_path, non_b_path)
    global_train, global_val, global_test = load_data_with_downsample_bdna(bdna_path, non_b_path)
    inputs = [[global_train, global_val, global_test, save_path, nonb, method] for nonb in non_b_types]
    pool = Pool(7)
    pool.map(train_svm_nonb_sim, inputs)
    

if __name__ == '__main__':
    
    one_class_svm_with_cross_val_experimental_data()
    # one_class_svm_with_cross_val_synthetic_data()
