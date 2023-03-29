import time
import sys
from utils import *
from multiprocessing import Pool
import math


def one_class_svm_model(dataset, folder, results_path, n_bdna, n_nonb, thread):

    if not os.path.exists(results_path):
        os.mkdir(results_path)
    non_b_types = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Z_DNA_Motif', 'Mirror_Repeat', 'Direct_Repeat',
                   'Short_Tandem_Repeat', 'Inverted_Repeat']
    method = 'SVM'
    if dataset == 'exp':

        results_name = '_'.join([dataset, method])
        methods_results_path = os.path.join(results_path, results_name)
        if not os.path.exists(methods_results_path):
            os.mkdir(methods_results_path)
        
        win_sizes = [50]
        
        inputs = [[nonb, method, dataset, wins, folder, methods_results_path] for nonb in non_b_types
                  for wins in win_sizes]
        pool = Pool(thread)
        pool.map(train_svm_nonb_exp, inputs)
        # for inp in inputs:
        #     train_svm_nonb_exp(inp)
        
        results_exp_pd = collect_results(results_path, results_name)
    
    if dataset == 'sim':
    
        results_name = '_'.join([dataset, method])
        methods_results_path = os.path.join(results_path, results_name)
        if not os.path.exists(methods_results_path):
            os.mkdir(methods_results_path)
        
        # win_sizes = [25, 50, 75, 100]
        win_sizes = [50]  # [25, 50, 75, 100]
        nonb_ratios = [0.05, 0.1, 0.25]
        inputs = [[nonb, method, dataset, wins, n_bdna, n_nonb, nbr, data_path,
                   methods_results_path] for nonb in non_b_types for wins in win_sizes for nbr in nonb_ratios]
        pool = Pool(thread)
        pool.map(train_svm_nonb_sim, inputs)
        # for inp in inputs:
        #     print(inp[0], inp[1], inp[2], inp[3])
        #     train_if_nonb_sim(inp)
        results_sim_pd = collect_results(results_path, results_name)
        # plot_sim(results_path, results_name)


def train_svm_nonb_exp(inp):
    model_sel = True
    nonb, method, dataset, winsize, folder, main_folder = inp
    result_name = '_'.join([dataset, method, nonb])
    save_path = os.path.join(main_folder, result_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print('--------------- Model:', nonb)
    print('Folder ' + save_path + ' created.')
    
    train, val, test, train_bed, val_bed, test_bed = load_data3(folder, nonb)
    
    alpha_list = np.arange(0.05, 1, 0.05)
    # alpha_list = np.arange(0.01, 1, 0.01)
    tails = ['upper', 'lower']
    
    final_results_df = pd.DataFrame(columns=['dataset', 'method', 'label', 'alpha', 'tail', 'potential_nonb_counts',
                                             'training_time'], index=range(len(alpha_list) * len(tails)))
    counter = 0
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
    
    train, val = make_new_train_validation(train, val, ratio=5)
    train_x = train.drop(['label'], axis=1).to_numpy()
    
    print(nonb, '# training samples:', len(train))
    if model_sel:
        print('Model selection ...')
        best_param = svm_hyperparam_tuning(train, val, save_path)
        start = time.time()
        svm_model = OneClassSVM(kernel=best_param, cache_size=10000, max_iter=10000).fit(train_x)
        duration = round(time.time() - start, 3)
    else:
        start = time.time()
        svm_model = OneClassSVM(cache_size=10000, max_iter=10000).fit(train_x)
        duration = round(time.time() - start, 3)
    
    null_dist_scores, eval_scores = calc_null_eval_distributions(test, svm_model)
    
    print('Evaluation ... ')
    
    for tail in ['upper', 'lower']:
        for alpha in list(np.arange(0.05, 0.55, 0.05)):
            sel_test_bed, non_b_count = evaluation_exp(test, test_bed, null_dist_scores, eval_scores, alpha, tail)
            if any([math.isclose(alpha, num) for num in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]):
                print(method, tail, alpha, len(sel_test_bed))
                if len(sel_test_bed) > 0:
                    sel_test_bed.to_csv(os.path.join(save_path, 'test_alpha_{}_{}.bed'.format(alpha, tail)), sep='\t',
                                        index=False, header=False)
            final_results_df.loc[counter, :] = 'experimental', method, nonb, alpha, tail, non_b_count, duration
            counter += 1
            final_results_df.to_csv(os.path.join(save_path, 'final_results.csv'))
    print(os.path.join(save_path, 'final_results.csv'), ' saved.')


def train_svm_nonb_sim(inp):
    model_sel = True
    nonb, method, dataset, winsize, n_bdna, n_nonb, nonb_ratio, folder, main_folder = inp
    print('-----------Model: ', nonb, nonb_ratio)
    result_name = '_'.join([dataset, method, nonb, str(nonb_ratio)])
    save_path = os.path.join(main_folder, result_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print('Folder ' + save_path + ' created.')
    global_train, global_val, global_test = load_data2(folder, winsize, n_bdna, n_nonb, nonb_ratio=nonb_ratio)
    
    # alpha_list = np.arange(0.01, 1, 0.01)
    alpha_list = np.arange(0.05, 1, 0.05)
    tails = ['upper', 'lower']
    final_results_df = pd.DataFrame(columns=['dataset', 'method', 'label', 'window_size', 'nonb_ratio', 'alpha', 'tail',
                                             'tp', 'tn', 'fp', 'fn', 'precision', 'recall', 'fpr', 'tpr', 'fscore',
                                             'duration', 'function'], index=range(len(alpha_list) * len(tails) * 2))
    counter = 0
    train = global_train[(global_train['label'] == nonb) | (global_train['label'] == 'bdna')].reset_index(drop=True)
    val = global_val[(global_val['label'] == nonb) | (global_val['label'] == 'bdna')].reset_index(drop=True)
    test = global_test[(global_test['label'] == nonb) | (global_test['label'] == 'bdna')].reset_index(drop=True)
    
    train.loc[train['label'] == nonb, 'label'] = 1
    train.loc[train['label'] == 'bdna', 'label'] = 0
    
    val.loc[val['label'] == nonb, 'label'] = 1
    val.loc[val['label'] == 'bdna', 'label'] = 0
    
    test.loc[test['label'] == nonb, 'label'] = 1
    test.loc[test['label'] == 'bdna', 'label'] = 0
    
    convert_dict = {'label': int}
    
    train = train.astype(convert_dict)
    val = val.astype(convert_dict)
    test = test.astype(convert_dict)
    
    train_x = train.drop(['label', 'true_label'], axis=1).to_numpy()
    
    # hyperparam tunning:
    print(nonb, '# training samples:', len(train))
    if model_sel:
        print('Model selection ...')
        best_param = svm_hyperparam_tuning(train, val, save_path)
        start = time.time()
        svm_model = OneClassSVM(kernel=best_param, cache_size=10000, max_iter=1000).fit(train_x)
        duration = round(time.time() - start, 3)
    else:
        start = time.time()
        svm_model = OneClassSVM(cache_size=10000, max_iter=1000).fit(train_x)
        duration = round(time.time() - start, 3)
    
    print('Compute distributions ... ')
    null_dist_scores, eval_scores = calc_null_eval_distributions(test, svm_model)
    null_dist_scores_scores, eval_scores_scores = calc_null_eval_distributions_scores(test, svm_model)
    plot_histogram(null_dist_scores, method + '_' + nonb + '_test_bdna_decision_function', save_path)
    plot_histogram(eval_scores, method + '_' + nonb + '_test_nonb_decision_function', save_path)
    plot_histogram(null_dist_scores_scores, method + '_' + nonb + '_test_bdna_scores', save_path)
    plot_histogram(eval_scores_scores, method + '_' + nonb + '_test_nonb_scores', save_path)
    print('Evaluation ... ')
    for tail in tails:
        for alpha in alpha_list:
            # print(tail, alpha)
            p_values, tn, fp, fn, tp = evaluation_sim(test, null_dist_scores, eval_scores, alpha, tail)
            plot_histogram(p_values, '_'.join([method, nonb, tail, str(round(alpha, 2)), 'decision_p_values']),
                           save_path)
            precision, recall, tpr, fpr, fscore = compute_accuracy_metrics(tn, fp, fn, tp)
            final_results_df.loc[counter, :] = dataset, method, nonb, winsize, nonb_ratio, alpha, tail, tp, tn, fp, \
                                               fn, precision, recall, fpr, tpr, fscore, duration, 'decision_function'
            counter += 1
            p_values_2, tn, fp, fn, tp = evaluation_sim(test, null_dist_scores_scores, eval_scores_scores, alpha, tail)
            plot_histogram(p_values_2, '_'.join([method, nonb, tail, str(round(alpha, 2)), 'scores_p_values']),
                           save_path)
            precision, recall, tpr, fpr, fscore = compute_accuracy_metrics(tn, fp, fn, tp)
            final_results_df.loc[counter, :] = dataset, method, nonb, winsize, nonb_ratio, alpha, tail, tp, tn, fp, \
                                               fn, precision, recall, fpr, tpr, fscore, duration, 'scores'
            counter += 1
    final_results_df.to_csv(os.path.join(save_path, 'final_results.csv'))


if __name__ == '__main__':
    if '-d' in sys.argv:
        dataset = sys.argv[sys.argv.index('-d') + 1]
    else:
        # dataset = 'exp', 'sim'
        dataset = ''
    
    if '-f' in sys.argv:
        data_path = sys.argv[sys.argv.index('-f') + 1]
    else:
        data_path = ''

    
    if '-r' in sys.argv:
        results_path = sys.argv[sys.argv.index('-r') + 1]
    else:
        results_path = ''

    
    if '-t' in sys.argv:
        thread = int(sys.argv[sys.argv.index('-t') + 1])
    else:
        thread = 1
    
    if '-nb' in sys.argv:
        n_nonb = int(sys.argv[sys.argv.index('-nb') + 1])
    else:
        n_nonb = 20_000
    
    # total number of bdna simulated
    if '-b' in sys.argv:
        n_bdna = int(sys.argv[sys.argv.index('-b') + 1])
    else:
        n_bdna = 200_000
    
    one_class_svm_model(dataset, data_path, results_path, n_bdna, n_nonb, thread)


