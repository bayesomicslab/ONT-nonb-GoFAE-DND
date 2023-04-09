import time
import sys
sys.path.append('../')
from utils import *
from multiprocessing import Pool


def nearest_neighbors_model(dataset, folder, results_path, n_bdna, n_nonb, thread):

    if not os.path.exists(results_path):
        os.mkdir(results_path)
    non_b_types = ['G_Quadruplex_Motif', 'Short_Tandem_Repeat']
    method = 'KNN'
    
    if dataset == 'sim':
        
        results_name = '_'.join([dataset, method])
        methods_results_path = os.path.join(results_path, results_name)
        if not os.path.exists(methods_results_path):
            os.mkdir(methods_results_path)
        
        # win_sizes = [25, 50, 75, 100]
        win_sizes = [50]  # [25, 50, 75, 100]
        nonb_ratios = [0.05, 0.1, 0.25] # [0.025, 0.05, 0.075, 0.1, 0.25, 0.5]
        inputs = [[nonb, method, dataset, wins, n_bdna, n_nonb, nbr, data_path,
                   methods_results_path] for nonb in non_b_types for wins in win_sizes for nbr in nonb_ratios]
        pool = Pool(thread)
        pool.map(train_nn_nonb_sim, inputs)
        # for inp in inputs:
        #     print(inp[0], inp[1], inp[2], inp[3])
        #     train_if_nonb_sim(inp)
        results_sim_pd = collect_results(results_path, results_name)
        # plot_sim(results_path, results_name)
    else:
        exit(0)

def train_nn_nonb_sim(inp):
    model_sel = True
    nonb, method, dataset, winsize, n_bdna, n_nonb, nonb_ratio, folder, main_folder = inp
    print('-----------Model: ', nonb, nonb_ratio)
    result_name = '_'.join([dataset, method, nonb, str(nonb_ratio)])
    save_path = os.path.join(main_folder, result_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print('Folder ' + save_path + ' created.')
    global_train, global_val, global_test = load_data2(folder, winsize, n_bdna, n_nonb, nonb_ratio=nonb_ratio)
    
    train = global_train[(global_train['label'] == nonb) | (global_train['label'] == 'bdna')].reset_index(drop=True)
    val = global_val[(global_val['label'] == nonb) | (global_val['label'] == 'bdna')].reset_index(drop=True)
    test = global_test[(global_test['label'] == nonb) | (global_test['label'] == 'bdna')].reset_index(drop=True)
    
    train.loc[train['label'] == nonb, 'label'] = 1
    train.loc[train['true_label'] == nonb, 'true_label'] = 1
    
    train.loc[train['label'] == 'bdna', 'label'] = 0
    train.loc[train['true_label'] == 'bdna', 'true_label'] = 0
    
    val.loc[val['label'] == nonb, 'label'] = 1
    val.loc[val['true_label'] == nonb, 'true_label'] = 1
    
    val.loc[val['label'] == 'bdna', 'label'] = 0
    val.loc[val['true_label'] == 'bdna', 'true_label'] = 0
    
    test.loc[test['label'] == nonb, 'label'] = 1
    test.loc[test['true_label'] == nonb, 'true_label'] = 1
    
    test.loc[test['label'] == 'bdna', 'label'] = 0
    test.loc[test['true_label'] == 'bdna', 'true_label'] = 0
    
    convert_dict = {'label': int, 'true_label': int}
    
    train = train.astype(convert_dict)
    val = val.astype(convert_dict)
    test = test.astype(convert_dict)
    
    train, val = make_new_train_validation(train, val, ratio=1)
    
    train_x = train.drop(['label', 'true_label'], axis=1).to_numpy()
    train_y = train[['label']].to_numpy().flatten()
    
    test_x = test.drop(['label', 'true_label'], axis=1).to_numpy()
    test_y = test[['label']].to_numpy().flatten()
    test_y_true = test[['true_label']].to_numpy().flatten()
    
    # hyperparam tunning:
    print(nonb, '# training samples:', len(train))
    if model_sel:
        print('Model selection ...')
        best_param = nn_hyperparam_tuning(train, val, save_path)
        start = time.time()
        nn_model = KNeighborsClassifier(n_neighbors=best_param).fit(train_x, train_y)
        duration = round(time.time() - start, 3)
    else:
        start = time.time()
        nn_model = KNeighborsClassifier().fit(train_x, train_y)
        duration = round(time.time() - start, 3)
    
    print('Evaluation ... ')
    
    evaluate_classifiers(nn_model, test_x, test_y_true, dataset, method, nonb, nonb_ratio, save_path, winsize,
                         duration)


if __name__ == '__main__':
    if '-d' in sys.argv:
        dataset = sys.argv[sys.argv.index('-d') + 1]
    else:
        # dataset = 'exp'
        # dataset = 'sim'
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
        n_nonb = 20000
    
    # total number of bdna simulated
    if '-b' in sys.argv:
        n_bdna = int(sys.argv[sys.argv.index('-b') + 1])
    else:
        n_bdna = 200000
    
    nearest_neighbors_model(dataset, data_path, results_path, n_bdna, n_nonb, thread)


