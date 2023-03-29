import sys

import matplotlib.pyplot as plt
import os
import numpy as np
import itertools
import math
import pandas as pd

def plot_combinations():
    w_size = 100
    non_b = 'Short_Tandem_Repeat'
    frquencies = [0.3]
    intensities = [0.5, 0.75, 1]
    limits = [5, 10, 20]
    N = int(np.ceil(n_nonb/(len(frquencies) * len(intensities) * len(limits))))
    parameters = [frquencies, intensities, limits]
    combinations = list(itertools.product(*parameters))

    colors = [plt.cm.Spectral(i) for i in np.linspace(0, 1, 27)]


    fig, axis = plt.subplots(1, 2, figsize=(17, 7), sharey=True)
    axis[0].set_prop_cycle('color', plt.cm.Spectral(np.linspace(0, 1, 27)))
    axis[1].set_prop_cycle('color', plt.cm.Spectral(np.linspace(0, 1, 27)))

    x = np.linspace(-int(w_size/2), int(w_size/2), w_size)
    for comb in combinations:
        freq, intensity, lim = comb
        sample_f = [0 if (p < -lim) or (p > lim) else intensity * np.sin(freq * p) for p in x]
        sample_r = [0 if (p < -lim) or (p >= lim) else -1 * intensity * np.sin(freq * p) for p in x]
        axis[0].plot(x, sample_f, label=str(comb), linewidth=1.5)
        axis[1].plot(x, sample_r, label=str(comb), linewidth=1.5)
    axis[0].set_xlabel('Position in Window', fontsize=18)
    axis[1].set_xlabel('Position in Window', fontsize=18)
    axis[0].set_ylabel('Translocation Time', fontsize=18)
    
    plt.legend()
    plt.setp(axis[0].get_xticklabels(), Fontsize=13)
    plt.setp(axis[0].get_yticklabels(), Fontsize=13)
    plt.setp(axis[1].get_xticklabels(), Fontsize=13)
    plt.setp(axis[1].get_yticklabels(), Fontsize=13)
    # ax.text(40, max_val+0.0001, direction_names[dir], fontsize=25, bbox={'alpha': 0, 'pad': 2})
    # ax.set_ylim(top=max_val+0.00053)
    axis[0].set_title('Forward', fontsize=20)
    axis[1].set_title('Reverse', fontsize=20)
    plt.tight_layout()
    plt.show()

def plot_functions(folder):
    elements = ['Short_Tandem_Repeat', 'G_Quadruplex_Motif']
    elements_name = ['Short Tandem Repeat (Synthetic Data)', 'G Quadruplex Motif (Synthetic Data)']
    elements_path = {non_b: os.path.join(folder, non_b + '_centered.csv') for non_b in elements}
    
    control_same_path = os.path.join(folder, 'forward_bdna.npy')
    control_opposite_path = os.path.join(folder, 'reverse_bdna.npy')
    control_signals = {'same': np.load(control_same_path), 'opposite': np.load(control_opposite_path)}
    
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    x = range(0, 100)
    direction = ['same', 'opposite']
    direction_cols = {'same': ['forward_' + str(i) for i in range(100)],
                      'opposite': ['reverse_' + str(i) for i in range(100)]}
    direction_names = {'same': 'Forward', 'opposite': 'Reverse'}
    plot_path = os.path.join(folder, 'plots')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    
    for elem_id in range(len(elements)):
        elem_name = elements_name[elem_id]
        elem = elements[elem_id]
        df = pd.read_csv(elements_path[elem], index_col=0)
        for dire in direction:
            cols = direction_cols[dire]
            signal = df.loc[:, cols].to_numpy()
            control_signal = control_signals[dire]
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            for quant_id in range(len(quantiles)):
                quant = quantiles[quant_id]
                color = colors[quant_id]
                signal_quantile = np.quantile(signal, quant, axis=0)
                control_signal_quantile = np.quantile(control_signal, quant, axis=0)
                ax.plot(x, signal_quantile, color=color, label=str(quant), linewidth=3)
                ax.plot(x, control_signal_quantile, color=color, linestyle='--', linewidth=3)
            # max_val = np.max(np.quantile(signal, 0.95, axis=0))
            ax.set_xlabel('Position in Window', fontsize=18)
            ax.set_ylabel('Translocation Time', fontsize=18)
            leg = plt.legend(loc=(1.03, 0.50), title="Quantile", prop={'size': 14}, title_fontsize=15)
            ax.add_artist(leg)
            h = [plt.plot([], [], color="gray", linestyle=ls, linewidth=3)[0] for ls in ['-', '--']]
            plt.legend(handles=h, labels=['Non-B DNA', 'Control'], loc=(1.03, 0.85),
                       title="Structure", prop={'size': 14}, title_fontsize=15)
            plt.setp(ax.get_xticklabels(), Fontsize=13)
            plt.setp(ax.get_yticklabels(), Fontsize=13)
            # ax.text(40, max_val+0.0001, direction_names[dir], fontsize=25, bbox={'alpha': 0, 'pad': 2})
            # ax.set_ylim(top=max_val+0.00053)
            ax.set_title(elem_name + ' ' + direction_names[dire], fontsize=20)
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(plot_path, elem + '_Control_' + direction_names[dire] + '.png'), dpi=1200)
            plt.savefig(os.path.join(plot_path, elem + '_Control_' + direction_names[dire] + '.tif'), dpi=1200)
            plt.savefig(os.path.join(plot_path, elem + '_Control_' + direction_names[dire] + '.tiff'), dpi=1200)
            plt.savefig(os.path.join(plot_path, elem + '_Control_' + direction_names[dire] + '.pdf'), dpi=1200)


def plot_function_5samples(folder):
    elements = ['Short_Tandem_Repeat', 'G_Quadruplex_Motif']
    elements_name = ['Short_Tandem_Repeat (Sim.)', 'G_Quadruplex_Motif (Sim.)']
    elements_path = {non_b: os.path.join(folder, non_b + '_centered.csv') for non_b in elements}
    
    control_same_path = os.path.join(folder, 'forward_bdna.npy')
    control_opposite_path = os.path.join(folder, 'reverse_bdna.npy')
    control_signals = {'same': np.load(control_same_path), 'opposite': np.load(control_opposite_path)}
    n_sample = 5

    x = range(0, 100)
    direction = ['same', 'opposite']
    direction_cols = {'same': ['forward_' + str(i) for i in range(100)],
                      'opposite': ['reverse_' + str(i) for i in range(100)]}
    direction_names = {'same': 'Forward', 'opposite': 'Reverse'}
    plot_path = os.path.join(folder, 'plots')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    
    for elem_id in range(len(elements)):
        elem_name = elements_name[elem_id]
        elem = elements[elem_id]
        df = pd.read_csv(elements_path[elem], index_col=0)
        for dire in direction:
            
            cols = direction_cols[dire]
            signal = df.loc[:, cols].to_numpy()
            control_signal = control_signals[dire]
            
            random_indices_signal = np.random.choice(signal.shape[0], size=n_sample, replace=False)
            random_indices_control = np.random.choice(control_signal.shape[0], size=n_sample, replace=False)

            signal = signal[random_indices_signal, :]
            control_signal = control_signal[random_indices_control, :]
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            for i in range(n_sample - 1):
                ax.plot(x, signal[i, :], color='red', linewidth=1)
                ax.plot(x, control_signal[i, :], color='gray', linestyle='--', linewidth=1)
            ax.plot(x, signal[i, :], color='red', label='Non-B DNA', linewidth=1)
            ax.plot(x, control_signal[i, :], color='gray', label='B-DNA', linestyle='--', linewidth=1)
            # max_val = np.max(np.quantile(signal, 0.95, axis=0))
            ax.set_xlabel('Position in Window', fontsize=18)
            ax.set_ylabel('Translocation Time', fontsize=18)
            leg = plt.legend(loc=(1.03, 0.50), title="Quantile", prop={'size': 14}, title_fontsize=15)
            ax.add_artist(leg)
            # h = [plt.plot([], [], color="gray", linestyle=ls, linewidth=3)[0] for ls in ['-', '--']]
            # plt.legend(handles=h, labels=['Non-B DNA', 'Control'], loc=(1.03, 0.85),
            #            title="Structure", prop={'size': 14}, title_fontsize=15)
            plt.legend()
            plt.setp(ax.get_xticklabels(), Fontsize=13)
            plt.setp(ax.get_yticklabels(), Fontsize=13)
            # ax.text(40, max_val+0.0001, direction_names[dir], fontsize=25, bbox={'alpha': 0, 'pad': 2})
            # ax.set_ylim(top=max_val+0.00053)
            ax.set_title(elem_name + ' ' + direction_names[dire], fontsize=20)
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(plot_path, elem + 'Samples_with_noise_' + direction_names[dire] + '.png'), dpi=200)
            plt.savefig(os.path.join(plot_path, elem + 'Samples_with_noise_' + direction_names[dire] + '.pdf'), dpi=200)


def plot_exp_data_5samples():
    n_sample = 5
    folder = 'Data/dataset/final_dataset_splitted'
    nonbs = ['Short_Tandem_Repeat', 'G_Quadruplex_Motif']
    plot_path = os.path.join(folder, 'plots')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
        
    x = np.arange(0, 50)
    
    for elem in nonbs:
        global_train = pd.read_csv(os.path.join(folder, elem + '_train.csv'), index_col=0)
        nonb_signal = global_train[global_train['label'] != 'bdna'].reset_index(drop=True)
        bdna_signal = global_train[global_train['label'] == 'bdna'].reset_index(drop=True)
        nonb_signal = nonb_signal.sample(n=n_sample, random_state=1).reset_index(drop=True)
        bdna_signal = bdna_signal.sample(n=n_sample, random_state=1).reset_index(drop=True)
        
        nonb_signal_np_f = nonb_signal[['forward_' + str(i) for i in range(50)]].to_numpy()
        nonb_signal_np_r = nonb_signal[['reverse_' + str(i) for i in range(50)]].to_numpy()
        
        bdna_signal_np_f = bdna_signal[['forward_' + str(i) for i in range(50)]].to_numpy()
        bdna_signal_np_r = bdna_signal[['reverse_' + str(i) for i in range(50)]].to_numpy()
        
        nonb_signals = {'same': nonb_signal_np_f, 'opposite': nonb_signal_np_r}
        controls = {'same': bdna_signal_np_f, 'opposite': bdna_signal_np_r}
        
        direction_names = {'same': 'Forward', 'opposite': 'Reverse'}
        
        for dire in direction_names.keys():
            
            signal = nonb_signals[dire]
            control_signal = controls[dire]
    
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            for i in range(n_sample - 1):
                ax.plot(x, signal[i, :], color='red', linewidth=1)
                ax.plot(x, control_signal[i, :], color='gray', linestyle='--', linewidth=1)
                
            ax.plot(x, signal[i, :], color='red', label='Non-B DNA', linewidth=1)
            ax.plot(x, control_signal[i, :], color='gray', label='B-DNA', linestyle='--', linewidth=1)
            # max_val = np.max(np.quantile(signal, 0.95, axis=0))
            ax.set_xlabel('Position in Window', fontsize=18)
            ax.set_ylabel('Translocation Time', fontsize=18)
            leg = plt.legend(loc=(1.03, 0.50), title="Quantile", prop={'size': 14}, title_fontsize=15)
            ax.add_artist(leg)
            plt.legend()
            plt.setp(ax.get_xticklabels(), Fontsize=13)
            plt.setp(ax.get_yticklabels(), Fontsize=13)
            ax.set_title(elem + ' ' + direction_names[dire], fontsize=20)
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(plot_path, elem + 'Samples_exp_' + direction_names[dire] + '.png'), dpi=200)
            plt.savefig(os.path.join(plot_path, elem + 'Samples_exp_' + direction_names[dire] + '.pdf'), dpi=200)




def simulate_nonb_gaussian_derivatives():
    x = np.linspace(-25, 25, 100)
    mu = 0
    st = 0.01
    noise = np.random.normal(loc=mu, scale=st, size=len(x))
    pi = math.pi
    fig, ax = plt.subplots(1, 1)
    for sigma in [1]:
        for alpha in [0.5]:
            for beta in [0.5]:
                y = -1 * alpha * np.exp((- 1 * (beta * x) ** 2) / (np.sqrt(2 * pi) * (sigma ** 2))) * (
                            1 * (beta * x)) / (sigma ** 2) + noise
                ax.plot(x, y, label=r'$\sigma = {}, \alpha = {}, \beta = {}$ '.format(sigma, alpha, beta))
    ax.set_ylabel(
        r'$- \alpha e^(\frac{-(\beta x)^2}{\sqrt{2\pi}\sigma^2}) \times \frac{\beta x}{\sigma^2} + \mathcal{N}(0, 0.02)$')
    ax.set_xlabel('x')
    plt.tight_layout()
    plt.legend()
    plt.show()


def simulate_split_bdna_gaussian_noise(f_std, r_std, s_path, nbdna):
    mu = 0
    train_portion = 0.8
    val_portion = 0.1
    forward_np = np.random.normal(loc=mu, scale=f_std, size=[nbdna, 100])
    reverse_np = np.random.normal(loc=mu, scale=r_std, size=[nbdna, 100])
    np.save(os.path.join(s_path, 'forward_bdna'), forward_np)
    np.save(os.path.join(s_path, 'reverse_bdna'), reverse_np)

    n_tr = int(nbdna * train_portion)
    n_val = int(nbdna * val_portion)

    indices = np.random.permutation(nbdna)

    train_idx, val_idx, test_idx = indices[:n_tr], indices[n_tr: n_tr + n_val], indices[n_tr + n_val:]
    forward_np_train = forward_np[train_idx, :]
    reverse_np_train = reverse_np[train_idx, :]

    forward_np_val = forward_np[val_idx, :]
    reverse_np_val = reverse_np[val_idx, :]

    forward_np_te = forward_np[test_idx, :]
    reverse_np_te = reverse_np[test_idx, :]

    np.save(os.path.join(s_path, 'forward_train_100'), forward_np_train)
    np.save(os.path.join(s_path, 'forward_val_100'), forward_np_val)
    np.save(os.path.join(s_path, 'forward_test_100'), forward_np_te)
    np.save(os.path.join(s_path, 'reverse_train_100'), reverse_np_train)
    np.save(os.path.join(s_path, 'reverse_val_100'), reverse_np_val)
    np.save(os.path.join(s_path, 'reverse_test_100'), reverse_np_te)


def simulate_split_bdna_dirac_noise(value, s_path, nbdna):
    value = 1
    mu = 0
    train_portion = 0.8
    val_portion = 0.1

    peak_f = np.random.choice(100, size=nbdna)
    peak_r = np.random.choice(100, size=nbdna)

    forward_np = np.zeros([nbdna, 100])
    reverse_np = np.zeros([nbdna, 100])
    for i in range(nbdna):
        forward_np[i, peak_f[i]] = value
        reverse_np[i, peak_r[i]] = value

    np.save(os.path.join(s_path, 'forward_bdna'), forward_np)
    np.save(os.path.join(s_path, 'reverse_bdna'), reverse_np)

    n_tr = int(nbdna * train_portion)
    n_val = int(nbdna * val_portion)

    indices = np.random.permutation(nbdna)

    train_idx, val_idx, test_idx = indices[:n_tr], indices[n_tr: n_tr + n_val], indices[n_tr + n_val:]
    forward_np_train = forward_np[train_idx, :]
    reverse_np_train = reverse_np[train_idx, :]

    forward_np_val = forward_np[val_idx, :]
    reverse_np_val = reverse_np[val_idx, :]

    forward_np_te = forward_np[test_idx, :]
    reverse_np_te = reverse_np[test_idx, :]

    np.save(os.path.join(s_path, 'forward_train_100'), forward_np_train)
    np.save(os.path.join(s_path, 'forward_val_100'), forward_np_val)
    np.save(os.path.join(s_path, 'forward_test_100'), forward_np_te)
    np.save(os.path.join(s_path, 'reverse_train_100'), reverse_np_train)
    np.save(os.path.join(s_path, 'reverse_val_100'), reverse_np_val)
    np.save(os.path.join(s_path, 'reverse_test_100'), reverse_np_te)


def simulate_split_nonb_sinus(n_nonb, s_path, st):
    w_size = 100
    non_b = 'Short_Tandem_Repeat'
    frquencies = [0.3]
    intensities = [0.5, 0.75, 1]
    limits = [5, 10, 20]
    N = int(np.ceil(n_nonb/(len(frquencies) * len(intensities) * len(limits))))
    parameters = [frquencies, intensities, limits]
    combinations = list(itertools.product(*parameters))

    x = np.linspace(-int(w_size/2), int(w_size/2), w_size)
    mu = 0
    noise_f = np.random.normal(loc=mu, scale=st, size=[N * len(combinations), len(x)])
    noise_r = np.random.normal(loc=mu, scale=st, size=[N * len(combinations), len(x)])
    

    functions_y_f = []
    functions_y_r = []
    masks = []
    combs = []
    for comb in combinations:
        freq, intensity, lim = comb
        sample_f = np.vstack([[0 if (p < -lim) or (p > lim) else intensity * np.sin(freq * p) for p in x] for n in range(N)])
        sample_r = np.vstack([[0 if (p < -lim) or (p >= lim) else intensity * np.sin(freq * p) for p in x] for n in range(N)])
        mask = np.vstack([[0 if (p < -lim) or (p >= lim) else 1 for p in x] for n in range(N)])
        comb_np = np.vstack([list(comb) for n in range(N)])
        functions_y_f.append(sample_f)
        functions_y_r.append(sample_r)
        masks.append(mask)
        combs.append(comb_np)
        
    function_y_f = np.vstack(functions_y_f)
    function_y_r = np.vstack(functions_y_r)
    all_masks = np.vstack(masks)
    all_combs = np.vstack(combs)
    # function_y_f = np.vstack([[0 if (p < -func_lim) or (p > func_lim) else np.sin(0.3 * p) for p in x] for n in range(n_nonb)])
    # function_y_r = np.vstack([[0 if (p < -func_lim) or (p >= func_lim) else -1 * np.sin(0.3 * p) for p in x] for n in range(n_nonb)])
    nonb_forward = noise_f + function_y_f
    nonb_reverse = noise_r + function_y_r
    # nonb_forward =  function_y_f
    # nonb_reverse =  function_y_r
    
    # mask = np.zeros([n_nonb, len(x)], dtype=int)
    # mask[:, int(w_size/2 - func_lim):-int(w_size/2 - func_lim)] = 1

    cols = ['id', 'chr', 'strand', 'start', 'end', 'label', 'motif_proportion'] + \
           ['forward_' + str(i) for i in range(w_size)] + \
           ['reverse_' + str(i) for i in range(w_size)] + \
           ['mask_' + str(i) for i in range(w_size)]
    param_cols = ['intensity', 'frequency', 'limit']
    
    non_b_df = pd.DataFrame(columns=cols + param_cols, index=range(N * len(combinations)))
    starts = list(range(1000, N * len(combinations) * 150 + 1000, 150))
    ends = [s + w_size - 1 for s in starts]
    ids = range(N * len(combinations))
    non_b_df['chr'] = 'chr'
    non_b_df['strand'] = '+'
    non_b_df['label'] = non_b
    non_b_df['motif_proportion'] = np.sum(all_masks, axis=1) / w_size
    non_b_df['id'] = ids
    non_b_df['start'] = starts
    non_b_df['end'] = ends
    non_b_df.iloc[:, 7:107] = nonb_forward
    non_b_df.iloc[:, 107:207] = nonb_reverse
    non_b_df.iloc[:, 207:307] = all_masks
    non_b_df.iloc[:, 307:310] = all_combs
    non_b_df = non_b_df.dropna()
    non_b_df.to_csv(os.path.join(s_path, non_b + '_centered.csv'))
    
    train_portion = 0.1
    val_portion = 0.1
    non_b_df = non_b_df.sample(frac=1).reset_index(drop=True)
    
    n_samples = len(non_b_df)
    n_tr = int(n_samples * train_portion)
    n_val = int(n_samples * val_portion)
    
    df_tr = non_b_df.iloc[0: n_tr, :].reset_index(drop=True)
    df_val = non_b_df.iloc[n_tr: n_tr + n_val, :].reset_index(drop=True)
    df_te = non_b_df.iloc[n_tr + n_val:, :].reset_index(drop=True)
    
    df_tr_param = df_tr[param_cols]
    df_val_param = df_val[param_cols]
    df_te_param = df_te[param_cols]
    
    df_tr_data = df_tr[cols]
    df_val_data = df_val[cols]
    df_te_data = df_te[cols]

    df_tr_data.to_csv(os.path.join(s_path, non_b + '_centered_train.csv'))
    df_val_data.to_csv(os.path.join(s_path, non_b + '_centered_validation.csv'))
    df_te_data.to_csv(os.path.join(s_path, non_b + '_centered_test.csv'))

    df_tr_param.to_csv(os.path.join(s_path, non_b + '_centered_train_params.csv'))
    df_val_param.to_csv(os.path.join(s_path, non_b + '_centered_validation_params.csv'))
    df_te_param.to_csv(os.path.join(s_path, non_b + '_centered_test_params.csv'))


def simulate_split_nonb_poly_d2(n_nonb, s_path, st):
    w_size = 100
    non_b = 'G_Quadruplex_Motif'
    frquencies = [0.05]
    intensities = [0.5, 0.75, 1]
    limits = [5, 10, 20]
    N = int(np.ceil(n_nonb / (len(frquencies) * len(intensities) * len(limits))))
    parameters = [frquencies, intensities, limits]
    combinations = list(itertools.product(*parameters))
    
    x = np.linspace(-int(w_size / 2), int(w_size / 2), w_size)
    mu = 0
    noise_f = np.random.normal(loc=mu, scale=st, size=[N * len(combinations), len(x)])
    noise_r = np.random.normal(loc=mu, scale=st, size=[N * len(combinations), len(x)])
    
    functions_y_f = []
    functions_y_r = []
    masks = []
    combs = []
    for comb in combinations:
        freq, intensity, lim = comb

        y_list_f = [[0 if (p < -lim) or (p > lim) else -1 * (freq * p) ** 2 + intensity for p in x] for n in range(N)]
        sample_f = np.vstack([[0 if (p < 0) else p for p in y] for y in y_list_f])
        
        y_list_r = [[0 if (p < -lim) or (p > lim) else 1 * (freq * p) ** 2 - intensity for p in x] for n in range(N)]
        sample_r = np.vstack( [[0 if (p > 0) else p for p in y] for y in y_list_r])

        mask = np.vstack([[0 if (p < -lim) or (p >= lim) else 1 for p in x] for n in range(N)])
        comb_np = np.vstack([list(comb) for n in range(N)])
        functions_y_f.append(sample_f)
        functions_y_r.append(sample_r)
        masks.append(mask)
        combs.append(comb_np)
    
    function_y_f = np.vstack(functions_y_f)
    function_y_r = np.vstack(functions_y_r)
    all_masks = np.vstack(masks)
    all_combs = np.vstack(combs)
    # function_y_f = np.vstack([[0 if (p < -func_lim) or (p > func_lim) else np.sin(0.3 * p) for p in x] for n in range(n_nonb)])
    # function_y_r = np.vstack([[0 if (p < -func_lim) or (p >= func_lim) else -1 * np.sin(0.3 * p) for p in x] for n in range(n_nonb)])
    nonb_forward = noise_f + function_y_f
    nonb_reverse = noise_r + function_y_r
    # nonb_forward =  function_y_f
    # nonb_reverse =  function_y_r
    
    # mask = np.zeros([n_nonb, len(x)], dtype=int)
    # mask[:, int(w_size/2 - func_lim):-int(w_size/2 - func_lim)] = 1
    
    cols = ['id', 'chr', 'strand', 'start', 'end', 'label', 'motif_proportion'] + \
           ['forward_' + str(i) for i in range(w_size)] + \
           ['reverse_' + str(i) for i in range(w_size)] + \
           ['mask_' + str(i) for i in range(w_size)]
    param_cols = ['intensity', 'frequency', 'limit']
    
    non_b_df = pd.DataFrame(columns=cols + param_cols, index=range(N * len(combinations)))
    starts = list(range(1000, N * len(combinations) * 150 + 1000, 150))
    ends = [s + w_size - 1 for s in starts]
    ids = range(N * len(combinations))
    non_b_df['chr'] = 'chr'
    non_b_df['strand'] = '+'
    non_b_df['label'] = non_b
    non_b_df['motif_proportion'] = np.sum(all_masks, axis=1) / w_size
    non_b_df['id'] = ids
    non_b_df['start'] = starts
    non_b_df['end'] = ends
    non_b_df.iloc[:, 7:107] = nonb_forward
    non_b_df.iloc[:, 107:207] = nonb_reverse
    non_b_df.iloc[:, 207:307] = all_masks
    non_b_df.iloc[:, 307:310] = all_combs
    non_b_df = non_b_df.dropna()
    non_b_df.to_csv(os.path.join(s_path, non_b + '_centered.csv'))
    
    train_portion = 0.1
    val_portion = 0.1
    non_b_df = non_b_df.sample(frac=1).reset_index(drop=True)
    
    n_samples = len(non_b_df)
    n_tr = int(n_samples * train_portion)
    n_val = int(n_samples * val_portion)
    
    df_tr = non_b_df.iloc[0: n_tr, :].reset_index(drop=True)
    df_val = non_b_df.iloc[n_tr: n_tr + n_val, :].reset_index(drop=True)
    df_te = non_b_df.iloc[n_tr + n_val:, :].reset_index(drop=True)
    
    df_tr_param = df_tr[param_cols]
    df_val_param = df_val[param_cols]
    df_te_param = df_te[param_cols]
    
    df_tr_data = df_tr[cols]
    df_val_data = df_val[cols]
    df_te_data = df_te[cols]
    
    df_tr_data.to_csv(os.path.join(s_path, non_b + '_centered_train.csv'))
    df_val_data.to_csv(os.path.join(s_path, non_b + '_centered_validation.csv'))
    df_te_data.to_csv(os.path.join(s_path, non_b + '_centered_test.csv'))
    
    df_tr_param.to_csv(os.path.join(s_path, non_b + '_centered_train_params.csv'))
    df_val_param.to_csv(os.path.join(s_path, non_b + '_centered_validation_params.csv'))
    df_te_param.to_csv(os.path.join(s_path, non_b + '_centered_test_params.csv'))





if __name__ == '__main__':
    # total number of nonb simulated
    if '-nb' in sys.argv:
        n_nonb = int(sys.argv[sys.argv.index('-nb') + 1])
    else:
        n_nonb = 10000
    
    # total number of bdna simulated
    if '-b' in sys.argv:
        n_bdna = int(sys.argv[sys.argv.index('-b') + 1])
    else:
        n_bdna = 1000000

    if '-std' in sys.argv:
        noise_std = int(sys.argv[sys.argv.index('-std') + 1])
    else:
        noise_std = 0.1
    win_size = 100
    save_path = 'Data/simulated_data'

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # Simulate nonb
    simulate_split_nonb_sinus(n_nonb, save_path, 0.3)
    simulate_split_nonb_poly_d2(n_nonb, save_path, 0.3)
    
    # Simulate bdna
    simulate_split_bdna_gaussian_noise(0.3, 0.3, save_path, n_bdna)
    
    # Plot
    plot_functions(save_path)
    plot_function_5samples(save_path)
