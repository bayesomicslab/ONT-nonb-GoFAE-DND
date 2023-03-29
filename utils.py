import math
import os
import pickle
import gzip
import _pickle as cPickle
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from statsmodels.stats.multitest import fdrcorrection
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct
from multiprocessing import Pool


def chunk_it(lst, n):
    """Yield successive n-sized chunks from lst.
        Copyright: https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
    lst: input list
    num: number of chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_chunk_name(this_chunk):
    label = this_chunk[0][6]
    starts = []
    ends = []
    for ch in this_chunk:
        starts.append(ch[0])
        ends.append(ch[1])
    file_name = label + '_' + str(min(starts)) + '_' + str(max(ends)) + '.pkl.gz'
    return file_name


def read_window_with_pickle_gzip(path):
    with gzip.open(path, 'rb') as f:
        window = cPickle.load(f)
    return window


def write_window_with_pickle_gzip(window, path):
    with gzip.open(path, 'wb') as f:
        cPickle.dump(window, f)


def read_window_pickle(path):
    with open(path, 'rb') as f:
        x = pickle.load(f)
    return x


def compress_pickles_in_folder(folder):
    files = [os.path.join(folder, name) for name in os.listdir(folder) if '.pkl' in name and '.gz' not in name]
    for file in files:
        data = read_window_pickle(file)
        write_window_with_pickle_gzip(data, file + '.gz')
        os.remove(file)


def compress_pickles_chr_strand(chromosome, strand):
    main_folder = 'annotations/windows/initial_windows/' + chromosome + strand
    folders = sorted([os.path.join(main_folder, name) for name in os.listdir(main_folder) if
                      os.path.isdir(os.path.join(main_folder, name))])
    for folder in folders:
        compress_pickles_in_folder(folder)


def make_reverse_complete(seq):
    complement = {'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}
    com_seq = ''.join([complement[base] if base in complement.keys() else base for base in seq])
    reverse_comp_seq = com_seq[::-1]
    return reverse_comp_seq


def sliding_window_generator(iterable, size=100):
    i = iter(iterable)
    win = []
    for e in range(0, size):
        win.append(next(i))
    yield win
    for e in i:
        win = win[1:] + [e]
        yield win


def sliding_window(vector, window_size=100):
    n_sliding_windows = len(vector) - window_size + 1
    sliding_windows = np.zeros([n_sliding_windows, window_size])
    if len(vector) <= window_size:
        return vector
    for i in range(n_sliding_windows):
        sliding_windows[i, :] = vector[i:i + window_size]
    return sliding_windows


def prepare_bdna_dataset(path):

    forward_train_100 = np.load(os.path.join(path, 'forward_train_100.npy'))
    forward_val_100 = np.load(os.path.join(path, 'forward_val_100.npy'))
    forward_test_100 = np.load(os.path.join(path, 'forward_test_100.npy'))
    reverse_train_100 = np.load(os.path.join(path, 'reverse_train_100.npy'))
    reverse_val_100 = np.load(os.path.join(path, 'reverse_val_100.npy'))
    reverse_test_100 = np.load(os.path.join(path, 'reverse_test_100.npy'))
    return forward_train_100, forward_val_100, forward_test_100, reverse_train_100, reverse_val_100, reverse_test_100


def prepare_nonb_dataset_center(path):

    non_b_types = list(
        np.unique([name.split('_centered')[0] for name in os.listdir(path) if 'centered' in name and 'train' in name]))
    nonb_path_train = {nonb: os.path.join(path, nonb + '_centered_train.csv') for nonb in non_b_types}
    nonb_path_validation = {nonb: os.path.join(path, nonb + '_centered_validation.csv') for nonb in non_b_types}
    nonb_path_test = {nonb: os.path.join(path, nonb + '_centered_test.csv') for nonb in non_b_types}
    nonb_train_dfs = {nonb: pd.read_csv(nonb_path_train[nonb], index_col=0) for nonb in non_b_types}
    nonb_val_dfs = {nonb: pd.read_csv(nonb_path_validation[nonb], index_col=0) for nonb in non_b_types}
    nonb_test_dfs = {nonb: pd.read_csv(nonb_path_test[nonb], index_col=0) for nonb in non_b_types}
    return nonb_train_dfs, nonb_val_dfs, nonb_test_dfs


def prepare_nonb_dataset_sliding(nonb_type='all', min_motif_proportion=0.7):
    non_b_types = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                   'Short_Tandem_Repeat', 'Z_DNA_Motif']

    main_path = 'Data/prepared_windows/'
    nonbdna_sliding_path = os.path.join(main_path, 'outliers', 'sliding_extended_motifs')
    sliding_nonb_files_path = {nonb: os.path.join(nonbdna_sliding_path, nonb + '.csv') for nonb in non_b_types}
    cols = ['id', 'chr', 'strand', 'start', 'end', 'label', 'motif_proportion'] + \
           ['forward_' + str(i) for i in range(100)] + \
           ['reverse_' + str(i) for i in range(100)] + \
           ['mask_' + str(i) for i in range(100)]
    all_sliding = pd.DataFrame(columns=cols)
    if nonb_type == 'all':
        for nonb in non_b_types:
            this_df = pd.read_csv(sliding_nonb_files_path[nonb], index_col=0)
            this_df = this_df[this_df['motif_proportion'] == min_motif_proportion].reset_index(drop=True)
            all_sliding = all_sliding.append(this_df).reset_index(drop=True)
    else:
        if nonb_type in non_b_types:
            this_df = pd.read_csv(sliding_nonb_files_path[nonb_type], index_col=0)
            this_df = this_df[this_df['motif_proportion'] == min_motif_proportion].reset_index(drop=True)
            all_sliding = all_sliding.append(this_df).reset_index(drop=True)
    return all_sliding


def separate_df(df):
    train_portion = 0.8
    val_portion = 0.1
    train, validate, test = np.split(df.sample(frac=1, random_state=22),
                                     [int(train_portion * len(df)), int((train_portion + val_portion) * len(df))])
    return train, validate, test


def compute_accuracy_metrics(tn, fp, fn, tp):
    # accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        fscore = 0
    else:
        fscore = (2 * precision * recall) / (precision + recall)
    fpr = fp / (fp + tn)
    # fnr = fn / (fn + tp)
    # fdr = fp / (fp + tp) # False discovery rate
    tpr = tp / (tp + fn)
    return precision, recall, tpr, fpr, fscore


def split_dataframe(df, chunk_size=10000):
    chunks = []
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size:(i + 1) * chunk_size].reset_index(drop=True))
    return chunks


def load_data(bdna_folder, nonb_folder, win_size, frac=1):
    # bdna data:
    forward_cols = ['forward_' + str(i) for i in range(win_size)]
    reverse_cols = ['reverse_' + str(i) for i in range(win_size)]
    cols = forward_cols + reverse_cols + ['label']
    tr_bed, val_bed, te_bed = make_bed_files_for_splits(bdna_folder, nonb_folder, win_size, frac)
    tr_df_bdna, val_df_bdna, te_df_bdna, train_forward_mu, train_forward_std, train_reverse_mu, train_reverse_std = \
        reprocess_data2(bdna_folder, win_size, percent=frac, std_type='median/iqr', seed=42)
    #######################################################################
    # nonb data:
    nonb_tr_dfs, nonb_val_dfs, nonb_te_dfs = prepare_nonb_dataset_center(path=nonb_folder)
    nonb_types = list(nonb_tr_dfs.keys())
    nonb_tr_df = pd.DataFrame(columns=cols)
    nonb_val_df = pd.DataFrame(columns=cols)
    nonb_te_df = pd.DataFrame(columns=cols)
    start_idx_forward = (100 - win_size) // 2
    start_idx_reverse = (100 - win_size) // 2 + 100
    for non_b in nonb_types:
        nonb_train_forward = nonb_tr_dfs[non_b].iloc[:,
                             7 + start_idx_forward:7 + start_idx_forward + win_size].to_numpy()
        nonb_train_reverse = nonb_tr_dfs[non_b].iloc[:,
                             7 + start_idx_reverse:7 + start_idx_reverse + win_size].to_numpy()
        nonb_train_forward = (nonb_train_forward - train_forward_mu) / train_forward_std
        nonb_train_reverse = (nonb_train_reverse - train_reverse_mu) / train_reverse_std
        nonb_train = np.concatenate(
            (nonb_train_forward, nonb_train_reverse, np.ones(nonb_train_forward.shape[0]).reshape(-1, 1)), axis=1)
        this_tr_df_nonb = pd.DataFrame(data=nonb_train, columns=cols, index=range(nonb_train.shape[0]))
        this_tr_df_nonb['label'] = non_b
        nonb_tr_df = pd.concat([nonb_tr_df, this_tr_df_nonb], ignore_index=True)
        # nonb_tr_df = nonb_tr_df.append(this_tr_df_nonb).reset_index(drop=True)
        nonb_val_forward = nonb_val_dfs[non_b].iloc[:,
                           7 + start_idx_forward:7 + start_idx_forward + win_size].to_numpy()
        nonb_val_reverse = nonb_val_dfs[non_b].iloc[:,
                           7 + start_idx_reverse:7 + start_idx_reverse + win_size].to_numpy()
        nonb_val_forward = (nonb_val_forward - train_forward_mu) / train_forward_std
        nonb_val_reverse = (nonb_val_reverse - train_reverse_mu) / train_reverse_std
        nonb_val = np.concatenate(
            (nonb_val_forward, nonb_val_reverse, np.ones(nonb_val_forward.shape[0]).reshape(-1, 1)), axis=1)
        this_val_df_nonb = pd.DataFrame(data=nonb_val, columns=cols, index=range(nonb_val.shape[0]))
        this_val_df_nonb['label'] = non_b
        nonb_val_df = pd.concat([nonb_val_df, this_val_df_nonb], ignore_index=True)
        # nonb_val_df = nonb_val_df.append(this_val_df_nonb).reset_index(drop=True)
        nonb_test_forward = nonb_te_dfs[non_b].iloc[:,
                            7 + start_idx_forward:7 + start_idx_forward + win_size].to_numpy()
        nonb_test_reverse = nonb_te_dfs[non_b].iloc[:,
                            7 + start_idx_reverse:7 + start_idx_reverse + win_size].to_numpy()
        nonb_test_forward = (nonb_test_forward - train_forward_mu) / train_forward_std
        nonb_test_reverse = (nonb_test_reverse - train_reverse_mu) / train_reverse_std
        nonb_test = np.concatenate(
            (nonb_test_forward, nonb_test_reverse, np.ones(nonb_test_forward.shape[0]).reshape(-1, 1)), axis=1)
        this_test_df_nonb = pd.DataFrame(data=nonb_test, columns=cols, index=range(nonb_test.shape[0]))
        this_test_df_nonb['label'] = non_b
        nonb_te_df = pd.concat([nonb_te_df, this_test_df_nonb], ignore_index=True)
        # nonb_te_df = nonb_te_df.append(this_test_df_nonb).reset_index(drop=True)
    # nonb_tr_df = nonb_tr_df.astype(convert_dict)
    # nonb_val_df = nonb_val_df.astype(convert_dict)
    # nonb_te_df = nonb_te_df.astype(convert_dict)
    train = pd.concat([tr_df_bdna, nonb_tr_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(
        drop=True)
    val = pd.concat([val_df_bdna, nonb_val_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(
        drop=True)
    test = pd.concat([te_df_bdna, nonb_te_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    # train = tr_df_bdna.append(nonb_tr_df).sample(frac=1, random_state=42).reset_index(drop=True)
    # val = val_df_bdna.append(nonb_val_df).sample(frac=1, random_state=42).reset_index(drop=True)
    # test = te_df_bdna.append(nonb_te_df).sample(frac=1, random_state=42).reset_index(drop=True)
    # nonb_indices = train[train['label'] != 'bdna'].index
    # nonbinb_idx = random.sample(list(nonb_indices), nbinb)
    # train.loc[nonbinb_idx, 'label'] = 'bdna'
    return train, val, test, tr_bed, val_bed, te_bed


def save_data_split(bdna_folder, nonb_folder, win_size, frac=1):
    save_path = 'Data/dataset/final_dataset_splitted'
    # bdna data:
    forward_cols = ['forward_' + str(i) for i in range(win_size)]
    reverse_cols = ['reverse_' + str(i) for i in range(win_size)]
    cols = forward_cols + reverse_cols + ['label']
    bed_cols = ['id', 'chr', 'start', 'end', 'name', 'score', 'strand']
    shift = (100 - win_size) // 2
    # tr_bed, val_bed, te_bed = make_bed_files_for_splits(bdna_folder, nonb_folder, win_size, frac)
    tr_df_bdna, val_df_bdna, te_df_bdna, train_forward_mu, train_forward_std, train_reverse_mu, train_reverse_std = \
        reprocess_data2(bdna_folder, win_size, percent=frac, std_type='median/iqr', seed=42)
    tr_df_bdna['label'] = 'bdna'
    val_df_bdna['label'] = 'bdna'
    te_df_bdna['label'] = 'bdna'
    bdna_bed_tr = pd.DataFrame(columns=['chr', 'start', 'end', 'name', 'score', 'strand'], index=range(len(tr_df_bdna)))
    bdna_bed_val = pd.DataFrame(columns=['chr', 'start', 'end', 'name', 'score', 'strand'],
                                index=range(len(val_df_bdna)))
    bdna_bed_te = pd.DataFrame(columns=['chr', 'start', 'end', 'name', 'score', 'strand'], index=range(len(te_df_bdna)))
    #######################################################################
    # nonb data:
    nonb_tr_dfs, nonb_val_dfs, nonb_te_dfs = prepare_nonb_dataset_center(path=nonb_folder)
    nonb_types = list(nonb_tr_dfs.keys())
    start_idx_forward = (100 - win_size) // 2
    start_idx_reverse = (100 - win_size) // 2 + 100
    for non_b in nonb_types:
        nonb_train_forward = nonb_tr_dfs[non_b].iloc[:,
                             7 + start_idx_forward:7 + start_idx_forward + win_size].to_numpy()
        nonb_train_reverse = nonb_tr_dfs[non_b].iloc[:,
                             7 + start_idx_reverse:7 + start_idx_reverse + win_size].to_numpy()
        nonb_train_forward = (nonb_train_forward - train_forward_mu) / train_forward_std
        nonb_train_reverse = (nonb_train_reverse - train_reverse_mu) / train_reverse_std
        nonb_train = np.concatenate(
            (nonb_train_forward, nonb_train_reverse, np.ones(nonb_train_forward.shape[0]).reshape(-1, 1)), axis=1)
        this_tr_df_nonb = pd.DataFrame(data=nonb_train, columns=cols, index=range(nonb_train.shape[0]))
        this_tr_df_nonb['label'] = non_b
        this_nonb_train = pd.concat([tr_df_bdna, this_tr_df_nonb], ignore_index=True)
        this_nonb_train = this_nonb_train.sample(frac=1, random_state=42).reset_index(drop=True)
        this_nonb_train.to_csv(os.path.join(save_path, non_b + '_train.csv'))
        nonb_bed_tr = nonb_tr_dfs[non_b].iloc[:, 1:7]
        nonb_bed_tr = nonb_bed_tr[['chr', 'start', 'end', 'label', 'motif_proportion', 'strand']]
        nonb_bed_tr = nonb_bed_tr.rename(columns={'label': 'name', 'motif_proportion': 'score'})
        nonb_bed_tr['start'] = nonb_bed_tr['start'].apply(lambda x: x + shift + 100)
        nonb_bed_tr['end'] = nonb_bed_tr['start'].apply(lambda x: x + win_size)
        nonb_bed_tr['score'] = nonb_bed_tr['score'].apply(lambda x: int(1000 * min(1, x * (100 / win_size))))
        this_nonb_train_bed = pd.concat([bdna_bed_tr, nonb_bed_tr], ignore_index=True)
        this_nonb_train_bed = this_nonb_train_bed.sample(frac=1, random_state=42).reset_index(drop=True)
        this_nonb_train_bed['id'] = this_nonb_train_bed.index
        this_nonb_train_bed = this_nonb_train_bed[bed_cols]
        this_nonb_train_bed = this_nonb_train_bed.dropna().reset_index(drop=True)
        this_nonb_train_bed.to_csv(os.path.join(save_path, non_b + '_train_bed.csv'), index=False)
        nonb_val_forward = nonb_val_dfs[non_b].iloc[:,
                           7 + start_idx_forward:7 + start_idx_forward + win_size].to_numpy()
        nonb_val_reverse = nonb_val_dfs[non_b].iloc[:,
                           7 + start_idx_reverse:7 + start_idx_reverse + win_size].to_numpy()
        nonb_val_forward = (nonb_val_forward - train_forward_mu) / train_forward_std
        nonb_val_reverse = (nonb_val_reverse - train_reverse_mu) / train_reverse_std
        nonb_val = np.concatenate(
            (nonb_val_forward, nonb_val_reverse, np.ones(nonb_val_forward.shape[0]).reshape(-1, 1)), axis=1)
        this_val_df_nonb = pd.DataFrame(data=nonb_val, columns=cols, index=range(nonb_val.shape[0]))
        this_val_df_nonb['label'] = non_b
        this_nonb_val = pd.concat([val_df_bdna, this_val_df_nonb], ignore_index=True)
        this_nonb_val = this_nonb_val.sample(frac=1, random_state=42).reset_index(drop=True)
        this_nonb_val.to_csv(os.path.join(save_path, non_b + '_validation.csv'))
        nonb_bed_val = nonb_val_dfs[non_b].iloc[:, 1:7]
        nonb_bed_val = nonb_bed_val[['chr', 'start', 'end', 'label', 'motif_proportion', 'strand']]
        nonb_bed_val = nonb_bed_val.rename(columns={'label': 'name', 'motif_proportion': 'score'})
        nonb_bed_val['start'] = nonb_bed_val['start'].apply(lambda x: x + shift + 100)
        nonb_bed_val['end'] = nonb_bed_val['start'].apply(lambda x: x + win_size)
        nonb_bed_val['score'] = nonb_bed_val['score'].apply(lambda x: int(1000 * min(1, x * (100 / win_size))))
        this_nonb_val_bed = pd.concat([bdna_bed_val, nonb_bed_val], ignore_index=True)
        this_nonb_val_bed = this_nonb_val_bed.sample(frac=1, random_state=42).reset_index(drop=True)
        this_nonb_val_bed['id'] = this_nonb_val_bed.index
        this_nonb_val_bed = this_nonb_val_bed[bed_cols]
        this_nonb_val_bed = this_nonb_val_bed.dropna().reset_index(drop=True)
        this_nonb_val_bed.to_csv(os.path.join(save_path, non_b + '_validation_bed.csv'), index=False)
        nonb_test_forward = nonb_te_dfs[non_b].iloc[:,
                            7 + start_idx_forward:7 + start_idx_forward + win_size].to_numpy()
        nonb_test_reverse = nonb_te_dfs[non_b].iloc[:,
                            7 + start_idx_reverse:7 + start_idx_reverse + win_size].to_numpy()
        nonb_test_forward = (nonb_test_forward - train_forward_mu) / train_forward_std
        nonb_test_reverse = (nonb_test_reverse - train_reverse_mu) / train_reverse_std
        nonb_test = np.concatenate(
            (nonb_test_forward, nonb_test_reverse, np.ones(nonb_test_forward.shape[0]).reshape(-1, 1)), axis=1)
        this_test_df_nonb = pd.DataFrame(data=nonb_test, columns=cols, index=range(nonb_test.shape[0]))
        this_test_df_nonb['label'] = non_b
        this_nonb_te = pd.concat([te_df_bdna, this_test_df_nonb], ignore_index=True)
        this_nonb_te = this_nonb_te.sample(frac=1, random_state=42).reset_index(drop=True)
        this_nonb_te.to_csv(os.path.join(save_path, non_b + '_test.csv'))
        nonb_bed_te = nonb_te_dfs[non_b].iloc[:, 1:7]
        nonb_bed_te = nonb_bed_te[['chr', 'start', 'end', 'label', 'motif_proportion', 'strand']]
        nonb_bed_te = nonb_bed_te.rename(columns={'label': 'name', 'motif_proportion': 'score'})
        nonb_bed_te['start'] = nonb_bed_te['start'].apply(lambda x: x + shift + 100)
        nonb_bed_te['end'] = nonb_bed_te['start'].apply(lambda x: x + win_size)
        nonb_bed_te['score'] = nonb_bed_te['score'].apply(lambda x: int(1000 * min(1, x * (100 / win_size))))
        this_nonb_te_bed = pd.concat([bdna_bed_te, nonb_bed_te], ignore_index=True)
        this_nonb_te_bed = this_nonb_te_bed.sample(frac=1, random_state=42).reset_index(drop=True)
        this_nonb_te_bed['id'] = this_nonb_te_bed.index
        this_nonb_te_bed = this_nonb_te_bed[bed_cols]
        this_nonb_te_bed = this_nonb_te_bed.dropna().reset_index(drop=True)
        this_nonb_te_bed.to_csv(os.path.join(save_path, non_b + '_test_bed.csv'), index=False)


def normalize_nonb_df(all_nonb_dfs, non_b, cols, win_size, f_mu, f_std, r_mu, r_std):
    start_idx_forward = (100 - win_size) // 2
    start_idx_reverse = (100 - win_size) // 2 + 100
    nonb_train_forward = all_nonb_dfs[non_b].iloc[:,
                         7 + start_idx_forward:7 + start_idx_forward + win_size].to_numpy()
    nonb_train_reverse = all_nonb_dfs[non_b].iloc[:,
                         7 + start_idx_reverse:7 + start_idx_reverse + win_size].to_numpy()
    nonb_train_forward = (nonb_train_forward - f_mu) / f_std
    nonb_train_reverse = (nonb_train_reverse - r_mu) / r_std
    nonb_np = np.concatenate(
        (nonb_train_forward, nonb_train_reverse, np.ones(nonb_train_forward.shape[0]).reshape(-1, 1)), axis=1)
    nonb_df = pd.DataFrame(data=nonb_np, columns=cols, index=range(nonb_np.shape[0]))
    nonb_df['label'] = non_b
    return nonb_df


def load_data2(folder, win_size, n_bdna, n_nonb, nonb_ratio):
    """folder: path to data folder,
    win_size: could be 25, 50, 75, 100,
    n_bdna: number of bdna labels
    n_nonb: number of nonb labels
    nonb_ratio: (example: 0.05, i.e. we have to pull 0.95 actual bdna data but label them as nonb)"""
    # tr_df_bdna, val_df_bdna, te_df_bdna, train_forward_mu, train_forward_std, train_reverse_mu, train_reverse_std = \
    #     reprocess_data2(folder, win_size)
    tr_frac = 0.3
    val_frac = 0.2
    forward_cols = ['forward_' + str(i) for i in range(win_size)]
    reverse_cols = ['reverse_' + str(i) for i in range(win_size)]
    cols = forward_cols + reverse_cols + ['label']
    train_forward = np.load(os.path.join(folder, 'forward_train_100.npy'))
    train_reverse = np.load(os.path.join(folder, 'reverse_train_100.npy'))
    val_forward = np.load(os.path.join(folder, 'forward_val_100.npy'))
    val_reverse = np.load(os.path.join(folder, 'reverse_val_100.npy'))
    test_forward = np.load(os.path.join(folder, 'forward_test_100.npy'))
    test_reverse = np.load(os.path.join(folder, 'reverse_test_100.npy'))
    length = (train_forward.shape[1] * 2) // 2
    start_idx_forward = (length - win_size) // 2
    # start_idx_reverse = (length - win_size) // 2 + length
    start_idx_reverse = (length - win_size) // 2
    train_forward = train_forward[:, start_idx_forward: start_idx_forward + win_size]
    train_reverse = train_reverse[:, start_idx_reverse: start_idx_reverse + win_size]
    val_forward = val_forward[:, start_idx_forward: start_idx_forward + win_size]
    val_reverse = val_reverse[:, start_idx_reverse: start_idx_reverse + win_size]
    test_forward = test_forward[:, start_idx_forward: start_idx_forward + win_size]
    test_reverse = test_reverse[:, start_idx_reverse: start_idx_reverse + win_size]
    tr = np.concatenate((train_forward, train_reverse, np.zeros(train_forward.shape[0]).reshape(-1, 1)), axis=1)
    val = np.concatenate((val_forward, val_reverse, np.zeros(val_forward.shape[0]).reshape(-1, 1)), axis=1)
    te = np.concatenate((test_forward, test_reverse, np.zeros(test_forward.shape[0]).reshape(-1, 1)), axis=1)
    ####################################################
    tr_df_bdna = pd.DataFrame(data=tr, columns=cols, index=range(tr.shape[0]))
    val_df_bdna = pd.DataFrame(data=val, columns=cols, index=range(val.shape[0]))
    te_df_bdna = pd.DataFrame(data=te, columns=cols, index=range(te.shape[0]))
    bdna_df = pd.concat([tr_df_bdna, val_df_bdna, te_df_bdna], ignore_index=True)
    bdna_df = bdna_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_sample = len(bdna_df)
    n_tr = tr_frac * n_bdna
    n_val = val_frac * n_bdna
    tr_df_bdna = bdna_df.loc[0:n_tr - 1].reset_index(drop=True)
    val_df_bdna = bdna_df.loc[n_tr: n_tr + n_val - 1].reset_index(drop=True)
    te_df_bdna = bdna_df.loc[n_tr + n_val:n_bdna - 1].reset_index(drop=True)
    tr_df_bdna['label'] = 'bdna'
    val_df_bdna['label'] = 'bdna'
    te_df_bdna['label'] = 'bdna'
    tr_f_np = tr_df_bdna[forward_cols].to_numpy()
    tr_r_np = tr_df_bdna[reverse_cols].to_numpy()
    val_f_np = val_df_bdna[forward_cols].to_numpy()
    val_r_np = val_df_bdna[reverse_cols].to_numpy()
    te_f_np = te_df_bdna[forward_cols].to_numpy()
    te_r_np = te_df_bdna[reverse_cols].to_numpy()
    tfd_mu, tfd_std, trd_mu, trd_std = compute_stats(tr_f_np, tr_r_np, std_type='median/iqr')
    train_forward = (tr_f_np - tfd_mu) / tfd_std
    train_reverse = (tr_r_np - trd_mu) / trd_std
    val_forward = (val_f_np - tfd_mu) / tfd_std
    val_reverse = (val_r_np - trd_mu) / trd_std
    test_forward = (te_f_np - tfd_mu) / tfd_std
    test_reverse = (te_r_np - trd_mu) / trd_std
    tr_df_bdna[forward_cols] = train_forward
    tr_df_bdna[reverse_cols] = train_reverse
    val_df_bdna[forward_cols] = val_forward
    val_df_bdna[reverse_cols] = val_reverse
    te_df_bdna[forward_cols] = test_forward
    te_df_bdna[reverse_cols] = test_reverse
    tr_df_bdna['true_label'] = 'bdna'
    val_df_bdna['true_label'] = 'bdna'
    te_df_bdna['true_label'] = 'bdna'
    unused_indices = range(n_bdna, n_sample)
    unused_bdna = bdna_df.loc[n_bdna:].reset_index(drop=True)
    unused_f_np = unused_bdna[forward_cols].to_numpy()
    unused_r_np = unused_bdna[reverse_cols].to_numpy()
    unused_forward_norm = (unused_f_np - tfd_mu) / tfd_std
    unused_reverse_norm = (unused_r_np - trd_mu) / trd_std
    unused_bdna[forward_cols] = unused_forward_norm
    unused_bdna[reverse_cols] = unused_reverse_norm
    nbdna_needed_in_nonb = n_nonb - int(nonb_ratio * n_nonb)
    actual_nonb = n_nonb - nbdna_needed_in_nonb
    # nbdna_needed_in_nonb = int(n_nonb / nonb_ratio - n_nonb)
    assert nbdna_needed_in_nonb <= len(unused_indices)
    binnonb_df = unused_bdna.sample(n=nbdna_needed_in_nonb, random_state=42).reset_index(drop=True)
    #######################################################################
    # nonb data:
    nonb_tr_dfs, nonb_val_dfs, nonb_te_dfs = prepare_nonb_dataset_center(path=folder)
    nonb_types = list(nonb_tr_dfs.keys())
    all_nonb_dfs = {nonb: pd.concat([nonb_tr_dfs[nonb], nonb_val_dfs[nonb], nonb_te_dfs[nonb]]) for nonb in nonb_types}
    nonb_tr_df = pd.DataFrame(columns=cols)
    nonb_val_df = pd.DataFrame(columns=cols)
    nonb_te_df = pd.DataFrame(columns=cols)
    # start_idx_forward = (100 - win_size) // 2
    # start_idx_reverse = (100 - win_size) // 2 + 100
    # last_bdna_id = 0
    for non_b in nonb_types:
        # print(non_b)
        normalized_dfs = normalize_nonb_df(all_nonb_dfs, non_b, cols, win_size, tfd_mu, tfd_std, trd_mu, trd_std)
        normalized_dfs = normalized_dfs.sample(n=actual_nonb, random_state=42).reset_index(drop=True)
        # binnonb_df = unused_bdna.loc[last_bdna_id: last_bdna_id+nbdna_needed_in_nonb-1]
        normalized_dfs['true_label'] = non_b
        binnonb_df['true_label'] = 'bdna'
        this_nonb_df = pd.concat([normalized_dfs, binnonb_df], ignore_index=True)
        this_nonb_df['label'] = non_b
        this_nonb_df = this_nonb_df.sample(frac=1, random_state=42).reset_index(drop=True)
        y = this_nonb_df[['true_label']]
        x = this_nonb_df.drop(['true_label'], axis=1)
        x_tr_val, x_test, y_tr_val, y_test = train_test_split(x, y, stratify=y, train_size=0.2, random_state=42)
        x_tr, x_val, y_tr, y_val = train_test_split(x_tr_val, y_tr_val, stratify=y_tr_val, train_size=0.5,
                                                    random_state=42)
        nonb_tr_new_df = pd.concat([x_tr, y_tr], axis=1)
        nonb_val_new_df = pd.concat([x_val, y_val], axis=1)
        nonb_te_new_df = pd.concat([x_test, y_test], axis=1)
        # last_bdna_id += nbdna_needed_in_nonb
        nonb_tr_df = pd.concat([nonb_tr_df, nonb_tr_new_df], ignore_index=True)
        nonb_val_df = pd.concat([nonb_val_df, nonb_val_new_df], ignore_index=True)
        nonb_te_df = pd.concat([nonb_te_df, nonb_te_new_df], ignore_index=True)
    # nonb_tr_df = nonb_tr_df.astype(convert_dict)
    # nonb_val_df = nonb_val_df.astype(convert_dict)
    # nonb_te_df = nonb_te_df.astype(convert_dict)
    train = pd.concat([tr_df_bdna, nonb_tr_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(
        drop=True)
    val = pd.concat([val_df_bdna, nonb_val_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(
        drop=True)
    test = pd.concat([te_df_bdna, nonb_te_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return train, val, test


def load_data_with_downsample_bdna(bdna_folder, nonb_folder, win_size):
    forward_tr, reverse_tr, forward_val, reverse_val, forward_te, reverse_te, \
    train_forward_mu, train_forward_std, train_reverse_mu, train_reverse_std = reprocess_data(bdna_folder, win_size)
    win_size = forward_tr.shape[1]
    # forward_tr, forward_val, forward_te, reverse_tr, reverse_val, reverse_te = prepare_bdna_dataset(path=bdna_folder)
    # bdna data:
    forward_cols = ['forward_' + str(i) for i in range(win_size)]
    reverse_cols = ['reverse_' + str(i) for i in range(win_size)]
    cols = forward_cols + reverse_cols + ['label']
    tr = np.concatenate((forward_tr, reverse_tr, np.zeros(forward_tr.shape[0]).reshape(-1, 1)), axis=1)
    val = np.concatenate((forward_val, reverse_val, np.zeros(forward_val.shape[0]).reshape(-1, 1)), axis=1)
    te = np.concatenate((forward_te, reverse_te, np.zeros(forward_te.shape[0]).reshape(-1, 1)), axis=1)
    ####################################################
    tr_df_bdna = pd.DataFrame(data=tr, columns=cols, index=range(tr.shape[0]))
    val_df_bdna = pd.DataFrame(data=val, columns=cols, index=range(val.shape[0]))
    te_df_bdna = pd.DataFrame(data=te, columns=cols, index=range(te.shape[0]))
    tr_df_bdna = tr_df_bdna.sample(frac=0.1, random_state=42).reset_index(drop=True)
    val_df_bdna = val_df_bdna.sample(frac=0.1, random_state=42).reset_index(drop=True)
    te_df_bdna = te_df_bdna.sample(frac=0.1, random_state=42).reset_index(drop=True)
    convert_dict = {'label': int}
    tr_df_bdna = tr_df_bdna.astype(convert_dict)
    val_df_bdna = val_df_bdna.astype(convert_dict)
    te_df_bdna = te_df_bdna.astype(convert_dict)
    tr_df_bdna['label'] = 'bdna'
    val_df_bdna['label'] = 'bdna'
    te_df_bdna['label'] = 'bdna'
    #######################################################################
    # nonb data:
    nonb_tr_dfs, nonb_val_dfs, nonb_te_dfs = prepare_nonb_dataset_center(path=nonb_folder)
    nonb_types = list(nonb_tr_dfs.keys())
    nonb_tr_df = pd.DataFrame(columns=cols)
    nonb_val_df = pd.DataFrame(columns=cols)
    nonb_te_df = pd.DataFrame(columns=cols)
    start_idx_forward = (100 - win_size) // 2
    start_idx_reverse = (100 - win_size) // 2 + 100
    for non_b in nonb_types:
        nonb_train_forward = nonb_tr_dfs[non_b].iloc[:,
                             7 + start_idx_forward:7 + start_idx_forward + win_size].to_numpy()
        nonb_train_reverse = nonb_tr_dfs[non_b].iloc[:,
                             7 + start_idx_reverse:7 + start_idx_reverse + win_size].to_numpy()
        nonb_train_forward = (nonb_train_forward - train_forward_mu) / train_forward_std
        nonb_train_reverse = (nonb_train_reverse - train_reverse_mu) / train_reverse_std
        nonb_train = np.concatenate(
            (nonb_train_forward, nonb_train_reverse, np.ones(nonb_train_forward.shape[0]).reshape(-1, 1)), axis=1)
        this_tr_df_nonb = pd.DataFrame(data=nonb_train, columns=cols, index=range(nonb_train.shape[0]))
        this_tr_df_nonb['label'] = non_b
        nonb_tr_df = nonb_tr_df.append(this_tr_df_nonb).reset_index(drop=True)
        nonb_val_forward = nonb_val_dfs[non_b].iloc[:,
                           7 + start_idx_forward:7 + start_idx_forward + win_size].to_numpy()
        nonb_val_reverse = nonb_val_dfs[non_b].iloc[:,
                           7 + start_idx_reverse:7 + start_idx_reverse + win_size].to_numpy()
        nonb_val_forward = (nonb_val_forward - train_forward_mu) / train_forward_std
        nonb_val_reverse = (nonb_val_reverse - train_reverse_mu) / train_reverse_std
        nonb_val = np.concatenate(
            (nonb_val_forward, nonb_val_reverse, np.ones(nonb_val_forward.shape[0]).reshape(-1, 1)), axis=1)
        this_val_df_nonb = pd.DataFrame(data=nonb_val, columns=cols, index=range(nonb_val.shape[0]))
        this_val_df_nonb['label'] = non_b
        nonb_val_df = nonb_val_df.append(this_val_df_nonb).reset_index(drop=True)
        nonb_test_forward = nonb_te_dfs[non_b].iloc[:,
                            7 + start_idx_forward:7 + start_idx_forward + win_size].to_numpy()
        nonb_test_reverse = nonb_te_dfs[non_b].iloc[:,
                            7 + start_idx_reverse:7 + start_idx_reverse + win_size].to_numpy()
        nonb_test_forward = (nonb_test_forward - train_forward_mu) / train_forward_std
        nonb_test_reverse = (nonb_test_reverse - train_reverse_mu) / train_reverse_std
        nonb_test = np.concatenate(
            (nonb_test_forward, nonb_test_reverse, np.ones(nonb_test_forward.shape[0]).reshape(-1, 1)), axis=1)
        this_test_df_nonb = pd.DataFrame(data=nonb_test, columns=cols, index=range(nonb_test.shape[0]))
        this_test_df_nonb['label'] = non_b
        nonb_te_df = nonb_te_df.append(this_test_df_nonb).reset_index(drop=True)
    # nonb_tr_df = nonb_tr_df.astype(convert_dict)
    # nonb_val_df = nonb_val_df.astype(convert_dict)
    # nonb_te_df = nonb_te_df.astype(convert_dict)
    train = tr_df_bdna.append(nonb_tr_df).sample(frac=1, random_state=42).reset_index(drop=True)
    val = val_df_bdna.append(nonb_val_df).sample(frac=1, random_state=42).reset_index(drop=True)
    test = te_df_bdna.append(nonb_te_df).sample(frac=1, random_state=42).reset_index(drop=True)
    return train, val, test


def load_data_sim(bdna_folder, nonb_folder):
    forward_tr, reverse_tr, forward_val, reverse_val, forward_te, reverse_te, \
    train_forward_mu, train_forward_std, train_reverse_mu, train_reverse_std = reprocess_data(bdna_folder)
    # forward_tr, forward_val, forward_te, reverse_tr, reverse_val, reverse_te = prepare_bdna_dataset(path=bdna_folder)
    # bdna data:
    forward_cols = ['forward_' + str(i) for i in range(100)]
    reverse_cols = ['reverse_' + str(i) for i in range(100)]
    cols = forward_cols + reverse_cols + ['label']
    tr = np.concatenate((forward_tr, reverse_tr, np.zeros(forward_tr.shape[0]).reshape(-1, 1)), axis=1)
    val = np.concatenate((forward_val, reverse_val, np.zeros(forward_val.shape[0]).reshape(-1, 1)), axis=1)
    te = np.concatenate((forward_te, reverse_te, np.zeros(forward_te.shape[0]).reshape(-1, 1)), axis=1)
    ####################################################
    tr_df_bdna = pd.DataFrame(data=tr, columns=cols, index=range(tr.shape[0]))
    val_df_bdna = pd.DataFrame(data=val, columns=cols, index=range(val.shape[0]))
    te_df_bdna = pd.DataFrame(data=te, columns=cols, index=range(te.shape[0]))
    convert_dict = {'label': int}
    tr_df_bdna = tr_df_bdna.astype(convert_dict)
    val_df_bdna = val_df_bdna.astype(convert_dict)
    te_df_bdna = te_df_bdna.astype(convert_dict)
    tr_df_bdna['label'] = 'bdna'
    val_df_bdna['label'] = 'bdna'
    te_df_bdna['label'] = 'bdna'
    #######################################################################
    # nonb data:
    nonb_tr_dfs, nonb_val_dfs, nonb_te_dfs = prepare_nonb_dataset_center(path=nonb_folder)
    nonb_types = list(nonb_tr_dfs.keys())
    nonb_tr_df = pd.DataFrame(columns=cols)
    nonb_val_df = pd.DataFrame(columns=cols)
    nonb_te_df = pd.DataFrame(columns=cols)
    for non_b in nonb_types:
        nonb_train_forward = nonb_tr_dfs[non_b].iloc[:, 7:107].to_numpy()
        nonb_train_reverse = nonb_tr_dfs[non_b].iloc[:, 107:207].to_numpy()
        nonb_train_forward = (nonb_train_forward - train_forward_mu) / train_forward_std
        nonb_train_reverse = (nonb_train_reverse - train_reverse_mu) / train_reverse_std
        nonb_train = np.concatenate(
            (nonb_train_forward, nonb_train_reverse, np.ones(nonb_train_forward.shape[0]).reshape(-1, 1)), axis=1)
        this_tr_df_nonb = pd.DataFrame(data=nonb_train, columns=cols, index=range(nonb_train.shape[0]))
        this_tr_df_nonb['label'] = non_b
        nonb_tr_df = nonb_tr_df.append(this_tr_df_nonb).reset_index(drop=True)
        nonb_val_forward = nonb_val_dfs[non_b].iloc[:, 7:107].to_numpy()
        nonb_val_reverse = nonb_val_dfs[non_b].iloc[:, 107:207].to_numpy()
        nonb_val_forward = (nonb_val_forward - train_forward_mu) / train_forward_std
        nonb_val_reverse = (nonb_val_reverse - train_reverse_mu) / train_reverse_std
        nonb_val = np.concatenate(
            (nonb_val_forward, nonb_val_reverse, np.ones(nonb_val_forward.shape[0]).reshape(-1, 1)), axis=1)
        this_val_df_nonb = pd.DataFrame(data=nonb_val, columns=cols, index=range(nonb_val.shape[0]))
        this_val_df_nonb['label'] = non_b
        nonb_val_df = nonb_val_df.append(this_val_df_nonb).reset_index(drop=True)
        nonb_test_forward = nonb_te_dfs[non_b].iloc[:, 7:107].to_numpy()
        nonb_test_reverse = nonb_te_dfs[non_b].iloc[:, 107:207].to_numpy()
        nonb_test_forward = (nonb_test_forward - train_forward_mu) / train_forward_std
        nonb_test_reverse = (nonb_test_reverse - train_reverse_mu) / train_reverse_std
        nonb_test = np.concatenate(
            (nonb_test_forward, nonb_test_reverse, np.ones(nonb_test_forward.shape[0]).reshape(-1, 1)), axis=1)
        this_test_df_nonb = pd.DataFrame(data=nonb_test, columns=cols, index=range(nonb_test.shape[0]))
        this_test_df_nonb['label'] = non_b
        nonb_te_df = nonb_te_df.append(this_test_df_nonb).reset_index(drop=True)
    # nonb_tr_df = nonb_tr_df.astype(convert_dict)
    # nonb_val_df = nonb_val_df.astype(convert_dict)
    # nonb_te_df = nonb_te_df.astype(convert_dict)
    train = tr_df_bdna.append(nonb_tr_df).sample(frac=1, random_state=42).reset_index(drop=True)
    val = val_df_bdna.append(nonb_val_df).sample(frac=1, random_state=42).reset_index(drop=True)
    test = te_df_bdna.append(nonb_te_df).sample(frac=1, random_state=42).reset_index(drop=True)
    return train, val, test


def compute_stats(forward, reverse, std_type='median/iqr'):
    """Aaron function"""
    if std_type == 'mean/std':
        # print('Using mean/std\n')
        tfd_mu, tfd_std = forward.mean(), reverse.std()
        trd_mu, trd_std = forward.mean(), reverse.std()
        return (tfd_mu, tfd_std, trd_mu, trd_std)
    elif std_type == 'median/iqr':
        # print('Using median/iqr\n')
        iqr_fd = np.subtract(*np.percentile(forward, [75, 25], interpolation='midpoint'))
        iqr_rd = np.subtract(*np.percentile(reverse, [75, 25], interpolation='midpoint'))
        tfd_mu, tfd_std = np.median(forward), iqr_fd
        trd_mu, trd_std = np.median(reverse), iqr_rd
        return (tfd_mu, tfd_std, trd_mu, trd_std)
    else:
        print('Not implemented\n')


def reprocess_data(bdna_path, win_size, percent=0.3, std_type='median/iqr', seed=42):
    """Aaron function"""
    train_forward = np.load(os.path.join(bdna_path, 'forward_train_100.npy'))
    train_reverse = np.load(os.path.join(bdna_path, 'reverse_train_100.npy'))
    val_forward = np.load(os.path.join(bdna_path, 'forward_val_100.npy'))
    val_reverse = np.load(os.path.join(bdna_path, 'reverse_val_100.npy'))
    test_forward = np.load(os.path.join(bdna_path, 'forward_test_100.npy'))
    test_reverse = np.load(os.path.join(bdna_path, 'reverse_test_100.npy'))
    forward = np.vstack((train_forward, val_forward, test_forward))
    reverse = np.vstack((train_reverse, val_reverse, test_reverse))
    full_bdna = np.hstack((forward, reverse))
    full_bdna_df = pd.DataFrame(full_bdna)
    full_bdna_df = full_bdna_df.sample(frac=1, random_state=seed)
    train_size = int(percent * len(full_bdna_df))
    full_train0, full_val0 = np.array(full_bdna_df[:train_size]), full_bdna_df[train_size:]
    val_size = int(percent * len(full_val0))
    full_val0, full_test0 = np.array(full_val0[:val_size]), np.array(full_val0[val_size:])
    length = full_val0.shape[1] // 2
    start_idx_forward = (length - win_size) // 2
    start_idx_reverse = (length - win_size) // 2 + length
    train_forward = full_train0[:, start_idx_forward: start_idx_forward + win_size]
    train_reverse = full_train0[:, start_idx_reverse: start_idx_reverse + win_size]
    val_forward = full_val0[:, start_idx_forward: start_idx_forward + win_size]
    val_reverse = full_val0[:, start_idx_reverse: start_idx_reverse + win_size]
    test_forward = full_test0[:, start_idx_forward: start_idx_forward + win_size]
    test_reverse = full_test0[:, start_idx_reverse: start_idx_reverse + win_size]
    tfd_mu, tfd_std, trd_mu, trd_std = compute_stats(train_forward, train_reverse, std_type=std_type)
    train_forward = (train_forward - tfd_mu) / tfd_std
    train_reverse = (train_reverse - trd_mu) / trd_std
    val_forward = (val_forward - tfd_mu) / tfd_std
    val_reverse = (val_reverse - trd_mu) / trd_std
    test_forward = (test_forward - tfd_mu) / tfd_std
    test_reverse = (test_reverse - trd_mu) / trd_std
    # print('New Train Set {:.2f}'.format(len(full_train0)/len(full_bdna)))
    # print('New Validation Set {:.2f}'.format(len(full_val0)/len(full_bdna)))
    # print('New Test Set {:.2f}'.format(len(full_test0)/len(full_bdna)))
    return train_forward, train_reverse, val_forward, val_reverse, test_forward, test_reverse, tfd_mu, tfd_std, trd_mu, trd_std


def reprocess_data2(bdna_path, win_size, percent=1, std_type='median/iqr', seed=42):
    """Marjan function"""
    tr_frac = 0.3
    val_frac = 0.2
    forward_cols = ['forward_' + str(i) for i in range(win_size)]
    reverse_cols = ['reverse_' + str(i) for i in range(win_size)]
    cols = forward_cols + reverse_cols + ['label']
    train_forward = np.load(os.path.join(bdna_path, 'forward_train_100.npy'))
    train_reverse = np.load(os.path.join(bdna_path, 'reverse_train_100.npy'))
    val_forward = np.load(os.path.join(bdna_path, 'forward_val_100.npy'))
    val_reverse = np.load(os.path.join(bdna_path, 'reverse_val_100.npy'))
    test_forward = np.load(os.path.join(bdna_path, 'forward_test_100.npy'))
    test_reverse = np.load(os.path.join(bdna_path, 'reverse_test_100.npy'))
    length = (train_forward.shape[1] * 2) // 2
    start_idx_forward = (length - win_size) // 2
    # start_idx_reverse = (length - win_size) // 2 + length
    start_idx_reverse = (length - win_size) // 2
    train_forward = train_forward[:, start_idx_forward: start_idx_forward + win_size]
    train_reverse = train_reverse[:, start_idx_reverse: start_idx_reverse + win_size]
    val_forward = val_forward[:, start_idx_forward: start_idx_forward + win_size]
    val_reverse = val_reverse[:, start_idx_reverse: start_idx_reverse + win_size]
    test_forward = test_forward[:, start_idx_forward: start_idx_forward + win_size]
    test_reverse = test_reverse[:, start_idx_reverse: start_idx_reverse + win_size]
    tr = np.concatenate((train_forward, train_reverse, np.zeros(train_forward.shape[0]).reshape(-1, 1)), axis=1)
    val = np.concatenate((val_forward, val_reverse, np.zeros(val_forward.shape[0]).reshape(-1, 1)), axis=1)
    te = np.concatenate((test_forward, test_reverse, np.zeros(test_forward.shape[0]).reshape(-1, 1)), axis=1)
    ####################################################
    tr_df_bdna = pd.DataFrame(data=tr, columns=cols, index=range(tr.shape[0]))
    val_df_bdna = pd.DataFrame(data=val, columns=cols, index=range(val.shape[0]))
    te_df_bdna = pd.DataFrame(data=te, columns=cols, index=range(te.shape[0]))
    
    bdna_df = pd.concat([tr_df_bdna, val_df_bdna, te_df_bdna], ignore_index=True)
    bdna_df = bdna_df.sample(frac=percent, random_state=seed).reset_index(drop=True)
    n_samples = len(bdna_df)
    n_tr = tr_frac * n_samples
    n_val = val_frac * n_samples
    tr_df_bdna = bdna_df.loc[0:n_tr - 1].reset_index(drop=True)
    val_df_bdna = bdna_df.loc[n_tr: n_tr + n_val - 1].reset_index(drop=True)
    te_df_bdna = bdna_df.loc[n_tr + n_val:].reset_index(drop=True)
    # tr_df_bdna = tr_df_bdna.sample(frac=percent, random_state=seed).reset_index(drop=True)
    # val_df_bdna = val_df_bdna.sample(frac=percent, random_state=seed).reset_index(drop=True)
    # te_df_bdna = te_df_bdna.sample(frac=percent, random_state=seed).reset_index(drop=True)
    convert_dict = {'label': int}
    tr_df_bdna = tr_df_bdna.astype(convert_dict)
    val_df_bdna = val_df_bdna.astype(convert_dict)
    te_df_bdna = te_df_bdna.astype(convert_dict)
    tr_df_bdna['label'] = 'bdna'
    val_df_bdna['label'] = 'bdna'
    te_df_bdna['label'] = 'bdna'
    tr_f_np = tr_df_bdna[forward_cols].to_numpy()
    tr_r_np = tr_df_bdna[reverse_cols].to_numpy()
    val_f_np = val_df_bdna[forward_cols].to_numpy()
    val_r_np = val_df_bdna[reverse_cols].to_numpy()
    te_f_np = te_df_bdna[forward_cols].to_numpy()
    te_r_np = te_df_bdna[reverse_cols].to_numpy()
    tfd_mu, tfd_std, trd_mu, trd_std = compute_stats(tr_f_np, tr_r_np, std_type=std_type)
    train_forward = (tr_f_np - tfd_mu) / tfd_std
    train_reverse = (tr_r_np - trd_mu) / trd_std
    val_forward = (val_f_np - tfd_mu) / tfd_std
    val_reverse = (val_r_np - trd_mu) / trd_std
    test_forward = (te_f_np - tfd_mu) / tfd_std
    test_reverse = (te_r_np - trd_mu) / trd_std
    tr_df_bdna[forward_cols] = train_forward
    tr_df_bdna[reverse_cols] = train_reverse
    val_df_bdna[forward_cols] = val_forward
    val_df_bdna[reverse_cols] = val_reverse
    te_df_bdna[forward_cols] = test_forward
    te_df_bdna[reverse_cols] = test_reverse
    return tr_df_bdna, val_df_bdna, te_df_bdna, tfd_mu, tfd_std, trd_mu, trd_std


def compute_empirical(null_dist, eval_data, tail='two_sided'):
    store_emp_pval = []
    if int(scipy.__version__.split('.')[1]) < 9 or len(eval_data) > 700000:
        # if int(scipy.__version__.split('.')[1]) < 9:
        if tail == 'lower':
            for i in range(len(eval_data)):
                temp = stats.percentileofscore(null_dist, eval_data[i]) / 100.
                store_emp_pval.append(temp)
        elif tail == 'upper':
            for i in range(len(eval_data)):
                temp = 1. - stats.percentileofscore(null_dist, eval_data[i]) / 100.
                store_emp_pval.append(temp)
        else:
            print('Not defined')
    else:
        if tail == 'lower':
            store_emp_pval = stats.percentileofscore(null_dist, eval_data) / 100.
        elif tail == 'upper':
            store_emp_pval = 1. - stats.percentileofscore(null_dist, eval_data) / 100.
        else:
            print('Not defined')
    emp_dist = np.sort(np.array(store_emp_pval))
    indices = np.argsort(np.array(store_emp_pval))
    return emp_dist, indices


def list_comp_emp(inp):
    null_dist, this_eval, tail, ids, path = inp
    store_emp_pval = []
    if tail == 'lower':
        for i in range(len(this_eval)):
            temp = stats.percentileofscore(null_dist, this_eval[i]) / 100.
            store_emp_pval.append(temp)
    elif tail == 'upper':
        for i in range(len(this_eval)):
            temp = 1. - stats.percentileofscore(null_dist, this_eval[i]) / 100.
            store_emp_pval.append(temp)
    else:
        print('Not defined')
    window = {ids: store_emp_pval}
    write_window_with_pickle_gzip(window, os.path.join(path, str(ids) + '.gzip'))


def list_comp_emp2(inp):
    null_dist, this_eval, tail, ids, path = inp
    lll = []
    if int(scipy.__version__.split('.')[1]) < 9:
        # if int(scipy.__version__.split('.')[1]) < 9:
        if tail == 'lower':
            for i in range(len(this_eval)):
                temp = stats.percentileofscore(null_dist, this_eval[i]) / 100.
                lll.append(temp)
        elif tail == 'upper':
            for i in range(len(this_eval)):
                temp = 1. - stats.percentileofscore(null_dist, this_eval[i]) / 100.
                lll.append(temp)
        else:
            print('Not defined')
    else:
        if tail == 'lower':
            lll = stats.percentileofscore(null_dist, this_eval) / 100.
        elif tail == 'upper':
            lll = 1. - stats.percentileofscore(null_dist, this_eval) / 100.
        else:
            print('Not defined')
    
    window = {ids: lll}
    write_window_with_pickle_gzip(window, os.path.join(path, str(ids) + '.gzip'))



def FDR_BHP(dist, alpha=0.5):
    BH_corrected = alpha * np.arange(1, len(dist) + 1) / len(dist)
    check = (dist <= BH_corrected)
    if check.sum() != 0:
        valemp = np.max(np.argwhere(check)) + 1
        estimated_proportion = valemp * (1. - alpha) / len(dist)
    else:
        valemp = 0
        estimated_proportion = 0
    return valemp, estimated_proportion


def FDR_BHP2(dist, alpha=0.5):
    BH_corrected = alpha * np.arange(1, len(dist) + 1) / len(dist)
    check = (dist <= BH_corrected)
    if check.sum() != 0:
        valemp = np.max(np.argwhere(check)) + 1
        estimated_proportion = valemp * (1. - alpha) / len(dist)
    else:
        valemp = 0
        estimated_proportion = 0
    return valemp, estimated_proportion


def plot_histogram(lst, name, save_path):
    plt.cla()
    plt.clf()
    plt.close()
    plt.hist(lst, bins=30)
    plt.xlabel(name)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, name + '.png'))


def plot_roc_curve(fpr, tpr, roc_auc_score, dataset, method, nonb, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(fpr, tpr, linewidth=4, label='ROC Curve')
    plt.xlabel('False Positive Rate', fontsize=15)  # Add an x-label to the axes.
    plt.ylabel('True Negative Rate', fontsize=15)  # Add a y-label to the axes.
    ax.legend()
    title = '{} \nROC-AUC: {:.4f}'.format(method, roc_auc_score)
    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, dataset + '_' + method + '_' + nonb + '_eval_ROC.png'))


def plot_PR_curve(recall, precision, pr_auc, dataset, method, nonb, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(recall, precision, linewidth=4, label='PR Curve')
    plt.xlabel('Recall', fontsize=15)  # Add an x-label to the axes.
    plt.ylabel('Precision', fontsize=15)  # Add a y-label to the axes.
    ax.legend()
    title = '{}\nPR-AUC: {:.4f}'.format(method, pr_auc)
    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, dataset + '_' + method + '_' + nonb + '_eval_PR.png'))


def calc_null_eval_distributions(test, if_model):
    test_bdna = test[test['label'] == 0].reset_index(drop=True)
    test_nonb = test[test['label'] == 1].reset_index(drop=True)
    if 'true_label' in list(test.columns.values):
        test_bdna_x = test_bdna.drop(['label', 'true_label'], axis=1).to_numpy()
        test_nonb_x = test_nonb.drop(['label', 'true_label'], axis=1).to_numpy()
    else:
        test_bdna_x = test_bdna.drop(['label'], axis=1).to_numpy()
        test_nonb_x = test_nonb.drop(['label'], axis=1).to_numpy()
    null_dist_scores = if_model.decision_function(test_bdna_x)
    eval_scores = if_model.decision_function(test_nonb_x)
    return null_dist_scores, eval_scores


def calc_null_eval_distributions_scores(test, if_model):
    test_bdna = test[test['label'] == 0].reset_index(drop=True)
    test_nonb = test[test['label'] == 1].reset_index(drop=True)
    if 'true_label' in list(test.columns.values):
        test_bdna_x = test_bdna.drop(['label', 'true_label'], axis=1).to_numpy()
        test_nonb_x = test_nonb.drop(['label', 'true_label'], axis=1).to_numpy()
    else:
        test_bdna_x = test_bdna.drop(['label'], axis=1).to_numpy()
        test_nonb_x = test_nonb.drop(['label'], axis=1).to_numpy()
    null_dist_scores = if_model.score_samples(test_bdna_x)
    eval_scores = if_model.score_samples(test_nonb_x)
    return null_dist_scores, eval_scores


def calc_confusion_matrix_sim(test, null_dist_scores, eval_scores, alpha, tail):
    """This function uses the nonb part of test data for evaluation"""
    # indices of nonb abd bdna true and labels
    test_nonb_indices = test[test['label'] == 1].index
    test_bdna_indices = test[test['label'] == 0].index
    test_nonb_indices_true = test[test['true_label'] != 'bdna'].index
    # Nonb detected by model in nonb labelled part
    p_values, indices = compute_empirical(null_dist_scores, eval_scores, tail=tail)
    fdr_check, _ = fdrcorrection(p_values, alpha=alpha, is_sorted=True)
    rejected_counts = np.max(np.where(fdr_check)) + 1
    y_hat_nonb_indices_in_pval = indices[0:rejected_counts]
    y_hat_nonb_indices_in_nonb = test_nonb_indices[y_hat_nonb_indices_in_pval].to_list()
    y_hat_bdna_indices_in_pval = indices[rejected_counts:]
    y_hat_bdna_indices_in_nonb = test_nonb_indices[y_hat_bdna_indices_in_pval].to_list()
    # Construct y and y_hat
    y = np.zeros((len(test_nonb_indices) + len(test_bdna_indices)), dtype=int)
    y[test_nonb_indices_true] = 1
    y_hat = np.ones((len(test_nonb_indices) + len(test_bdna_indices))) * -1
    y_hat[y_hat_nonb_indices_in_nonb] = 1
    y_hat[y_hat_bdna_indices_in_nonb] = 0
    y_nonb = y[test_nonb_indices]
    y_hat_nonb = y_hat[test_nonb_indices]
    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_nonb, y_hat_nonb).ravel()
    return tn, fp, fn, tp


def evaluation_sim(test, null_dist_scores, eval_scores, alpha, tail):
    """This function uses the nonb part of test data for evaluation"""
    # indices of nonb abd bdna true and labels
    test_nonb_indices = test[test['label'] == 1].index
    test_bdna_indices = test[test['label'] == 0].index
    test_nonb_indices_true = test[test['true_label'] != 'bdna'].index
    # Nonb detected by model in nonb labelled part
    p_values, indices = compute_empirical(null_dist_scores, eval_scores, tail=tail)
    fdr_check, _ = fdrcorrection(p_values, alpha=alpha, is_sorted=True)
    if np.sum(fdr_check) > 0:
        rejected_counts = np.max(np.where(fdr_check))
    else:
        rejected_counts = 0
    y_hat_nonb_indices_in_pval = indices[0:rejected_counts]
    y_hat_nonb_indices_in_nonb = test_nonb_indices[y_hat_nonb_indices_in_pval].to_list()
    y_hat_bdna_indices_in_pval = indices[rejected_counts:]
    y_hat_bdna_indices_in_nonb = test_nonb_indices[y_hat_bdna_indices_in_pval].to_list()
    # Construct y and y_hat
    y = np.zeros((len(test_nonb_indices) + len(test_bdna_indices)), dtype=int)
    y[test_nonb_indices_true] = 1
    y_hat = np.ones((len(test_nonb_indices) + len(test_bdna_indices))) * -1
    y_hat[y_hat_nonb_indices_in_nonb] = 1
    y_hat[y_hat_bdna_indices_in_nonb] = 0
    y_nonb = y[test_nonb_indices]
    y_hat_nonb = y_hat[test_nonb_indices]
    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_nonb, y_hat_nonb).ravel()
    return p_values, tn, fp, fn, tp


def evaluation_sim_classifier():
    sss = 'g'


def evaluation_exp(test, test_bed, null_dist_scores, eval_scores, alpha, tail, write_bed=True):
    """ Positive samples: bdna samples, Negative Samples: nonb samples"""
    # indices of nonb and bdna labels
    test_nonb_indices = test[test['label'] == 1].index
    # test_bdna_indices = test[test['label'] == 0].index
    # Nonb detected by model in nonb labelled part
    print(tail, alpha)
    # print(tail, alpha, 'compute_empirical ...')
    p_values, indices = compute_empirical(null_dist_scores, eval_scores, tail=tail)
    # print(tail, alpha, 'fdr correction ...')
    fdr_check, _ = fdrcorrection(p_values, alpha=alpha, is_sorted=True)
    if np.sum(fdr_check) > 0:
        rejected_counts = np.max(np.where(fdr_check))
    else:
        rejected_counts = 0
    nonb_indices_in_pval = indices[0:rejected_counts]
    nonb_indices_in_nonb = test_nonb_indices[nonb_indices_in_pval].to_list()  # TN samples
    # bdna_indices_in_pval = indices[rejected_counts:]
    # bdna_indices_in_nonb = test_nonb_indices[bdna_indices_in_pval].to_list()  # FP samples
    if write_bed:
        test_bed_id = test_bed.set_index('id')
        sel_test_bed = test_bed_id.loc[nonb_indices_in_nonb]
    else:
        sel_test_bed = pd.DataFrame(columns=['1'])
    # Nonb detected by model in bdna labelled part
    # print(tail, alpha, 'compute_empirical ...')
    selected_null = np.random.choice(null_dist_scores, size=len(eval_scores), replace=False)
    p_values2, indices2 = compute_empirical(null_dist_scores, selected_null, tail=tail)
    # print(tail, alpha, 'fdr correction ...')
    fdr_check2, _ = fdrcorrection(p_values2, alpha=alpha, is_sorted=True)
    if np.sum(fdr_check2) > 0:
        rejected_counts2 = np.max(np.where(fdr_check2)) + 1
    else:
        rejected_counts2 = 0
    # nonb_indices_in_pval2 = indices2[0:rejected_counts2]
    # nonb_indices_in_bdna = test_bdna_indices[nonb_indices_in_pval2].to_list()  # FN samples
    # bdna_indices_in_pval2 = indices2[rejected_counts2:]
    # bdna_indices_in_bdna = test_bdna_indices[bdna_indices_in_pval2].to_list()  # TP samples
    tn = len(nonb_indices_in_nonb)
    fp = len(indices) - tn
    fn = rejected_counts2
    tp = len(indices2) - fn
    # print(tail, alpha, 'bias correction ...')
    acc_corrected, bacc_corrected, f_corrected = biased_corrected_criteria(tn, fp, fn, tp, test, g_p=0.05, beta=0)
    return sel_test_bed, tn, fp, fn, tp, acc_corrected, bacc_corrected, f_corrected


def evaluation_exp2(test, test_bed, null_dist_scores, eval_scores, alpha, tail, write_bed=False):
    """ Positive samples: bdna samples, Negative Samples: nonb samples"""
    # indices of nonb and bdna labels
    test_nonb_indices = test[test['label'] == 1].index
    # test_bdna_indices = test[test['label'] == 0].index
    # Nonb detected by model in nonb labelled part
    # print(tail, alpha)
    # print(tail, alpha, 'compute_empirical ...')
    p_values, indices = compute_empirical(null_dist_scores, eval_scores, tail=tail)
    # p_values, indices = compute_empirical_parallel(null_dist_scores, eval_scores, tail=tail)
    # print(tail, alpha, 'fdr correction ...')
    fdr_check, _ = fdrcorrection(p_values, alpha=alpha, is_sorted=True)
    if np.sum(fdr_check) > 0:
        rejected_counts = np.max(np.where(fdr_check))
    else:
        rejected_counts = 0
    nonb_indices_in_pval = indices[0:rejected_counts]
    nonb_indices_in_nonb = test_nonb_indices[nonb_indices_in_pval].to_list()  # TN samples
    # bdna_indices_in_pval = indices[rejected_counts:]
    # bdna_indices_in_nonb = test_nonb_indices[bdna_indices_in_pval].to_list()  # FP samples
    if len(nonb_indices_in_nonb) > 0:
        test_bed_id = test_bed.set_index('id')
        sel_test_bed = test_bed_id.loc[nonb_indices_in_nonb]
    else:
        sel_test_bed = pd.DataFrame(columns=['1'])
    # Nonb detected by model in bdna labelled part
    # print(tail, alpha, 'compute_empirical ...')
    return sel_test_bed, rejected_counts


def biased_corrected_criteria(tn, fp, fn, tp, test, g_p, beta=0):
    """ Positive samples: bdna samples, Negative Samples: nonb samples
    tn: nonb predicted as nonb
    fp: nonb predicted as bdna
    fn: bdna predicted as nonb
    tp: bdna predicted as bdna
    g_p :proportion of the nonB assumed to actually form nonB structures
    c: proportion of labeled to total
    """
    x_n = len(test[test['label'] == 1])  # x_nonb
    x_p = len(test[test['label'] == 0])  # x_bdna
    c = x_p / (x_n + x_p)  # proportion of labeled to total
    alpha_hat = (g_p * x_n) / (x_n + x_p)
    gamma = tp / (tp + fn)
    eta = fp / (tn + fp)
    pi = (tp + fn) / (tp + fn + tn + fp)
    theta = (tp + fp) / (tp + fn + tn + fp)
    gamma_corrected = (((1 - alpha_hat) * gamma) - ((1 - beta) * eta)) / (beta - alpha_hat)
    eta_corrected = (beta * eta - alpha_hat * gamma) / (beta - alpha_hat)
    pi_corrected = (c * beta) + (1 - c) * alpha_hat
    acc_corrected = (pi_corrected * gamma_corrected) + (1 - pi_corrected) * (1 - eta_corrected)
    bacc_corrected = (1 + gamma_corrected - eta_corrected) / 2
    f_corrected = (2 * pi_corrected * gamma_corrected) / (pi_corrected + theta)
    return acc_corrected, bacc_corrected, f_corrected



def plot_results_roc_pr(results_pd, plot_path, plot_type):
    """Plot roc or pr for sim data"""
    # path = 'results/sim_IF/final_results_sim_IF.csv'
    # results_pd = pd.read_csv(path, index_col=0)
    filtereted_results_pd = results_pd[results_pd['tail'] == 'upper'].reset_index(drop=True)
    train_sets = list(filtereted_results_pd['train_set'].unique())
    windows = list(filtereted_results_pd['window_size'].unique())
    # windows = [25, 50, 75, 100]
    non_bs = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
              'Short_Tandem_Repeat', 'Z_DNA_Motif']
    for win_id, win in enumerate(windows):
        for ts in train_sets:
            thisdf = filtereted_results_pd[(filtereted_results_pd['window_size'] == win) &
                                           (filtereted_results_pd['train_set'] == ts)].reset_index(drop=True)
            labels = list(thisdf['label'].unique())
            for label in labels:
                lb_df = thisdf[thisdf['label'] == label].reset_index(drop=True)
                nonb_ratios = list(lb_df['nonb_ratio'].unique())
                
                # if len(lb_df) < 19:
                #     print(win, ts, label)
                plot_name = ''
                x_labeb = ''
                y_labeb = ''
                fig, ax = plt.subplots(1, figsize=(7, 7))
                for nbr in nonb_ratios:
                    lb_nbratio = lb_df[lb_df['nonb_ratio'] == nbr].reset_index(drop=True)
                    if plot_type == 'ROC':
                        x = list(lb_nbratio['fpr'])
                        y = list(lb_nbratio['tpr'])
                        plot_name = '_'.join([ts, label, str(win)]) + '_ROC_plot.png'
                        x_labeb = 'FPR'
                        y_labeb = 'TPR'
                    elif plot_type == 'PR':
                        x = list(lb_nbratio['recall'])
                        y = list(lb_nbratio['precision'])
                        plot_name = '_'.join([ts, label, str(win)]) + '_PR_plot.png'
                        x_labeb = 'Recall'
                        y_labeb = 'Precision'
                    else:
                        return
                    # new_fpr, new_tpr = make_smooth(fpr, tpr, degree=3)
                    ax.plot(x, y, linewidth=3, label='Nonb ratio: ' + str(nbr))
                ax.set_xlabel(x_labeb, fontsize=13)  # Add an x-label to the axes.
                ax.set_ylabel(y_labeb, fontsize=13)  # Add a y-label to the axes.
                title = 'Window size: {}, \nTrained on: {}, \nTest on: {}'.format(win, ts, label)
                ax.set_title(title, fontsize=13)
                ax.legend(bbox_to_anchor=(1.1, 1.1))
                plt.tight_layout()
                # plt.show()
                plt.savefig(os.path.join(plot_path, plot_name))
    return


def collect_results(results_path, results_name):
    methods_results_path = os.path.join(results_path, results_name)
    non_b_types = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                   'Short_Tandem_Repeat', 'Z_DNA_Motif']
    results_files = [os.path.join(methods_results_path, elem, 'final_results.csv') for elem in
                     os.listdir(methods_results_path)
                     if os.path.isdir(os.path.join(methods_results_path, elem)) and any(
            nonb in elem for nonb in non_b_types)]
    results_pd_list = [pd.read_csv(file, index_col=0) for file in results_files if os.path.exists(file)]
    results_pd = pd.concat(results_pd_list, axis=0, ignore_index=True)
    results_pd.to_csv(os.path.join(methods_results_path, 'final_results_' + results_name + '.csv'))
    return results_pd


def make_smooth(x, y, degree=3):
    xnew = np.linspace(min(x), max(x), 200)
    xnew2 = sorted(list(xnew) + list(x))
    spl = make_interp_spline(x, y, k=degree)
    y_smooth = spl(xnew2)
    return xnew, y_smooth


def plot_sim(results_path, results_name):
    results_sim_pd = collect_results(results_path, results_name)
    methods_results_path = os.path.join(results_path, results_name)
    plot_path = os.path.join(methods_results_path, 'plots')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    plot_results_roc_pr(results_sim_pd, plot_path, 'ROC')
    plot_results_roc_pr(results_sim_pd, plot_path, 'PR')


def make_bed_files_for_splits(bdna_folder, nonb_folder, win_size, frac):
    cols = ['id', 'chr', 'start', 'end', 'name', 'score', 'strand']
    tr, val, te, _, _, _, _ = reprocess_data2(bdna_folder, win_size, percent=frac, std_type='median/iqr', seed=42)
    tr_df_bdna = pd.DataFrame(columns=['chr', 'start', 'end', 'name', 'score', 'strand'], index=range(tr.shape[0]))
    val_df_bdna = pd.DataFrame(columns=['chr', 'start', 'end', 'name', 'score', 'strand'], index=range(val.shape[0]))
    te_df_bdna = pd.DataFrame(columns=['chr', 'start', 'end', 'name', 'score', 'strand'], index=range(te.shape[0]))
    ####################################################
    nonb_tr_dfs, nonb_val_dfs, nonb_te_dfs = prepare_nonb_dataset_center(path=nonb_folder)
    nonb_types = list(nonb_tr_dfs.keys())
    nonb_tr_df = pd.DataFrame(columns=['chr', 'start', 'end', 'label', 'motif_proportion', 'strand'])
    nonb_val_df = pd.DataFrame(columns=['chr', 'start', 'end', 'label', 'motif_proportion', 'strand'])
    nonb_te_df = pd.DataFrame(columns=['chr', 'start', 'end', 'label', 'motif_proportion', 'strand'])
    shift = (100 - win_size) // 2
    # end_idx = start_idx + win_size
    for non_b in nonb_types:
        this_train_df_nonb = nonb_tr_dfs[non_b][['chr', 'start', 'end', 'label', 'motif_proportion', 'strand']]
        this_val_df_nonb = nonb_val_dfs[non_b][['chr', 'start', 'end', 'label', 'motif_proportion', 'strand']]
        this_test_df_nonb = nonb_te_dfs[non_b][['chr', 'start', 'end', 'label', 'motif_proportion', 'strand']]
        nonb_tr_df = pd.concat([nonb_tr_df, this_train_df_nonb], ignore_index=True)
        nonb_val_df = pd.concat([nonb_val_df, this_val_df_nonb], ignore_index=True)
        nonb_te_df = pd.concat([nonb_te_df, this_test_df_nonb], ignore_index=True)
    nonb_tr_df = nonb_tr_df.rename(columns={'label': 'name', 'motif_proportion': 'score'})
    nonb_tr_df['start'] = nonb_tr_df['start'].apply(lambda x: x + shift + 100)
    nonb_tr_df['end'] = nonb_tr_df['start'].apply(lambda x: x + win_size)
    nonb_tr_df['score'] = nonb_tr_df['score'].apply(lambda x: int(1000 * min(1, x * (100 / win_size))))
    nonb_val_df = nonb_val_df.rename(columns={'label': 'name', 'motif_proportion': 'score'})
    nonb_val_df['start'] = nonb_val_df['start'].apply(lambda x: x + shift + 100)
    nonb_val_df['end'] = nonb_val_df['start'].apply(lambda x: x + win_size)
    nonb_val_df['score'] = nonb_val_df['score'].apply(lambda x: int(1000 * min(1, x * (100 / win_size))))
    nonb_te_df = nonb_te_df.rename(columns={'label': 'name', 'motif_proportion': 'score'})
    nonb_te_df['start'] = nonb_te_df['start'].apply(lambda x: x + shift + 100)
    nonb_te_df['end'] = nonb_te_df['start'].apply(lambda x: x + win_size)
    nonb_te_df['score'] = nonb_te_df['score'].apply(lambda x: int(1000 * min(1, x * (100 / win_size))))
    tr_bed_df = pd.concat([tr_df_bdna, nonb_tr_df], ignore_index=True)
    tr_bed_df = tr_bed_df.sample(frac=1, random_state=42).reset_index(drop=True)
    tr_bed_df['id'] = tr_bed_df.index
    tr_bed_df = tr_bed_df[cols]
    tr_bed_df = tr_bed_df.dropna()
    tr_bed_df = tr_bed_df.sort_values(by=['id']).reset_index(drop=True)
    val_bed_df = pd.concat([val_df_bdna, nonb_val_df], ignore_index=True)
    val_bed_df = val_bed_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_bed_df['id'] = val_bed_df.index
    val_bed_df = val_bed_df[cols]
    val_bed_df = val_bed_df.dropna()
    val_bed_df = val_bed_df.sort_values(by=['id']).reset_index(drop=True)
    test_bed_df = pd.concat([te_df_bdna, nonb_te_df], ignore_index=True)
    test_bed_df = test_bed_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_bed_df['id'] = test_bed_df.index
    test_bed_df = test_bed_df[cols]
    test_bed_df = test_bed_df.dropna()
    test_bed_df = test_bed_df.sort_values(by=['id']).reset_index(drop=True)
    return tr_bed_df, val_bed_df, test_bed_df


def load_data3(folder, nonb):
    global_train = pd.read_csv(os.path.join(folder, nonb + '_train.csv'), index_col=0)
    train_bed = pd.read_csv(os.path.join(folder, nonb + '_train.bed'), sep='\t')
    global_val = pd.read_csv(os.path.join(folder, nonb + '_validation.csv'), index_col=0)
    val_bed = pd.read_csv(os.path.join(folder, nonb + '_validation.bed'), sep='\t')
    global_test = pd.read_csv(os.path.join(folder, nonb + '_test.csv'), index_col=0)
    test_bed = pd.read_csv(os.path.join(folder, nonb + '_test.bed'), sep='\t')
    return global_train, global_val, global_test, train_bed, val_bed, test_bed


def make_new_train_validation(train, val, ratio=10):
    train_val = pd.concat([train, val], ignore_index=True)
    nonb_set = train_val[train_val['label'] == 1].reset_index(drop=True)
    bdna_set = train_val[train_val['label'] == 0].reset_index(drop=True)
    n_nonb = len(nonb_set)
    n_bdna = int(ratio * n_nonb)
    needed = min(n_bdna, len(bdna_set))
    bdna_set = bdna_set.sample(n=needed, random_state=42).reset_index(drop=True)
    train_val = pd.concat([bdna_set, nonb_set], ignore_index=True)
    train_val = train_val.sample(frac=1, random_state=42).reset_index(drop=True)
    n_tr = len(train_val) * 9 // 10
    new_tr = train_val.loc[0:n_tr - 1].reset_index(drop=True)
    new_val = train_val.loc[n_tr:].reset_index(drop=True)
    return new_tr, new_val



def validation(new_val, val_model, alpha, tail):
    nonb_val_orig = new_val[new_val['label'] == 1].reset_index(drop=True)
    bdna_val_orig = new_val[new_val['label'] == 0].reset_index(drop=True)
    bdna_poison_count = len(nonb_val_orig)
    bdna_poison = bdna_val_orig[:bdna_poison_count].reset_index(drop=True)
    bdna_val_new = bdna_val_orig[bdna_poison_count:].reset_index(drop=True)
    if 'true_label' in list(new_val.columns.values):
        nonb_val_x = nonb_val_orig.drop(['label', 'true_label'], axis=1).to_numpy()
        poison_x = bdna_poison.drop(['label', 'true_label'], axis=1).to_numpy()
        bdna_val_x = bdna_val_new.drop(['label', 'true_label'], axis=1).to_numpy()
    else:
        nonb_val_x = nonb_val_orig.drop(['label'], axis=1).to_numpy()
        poison_x = bdna_poison.drop(['label'], axis=1).to_numpy()
        bdna_val_x = bdna_val_new.drop(['label'], axis=1).to_numpy()
    
    null_dist = val_model.decision_function(bdna_val_x)
    eval_dist = val_model.decision_function(nonb_val_x)
    posion_dist = val_model.decision_function(poison_x)
    eval_mix = np.hstack([eval_dist, posion_dist])
    
    print(eval_mix.shape[0])
    # p_values, indices = compute_empirical_parallel(null_dist, eval_mix, tail=tail)
    p_values, indices = compute_empirical(null_dist, eval_mix, tail=tail)
    fdr_check, _ = fdrcorrection(p_values, alpha=alpha, is_sorted=True)
    if np.sum(fdr_check) > 0:
        rejected_counts = np.max(np.where(fdr_check))
    else:
        rejected_counts = 0
    nonb_indices_in_pval = indices[0:rejected_counts]
    nonb_novelties = np.sum(nonb_indices_in_pval < len(eval_dist))
    bdna_novelties = np.sum(nonb_indices_in_pval >= len(eval_dist))
    tp = nonb_novelties
    fn = len(eval_dist) - tp
    fp = bdna_novelties
    tn = len(posion_dist) - fp
    return tn, fp, fn, tp


def evaluate_classifiers(svm_model, test_x, test_y_true, dataset, method, nonb, nonb_ratio, save_path, winsize,
                         duration):
    y_score = svm_model.predict_proba(test_x)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(test_y_true, y_score, pos_label=1)
    roc_curve_df = pd.DataFrame({'dataset': dataset, 'method': method, 'label': nonb, 'nonb_ratio': nonb_ratio,
                                 'fpr': fpr, 'tpr': tpr})
    roc_curve_df.to_csv(os.path.join(save_path, 'roc_df.csv'))
    roc_auc = metrics.auc(fpr, tpr)
    
    precision, recall, thresholds = metrics.precision_recall_curve(test_y_true, y_score)
    pr_curve_df = pd.DataFrame({'dataset': dataset, 'method': method, 'label': nonb, 'nonb_ratio': nonb_ratio,
                                'precision': precision, 'recall': recall})
    pr_curve_df.to_csv(os.path.join(save_path, 'pr_df.csv'))
    pr_auc = metrics.auc(recall, precision)
    
    # alpha_list = np.arange(min(y_score), max(y_score), 0.01)
    alpha_list = [0.5]
    final_results_df = pd.DataFrame(columns=['dataset', 'method', 'label', 'window_size', 'nonb_ratio', 'roc_auc',
                                             'pr_auc', 'prob', 'tp', 'tn', 'fp', 'fn', 'precision', 'recall', 'fpr',
                                             'tpr', 'fscore', 'duration'], index=[0])
    counter = 0
    for prob in alpha_list:
        print(prob)
        y_score[y_score > prob] = 1
        y_score[y_score <= prob] = 0
        tn, fp, fn, tp = confusion_matrix(test_y_true, y_score).ravel()
        precision, recall, tpr, fpr, fscore = compute_accuracy_metrics(tn, fp, fn, tp)
        final_results_df.loc[counter, :] = dataset, method, nonb, winsize, nonb_ratio, roc_auc, pr_auc, prob, tp, tn, \
                                           fp, fn, precision, recall, fpr, tpr, fscore, duration
        counter += 1
    final_results_df.to_csv(os.path.join(save_path, 'final_results.csv'))


def if_hyperparam_tuning(new_tr, new_val, save_folder):
    # params = ['auto', 0.01, 0.05, 0.1, 0.2, 0.4]
    params = ['auto', 500, 1000, 2000, 3000]
    alpha_range = list(np.arange(0.25, 0.95, 0.05))
    if 'true_label' in list(new_val.columns.values):
        new_train_x = new_tr.drop(['label', 'true_label'], axis=1).to_numpy()
    else:
        new_train_x = new_tr.drop(['label'], axis=1).to_numpy()
    val_results = pd.DataFrame(columns=['parameter', 'alpha', 'criteria', 'tn', 'fp', 'fn', 'tp', 'precision', 'recall',
                                        'fscore'], index=range(len(params) * len(alpha_range)))
    count = 0
    for alpha in alpha_range:
        for par in params:
            print('Hyper param tuninig --->', alpha, par, count)
            val_model = IsolationForest(max_samples=par).fit(new_train_x)
            tn, fp, fn, tp = validation(new_val, val_model, alpha, 'lower')
            criteria = tp / (fp + 1)
            precision, recall, _, _, fscore = compute_accuracy_metrics(tn, fp, fn, tp)
            val_results.loc[count, :] = par, alpha, criteria, tn, fp, fn, tp, precision, recall, fscore
            count += 1
        if np.sum(val_results['tp']) > 0:
            break
    # val_results = val_results.dropna().reset_index(drop=True)
    print('df', val_results)
    print('----------------------------------------------------')
    val_results.to_csv(os.path.join(save_folder, 'model_selection.csv'))
    # find the best params in validation data
    max_criteria = np.max(val_results['criteria'])
    print('===>')
    best_row = val_results[val_results['criteria'] == max_criteria]
    # select the best param:
    best_param = best_row['parameter'].values[0]
    print('best_param ===>', best_param)
    return best_param


def svm_hyperparam_tuning(new_tr, new_val, save_folder):
    params = ['linear', 'poly', 'rbf', 'sigmoid']
    alpha_range = list(np.arange(0.25, 0.95, 0.05))
    if 'true_label' in list(new_val.columns.values):
        new_train_x = new_tr.drop(['label', 'true_label'], axis=1).to_numpy()
    else:
        new_train_x = new_tr.drop(['label'], axis=1).to_numpy()
    val_results = pd.DataFrame(columns=['parameter', 'alpha', 'criteria', 'tn', 'fp', 'fn', 'tp', 'precision', 'recall',
                                        'fscore'], index=range(len(params) * len(alpha_range)))
    count = 0
    for alpha in alpha_range:
        for par in params:
            # print(alpha, par, count)
            val_model = OneClassSVM(kernel=par, cache_size=10000, max_iter=10000).fit(new_train_x)
            tn, fp, fn, tp = validation(new_val, val_model, alpha, 'lower')
            criteria = tp / (fp + 1)
            precision, recall, _, _, fscore = compute_accuracy_metrics(tn, fp, fn, tp)
            val_results.loc[count, :] = par, alpha, criteria, tn, fp, fn, tp, precision, recall, fscore
            count += 1
        if np.sum(val_results['tp']) > 0:
            break
    val_results = val_results.dropna().reset_index(drop=True)
    val_results.to_csv(os.path.join(save_folder, 'model_selection.csv'))
    # find the best params in validation data
    if len(val_results) > 0:
        max_criteria = np.max(val_results['criteria'])
        best_row = val_results[val_results['criteria'] == max_criteria]
        # select the best param:
        best_param = best_row['parameter'].values[0]
    else:
        best_param = 'linear'
    return best_param


def lof_hyperparam_tuning(new_tr, new_val, save_folder):
    # params = ['auto', 0.01, 0.05, 0.1, 0.2, 0.4]
    params = [10, 20, 40, 100, 200]
    alpha_range = list(np.arange(0.25, 0.95, 0.05))
    if 'true_label' in list(new_val.columns.values):
        new_train_x = new_tr.drop(['label', 'true_label'], axis=1).to_numpy()
    else:
        new_train_x = new_tr.drop(['label'], axis=1).to_numpy()
    val_results = pd.DataFrame(columns=['parameter', 'alpha', 'criteria', 'tn', 'fp', 'fn', 'tp', 'precision', 'recall',
                                        'fscore'], index=range(len(params) * len(alpha_range)))
    count = 0
    for alpha in alpha_range:
        for par in params:
            # print(alpha, par, count)
            val_model = LocalOutlierFactor(n_jobs=-1, novelty=True, n_neighbors=par).fit(new_train_x)
            tn, fp, fn, tp = validation(new_val, val_model, alpha, 'lower')
            criteria = tp / (fp + 1)
            precision, recall, _, _, fscore = compute_accuracy_metrics(tn, fp, fn, tp)
            val_results.loc[count, :] = par, alpha, criteria, tn, fp, fn, tp, precision, recall, fscore
            count += 1
        if np.sum(val_results['tp']) > 0:
            break
    val_results = val_results.dropna().reset_index(drop=True)
    val_results.to_csv(os.path.join(save_folder, 'model_selection.csv'))
    if len(val_results) > 0:
        # find the best params in validation data
        max_criteria = np.max(val_results['criteria'])
        best_row = val_results[val_results['criteria'] == max_criteria]
        # select the best param:
        best_param = best_row['parameter'].values[0]
    else:
        best_param = 100
    return best_param


def svc_hyperparam_tuning(new_tr, new_val, save_folder):
    params = ['linear', 'poly', 'rbf', 'sigmoid']
    alpha_range = list(np.arange(0.25, 0.95, 0.05))
    new_train_y = new_tr[['label']].to_numpy().flatten()
    new_val_y = new_val[['label']].to_numpy().flatten()
    
    if 'true_label' in list(new_val.columns.values):
        new_train_x = new_tr.drop(['label', 'true_label'], axis=1).to_numpy()
        new_val_x = new_val.drop(['label', 'true_label'], axis=1).to_numpy()
    else:
        new_train_x = new_tr.drop(['label'], axis=1).to_numpy()
        new_val_x = new_val.drop(['label'], axis=1).to_numpy()
    
    val_results = pd.DataFrame(columns=['parameter', 'alpha', 'criteria', 'tn', 'fp', 'fn', 'tp', 'precision', 'recall',
                                        'fscore'], index=range(len(params) * len(alpha_range)))
    count = 0
    for alpha in alpha_range:
        for par in params:
            print(alpha, par, count)
            clf = SVC(kernel=par, C=0.01, probability=True)
            clf.fit(new_train_x, new_train_y)
            y_pred = clf.predict_proba(new_val_x)[:, 1]
            y_pred[y_pred > alpha] = 1
            y_pred[y_pred <= alpha] = 0
            tn, fp, fn, tp = confusion_matrix(new_val_y, y_pred).ravel()
            criteria = tp / (fp + 1)
            precision, recall, _, _, fscore = compute_accuracy_metrics(tn, fp, fn, tp)
            val_results.loc[count, :] = par, alpha, criteria, tn, fp, fn, tp, precision, recall, fscore
            count += 1
        if np.sum(val_results['tp']) > 0:
            break
    val_results = val_results.dropna().reset_index(drop=True)
    val_results.to_csv(os.path.join(save_folder, 'model_selection.csv'))
    # find the best params in validation data
    max_criteria = np.max(val_results['criteria'])
    best_row = val_results[val_results['criteria'] == max_criteria]
    # select the best param:
    best_param = best_row['parameter'].values[0]
    print('---------------------------------')
    print(val_results)
    print('best param --->', best_param)
    return best_param


def lr_hyperparam_tuning(new_tr, new_val, save_folder):
    params = [0.001, 0.01, 0.5, 1]
    alpha_range = list(np.arange(0.25, 0.95, 0.05))
    new_train_y = new_tr[['label']].to_numpy().flatten()
    new_val_y = new_val[['label']].to_numpy().flatten()
    
    if 'true_label' in list(new_val.columns.values):
        new_train_x = new_tr.drop(['label', 'true_label'], axis=1).to_numpy()
        new_val_x = new_val.drop(['label', 'true_label'], axis=1).to_numpy()
    else:
        new_train_x = new_tr.drop(['label'], axis=1).to_numpy()
        new_val_x = new_val.drop(['label'], axis=1).to_numpy()
    
    val_results = pd.DataFrame(columns=['parameter', 'alpha', 'criteria', 'tn', 'fp', 'fn', 'tp', 'precision', 'recall',
                                        'fscore'], index=range(len(params) * len(alpha_range)))
    count = 0
    for alpha in alpha_range:
        for par in params:
            # print(alpha, par, count)
            clf = LogisticRegression(C=par)
            clf.fit(new_train_x, new_train_y)
            y_pred = clf.predict_proba(new_val_x)[:, 1]
            y_pred[y_pred > alpha] = 1
            y_pred[y_pred <= alpha] = 0
            tn, fp, fn, tp = confusion_matrix(new_val_y, y_pred).ravel()
            criteria = tp / (fp + 1)
            precision, recall, _, _, fscore = compute_accuracy_metrics(tn, fp, fn, tp)
            val_results.loc[count, :] = par, alpha, criteria, tn, fp, fn, tp, precision, recall, fscore
            count += 1
        if np.sum(val_results['tp']) > 0:
            break
    val_results = val_results.dropna().reset_index(drop=True)
    val_results.to_csv(os.path.join(save_folder, 'model_selection.csv'))
    # find the best params in validation data
    max_criteria = np.max(val_results['criteria'])
    best_row = val_results[val_results['criteria'] == max_criteria]
    # select the best param:
    best_param = best_row['parameter'].values[0]
    return best_param


def nn_hyperparam_tuning(new_tr, new_val, save_folder):
    params = [3, 5, 10, 15, 50, 100]
    alpha_range = list(np.arange(0.25, 0.95, 0.05))
    new_train_y = new_tr[['label']].to_numpy().flatten()
    new_val_y = new_val[['label']].to_numpy().flatten()
    
    if 'true_label' in list(new_val.columns.values):
        new_train_x = new_tr.drop(['label', 'true_label'], axis=1).to_numpy()
        new_val_x = new_val.drop(['label', 'true_label'], axis=1).to_numpy()
    else:
        new_train_x = new_tr.drop(['label'], axis=1).to_numpy()
        new_val_x = new_val.drop(['label'], axis=1).to_numpy()
    
    val_results = pd.DataFrame(columns=['parameter', 'alpha', 'criteria', 'tn', 'fp', 'fn', 'tp', 'precision', 'recall',
                                        'fscore'], index=range(len(params) * len(alpha_range)))
    count = 0
    for alpha in alpha_range:
        for par in params:
            # print(alpha, par, count)
            clf = KNeighborsClassifier(n_neighbors=par)
            clf.fit(new_train_x, new_train_y)
            y_pred = clf.predict_proba(new_val_x)[:, 1]
            y_pred[y_pred > alpha] = 1
            y_pred[y_pred <= alpha] = 0
            tn, fp, fn, tp = confusion_matrix(new_val_y, y_pred).ravel()
            criteria = tp / (fp + 1)
            precision, recall, _, _, fscore = compute_accuracy_metrics(tn, fp, fn, tp)
            val_results.loc[count, :] = par, alpha, criteria, tn, fp, fn, tp, precision, recall, fscore
            count += 1
        if np.sum(val_results['tp']) > 0:
            break
    val_results = val_results.dropna().reset_index(drop=True)
    val_results.to_csv(os.path.join(save_folder, 'model_selection.csv'))
    # find the best params in validation data
    max_criteria = np.max(val_results['criteria'])
    best_row = val_results[val_results['criteria'] == max_criteria]
    # select the best param:
    best_param = best_row['parameter'].values[0]
    return best_param


def gp_hyperparam_tuning(new_tr, new_val, save_folder):
    # params = [RBF(length_scale=2.0), WhiteKernel(noise_level=0.5), DotProduct()]
    # length_scales = [1.0, 0.5, 2.0, 3.0, 10.0]
    # noise_levels = [0.1, 0.3, 0.5, 1, 2]
    sigma_0_levels = [0.5, 1.0, 0.2]
    # params = [RBF(length_scale=p) for p in length_scales]
    # params = [WhiteKernel(noise_level=p) for p in noise_levels]
    params = [DotProduct(sigma_0=p) for p in sigma_0_levels]
    
    # params = [100, 200, 500, 1000]
    alpha_range = list(np.arange(0.25, 0.95, 0.05))
    new_train_y = new_tr[['label']].to_numpy().flatten()
    new_val_y = new_val[['label']].to_numpy().flatten()
    
    if 'true_label' in list(new_val.columns.values):
        new_train_x = new_tr.drop(['label', 'true_label'], axis=1).to_numpy()
        new_val_x = new_val.drop(['label', 'true_label'], axis=1).to_numpy()
    else:
        new_train_x = new_tr.drop(['label'], axis=1).to_numpy()
        new_val_x = new_val.drop(['label'], axis=1).to_numpy()
    
    val_results = pd.DataFrame(columns=['parameter', 'alpha', 'criteria', 'tn', 'fp', 'fn', 'tp', 'precision', 'recall',
                                        'fscore'], index=range(len(params) * len(alpha_range)))
    count = 0
    for alpha in alpha_range:
        for par in params:
            print(alpha, par, count)
            clf = GaussianProcessClassifier(kernel=par)
            # clf = GaussianProcessClassifier(max_iter_predict=par)
            clf.fit(new_train_x, new_train_y)
            y_pred = clf.predict_proba(new_val_x)[:, 1]
            y_pred[y_pred > alpha] = 1
            y_pred[y_pred <= alpha] = 0
            tn, fp, fn, tp = confusion_matrix(new_val_y, y_pred).ravel()
            criteria = tp / (fp + 1)
            precision, recall, _, _, fscore = compute_accuracy_metrics(tn, fp, fn, tp)
            val_results.loc[count, :] = par, alpha, criteria, tn, fp, fn, tp, precision, recall, fscore
            count += 1
        if np.sum(val_results['tp']) > 0:
            break
    val_results = val_results.dropna().reset_index(drop=True)
    val_results.to_csv(os.path.join(save_folder, 'model_selection.csv'))
    # find the best params in validation data
    max_criteria = np.max(val_results['criteria'])
    best_row = val_results[val_results['criteria'] == max_criteria]
    # select the best param:
    best_param = best_row['parameter'].values[0]
    print('---------------------------------')
    print(val_results)
    print('best param --->', best_param)
    return best_param


def rf_hyperparam_tuning(new_tr, new_val, save_folder):
    params = ['gini', 'entropy', 'log_loss']
    alpha_range = list(np.arange(0.25, 0.95, 0.05))
    new_train_y = new_tr[['label']].to_numpy().flatten()
    new_val_y = new_val[['label']].to_numpy().flatten()
    
    if 'true_label' in list(new_val.columns.values):
        new_train_x = new_tr.drop(['label', 'true_label'], axis=1).to_numpy()
        new_val_x = new_val.drop(['label', 'true_label'], axis=1).to_numpy()
    else:
        new_train_x = new_tr.drop(['label'], axis=1).to_numpy()
        new_val_x = new_val.drop(['label'], axis=1).to_numpy()
    
    val_results = pd.DataFrame(columns=['parameter', 'alpha', 'criteria', 'tn', 'fp', 'fn', 'tp', 'precision', 'recall',
                                        'fscore'], index=range(len(params) * len(alpha_range)))
    count = 0
    for alpha in alpha_range:
        for par in params:
            # print(alpha, par, count)
            clf = RandomForestClassifier(criterion=par)
            clf.fit(new_train_x, new_train_y)
            y_pred = clf.predict_proba(new_val_x)[:, 1]
            y_pred[y_pred > alpha] = 1
            y_pred[y_pred <= alpha] = 0
            tn, fp, fn, tp = confusion_matrix(new_val_y, y_pred).ravel()
            criteria = tp / (fp + 1)
            precision, recall, _, _, fscore = compute_accuracy_metrics(tn, fp, fn, tp)
            val_results.loc[count, :] = par, alpha, criteria, tn, fp, fn, tp, precision, recall, fscore
            count += 1
        if np.sum(val_results['tp']) > 0:
            break
    val_results = val_results.dropna().reset_index(drop=True)
    val_results.to_csv(os.path.join(save_folder, 'model_selection.csv'))
    # find the best params in validation data
    max_criteria = np.max(val_results['criteria'])
    best_row = val_results[val_results['criteria'] == max_criteria]
    # select the best param:
    best_param = best_row['parameter'].values[0]
    return best_param


def plot_results_sim():
    plt.close()
    plt.clf()
    plt.cla
    path = 'results'
    sim_files = [os.path.join(path, name) for name in os.listdir(path) if
                 'final_results_sim' in name and 'old' not in name and '.csv' in name]
    results_list = [pd.read_csv(f, index_col=0) for f in sim_files]
    results_df = pd.concat(results_list, ignore_index=True)
    results_df = results_df[results_df['tail'] == 'lower'].reset_index(drop=True)
    nonbs = list(results_df['label'].unique())
    nonb = 'G_Quadruplex_Motif'
    this_lable = results_df[results_df['label'] == nonb].reset_index(drop=True)
    
    plt.close()
    plt.clf()
    plt.cla
    sns.boxplot(data=results_df, y='duration', x='nonb_ratio', hue='method')
    plt.show()
    
    exp_folders = [os.path.join(path, name) for name in os.listdir(path) if
                   'old' not in name and 'exp' in name and os.path.isdir(os.path.join(path, name))]
    
    results_files = [os.path.join(f, elem, 'final_results.csv') for f in exp_folders for elem in os.listdir(f)
                     if os.path.isdir(os.path.join(f, elem)) and any(nonb in elem for nonb in nonbs)]
    results_pd_list = [pd.read_csv(file, index_col=0) for file in results_files if os.path.exists(file)]
    results_pd = pd.concat(results_pd_list, axis=0, ignore_index=True)


def plot_classifiers_metrics():
    path = 'results'
    classifiers_folders = [os.path.join(path, name) for name in os.listdir(path) if 'sim' in name and
                           'old' not in name and 'sim' in name and os.path.isdir(os.path.join(path, name))]
    nonbs = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
             'Short_Tandem_Repeat', 'Z_DNA_Motif']
    classifiers = ['KNN', 'SVC', 'LR']
    results_files = [os.path.join(f, elem, 'final_results.csv') for f in classifiers_folders for elem in os.listdir(f)
                     if os.path.isdir(os.path.join(f, elem)) and any(nonb in elem for nonb in classifiers)]
    results_pd_list = [pd.read_csv(file, index_col=0) for file in results_files if os.path.exists(file)]
    results_pd = pd.concat(results_pd_list, axis=0, ignore_index=True)
    
    plt.cla()
    plt.close()
    plt.clf()
    sns.boxplot(data=results_pd, y='roc_auc', x='nonb_ratio', hue='method')
    plt.show()


def plot_classifiers_roc_curves():
    path = 'results'
    plot_path = os.path.join(path, 'plots')
    # classifiers_folders = [os.path.join(path, name) for name in os.listdir(path) if 'sim' in name and
    #                        'old' not in name and 'sim' in name and os.path.isdir(os.path.join(path, name))]
    nonbs = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
             'Short_Tandem_Repeat', 'Z_DNA_Motif']
    classifiers = ['KNN', 'SVC', 'LR', 'GP', 'RF']
    
    results_files = [os.path.join(path, elem) for elem in os.listdir(path) if os.path.isdir(os.path.join(path, elem))
                     if 'sim' in elem and any(nonb in elem for nonb in classifiers)]
    
    sub_folders = [os.path.join(fold, elem) for fold in results_files for elem in os.listdir(fold)
                   if os.path.isdir(os.path.join(fold, elem))]
    
    pr_dfs = [pd.read_csv(os.path.join(f, 'pr_df.csv'), index_col=0) for f in sub_folders if
              os.path.exists(os.path.join(f, 'pr_df.csv'))]
    
    roc_dfs = [pd.read_csv(os.path.join(f, 'roc_df.csv'), index_col=0) for f in sub_folders if
               os.path.exists(os.path.join(f, 'roc_df.csv'))]
    
    pr_df = pd.concat(pr_dfs, axis=0, ignore_index=True)
    roc_df = pd.concat(roc_dfs, axis=0, ignore_index=True)
    
    # nonb_ratio = pr_df['nonb_ratio'].unique()
    nonb_ratios = [0.025, 0.1, 0.5]
    
    pr_df = pr_df[pr_df['nonb_ratio'].isin(nonb_ratios)]
    roc_df = roc_df[roc_df['nonb_ratio'].isin(nonb_ratios)]
    
    methods = list(roc_df['method'].unique())
    # methods_colors = {'Isolation Forest': 'tab:blue', 'One Class SVM': 'tab:orange', 'GoFAE-COD': 'tab:green',
    #                   'Outlier AE (64)': 'tab:red', 'Outlier AE (128)': 'tab:purple'}
    methods_colors = {'LR': 'tab:blue', 'KNN': 'tab:orange', 'SVC': 'tab:green', 'GP': 'tab:red'}
    nonb_ratio_styles = {0.025: '-', 0.1: '--', 0.5: '-.'}
    
    for nonb in nonbs:
        prdf = pr_df[pr_df['label'] == nonb].reset_index(drop=True)
        rocdf = roc_df[roc_df['label'] == nonb].reset_index(drop=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        
        for method in methods:
            for nbr in nonb_ratios:
                this_df = rocdf[(rocdf['method'] == method) & (rocdf['nonb_ratio'] == nbr)].reset_index(drop=True)
                # tuples = list(zip(this_df.fpr, this_df.tpr))
                fpr = this_df['fpr']
                tpr = this_df['tpr']
                
                # fdrs = [i[0] for i in tuples]
                # n_nonb = [i[1] for i in tuples]
                # ax.scatter(fdrs, n_nonb, color=color, marker='o', label=method)
                
                ax.plot(fpr, tpr, color=methods_colors[method], linestyle=nonb_ratio_styles[nbr],
                        linewidth=2, label=method + ', ' + 'nonb ratio: ' + str(nbr))
        
        ax.set_xlabel('FPR', fontsize=27)
        ax.set_ylabel('TPR', fontsize=27)
        # leg = plt.legend(loc=(0.9, 0.03), title='Method', prop={'size': 16}, title_fontsize=18)
        # ax.add_artist(leg)
        # h = [plt.plot([], [], color="gray", linestyle=ls, linewidth=3)[0] for ls in ['-', '--']]
        # plt.legend(handles=h, labels=['Non-B DNA', 'Control'], loc=(1.03, 0.85),
        #            title="Structure", prop={'size': 14}, title_fontsize=15)
        plt.setp(ax.get_xticklabels(), Fontsize=22)
        plt.setp(ax.get_yticklabels(), Fontsize=22)
        # ax.text(40, max_val+0.0001, direction_names[dir], fontsize=25, bbox={'alpha': 0, 'pad': 2})
        # ax.set_ylim(top=max_val+0.00053)
        ax.set_title(nonb, fontsize=20)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(plot_path, 'classifiers_' + nonb + '.png'))
    
    this_pr_nbr = pr_df[pr_df['nonb_ratio'] == 0.5].reset_index(drop=True)
    this_roc_nbr = roc_df[roc_df['nonb_ratio'] == 0.5].reset_index(drop=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    for method in methods:
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        for ni, nonb in enumerate(nonbs):
            this_df_pr = this_pr_nbr[(this_pr_nbr['label'] == nonb) & (this_pr_nbr['method'] == method)].reset_index(
                drop=True)
            this_df_roc = this_roc_nbr[
                (this_roc_nbr['label'] == nonb) & (this_roc_nbr['method'] == method)].reset_index(drop=True)
            fpr = this_df_roc['fpr']
            tpr = this_df_roc['tpr']
            
            # fdrs = [i[0] for i in tuples]
            # n_nonb = [i[1] for i in tuples]
            # ax.scatter(fdrs, n_nonb, color=color, marker='o', label=method)
            
            ax.plot(fpr, tpr, color=colors[ni], linewidth=3, label=nonb)
        
        ax.set_xlabel('FPR', fontsize=27)
        ax.set_ylabel('TPR', fontsize=27)
        # leg = plt.legend(loc=(0.9, 0.03), title='Method', prop={'size': 16}, title_fontsize=18)
        # ax.add_artist(leg)
        # h = [plt.plot([], [], color="gray", linestyle=ls, linewidth=3)[0] for ls in ['-', '--']]
        # plt.legend(handles=h, labels=['Non-B DNA', 'Control'], loc=(1.03, 0.85),
        #            title="Structure", prop={'size': 14}, title_fontsize=15)
        plt.setp(ax.get_xticklabels(), Fontsize=22)
        plt.setp(ax.get_yticklabels(), Fontsize=22)
        # ax.text(40, max_val+0.0001, direction_names[dir], fontsize=25, bbox={'alpha': 0, 'pad': 2})
        # ax.set_ylim(top=max_val+0.00053)
        ax.set_title(method + ' - nonb ratio : 0.5', fontsize=20)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(plot_path, 'xclassifiers_' + method + '.png'))


def plot_input_all_separated(folder):
    # folder = 'Data/noisy_simulated_data/dataset'
    plot_path = os.path.join(folder, 'plots_input_sep')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    
    n_bdna = 20000
    n_nonb = 2000
    winsize = 50
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif']
    elements_name = ['A Phased Repeat', 'G Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z DNA']
    
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    x = range(0, winsize)
    direction = ['same', 'opposite']
    direction_cols = {'same': ['forward_' + str(i) for i in range(winsize)],
                      'opposite': ['reverse_' + str(i) for i in range(winsize)]}
    direction_names = {'same': 'Forward', 'opposite': 'Reverse'}
    
    for nonb_ratio in [0.05, 0.1, 0.25]:
        global_train, global_val, global_test = load_data2(folder, winsize, n_bdna, n_nonb, nonb_ratio=nonb_ratio)
        data_dict = {'train': global_train, 'validation': global_val, 'test': global_test}
        # df = pd.concat([global_train, global_val, global_test])
        for data in data_dict:
            df = data_dict[data]
            control_signals = df[df['true_label'] == 'bdna'].reset_index(drop=True)
            for elem_id in range(len(elements)):
                elem_name = elements_name[elem_id]
                elem = elements[elem_id]
                for dire in direction:
                    control_signal = control_signals[direction_cols[dire]]
                    nonb_signals = df[df['true_label'] == elem][direction_cols[dire]].reset_index(drop=True)
                    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                    for quant_id in range(len(quantiles)):
                        quant = quantiles[quant_id]
                        color = colors[quant_id]
                        signal_quantile = np.quantile(nonb_signals, quant, axis=0)
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
                    ax.set_title(
                        elem_name + ' ' + direction_names[dire] + ' ' + data + ' nonb ratio: ' + str(nonb_ratio),
                        fontsize=20)
                    plt.tight_layout()
                    # plt.show()
                    plt.savefig(
                        os.path.join(plot_path,
                                     elem + '_' + direction_names[dire] + '_' + data + '_' + str(nonb_ratio) + '.png'))


def plot_input_all(folder):
    # folder = 'Data/noisy_simulated_data/dataset'
    plot_path = os.path.join(folder, 'plots_input')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    
    n_bdna = 20000
    n_nonb = 2000
    winsize = 50
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif']
    elements_name = ['A Phased Repeat', 'G Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z DNA']
    
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    x = range(0, winsize)
    direction = ['same', 'opposite']
    direction_cols = {'same': ['forward_' + str(i) for i in range(winsize)],
                      'opposite': ['reverse_' + str(i) for i in range(winsize)]}
    direction_names = {'same': 'Forward', 'opposite': 'Reverse'}
    
    for nonb_ratio in [0.05, 0.1, 0.25]:
        global_train, global_val, global_test = load_data2(folder, winsize, n_bdna, n_nonb, nonb_ratio=nonb_ratio)
        data_dict = {'train': global_train, 'validation': global_val, 'test': global_test}
        df = pd.concat([global_train, global_val, global_test])
        control_signals = df[df['true_label'] == 'bdna'].reset_index(drop=True)
        for elem_id in range(len(elements)):
            elem_name = elements_name[elem_id]
            elem = elements[elem_id]
            for dire in direction:
                control_signal = control_signals[direction_cols[dire]]
                nonb_signals = df[df['true_label'] == elem][direction_cols[dire]].reset_index(drop=True)
                
                fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                for quant_id in range(len(quantiles)):
                    quant = quantiles[quant_id]
                    color = colors[quant_id]
                    signal_quantile = np.quantile(nonb_signals, quant, axis=0)
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
                ax.set_title(elem_name + ' ' + direction_names[dire] + 'nonb ratio: ' + str(nonb_ratio),
                             fontsize=20)
                plt.tight_layout()
                # plt.show()
                plt.savefig(
                    os.path.join(plot_path, elem + '_' + direction_names[dire] + '_' + str(nonb_ratio) + '.png'))




def subsample(train, val, frac):
    train_bdna = train[train['label'] == 'bdna'].reset_index(drop=True)
    train_nonb = train[train['label'] != 'bdna'].reset_index(drop=True)
    
    val_bdna = val[val['label'] == 'bdna'].reset_index(drop=True)
    val_nonb = val[val['label'] != 'bdna'].reset_index(drop=True)
    
    train_bdna = train_bdna.sample(frac=frac, random_state=42).reset_index(drop=True)
    train_nonb = train_nonb.sample(frac=frac, random_state=42).reset_index(drop=True)
    val_bdna = val_bdna.sample(frac=frac, random_state=42).reset_index(drop=True)
    val_nonb = val_nonb.sample(frac=frac, random_state=42).reset_index(drop=True)
    train = pd.concat([train_bdna, train_nonb], ignore_index=True).sample(frac=1, random_state=42).reset_index(
        drop=True)
    val = pd.concat([val_bdna, val_nonb], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return train, val


def plot_experimental_methods_comparison(path):
    exp_folders = [os.path.join(path, name) for name in os.listdir(path) if
                   'old' not in name and 'exp' in name and os.path.isdir(os.path.join(path, name))]
    nonbs = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
             'Short_Tandem_Repeat', 'Z_DNA_Motif']
    
    results_files = [os.path.join(f, elem, 'final_results.csv') for f in exp_folders for elem in os.listdir(f)
                     if os.path.isdir(os.path.join(f, elem)) and any(nonb in elem for nonb in nonbs)]
    results_pd_list = [pd.read_csv(file, index_col=0) for file in results_files if os.path.exists(file)]
    results_pd = pd.concat(results_pd_list, axis=0, ignore_index=True)
    plot_path = 'results/plots_exp'
    tail_types = ['upper', 'lower']
    
    for tail_type in tail_types:
        elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                    'Short_Tandem_Repeat', 'Z_DNA_Motif']
        elements_name = ['A Phased Repeat', 'G-Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                         'Short Tandem Repeat', 'Z-DNA']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
        
        # ['tab:brown', 'tab:gray', 'tab:olive', 'tab:cyan']
        results_pd = results_pd[results_pd['tail'] == 'upper'].reset_index(drop=True)
        for elem_id in range(len(elements)):
            elem_name = elements_name[elem_id]
            elem = elements[elem_id]
            df = results_pd[results_pd['label'] == elem].reset_index(drop=True)
            plt.cla()
            plt.close()
            plt.clf()
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            methods = list(df['method'].unique())
            # methods_colors = {'Isolation Forest': 'tab:blue', 'One Class SVM': 'tab:orange', 'GoFAE-COD': 'tab:green',
            #                   'Outlier AE (64)': 'tab:red', 'Outlier AE (128)': 'tab:purple'}
            methods_colors = {'IF': 'tab:blue', 'LOF': 'tab:orange', 'SVM': 'tab:green'}
            for idx, method in enumerate(methods):
                this_df = df[df['method'] == method].reset_index(drop=True)
                tuples = list(zip(this_df.alpha, this_df.potential_nonb_counts))
                fdrs = [i[0] for i in tuples]
                n_nonb = [i[1] for i in tuples]
                color = methods_colors[method]
                ax.plot(fdrs, n_nonb, color=color, linewidth=3, marker='o', label=method)
                # ax.scatter(fdrs, n_nonb, color=color, marker='o', label=method)
            
            ax.set_xlabel('FDR', fontsize=27)
            ax.set_ylabel('Non-B DNA Count', fontsize=27)
            # leg = plt.legend(loc=(0.9, 0.03), title='Method', prop={'size': 16}, title_fontsize=18)
            # ax.add_artist(leg)
            # h = [plt.plot([], [], color="gray", linestyle=ls, linewidth=3)[0] for ls in ['-', '--']]
            # plt.legend(handles=h, labels=['Non-B DNA', 'Control'], loc=(1.03, 0.85),
            #            title="Structure", prop={'size': 14}, title_fontsize=15)
            plt.setp(ax.get_xticklabels(), Fontsize=22)
            plt.setp(ax.get_yticklabels(), Fontsize=22)
            # ax.text(40, max_val+0.0001, direction_names[dir], fontsize=25, bbox={'alpha': 0, 'pad': 2})
            # ax.set_ylim(top=max_val+0.00053)
            ax.set_title(elem_name + ', ' + tail_type, fontsize=20)
            plt.legend()
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(plot_path, 'exp_' + elem + '_' + tail_type + '.png'))


def plot_experimental_comparison_ismb(path):
    cols = ['dataset', 'method', 'label', 'alpha', 'potential_nonb_counts']
    
    IF_final = 'results/beagle2/exp_IF/final_results_exp_IF.csv'
    LOF_final = 'results/beagle2/exp_LOF/final_results_exp_LOF.csv'
    SVM_final = 'results/beagle2/exp_SVM/final_results_exp_SVM.csv'
    # GoFAE_DND_final = 'results/beagle2/exp_ourmodel/GoFAE-DND_final_results_0.35.csv'
    # AE_final = 'results/beagle/exp_AE/AE_final_results_exp.csv'
    
    if_df = pd.read_csv(IF_final, index_col=0)
    lof_df = pd.read_csv(LOF_final, index_col=0)
    svm_df = pd.read_csv(SVM_final, index_col=0)
    # gofae_df = pd.read_csv(GoFAE_DND_final, index_col=0)
    # ae_df = pd.read_csv(AE_final, index_col=0)
    
    if_df = if_df[if_df['tail'] == 'lower'].reset_index(drop=True)
    lof_df = lof_df[lof_df['tail'] == 'lower'].reset_index(drop=True)
    svm_df = svm_df[svm_df['tail'] == 'lower'].reset_index(drop=True)
    if_df = if_df.drop(['training_time', 'tail'], axis=1)
    lof_df = lof_df.drop(['training_time', 'tail'], axis=1)
    svm_df = svm_df.drop(['training_time', 'tail'], axis=1)
    
    # gofae_df['dataset'] = 'experimental'
    # gofae_df['method'] = 'GoFAE-DND'
    # gofae_df['dataset'] = 'experimental'
    # ae_df['dataset'] = 'experimental'
    
    if_df['method'] = 'Isolation Forest'
    lof_df['method'] = 'Local Outlier Factor'
    svm_df['method'] = 'One Class SVM'
    
    if_df = if_df[cols]
    lof_df = lof_df[cols]
    # gofae_df = gofae_df[cols]
    svm_df = svm_df[cols]
    # ae_df = ae_df[cols]
    
    # gofae_df = gofae_df[['dataset', 'method', 'label', 'alpha', 'potential_nonb_counts']]
    # if_df = if_df[['dataset', 'method', 'label', 'alpha', 'potential_nonb_counts']]
    # lof_df = lof_df[['dataset', 'method', 'label', 'alpha', 'potential_nonb_counts']]
    # svm_df = svm_df[['dataset', 'method', 'label', 'alpha', 'potential_nonb_counts']]
    
    # all_results = pd.concat([if_df, lof_df, svm_df, gofae_df, ae_df], ignore_index=True)
    all_results = pd.concat([if_df, lof_df, svm_df], ignore_index=True)
    all_results = all_results[all_results['alpha'] <= 0.5]
    all_results.to_csv('results/beagle2/all_experimental.csv')
    plot_path = 'results//beagle2/plots_exp'
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif']
    elements_name = ['A Phased Repeat', 'G-Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z-DNA']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    methods_colors = {'Isolation Forest': 'tab:blue', 'One Class SVM': 'tab:orange', 'GoFAE-DND': 'tab:green',
                      'AE': 'tab:red', 'Outlier AE (128)': 'tab:purple', 'Local Outlier Factor': 'tab:pink'}
    markers_markers = {'Isolation Forest': 'd', 'One Class SVM': '*', 'GoFAE-DND': 'o',
                       'AE': '|', 'Outlier AE (128)': 'x', 'Local Outlier Factor': 'P'}
    
    for elem_id in range(len(elements)):
        elem_name = elements_name[elem_id]
        elem = elements[elem_id]
        df = all_results[all_results['label'] == elem].reset_index(drop=True)
        plt.cla()
        plt.close()
        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        methods = list(df['method'].unique())
        
        for idx, method in enumerate(methods):
            this_df = df[df['method'] == method].reset_index(drop=True)
            tuples = list(zip(this_df.alpha, this_df.potential_nonb_counts))
            fdrs = [i[0] for i in tuples]
            n_nonb = [i[1] for i in tuples]
            color = methods_colors[method]
            marker = markers_markers[method]
            ax.plot(fdrs, n_nonb, color=color, linewidth=3, marker=marker, label=method, markersize=10, alpha=0.7)
            # ax.scatter(fdrs, n_nonb, color=color, marker='o', label=method)
        
        ax.set_xlabel('FDR', fontsize=27)
        ax.set_ylabel('Non-B DNA Count', fontsize=27)
        # leg = plt.legend(loc=(0.9, 0.03), title='Method', prop={'size': 16}, title_fontsize=18)
        # ax.add_artist(leg)
        # h = [plt.plot([], [], color="gray", linestyle=ls, linewidth=3)[0] for ls in ['-', '--']]
        # plt.legend(handles=h, labels=['Non-B DNA', 'Control'], loc=(1.03, 0.85),
        #            title="Structure", prop={'size': 14}, title_fontsize=15)
        plt.setp(ax.get_xticklabels(), Fontsize=22)
        plt.setp(ax.get_yticklabels(), Fontsize=22)
        # ax.text(40, max_val+0.0001, direction_names[dir], fontsize=25, bbox={'alpha': 0, 'pad': 2})
        # ax.set_ylim(top=max_val+0.00053)
        ax.set_title(elem_name, fontsize=20)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(plot_path, 'exp_' + elem + '.png'))


def plot_classifiers_ismb1():
    classifiers_path = 'results/classifiers'


def plot_roc():
    results_sim = 'D:/UCONN/nonBDNA/results/classifiers/simulation'
    out_liers = ['IF', 'LOF', 'SVM']
    classifiers = ['SVC', 'RF', 'GP', 'KNN', 'LR']
    # outfiles = [os.path.join(results_sim, name, 'final_results_' + name + '.csv') for name in os.listdir(results_sim)
    #            if 'sim' in name and any(ol in name for ol in out_liers)]
    needed_cols = ['dataset', 'method', 'label', 'nonb_ratio', 'fscore']
    needed_nonb_ratios = [0.05, 0.1, 0.25]
    classifiles = [os.path.join(results_sim, name, 'final_results_' + name + '.csv') for name in os.listdir(results_sim)
                   if 'sim' in name and any(cl in name for cl in classifiers)]
    cls = [pd.read_csv(f, index_col=0) for f in classifiles]
    all_classifiers = pd.concat(cls, ignore_index=True)


def find_mask_and_genomic_positions_in_100_wins(win_start, win_end, motif):
    ml = len(motif)
    mask = np.ones(100)
    if ml % 2 == 0:
        a = ml / 2
        b = 50 - a
        motif_start = win_start + b
        motif_end = win_end - b
        mask[0:int(b)] = 0
        mask[-int(b):] = 0
    # ml%2 == 1: # motif length is odd
    else:
        a = np.floor(ml / 2)
        b_start = 50 - (a + 1)
        b_end = 50 - a
        motif_start = win_start + b_start
        motif_end = win_end - b_end
        mask[0:int(b_start)] = 0
        mask[-int(b_end):] = 0
    return motif_start, motif_end, mask



def prepare_sim_all_results_ismb():
    alpha = 0.2
    results_sim = 'results/final_results_ismb'
    plot_path = os.path.join(results_sim, 'plots')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    out_liers = ['IF', 'LOF', 'SVM']
    classifiers = ['SVC', 'RF', 'GP', 'KNN', 'LR']
    elements = ['G_Quadruplex_Motif', 'Short_Tandem_Repeat']
    elements_name = ['G Quadruplex', 'Short Tandem Repeat']
    methods_colors = {'GoFAE-DND': 'tab:red', 'Isolation Forest': 'tab:blue', 'One Class SVM': 'tab:green',
                      'Local Outlier Factor': 'tab:orange',
                      'Isolation Forest (bdna)': '#acc2d9', 'One Class SVM (bdna)': 'black',
                      'Local Outlier Factor (bdna)': '#388004', 'GP': 'tab:purple', 'KNN': '#DBB40C',
                      'LR': 'tab:cyan', 'RF': 'tab:brown', 'SVC': 'tab:pink'}
    
    method_order = {'GoFAE-DND': 0, 'Isolation Forest': 1, 'Isolation Forest (bdna)': 2, 'Local Outlier Factor': 3,
                    'Local Outlier Factor (bdna)': 4, 'One Class SVM': 5, 'One Class SVM (bdna)': 6,
                    'KNN': 7, 'LR': 8, 'RF': 9, 'SVC': 10, 'GP': 11}
    # outlier_order = {'GoFAE-DND': 0, 'Isolation Forest': 1, 'Isolation Forest (bdna)': 2, 'Local Outlier Factor': 3,
    #                  'Local Outlier Factor (bdna)': 4, 'One Class SVM': 5, 'One Class SVM (bdna)': 6}
    #
    # classifiers_order = {'KNN': 7, 'LR': 8, 'RF': 9, 'SVC': 10, 'GP': 11}
    
    needed_cols = ['dataset', 'method', 'label', 'nonb_ratio', 'alpha', 'fscore']
    needed_cols_classifiers = ['dataset', 'method', 'label', 'nonb_ratio', 'fscore']
    needed_nonb_ratios = [0.05, 0.1, 0.25]
    classifiles = [os.path.join(results_sim, name, 'final_results_' + name + '.csv') for name in os.listdir(results_sim)
                   if 'sim' in name and any(cl in name for cl in classifiers)]
    cls = [pd.read_csv(f, index_col=0) for f in classifiles]
    all_classifiers = pd.concat(cls, ignore_index=True)
    all_classifiers = all_classifiers[needed_cols_classifiers]
    all_classifiers = all_classifiers[all_classifiers['nonb_ratio'].isin(needed_nonb_ratios)].reset_index(drop=True)
    all_classifiers.to_csv(os.path.join(results_sim, 'simulation_results_classifiers_methods.csv'))
    
    IF_final = os.path.join(results_sim, 'sim_IF/final_results_sim_IF.csv')
    LOF_final = os.path.join(results_sim, 'sim_LOF/final_results_sim_LOF.csv')
    SVM_final = os.path.join(results_sim, 'sim_SVM/final_results_sim_SVM.csv')
    IF_bdna_final = os.path.join(results_sim, 'sim_IF_bdna/final_results_sim_IF_bdna.csv')
    LOF_bdna_final = os.path.join(results_sim, 'sim_LOF_bdna/final_results_sim_LOF_bdna.csv')
    SVM_bdna_final = os.path.join(results_sim, 'sim_SVM_bdna/final_results_sim_SVM_bdna.csv')
    all_paths = [IF_final, LOF_final, SVM_final, IF_bdna_final, LOF_bdna_final, SVM_bdna_final]
    all_dfs = [pd.read_csv(path, index_col=0) for path in all_paths]
    dfs = pd.concat(all_dfs).reset_index(drop=True)
    dfs = dfs[dfs['tail'] == 'lower'].reset_index(drop=True)
    dfs = dfs[dfs['function'] == 'decision_function'].reset_index(drop=True)
    dfs = dfs.drop(['duration', 'tail', 'function'], axis=1)
    dfs.loc[dfs['method'] == 'IF', 'method'] = 'Isolation Forest'
    dfs.loc[dfs['method'] == 'LOF', 'method'] = 'Local Outlier Factor'
    dfs.loc[dfs['method'] == 'SVM', 'method'] = 'One Class SVM'
    dfs.loc[dfs['method'] == 'IF_bdna', 'method'] = 'Isolation Forest (bdna)'
    dfs.loc[dfs['method'] == 'LOF_bdna', 'method'] = 'Local Outlier Factor (bdna)'
    dfs.loc[dfs['method'] == 'SVM_bdna', 'method'] = 'One Class SVM (bdna)'
    dfs = dfs[needed_cols]
    
    gofae_path = os.path.join(results_sim, 'sim_GoFAE-DND')
    gofae_files = [pd.read_csv(os.path.join(gofae_path, name), index_col=0) for name in os.listdir(gofae_path) if
                   '.csv' in name]
    gofae_df = pd.concat(gofae_files, ignore_index=True)
    gofae_df = gofae_df.rename(columns={'nonb_type': 'label'})
    gofae_df = gofae_df.drop(['upper'], axis=1)
    gofae_df = gofae_df[needed_cols]
    
    all_out = pd.concat([dfs, gofae_df], ignore_index=True)
    all_out = all_out[(all_out['alpha'] > alpha - 0.01) & (all_out['alpha'] < alpha + 0.01)].reset_index(drop=True)
    
    all_out = all_out.drop(['alpha'], axis=1)
    all_out = all_out[needed_cols_classifiers]
    all_out['dataset'] = 'sim'
    all_out = all_out.sort_values(by=['dataset', 'method', 'label', 'nonb_ratio'])
    all_out = all_out[all_out['nonb_ratio'].isin(needed_nonb_ratios)].reset_index(drop=True)
    # all_out = all_out.fillna(0)
    # all_out = all_out.dropna()
    all_out.to_csv(os.path.join(results_sim, 'simulation_results_outlier_methods.csv'))
    everything = pd.concat([all_out, all_classifiers], ignore_index=True)
    everything = everything[everything['nonb_ratio'].isin(needed_nonb_ratios)].reset_index(drop=True)
    everything = everything.sort_values(by=['dataset', 'method', 'label', 'nonb_ratio']).reset_index(drop=True)
    everything.to_csv(os.path.join(results_sim, 'all_simulation_results.csv'))
    
    everything['method_order'] = everything['method'].apply(lambda x: method_order[x])
    everything = everything.sort_values(by='method_order', ascending=True)
    everything = everything.drop(['method_order'], axis=1).reset_index(drop=True)
    

    nonb_ratios = [0.05, 0.1, 0.25]
    
    outlier_order = {'Isolation Forest': 0, 'Local Outlier Factor': 1, 'One Class SVM': 2, 'GoFAE-DND': 3}
    classifiers_order = {'KNN': 4, 'GP': 5, 'RF': 6, 'LR': 7, 'SVC': 8}
    barWidth = 0.4
    for elem_id in range(2):
        elem_name = elements_name[elem_id]
        elem = elements[elem_id]
        plt.cla()
        plt.close()
        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        x = 0
        ax.axhline(0.1, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.2, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.3, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.4, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.5, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.6, color="gray", linestyle='--', zorder=0)
        # ax.axhline(0.7, color="gray", linestyle='--', zorder=0)
        # ax.axhline(0.8, color="gray", linestyle='--', zorder=0)
        for nbr in nonb_ratios[:-1]:
            
            for me in list(classifiers_order.keys()):
                this_df = everything[(everything['method'] == me) & (everything['label'] == elem) & (
                        everything['nonb_ratio'] == nbr)].reset_index(drop=True)
                y = this_df.fscore.values[0]
                plt.bar(x, y, color=methods_colors[me], width=barWidth, hatch='//', edgecolor='black')
                x += 0.5
            
            x += 0.5
            
            for me in list(outlier_order.keys()):
                this_df = everything[(everything['method'] == me) & (everything['label'] == elem) & (
                        everything['nonb_ratio'] == nbr)].reset_index(drop=True)
                y = this_df.fscore.values[0]
                plt.bar(x, y, color=methods_colors[me], width=barWidth, edgecolor='black')
                x += 0.5
            
            x += 2
        
        for nbr in [nonb_ratios[-1]]:
            
            for me in list(classifiers_order.keys()):
                this_df = everything[(everything['method'] == me) & (everything['label'] == elem) & (
                        everything['nonb_ratio'] == nbr)].reset_index(drop=True)
                y = this_df.fscore.values[0]
                plt.bar(x, y, color=methods_colors[me], width=barWidth, edgecolor='black', hatch='//', label=me)
                x += 0.5
            
            x += 0.5
            
            for me in list(outlier_order.keys()):
                this_df = everything[(everything['method'] == me) & (everything['label'] == elem) & (
                        everything['nonb_ratio'] == nbr)].reset_index(drop=True)
                y = this_df.fscore.values[0]
                plt.bar(x, y, color=methods_colors[me], width=barWidth, edgecolor='black', label=me)
                x += 0.5
        
        # leg = plt.legend(loc=(1.03, 0.50), title="Quantile", prop={'size': 14}, title_fontsize=15)
        # ax.add_artist(leg)
        # h = [plt.plot([], [], color="gray", linestyle=ls, linewidth=3)[0] for ls in ['-', '--']]
        # plt.legend(handles=h, labels=['Non-B DNA', 'Control'], loc=(1.03, 0.85),
        #            title="Structure", prop={'size': 14}, title_fontsize=15)
        
        # plt.xlabel('Non-B ratios', fontsize=27, labelpad=30)
        plt.xlabel('Non-B ratios', fontsize=27)
        plt.xticks([x / 6 - 1, x / 2, 5 * x / 6 + 0.5], [str(i) for i in nonb_ratios])
        plt.ylabel('F1 Score', fontsize=27)
        plt.setp(ax.get_xticklabels(), Fontsize=22)
        plt.setp(ax.get_yticklabels(), Fontsize=22)
        # plt.legend(labels = ['KNN', 'GP', 'RF', 'LR', 'SVC'], loc=(1.03, 0.2), title="Classifiers",
        #            prop={'size': 14}, title_fontsize=15)
        # plt.legend(labels = ['Isolation Forest', 'Local Outlier Factor', 'One Class SVM', 'GoFAE-DND'],
        #            loc=(1.03, 0.2), title="Novelty Detectors",
        #            prop={'size': 14}, title_fontsize=15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, elem + 'F1_score.tiff'), dpi=1200)
        plt.savefig(os.path.join(plot_path, elem + 'F1_score.png'), dpi=1200)
        plt.savefig(os.path.join(plot_path, elem + 'F1_score.tif'), dpi=1200)
        plt.savefig(os.path.join(plot_path, elem + 'F1_score.pdf'), dpi=1200)
        
        # plt.savefig(os.path.join(plot_path, 'leg2.png'), dpi=1200)


def prepare_sim_outliers_results_ismb():
    alpha = 0.2
    results_sim = 'results/final_results_ismb'
    plot_path = os.path.join(results_sim, 'plots')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    out_liers = ['IF', 'LOF', 'SVM']
    elements = ['G_Quadruplex_Motif', 'Short_Tandem_Repeat']
    elements_name = ['G Quadruplex', 'Short Tandem Repeat']
    methods_colors = {'GoFAE-DND': 'tab:red', 'Isolation Forest': 'tab:blue', 'One Class SVM': 'tab:green',
                      'Local Outlier Factor': 'tab:orange',
                      'Isolation Forest (bdna)': '#acc2d9', 'One Class SVM (bdna)': 'black',
                      'Local Outlier Factor (bdna)': '#388004', 'GP': 'tab:purple', 'KNN': '#DBB40C',
                      'LR': 'tab:cyan', 'RF': 'tab:brown', 'SVC': 'tab:pink'}
    
    outlier_order = {'Isolation Forest': 0, 'Isolation Forest (bdna)': 1, 'Local Outlier Factor': 2, 'GoFAE-DND': 3,
                     'Local Outlier Factor (bdna)': 4, 'One Class SVM': 5, 'One Class SVM (bdna)': 6}
    
    needed_cols = ['dataset', 'method', 'label', 'nonb_ratio', 'alpha', 'fscore']
    
    outlier_order = {'Isolation Forest': 0, 'Local Outlier Factor': 1, 'One Class SVM': 2, 'GoFAE-DND': 3}
    classifiers_order = {'KNN': 4, 'LR': 5, 'RF': 6, 'SVC': 7, 'GP': 8}
    
    nonb_ratios = [0.05, 0.1, 0.25]
    
    IF_final = os.path.join(results_sim, 'sim_IF/final_results_sim_IF.csv')
    LOF_final = os.path.join(results_sim, 'sim_LOF/final_results_sim_LOF.csv')
    SVM_final = os.path.join(results_sim, 'sim_SVM/final_results_sim_SVM.csv')
    IF_bdna_final = os.path.join(results_sim, 'sim_IF_bdna/final_results_sim_IF_bdna.csv')
    LOF_bdna_final = os.path.join(results_sim, 'sim_LOF_bdna/final_results_sim_LOF_bdna.csv')
    SVM_bdna_final = os.path.join(results_sim, 'sim_SVM_bdna/final_results_sim_SVM_bdna.csv')
    all_paths = [IF_final, LOF_final, SVM_final, IF_bdna_final, LOF_bdna_final, SVM_bdna_final]
    all_dfs = [pd.read_csv(path, index_col=0) for path in all_paths]
    dfs = pd.concat(all_dfs).reset_index(drop=True)
    dfs = dfs[dfs['tail'] == 'lower'].reset_index(drop=True)
    dfs = dfs[dfs['function'] == 'decision_function'].reset_index(drop=True)
    dfs = dfs.drop(['duration', 'tail', 'function'], axis=1)
    dfs.loc[dfs['method'] == 'IF', 'method'] = 'Isolation Forest'
    dfs.loc[dfs['method'] == 'LOF', 'method'] = 'Local Outlier Factor'
    dfs.loc[dfs['method'] == 'SVM', 'method'] = 'One Class SVM'
    dfs.loc[dfs['method'] == 'IF_bdna', 'method'] = 'Isolation Forest (bdna)'
    dfs.loc[dfs['method'] == 'LOF_bdna', 'method'] = 'Local Outlier Factor (bdna)'
    dfs.loc[dfs['method'] == 'SVM_bdna', 'method'] = 'One Class SVM (bdna)'
    dfs = dfs[needed_cols]
    
    gofae_path = os.path.join(results_sim, 'sim_GoFAE-DND')
    gofae_files = [pd.read_csv(os.path.join(gofae_path, name), index_col=0) for name in os.listdir(gofae_path) if
                   '.csv' in name]
    gofae_df = pd.concat(gofae_files, ignore_index=True)
    gofae_df = gofae_df.rename(columns={'nonb_type': 'label'})
    gofae_df = gofae_df.drop(['upper'], axis=1)
    gofae_df = gofae_df[needed_cols]
    
    all_out = pd.concat([dfs, gofae_df], ignore_index=True)
    all_out = all_out[(all_out['alpha'] > alpha - 0.01) & (all_out['alpha'] < alpha + 0.01)].reset_index(drop=True)
    # all_out = all_out.drop(['alpha'], axis=1)
    all_out['dataset'] = 'sim'
    all_out = all_out.sort_values(by=['dataset', 'method', 'label', 'nonb_ratio']).reset_index(drop=True)
    all_out.to_csv(os.path.join(results_sim, 'simulation_results_outlier_methods.csv'))
    everything = all_out
    
    barWidth = 0.3
    for elem_id in range(2):
        elem_name = elements_name[elem_id]
        elem = elements[elem_id]
        plt.cla()
        plt.close()
        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        x = 0
        ax.axhline(0.1, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.2, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.3, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.4, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.5, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.6, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.7, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.8, color="gray", linestyle='--', zorder=0)
        # ax.axhline(0.9, color="gray", linestyle='--', zorder=0)
        
        for nbr in nonb_ratios[:-1]:
            print(nbr)
            for me in list(outlier_order.keys()):
                this_df = everything[(everything['method'] == me) & (everything['label'] == elem) & (
                        everything['nonb_ratio'] == nbr)].reset_index(drop=True)
                y = this_df.fscore.values[0]
                print(nbr, me, y)
                plt.bar(x, y, color=methods_colors[me], width=barWidth, edgecolor='black')
                x += 0.3
            
            x += 0.6
        
        for nbr in [nonb_ratios[-1]]:
            print(nbr)
            for me in list(outlier_order.keys()):
                this_df = everything[(everything['method'] == me) & (everything['label'] == elem) & (
                        everything['nonb_ratio'] == nbr)].reset_index(drop=True)
                y = this_df.fscore.values[0]
                print(nbr, me, y)
                plt.bar(x, y, color=methods_colors[me], width=barWidth, edgecolor='black', label=me)
                x += 0.3
        
        # plt.xlabel('Non-B ratios', fontsize=27, labelpad=30)
        plt.xlabel('Non-B ratios', fontsize=27)
        # plt.xlabel("...", labelpad=20)
        plt.xticks([2 * 0.3 - 0.1, 8 * 0.3 - 0.1, 14 * 0.3 - 0.1], [str(i) for i in nonb_ratios])
        plt.ylabel('F1 Score', fontsize=27)
        plt.setp(ax.get_xticklabels(), Fontsize=22)
        plt.setp(ax.get_yticklabels(), Fontsize=22)
        plt.legend(loc=2, prop={'size': 17})
        # plt.annotate('Outliers', xy=(0, 2), xycoords='data', xytext=(1.5, 1.5),
        #              textcoords='offset points')
        plt.tight_layout()
        
        # plt.show()
        plt.savefig(os.path.join(plot_path, elem + 'F1_score_outliers.tiff'), dpi=1200)
        plt.savefig(os.path.join(plot_path, elem + 'F1_score_outliers.png'), dpi=1200)
        plt.savefig(os.path.join(plot_path, elem + 'F1_score_outliers.tif'), dpi=1200)
        plt.savefig(os.path.join(plot_path, elem + 'F1_score_outliers.pdf'), dpi=1200)

