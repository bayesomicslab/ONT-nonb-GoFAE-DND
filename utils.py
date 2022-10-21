import os
import pickle
import gzip
import _pickle as cPickle
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

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
    main_folder = '/labs/Aguiar/non_bdna/annotations/windows/initial_windows/' + chromosome + strand
    folders = sorted([os.path.join(main_folder, name) for name in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, name))])
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
        sliding_windows[i, :] = vector[i:i+window_size]
    return sliding_windows



def prepare_bdna_dataset(path):
    # main_path = '/labs/Aguiar/non_bdna/annotations/vae_windows/prepared_windows_req5_mean/dataset/bdna'
    # main_path = 'Data/prepared_windows/'
    # main_path = '/home/mah19006/projects/nonBDNA/data/prepared_windows_req5/dataset'
    # bdna_path = os.path.join(main_path, 'bdna')
    forward_train_100 = np.load(os.path.join(path, 'forward_train_100.npy'))
    forward_val_100 = np.load(os.path.join(path, 'forward_val_100.npy'))
    forward_test_100 = np.load(os.path.join(path, 'forward_test_100.npy'))
    reverse_train_100 = np.load(os.path.join(path, 'reverse_train_100.npy'))
    reverse_val_100 = np.load(os.path.join(path, 'reverse_val_100.npy'))
    reverse_test_100 = np.load(os.path.join(path, 'reverse_test_100.npy'))
    return forward_train_100, forward_val_100, forward_test_100, reverse_train_100, reverse_val_100, reverse_test_100


def prepare_nonb_dataset_center(path):
    # path = '/labs/Aguiar/non_bdna/annotations/vae_windows/prepared_windows_req5_mean/dataset/outliers/centered_windows'
    # path = '/home/mah19006/projects/nonBDNA/data/prepared_windows_req5/dataset/outliers/centered_windows'
    non_b_types = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                   'Short_Tandem_Repeat', 'Z_DNA_Motif']
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
    # main_path = '/labs/Aguiar/non_bdna/annotations/vae_windows/prepared_windows/'
    main_path = '/home/mah19006/projects/nonBDNA/data/prepared_windows/'
    # main_path = 'Data/prepared_windows/'
    nonbdna_sliding_path = os.path.join(main_path, 'outliers', 'sliding_extended_motifs')
    sliding_nonb_files_path = {nonb: os.path.join(nonbdna_sliding_path, nonb + '.csv') for nonb in non_b_types}
    cols = ['id', 'chr', 'strand', 'start', 'end', 'label', 'motif_proportion'] + \
           ['forward_'+str(i) for i in range(100)] + \
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
    train, validate, test = np.split(df.sample(frac=1, random_state=22), [int(train_portion*len(df)), int((train_portion + val_portion)*len(df))])
    return train, validate, test


def compute_accuracy_metrics(tn, fp, fn, tp):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = (2 * precision * recall) / (precision + recall)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    fdr = fp / (fp + tp) # False discovery rate
    return accuracy, precision, recall, fscore, fpr, fnr, fdr


def split_dataframe(df, chunk_size=10000):
    chunks = []
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size].reset_index(drop=True))
    return chunks


def load_data(bdna_folder, nonb_folder):
    forward_tr, reverse_tr, forward_val, reverse_val, forward_te, reverse_te, \
    train_forward_mu, train_forward_std, train_reverse_mu, train_reverse_std = reprocess_data(bdna_folder)
    win_size = forward_tr.shape[1]
    # forward_tr, forward_val, forward_te, reverse_tr, reverse_val, reverse_te = prepare_bdna_dataset(path=bdna_folder)
    # bdna data:
    forward_cols = ['forward_'+str(i) for i in range(win_size)]
    reverse_cols = ['reverse_'+str(i) for i in range(win_size)]
    cols = forward_cols + reverse_cols + ['label']
    tr = np. concatenate((forward_tr, reverse_tr, np.zeros(forward_tr.shape[0]).reshape(-1, 1)), axis=1)
    val = np. concatenate((forward_val, reverse_val, np.zeros(forward_val.shape[0]).reshape(-1, 1)), axis=1)
    te = np. concatenate((forward_te, reverse_te, np.zeros(forward_te.shape[0]).reshape(-1, 1)), axis=1)
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
        sample_df = nonb_tr_dfs[non_b]
        win_size = (len(list(sample_df.columns.values)) - 7) // 3
        nonb_train_forward = nonb_tr_dfs[non_b].iloc[:, 7: win_size + 7].to_numpy()
        nonb_train_reverse = nonb_tr_dfs[non_b].iloc[:, win_size + 7: (2 * win_size) + 7].to_numpy()
        nonb_train_forward = (nonb_train_forward - train_forward_mu)/train_forward_std
        nonb_train_reverse = (nonb_train_reverse - train_reverse_mu)/train_reverse_std
        nonb_train = np.concatenate((nonb_train_forward, nonb_train_reverse, np.ones(nonb_train_forward.shape[0]).reshape(-1, 1)), axis=1)
        this_tr_df_nonb = pd.DataFrame(data=nonb_train, columns=cols, index=range(nonb_train.shape[0]))
        this_tr_df_nonb['label'] = non_b
        nonb_tr_df = nonb_tr_df.append(this_tr_df_nonb).reset_index(drop=True)
        nonb_val_forward = nonb_val_dfs[non_b].iloc[:, 7: win_size + 7].to_numpy()
        nonb_val_reverse = nonb_val_dfs[non_b].iloc[:,  win_size + 7 : (2 * win_size) + 7].to_numpy()
        nonb_val_forward = (nonb_val_forward - train_forward_mu)/train_forward_std
        nonb_val_reverse = (nonb_val_reverse - train_reverse_mu)/train_reverse_std
        nonb_val = np.concatenate((nonb_val_forward, nonb_val_reverse, np.ones(nonb_val_forward.shape[0]).reshape(-1, 1)), axis=1)
        this_val_df_nonb = pd.DataFrame(data=nonb_val, columns=cols, index=range(nonb_val.shape[0]))
        this_val_df_nonb['label'] = non_b
        nonb_val_df = nonb_val_df.append(this_val_df_nonb).reset_index(drop=True)
        nonb_test_forward = nonb_te_dfs[non_b].iloc[:, 7: win_size + 7].to_numpy()
        nonb_test_reverse = nonb_te_dfs[non_b].iloc[:, win_size + 7:(2 * win_size) + 7].to_numpy()
        nonb_test_forward = (nonb_test_forward - train_forward_mu)/train_forward_std
        nonb_test_reverse = (nonb_test_reverse - train_reverse_mu)/train_reverse_std
        nonb_test = np.concatenate((nonb_test_forward, nonb_test_reverse, np.ones(nonb_test_forward.shape[0]).reshape(-1, 1)), axis=1)
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


def load_data_with_downsample_bdna(bdna_folder, nonb_folder):
    forward_tr, reverse_tr, forward_val, reverse_val, forward_te, reverse_te, \
    train_forward_mu, train_forward_std, train_reverse_mu, train_reverse_std = reprocess_data(bdna_folder)
    win_size = forward_tr.shape[1]
    # forward_tr, forward_val, forward_te, reverse_tr, reverse_val, reverse_te = prepare_bdna_dataset(path=bdna_folder)
    # bdna data:
    forward_cols = ['forward_'+str(i) for i in range(win_size)]
    reverse_cols = ['reverse_'+str(i) for i in range(win_size)]
    cols = forward_cols + reverse_cols + ['label']
    tr = np. concatenate((forward_tr, reverse_tr, np.zeros(forward_tr.shape[0]).reshape(-1, 1)), axis=1)
    val = np. concatenate((forward_val, reverse_val, np.zeros(forward_val.shape[0]).reshape(-1, 1)), axis=1)
    te = np. concatenate((forward_te, reverse_te, np.zeros(forward_te.shape[0]).reshape(-1, 1)), axis=1)
    ####################################################
    tr_df_bdna = pd.DataFrame(data=tr, columns=cols, index=range(tr.shape[0]))
    val_df_bdna = pd.DataFrame(data=val, columns=cols, index=range(val.shape[0]))
    te_df_bdna = pd.DataFrame(data=te, columns=cols, index=range(te.shape[0]))
    tr_df_bdna = tr_df_bdna.sample(frac=0.1).reset_index(drop=True)
    val_df_bdna = val_df_bdna.sample(frac=0.1).reset_index(drop=True)
    te_df_bdna = te_df_bdna.sample(frac=0.1).reset_index(drop=True)
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
        sample_df = nonb_tr_dfs[non_b]
        win_size = (len(list(sample_df.columns.values)) - 7) // 3
        nonb_train_forward = nonb_tr_dfs[non_b].iloc[:, 7: win_size + 7].to_numpy()
        nonb_train_reverse = nonb_tr_dfs[non_b].iloc[:, win_size + 7: (2 * win_size) + 7].to_numpy()
        nonb_train_forward = (nonb_train_forward - train_forward_mu)/train_forward_std
        nonb_train_reverse = (nonb_train_reverse - train_reverse_mu)/train_reverse_std
        nonb_train = np.concatenate((nonb_train_forward, nonb_train_reverse, np.ones(nonb_train_forward.shape[0]).reshape(-1, 1)), axis=1)
        this_tr_df_nonb = pd.DataFrame(data=nonb_train, columns=cols, index=range(nonb_train.shape[0]))
        this_tr_df_nonb['label'] = non_b
        nonb_tr_df = nonb_tr_df.append(this_tr_df_nonb).reset_index(drop=True)
        nonb_val_forward = nonb_val_dfs[non_b].iloc[:, 7: win_size + 7].to_numpy()
        nonb_val_reverse = nonb_val_dfs[non_b].iloc[:,  win_size + 7 : (2 * win_size) + 7].to_numpy()
        nonb_val_forward = (nonb_val_forward - train_forward_mu)/train_forward_std
        nonb_val_reverse = (nonb_val_reverse - train_reverse_mu)/train_reverse_std
        nonb_val = np.concatenate((nonb_val_forward, nonb_val_reverse, np.ones(nonb_val_forward.shape[0]).reshape(-1, 1)), axis=1)
        this_val_df_nonb = pd.DataFrame(data=nonb_val, columns=cols, index=range(nonb_val.shape[0]))
        this_val_df_nonb['label'] = non_b
        nonb_val_df = nonb_val_df.append(this_val_df_nonb).reset_index(drop=True)
        nonb_test_forward = nonb_te_dfs[non_b].iloc[:, 7: win_size + 7].to_numpy()
        nonb_test_reverse = nonb_te_dfs[non_b].iloc[:, win_size + 7:(2 * win_size) + 7].to_numpy()
        nonb_test_forward = (nonb_test_forward - train_forward_mu)/train_forward_std
        nonb_test_reverse = (nonb_test_reverse - train_reverse_mu)/train_reverse_std
        nonb_test = np.concatenate((nonb_test_forward, nonb_test_reverse, np.ones(nonb_test_forward.shape[0]).reshape(-1, 1)), axis=1)
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
    forward_cols = ['forward_'+str(i) for i in range(100)]
    reverse_cols = ['reverse_'+str(i) for i in range(100)]
    cols = forward_cols + reverse_cols + ['label']
    tr = np. concatenate((forward_tr, reverse_tr, np.zeros(forward_tr.shape[0]).reshape(-1, 1)), axis=1)
    val = np. concatenate((forward_val, reverse_val, np.zeros(forward_val.shape[0]).reshape(-1, 1)), axis=1)
    te = np. concatenate((forward_te, reverse_te, np.zeros(forward_te.shape[0]).reshape(-1, 1)), axis=1)
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
        nonb_train_forward = (nonb_train_forward - train_forward_mu)/train_forward_std
        nonb_train_reverse = (nonb_train_reverse - train_reverse_mu)/train_reverse_std
        nonb_train = np.concatenate((nonb_train_forward, nonb_train_reverse, np.ones(nonb_train_forward.shape[0]).reshape(-1, 1)), axis=1)
        this_tr_df_nonb = pd.DataFrame(data=nonb_train, columns=cols, index=range(nonb_train.shape[0]))
        this_tr_df_nonb['label'] = non_b
        nonb_tr_df = nonb_tr_df.append(this_tr_df_nonb).reset_index(drop=True)
        nonb_val_forward = nonb_val_dfs[non_b].iloc[:, 7:107].to_numpy()
        nonb_val_reverse = nonb_val_dfs[non_b].iloc[:, 107:207].to_numpy()
        nonb_val_forward = (nonb_val_forward - train_forward_mu)/train_forward_std
        nonb_val_reverse = (nonb_val_reverse - train_reverse_mu)/train_reverse_std
        nonb_val = np.concatenate((nonb_val_forward, nonb_val_reverse, np.ones(nonb_val_forward.shape[0]).reshape(-1, 1)), axis=1)
        this_val_df_nonb = pd.DataFrame(data=nonb_val, columns=cols, index=range(nonb_val.shape[0]))
        this_val_df_nonb['label'] = non_b
        nonb_val_df = nonb_val_df.append(this_val_df_nonb).reset_index(drop=True)
        nonb_test_forward = nonb_te_dfs[non_b].iloc[:, 7:107].to_numpy()
        nonb_test_reverse = nonb_te_dfs[non_b].iloc[:, 107:207].to_numpy()
        nonb_test_forward = (nonb_test_forward - train_forward_mu)/train_forward_std
        nonb_test_reverse = (nonb_test_reverse - train_reverse_mu)/train_reverse_std
        nonb_test = np.concatenate((nonb_test_forward, nonb_test_reverse, np.ones(nonb_test_forward.shape[0]).reshape(-1, 1)), axis=1)
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


def compute_stats(forward,reverse, std_type='median/iqr'):
    """Aaron function"""
    if std_type == 'mean/std':
        print('Using mean/std\n')
        tfd_mu, tfd_std = forward.mean(), reverse.std()
        trd_mu, trd_std = forward.mean(), reverse.std()
        return (tfd_mu, tfd_std, trd_mu, trd_std)
    elif std_type == 'median/iqr':
        print('Using median/iqr\n')
        iqr_fd = np.subtract(*np.percentile(forward, [75, 25], interpolation='midpoint'))
        iqr_rd = np.subtract(*np.percentile(reverse, [75, 25], interpolation='midpoint'))
        tfd_mu, tfd_std = np.median(forward), iqr_fd
        trd_mu, trd_std = np.median(reverse), iqr_rd
        return (tfd_mu, tfd_std, trd_mu, trd_std)
    else:
        print('Not implemented\n')

def reprocess_data(bdna_path,percent=0.3, std_type= 'median/iqr', seed=42):
    """Aaron function"""
    train_forward = np.load(os.path.join(bdna_path, 'forward_train_100.npy'))
    win_size = train_forward.shape[1]
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
    train_size = int(percent*len(full_bdna_df))
    full_train0, full_val0 = np.array(full_bdna_df[:train_size]), full_bdna_df[train_size:]
    val_size = int(percent*len(full_val0))
    full_val0, full_test0 = np.array(full_val0[:val_size]), np.array(full_val0[val_size:])
    train_forward = full_train0[:, :win_size]
    train_reverse = full_train0[:, win_size:]
    val_forward = full_val0[:, :win_size]
    val_reverse = full_val0[:, win_size:]
    test_forward = full_test0[:, :win_size]
    test_reverse = full_test0[:, win_size:]
    tfd_mu, tfd_std, trd_mu, trd_std = compute_stats(train_forward,train_reverse, std_type=std_type)
    train_forward = (train_forward - tfd_mu)/tfd_std
    train_reverse = (train_reverse - trd_mu)/trd_std
    val_forward = (val_forward - tfd_mu)/tfd_std
    val_reverse = (val_reverse - trd_mu)/trd_std
    test_forward = (test_forward - tfd_mu)/tfd_std
    test_reverse = (test_reverse - trd_mu)/trd_std
    print('New Train Set {:.2f}'.format(len(full_train0)/len(full_bdna)))
    print('New Validation Set {:.2f}'.format(len(full_val0)/len(full_bdna)))
    print('New Test Set {:.2f}'.format(len(full_test0)/len(full_bdna)))
    return train_forward, train_reverse, val_forward, val_reverse, test_forward, test_reverse, tfd_mu, tfd_std, trd_mu, trd_std


def compute_empirical(null_dist, eval_data, tail='two_sided'):
    store_emp_pval = []
    if tail == 'lower':
        for i in range(len(eval_data)):
            temp = stats.percentileofscore(null_dist, eval_data[i])/100.
            store_emp_pval.append(temp)
    elif tail == 'upper':
        for i in range(len(eval_data)):
            temp = 1. - stats.percentileofscore(null_dist, eval_data[i])/100.
            store_emp_pval.append(temp)
    else:
        print('Not defined')
    emp_dist = np.sort(np.array(store_emp_pval))
    indices = np.argsort(np.array(store_emp_pval))
    return emp_dist, indices


def FDR_BHP(dist,alpha=0.5):
    BH_corrected = alpha * np.arange(1,len(dist)+1)/len(dist)
    check = (dist <= BH_corrected)
    if check.sum() != 0:
        valemp = np.max(np.argwhere(check))+1
        FDR = valemp*(1. - alpha)/len(dist)
    else:
        valemp = 0
        FDR = 0
    return valemp, FDR


def plot_histogram(lst, name, save_path):
    plt.cla()
    plt.clf()
    plt.close()
    plt.hist(lst, bins=30)
    plt.xlabel(name)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, name + '.png'))



