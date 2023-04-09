import numpy as np
import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
import sys
from torchvision.transforms import transforms

torch.manual_seed(42)


def compute_stats(forward,reverse, std_type='median/iqr'):
    if std_type == 'mean/std':
        tfd_mu, tfd_std = forward.mean(), reverse.std()
        trd_mu, trd_std = forward.mean(), reverse.std()
        return (tfd_mu, tfd_std, trd_mu, trd_std)
    elif std_type == 'median/iqr':
        iqr_fd = np.subtract(*np.percentile(forward, [75, 25], interpolation='midpoint'))
        iqr_rd = np.subtract(*np.percentile(reverse, [75, 25], interpolation='midpoint'))
        tfd_mu, tfd_std = np.median(forward), iqr_fd
        trd_mu, trd_std = np.median(reverse), iqr_rd
        return (tfd_mu, tfd_std, trd_mu, trd_std)
    else:
        print('Not implemented\n')


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


def import_data(path, nonb_type, poison_train=False, poison_val=True):
    train = pd.read_csv(path + str(nonb_type) + '_train.csv', index_col=0)
    val = pd.read_csv(path + str(nonb_type) + '_validation.csv', index_col=0)
    test = pd.read_csv(path + str(nonb_type) + '_test.csv', index_col=0)
    train_bdna = train[train["label"] == 'bdna'].drop(['label'], axis=1).to_numpy()
    val_bdna = val[val["label"] == 'bdna'].drop(['label'], axis=1).to_numpy()
    test_bdna = test[test["label"] == 'bdna'].drop(['label'], axis=1).to_numpy()
    train_nonb = train[train["label"] == nonb_type].drop(['label'], axis=1).to_numpy()
    val_nonb = val[val["label"] == nonb_type].drop(['label'], axis=1).to_numpy()
    test_nonb = test[test["label"] == nonb_type].drop(['label'], axis=1).to_numpy()

    train_bdna_poison = []
    val_bdna_poison = []

    if poison_train:
        poison_train_count = len(train_nonb)
        train_bdna_poison = train_bdna[:poison_train_count]
        train_bdna = train_bdna[poison_train_count:]

        train_bdna_poison = np.array([train_bdna_poison[:, :50], train_bdna_poison[:, 50:]]).transpose(1, 0, 2)

    if poison_val:
        poison_val_count = len(val_nonb)
        val_bdna_poison = val_bdna[:poison_val_count]
        val_bdna = val_bdna[poison_val_count:]

        train_bdna = np.array([train_bdna[:, :50], train_bdna[:, 50:]]).transpose(1, 0, 2)
        val_bdna = np.array([val_bdna[:, :50], val_bdna[:, 50:]]).transpose(1, 0, 2)
        test_bdna = np.array([test_bdna[:, :50], test_bdna[:, 50:]]).transpose(1, 0, 2)

        train_nonb = np.array([train_nonb[:, :50], train_nonb[:, 50:]]).transpose(1, 0, 2)
        val_nonb = np.array([val_nonb[:, :50], val_nonb[:, 50:]]).transpose(1, 0, 2)
        test_nonb = np.array([test_nonb[:, :50], test_nonb[:, 50:]]).transpose(1, 0, 2)

        val_bdna_poison = np.array([val_bdna_poison[:, :50], val_bdna_poison[:, 50:]]).transpose(1, 0, 2)

    return train_bdna, val_bdna, test_bdna, train_nonb, val_nonb, test_nonb, train_bdna_poison, val_bdna_poison


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
    # all_bdna = pd.concat([tr_df_bdna, val_df_bdna, te_df_bdna], ignore_index=True)
    # all_bdna = all_bdna.sample(frac=1, random_state=42).reset_index(drop=True)
    # all_bdna_cut = all_bdna.loc[0: n_bdna - 1]
    # n_bdna_tr = int(n_bdna * 30 / 100)
    # n_bdna_val = int(n_bdna * 20 / 100)
    # # n_bdna_te = int(n_bdna*10/100)
    # tr_bdna = all_bdna_cut.loc[0:n_bdna_tr - 1].reset_index(drop=True)
    # val_bdna = all_bdna_cut.loc[n_bdna_tr:n_bdna_tr + n_bdna_val - 1].reset_index(drop=True)
    # te_bdna = all_bdna_cut.loc[n_bdna_tr + n_bdna_val:].reset_index(drop=True)
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
    # train = tr_df_bdna.append(nonb_tr_df).sample(frac=1, random_state=42).reset_index(drop=True)
    # val = val_df_bdna.append(nonb_val_df).sample(frac=1, random_state=42).reset_index(drop=True)
    # test = te_df_bdna.append(nonb_te_df).sample(frac=1, random_state=42).reset_index(drop=True)
    # nonb_indices = train[train['label'] != 'bdna'].index
    # nonbinb_idx = random.sample(list(nonb_indices), nbinb)
    # train.loc[nonbinb_idx, 'label'] = 'bdna'
    return train, val, test


def reprocess_data2(nonb_type, train, val, test, poison_train=False, poison_val=True):
    train_bdna = train[train["label"] == 'bdna'].drop(['label', 'true_label'], axis=1).to_numpy()
    val_bdna = val[val["label"] == 'bdna'].drop(['label', 'true_label'], axis=1).to_numpy()
    test_bdna = test[test["label"] == 'bdna'].drop(['label', 'true_label'], axis=1).to_numpy()
    train_nonb = train[train["label"] == nonb_type].drop(['label', 'true_label'], axis=1).to_numpy()
    val_nonb = val[val["label"] == nonb_type].drop(['label', 'true_label'], axis=1).to_numpy()
    test_nonb = test[test["label"] == nonb_type].drop(['label', 'true_label'], axis=1).to_numpy()

    train_bdna_poison = []
    val_bdna_poison = []

    if poison_train:
        poison_train_count = len(train_nonb)
        train_bdna_poison = train_bdna[:poison_train_count]
        train_bdna = train_bdna[poison_train_count:]

        train_bdna_poison = np.array([train_bdna_poison[:, :50], train_bdna_poison[:, 50:]]).transpose(1, 0, 2)

    if poison_val:
        poison_val_count = len(val_nonb)
        val_bdna_poison = val_bdna[:poison_val_count]
        val_bdna = val_bdna[poison_val_count:]

        train_bdna = np.array([train_bdna[:, :50], train_bdna[:, 50:]]).transpose(1, 0, 2)
        val_bdna = np.array([val_bdna[:, :50], val_bdna[:, 50:]]).transpose(1, 0, 2)
        test_bdna = np.array([test_bdna[:, :50], test_bdna[:, 50:]]).transpose(1, 0, 2)

        train_nonb = np.array([train_nonb[:, :50], train_nonb[:, 50:]]).transpose(1, 0, 2)
        val_nonb = np.array([val_nonb[:, :50], val_nonb[:, 50:]]).transpose(1, 0, 2)
        test_nonb = np.array([test_nonb[:, :50], test_nonb[:, 50:]]).transpose(1, 0, 2)

        val_bdna_poison = np.array([val_bdna_poison[:, :50], val_bdna_poison[:, 50:]]).transpose(1, 0, 2)
        # X_train = X_train.transpose(1,0,2) # [sample, strand, position] = [sample, forward/reverse, 100]

    return train_bdna, val_bdna, test_bdna, train_nonb, val_nonb, test_nonb, train_bdna_poison, val_bdna_poison



