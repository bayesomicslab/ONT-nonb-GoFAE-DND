import numpy as np
import pandas as pd


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

    return train_bdna, val_bdna, test_bdna, train_nonb, val_nonb, test_nonb, train_bdna_poison, val_bdna_poison



