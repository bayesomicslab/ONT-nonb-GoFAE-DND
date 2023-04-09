import numpy as np
import sys
import torch
import torch.nn.functional as F
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection


def compute_empirical(null_dist, eval_data, tail='lower'):
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

def compute_accuracy_metrics(tn, fp, fn, tp):
    if (tp + fp)!=0:
        precision = tp / (tp + fp)
    else:
        precision = 0
    if (tp + fn)!=0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    if (precision + recall)!=0:
        fscore = (2 * precision * recall) / (precision + recall)
    else:
        fscore = 0
    return precision, recall, fscore


def compute_true_sim_counts(test_sim, nonb_type, idxs):
    a = test_sim[test_sim["label"]==nonb_type].reset_index(drop=True)
    real_idx = a[a["true_label"]==nonb_type].index.tolist()
    tp_idxs = set(a[a["true_label"]==nonb_type].index.tolist()).intersection(set(idxs))
    fp_idxs = set(a[a["true_label"]=='bdna'].index.tolist()).intersection(set(idxs))
    pred_neg = (set(a.index.tolist()) - set(idxs))
    fn_idxs = set(a[a["true_label"]==nonb_type].index.tolist()).intersection(pred_neg)
    tn_idxs = set(a[a["true_label"]=='bdna'].index.tolist()).intersection(pred_neg)
    return tn_idxs, fp_idxs, fn_idxs, tp_idxs, real_idx


def get_novelty_info(nets, loader, latent_dim, loss_fn_eval, Stiefel, device):
    with torch.no_grad():

        if Stiefel:
            encoder_p1 = nets[0]
            encoder_p1.eval()
            encoder_p2 = nets[1]
            encoder_p2.eval()
            decoder = nets[2]
            decoder.eval()
        else:
            encoder_p1 = nets[0]
            encoder_p1.eval()
            decoder = nets[1]
            decoder.eval()

        store_recon_loss = []

        store_orig = []
        store_recon = []

        store_z_codes = np.empty((0, latent_dim), float)

        for batch_idx, all_val_data in enumerate(loader):

            images = all_val_data[0].to(device)

            #############################################
            # Encode and project for test if needed #
            #############################################
            # Encoder needs to project for univariate tests
            if Stiefel:
                code_s1 = encoder_p1(images)
                p2_out = encoder_p2(code_s1)
            else:
                p2_out = encoder_p1(images)

            x_recon = decoder(p2_out)

            # calculate reconstruction loss
            if loss_fn_eval == 'mse':
                recon_resid = F.mse_loss(x_recon, images, reduction='none')
                recon_loss = torch.sum(recon_resid, dim=1)
            elif loss_fn_eval == 'mae':
                recon_resid = F.l1_loss(x_recon, images, reduction='none')
                recon_loss = torch.sum(recon_resid, dim=(1, 2))
            elif loss_fn_eval == 'smae':
                recon_resid = F.smooth_l1_loss(x_recon, images, reduction='none')
                recon_loss = torch.sum(recon_resid, dim=1)
            elif loss_fn_eval == 'rmse':
                recon_loss = torch.sqrt(torch.sum(F.mse_loss(x_recon, images, reduction='none'), dim=1))
            elif loss_fn_eval == 'max':
                recon_loss = torch.amax(torch.abs(x_recon - images), dim=1)
            else:
                sys.exit("Reconstruction Loss Not Defined")

            # Store the original input
            store_orig.extend(images.detach().cpu().numpy())

            # Store the reconstruction
            store_recon.extend(x_recon.detach().cpu().numpy())

            # Store the total reconstruction loss per observation
            store_recon_loss.extend(recon_loss.detach().cpu().numpy())

            # Store the encodings
            store_z_codes = np.append(store_z_codes, p2_out.detach().cpu().numpy(), axis=0)

    return store_z_codes, np.array(store_recon_loss), (np.array(store_orig), np.array(store_recon))


class NoveltyStats:
    def __init__(self, bdna_data, nonb_data, Xval, fdr_level, poison_val_count):
        self.bdna_data = bdna_data
        self.nonb_data = nonb_data
        self.Xval = Xval
        self.fdr_level = fdr_level
        self.poison_val_count = poison_val_count

        # top poison_val_count are bdna
        self.bdna_indices = pd.DataFrame(Xval.reshape(Xval.shape[0], -1)).iloc[:poison_val_count].index.to_list()
        self.nonb_indices = pd.DataFrame(Xval.reshape(Xval.shape[0], -1)).iloc[poison_val_count:].index.to_list()

    def get_ns(self, tail):
        emp_dist, sorted_idx = compute_empirical(self.bdna_data, self.nonb_data, tail)
        rejected, _ = fdrcorrection(emp_dist, alpha=self.fdr_level, method='indep', is_sorted=True)

        disc_bdna = set(sorted_idx[rejected]).intersection(set(self.bdna_indices))
        disc_nonb = set(sorted_idx[rejected]).intersection(set(self.nonb_indices))

        return emp_dist, sorted_idx, rejected, disc_bdna, disc_nonb


def get_union(idx1, idx2):
    return set(idx1).union(set(idx2))


def get_intersection(idx1, idx2):
    return set(idx1).intersection(set(idx2))


def get_symdif(idx1, idx2):
    return set(idx1).symmetric_difference(set(idx2))


def get_jaccard(idx1, idx2):
    return len(get_intersection(idx1, idx2)) / len(get_union(idx1, idx2)) if len(get_union(idx1, idx2)) > 0 else 0


def get_idx(selection='union', data='all'):
    expression = str(selection) + '_' + str(data) + '_U'
    return list(eval(expression))


def disc_ratio(disc_bdna, disc_nonb):
    return len(disc_nonb) / max(1, len(disc_bdna))

