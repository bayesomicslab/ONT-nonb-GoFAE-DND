import numpy as np
import os
import pandas as pd
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from hypotests.CoreTests import SWF_weights
from utilities.TestUtilities import check_test, testsetup
from core.Manifolds import replacegrad, retract2manifold, sample_Stiefel
from core.Sampling import samples #, est_image_recon,
from hypotests.CoreTests import CramerVonMises, KolmogorovSmirnov, \
    ShapiroFrancia, ShapiroWilk, HenzeZirkler, MardiaSkew, Royston, EppsPulley1
from hypotests.UnifTests import ChiSquareUnif, KolmogorovSmirnovUnif
from utilities.IOUtilities import savemodel, eprint
from architecture.Encoders import Encoder_P1, Encoder_P2
from architecture.Decoders import Decoder
#from core.Dataset import load_dataset
from utilities.IOUtilities import save_data, save_summary
from core.Losses import discriminative_loss
from core.Dataset import import_data, load_data2, reprocess_data2
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from core.Evaluation import get_novelty_info, NoveltyStats, get_union, get_intersection, get_symdif, \
    get_jaccard, get_idx, disc_ratio, compute_empirical , compute_true_sim_counts, compute_accuracy_metrics
from sklearn.covariance import MinCovDet
from scipy import stats
from core.Plotting import trace_plots, plot_test, plot_test_sim, plot_test_sim_split
from statsmodels.stats.multitest import fdrcorrection
from matplotlib import pyplot as plt


class GoFAEDND():

    def __init__(self, device, args):

        self.device = device
        self.uvn = {'sw', 'sf', 'ad', 'cvm', 'ks', 'ep1', 'ep2'}
        self.mvn = {'hz', 'mardia_skew', 'royston'}
        self.min_test_stats = {'hz', 'ad', 'cvm', 'mardia_skew', 'royston', 'ks', 'ep2'}
        self.max_test_stats = {'sf', 'sw', 'ep1'}
        self.data_type = args.data_type
        self.nonb_type = args.nonb_type
        self.nonb_ratio = args.nonb_ratio
        self.test = args.test
        self.test_for_training = self.test
        self.new_stat = args.new_stat
        self.use_proj = args.use_proj
        self.n_z = args.n_z
        self.latent_dim_proj = args.latent_dim_proj
        self.num_projections = args.num_projections
        self.batch_size = args.batch_size
        self.outlier_batch_size = args.outlier_batch_size
        self.alpha = args.alpha
        self.epochs = args.epochs
        self.use_labels = args.use_labels
        self.poison_train = args.poison_train
        self.poison_val = args.poison_val
        self.use_rec_MH = args.use_rec_MH
        self.s1_pow = args.s1_pow
        self.s2_pow = args.s2_pow
        self.fdr_level = args.fdr_level
        self.unblock_test = args.unblock_test
        self.use_test_rob_mean_cov = args.use_test_rob_mean_cov
        self.inform_plots = args.inform_plots
        self.loss_fn = args.loss_fn
        self.discriminative_loss_fn = args.discriminative_loss_fn
        self.loss_fn_eval = args.loss_fn_eval
        self.discriminative_weight = args.discriminative_weight
        self.lambda_alpha = args.lambda_alpha
        self.omega = args.omega
        self.Stiefel = args.Stiefel
        self.lr_e_d = args.lr_e_d
        self.lr_adam_enc_p1 = args.lr_e_d  # double check this
        self.lr_cycle = args.lr_e_d
        self.lr_sgd = 5*args.lr_e_d
        self.beta1_enc_p1 = args.beta1_enc_p1
        self.beta2_enc_p1 = args.beta2_enc_p1
        self.lr_adam_dec = args.lr_e_d
        self.beta1_dec = args.beta1_dec
        self.beta2_dec = args.beta2_dec
        self.h_dim = args.h_dim
        self.momentum = args.momentum
        self.ncv = args.ncv
        self.num_workers = args.num_workers
        self.experiment = args.experiment
        self.seed = args.seed
        self.diststat_path = args.diststat_path
        self.output_path = args.output_path
        self.exp_data_path = args.exp_data_path
        self.sim_data_path = args.sim_data_path

        if self.data_type == 'experimental':
            self.path = os.path.join(self.output_path, str(self.test), str(self.data_type), 'experiment_'+str(self.experiment),
                                     'use_'+str(self.new_stat))
        elif self.data_type == 'simulated':
            self.path = os.path.join(self.output_path, str(self.test), str(self.data_type),'nonb_ratio_'+str(self.nonb_ratio),
                                     'experiment_'+str(self.experiment), 'use_'+str(self.new_stat))
        else:
            sys.exit('Choose either the \'experimental\' or \'simulated\' dataset.')

        if not os.path.exists(self.path):
            Path(self.path).mkdir(parents=True, exist_ok=True)

        self.img_path_all = os.path.join(self.path, 'imgs')
        if not os.path.exists(self.img_path_all):
            Path(self.img_path_all).mkdir(parents=True, exist_ok=True)

        self.nonb_path = os.path.join(self.path,str(self.nonb_type))
        if not os.path.exists(self.nonb_path):
            Path(self.nonb_path).mkdir(parents=True, exist_ok=True)

        self.img_path = os.path.join(self.nonb_path, 'imgs')
        if not os.path.exists(self.img_path):
            Path(self.img_path).mkdir(parents=True, exist_ok=True)

        print('Get data', flush=True)
        if self.data_type == 'experimental':
            self.train_bdna, self.val_bdna, self.test_bdna, self.train_nonb, self.val_nonb, \
            self.test_nonb, self.train_bdna_poison, self.val_bdna_poison = import_data(self.exp_data_path, nonb_type=self.nonb_type,
                                                                        poison_train=self.poison_train,
                                                                        poison_val=self.poison_val)
        elif self.data_type == 'simulated':
            self.train_sim, self.val_sim, self.test_sim = load_data2(folder=self.sim_data_path, win_size=50, n_bdna=200_000, n_nonb=20_000,
                                                      nonb_ratio=self.nonb_ratio)
            self.train_bdna, self.val_bdna, self.test_bdna, self.train_nonb, self.val_nonb, \
            self.test_nonb, self.train_bdna_poison, self.val_bdna_poison = reprocess_data2(nonb_type=self.nonb_type, train=self.train_sim,
                                                                            val=self.val_sim, test=self.test_sim, \
                                                                            poison_train=self.poison_train,
                                                                            poison_val=self.poison_val)
        else:
            sys.exit("Dataset doesn't exist")

        print("Data processed\n", flush=True)

        if self.use_labels:
            print('Number of training {}: {}'.format(self.nonb_type, self.train_nonb.shape[0]))
            print('Number of training BDNA: {}'.format(self.train_bdna.shape[0]))

            if self.poison_train:
                self.X_train_outlier = np.vstack((self.train_bdna_poison, self.train_nonb))
            else:
                self.X_train_outlier = self.train_nonb
            print('Number of training BDNA transferred as poison into {}: {}'.format(self.nonb_type, len(self.train_bdna_poison)))

            self.outlier_label_train = -np.ones((self.X_train_outlier.shape[0]))
            self.real_label_train = np.ones((self.train_bdna.shape[0]))

            self.outlier_dataset_train = TensorDataset(torch.tensor(self.X_train_outlier.astype(np.float32)), \
                                                  torch.tensor(self.outlier_label_train.astype(np.float32)))
            self.train_dataset = TensorDataset(torch.tensor(self.train_bdna.astype(np.float32)), \
                                          torch.tensor(self.real_label_train.astype(np.float32)))

            self.test_dataset = TensorDataset(torch.tensor(self.test_bdna.astype(np.float32)))

            # nonb
            self.train_outlier_loader = DataLoader(dataset=self.outlier_dataset_train, batch_size=self.outlier_batch_size, \
                                              pin_memory=True, num_workers=self.num_workers, shuffle=True, drop_last=True)
            # bdna
            self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, pin_memory=True, \
                                      num_workers=self.num_workers, shuffle=True, drop_last=True)

            self.train_loader_fixed = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, pin_memory=True, \
                                            num_workers=self.num_workers, shuffle=False, drop_last=True)

            # bdna
            self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, pin_memory=True, \
                                     num_workers=self.num_workers, shuffle=False, drop_last=False)
        else:
            self.train_dataset = TensorDataset(torch.tensor(self.train_bdna.astype(np.float32)))

            self.test_dataset = TensorDataset(torch.tensor(self.test_bdna.astype(np.float32)))

            # bdna
            self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, pin_memory=True, \
                                      num_workers=self.num_workers, shuffle=True, drop_last=True)

            # bdna
            self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, pin_memory=True, \
                                     num_workers=self.num_workers, shuffle=False, drop_last=False)

        # Validation Set
        if self.poison_val:
            self.X_val_outlier = np.vstack((self.val_bdna_poison, self.val_nonb))
        else:
            self.X_val_outlier = self.val_nonb

        print(
            'Number of validation BDNA transferred into {} validation set: {}'.format(self.nonb_type, len(self.val_bdna_poison)))
        print('Number of BNDA in validation set for null distribution: {}'.format(len(self.val_bdna)))

        self.outlier_dataset_val = TensorDataset(torch.tensor(self.X_val_outlier.astype(np.float32)))
        self.outlier_loader_val = DataLoader(dataset=self.outlier_dataset_val, batch_size=self.batch_size, pin_memory=True,
                                        num_workers=self.num_workers, shuffle=False, drop_last=False)

        self.val_dataset = TensorDataset(torch.tensor(self.val_bdna.astype(np.float32)))

        # bdna
        self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, pin_memory=True, \
                                num_workers=self.num_workers, shuffle=False, drop_last=True)

        # Testing Set
        self.outlier_dataset_test = TensorDataset(torch.tensor(self.test_nonb.astype(np.float32)))
        self.outlier_loader_test = DataLoader(dataset=self.outlier_dataset_test, batch_size=self.batch_size, pin_memory=True,
                                         num_workers=self.num_workers, shuffle=False, drop_last=False)

        print('Number of BNDA in test set for null distribution: {}'.format(len(self.test_bdna)))
        print('Number of {} in test set for evaluation: {}'.format(self.nonb_type, len(self.outlier_dataset_test)))

        print("Dataloaders created\n", flush=True)


    def get_stat_info(self, path, use_proj, test, n_z, batch_size, latent_dim_proj, num_projections, new_stat):
        # path: specifies path to empirical distribution information
        # use_proj: whether or not projection is used
        # test: which test to use (see possiblities)
        # n_z: dimension of latent variable
        # batch_size: size of batch, needed for calculating correct test statistic and p-values
        # latent_dim_proj: the dimensionality of the latent space after projecting it (for univariate tests, project to 1d)
        # num_projections: how many projections will be used when calculating the new statistic
        # alpha: if no projections are used, then alpha is needed to select the correct T_alpha (critical value)

        if use_proj:
            #emp_dist_path = os.path.join(path,
            #                test + '/latent_dim_' + str(n_z) + '/latent_dim_proj_' + str(
            #    latent_dim_proj) + '/batch_size_' + str(self.batch_size) + \
            #                '/inner_sim_' + str(num_projections)

            emp_dist_path = os.path.join(self.diststat_path, str(test), 'latent_dim_'+str(n_z),
                                         'latent_dim_proj_'+str(latent_dim_proj),
                                         'batch_size_'+str(self.batch_size), 'inner_sim_'+str(num_projections))

            # Tests using projection
            if test in self.max_test_stats and new_stat == 'min':

                emp_dist_path1 = os.path.join(emp_dist_path, 'emp_dist_min.csv')  # select path for the min empirical dist
                emp_dist = np.array(pd.read_csv(emp_dist_path1))  # import min emp dist
                T_alpha = np.quantile(emp_dist, self.alpha).astype(np.float32)  # Grab the correct T_alpha
                return emp_dist, T_alpha

            elif test in self.max_test_stats and new_stat == 'avg':

                emp_dist_path1 = os.path.join(emp_dist_path, 'emp_dist_avg.csv')  # select path for the avg empirical dist
                emp_dist = np.array(pd.read_csv(emp_dist_path1))  # import avg emp dist
                T_alpha = np.quantile(emp_dist, self.alpha).astype(np.float32)  # Grab the correct T_alpha
                return emp_dist, T_alpha

            elif test in self.min_test_stats and new_stat == 'max':

                emp_dist_path1 = os.path.join(emp_dist_path, 'emp_dist_max.csv')  # select path for the max empirical dist
                emp_dist = np.array(pd.read_csv(emp_dist_path1))  # import max emp dist
                T_alpha = np.quantile(emp_dist, 1 - self.alpha).astype(np.float32)  # Grab the correct T_alpha
                return emp_dist, T_alpha

            elif test in self.min_test_stats and new_stat == 'avg':

                emp_dist_path1 = os.path.join(emp_dist_path, 'emp_dist_avg.csv')  # select path for the avg empirical dist
                emp_dist = np.array(pd.read_csv(emp_dist_path1))  # import avg emp dist
                T_alpha = np.quantile(emp_dist, 1. - self.alpha).astype(np.float32)  # Grab the correct T_alpha
                return emp_dist, T_alpha
            else:
                sys.exit("Configuration incorrect for test statistic")
        else:
            if test in self.mvn and new_stat == 'none':
                #emp_cv_path = path + \
                #              test + '/latent_dim_' + str(n_z) + '/emp_cv_alpha_' + str(self.alpha) + '.csv.gz'

                emp_cv_path = os.path.join(self.diststat_path, str(test), 'latent_dim_' + str(n_z),
                                           'emp_cv_alpha_'+str(self.alpha) + '.csv')

                # import empirical cv
                T_alpha = np.array(pd.read_csv(emp_cv_path))[0]
                return None, T_alpha
            else:
                sys.exit("Configuration incorrect for test statistic")

    def run(self):
        # Encoder/Decoder
        # 3 modules, 2 encoders and 1 decoder

        encoder_p1 = Encoder_P1(n_inputs=2, out_dim=self.n_z, h_dim=self.h_dim, Stiefel=self.Stiefel).to(
            self.device)

        if self.Stiefel:
            encoder_p2 = Encoder_P2(in_dim=self.h_dim * 24, z_dim=self.n_z).to(self.device)  # 1536

        decoder = Decoder(n_inputs=self.n_z, n_outputs=2, h_dim=self.h_dim).to(self.device)

        htest_map = self.generate_test_map()

        test_dictionary = self.get_test_param_dict(self.device, self.test, self.new_stat)
        emp_dist, T_alpha = self.get_stat_info(path=self.diststat_path, use_proj=self.use_proj, test=self.test,
                                               n_z=self.n_z,batch_size=self.batch_size, latent_dim_proj=self.latent_dim_proj,
                                               num_projections=self.num_projections, new_stat=self.new_stat)

        self.hypothesis_test = testsetup(self.test, emp_dist, test_dictionary)

        if self.use_proj:
            check_test(self.hypothesis_test, self.latent_dim_proj)

        ##########
        # Optimizers
        ##########
        # Encoder Optimizer
        enc_p1_optim = optim.Adam(encoder_p1.parameters(), lr=self.lr_adam_enc_p1, betas=(self.beta1_enc_p1, self.beta2_enc_p1))
        dec_optim = optim.Adam(decoder.parameters(), lr=self.lr_adam_dec, betas=(self.beta1_dec, self.beta2_dec))

        # Schedulers for Encoder/Decoder
        enc_p1_scheduler = ReduceLROnPlateau(enc_p1_optim, mode='min', factor=0.5, patience=5,
                                             verbose=True)  # change factor to .1 (was .5)
        dec_scheduler = ReduceLROnPlateau(dec_optim, mode='min', factor=0.5, patience=5,
                                          verbose=True)  # change factor to .1 (was .5)

        if self.Stiefel:
            enc_p2_optim = optim.SGD(encoder_p2.parameters(), lr=self.lr_sgd)
            enc_p2_scheduler = optim.lr_scheduler.OneCycleLR(enc_p2_optim, self.lr_cycle, epochs=self.epochs,
                                                             steps_per_epoch=len(self.train_loader))

        # track training
        track_train_recon = []
        track_train_tstat = []
        track_train_pval = []
        track_train_total_loss = []

        track_train_recon_bdna = []
        track_train_recon_nonb = []

        # track validation
        track_val_recon = []
        track_val_tstat = []
        track_val_pval = []
        track_val_total_loss = []

        # track testing
        track_test_recon = []
        track_test_tstat = []
        track_test_pval = []
        track_test_total_loss = []

        best_recon = float("inf")
        start_time_train = time.time()

        # intialize latent stat monitoring
        running_mean = np.zeros((self.n_z, 1))
        running_cov = np.zeros((self.n_z, self.n_z))

        print('Begin Training\n', flush=True)
        for epoch in range(self.epochs):

            if self.use_labels:
                outlier_iterator = iter(self.train_outlier_loader)
            step = 0
            temp_train_recon = []
            temp_train_recon_bdna = []
            temp_train_recon_nonb = []

            temp_train_tstat = []
            temp_train_pval = []
            temp_train_total_loss = []

            for batch_idx, all_data in enumerate(self.train_loader):

                if self.use_labels:

                    try:
                        o_data = next(outlier_iterator)
                    except StopIteration:
                        outlier_iterator = iter(self.train_outlier_loader)
                        o_data = next(outlier_iterator)
                    images = all_data[0].to(self.device)
                    real_labs = all_data[1].to(self.device)
                    o_data_img = o_data[0].to(self.device)
                    o_data_labs = o_data[1].to(self.device)

                else:
                    images = all_data[0].to(self.device)
                    # noisy_images = dropout(torch.ones(images.shape, device=device)) * images
                # print(images.shape)
                # print(o_data_img.shape)

                onebatch_start_time = time.time()
                if (step % 100) == 0:
                    hundbatch_start_time = time.time()

                if self.use_labels and self.Stiefel:
                    ### bdna
                    code_s1 = encoder_p1(images)
                    p2_out = encoder_p2(code_s1)
                    ### nonb
                    code_s1_o = encoder_p1(o_data_img)
                    p2_out_o = encoder_p2(code_s1_o)
                elif self.use_labels and not self.Stiefel:
                    p2_out = encoder_p1(images)
                    p2_out_o = encoder_p1(o_data_img)
                elif (not self.use_labels) and self.Stiefel:
                    code_s1 = encoder_p1(images)
                    p2_out = encoder_p2(code_s1)
                elif (not self.use_labels) and not self.Stiefel:
                    p2_out = encoder_p1(images)

                code_mean = np.mean(p2_out.detach().cpu().numpy(), axis=0).reshape(-1,
                                                                                   1)  # change to using dbn change back after
                code_cov = np.cov(p2_out.detach().cpu().numpy(), rowvar=False).reshape(self.n_z, self.n_z)

                running_mean = (1. - self.momentum) * running_mean + self.momentum * code_mean
                running_cov = (1. - self.momentum) * running_cov + self.momentum * code_cov

                if self.test in self.uvn:
                    temp = []  # empty list
                    stiefel_sample = sample_Stiefel(self.n_z, self.num_projections, self.device)  # Sample Stiefel
                    projected = torch.matmul(p2_out, stiefel_sample)
                    for direction in range(self.num_projections):
                        temp_stat = self.hypothesis_test.teststat(x=projected[:, direction])
                        temp.append(temp_stat.view(-1, 1))
                # Option for multivariate tests
                elif self.test in self.mvn:
                    # multivariate tests can be projected
                    if self.use_proj:
                        temp = []
                        for direction in range(self.num_projections):
                            stiefel_sample = sample_Stiefel(self.n_z, self.latent_dim_proj, self.device)
                            projected = torch.matmul(p2_out, stiefel_sample)
                            if self.test == 'royston':
                                temp_stat, H_e = self.hypothesis_test.teststat(x=projected)
                            else:
                                temp_stat = self.hypothesis_test.teststat(x=projected)
                            temp.append(temp_stat.view(-1, 1))
                    # multivariate tests don't need to be projected
                    else:
                        if self.test == 'royston':
                            temp_stat, H_e = self.hypothesis_test.teststat(x=p2_out)
                        else:
                            temp_stat = self.hypothesis_test.teststat(x=p2_out)
                else:
                    sys.exit("Test not implemented")

                #############################################
                # Calculate final test statistic #
                #############################################
                if self.test in self.max_test_stats and self.new_stat == 'min':
                    out_tensor = torch.cat(temp, dim=1)
                    stat = torch.min(out_tensor)
                elif self.test in self.max_test_stats and self.new_stat == 'avg':
                    out_tensor = torch.cat(temp, dim=1)
                    stat = torch.mean(out_tensor)
                elif self.test in self.min_test_stats and self.new_stat == 'max':
                    out_tensor = torch.cat(temp, dim=1)
                    stat = torch.max(out_tensor)
                elif self.test in self.min_test_stats and self.new_stat == 'avg':
                    out_tensor = torch.cat(temp, dim=1)
                    stat = torch.mean(out_tensor)
                elif self.test in self.mvn and self.new_stat == 'none':
                    stat = temp_stat
                else:
                    sys.exit("Configuration incorrect for test statistic")

                #############################################
                # Calculate P-values #
                #############################################
                pval = self.hypothesis_test.pval(inputs=stat)

                #############################################
                # Decode #
                #############################################

                if self.use_labels and self.loss_fn == 'discriminative':
                    dec_input = torch.cat((p2_out, p2_out_o), dim=0)
                    all_dat_labs = torch.cat((real_labs, o_data_labs), dim=0)
                    x_true = torch.cat((images, o_data_img), dim=0)
                else:
                    dec_input = p2_out
                    x_true = images

                # print(dec_input.shape)
                x_recon = decoder(dec_input)

                # calculate reconstruction loss
                if self.loss_fn == 'mse':
                    recon_loss = F.mse_loss(x_recon, x_true)  # , reduction='sum').div(batch_size)
                elif self.loss_fn == 'mae':
                    recon_loss = F.l1_loss(x_recon, x_true)
                elif self.loss_fn == 'smae':
                    recon_loss = F.smooth_l1_loss(x_recon, x_true)
                elif self.loss_fn == 'max':
                    recon_loss = torch.mean(torch.amax(torch.abs(x_recon - x_true), dim=1))
                elif self.loss_fn == 'rmse':
                    recon_loss = torch.mean(torch.sqrt(torch.sum(F.mse_loss(x_recon, x_true, reduction='none'), dim=1)))
                elif self.loss_fn == 'discriminative':
                    # loss, bdna_loss, nonb_loss
                    recon_loss, bdna_recon_loss, nonb_recon_loss = discriminative_loss(x_true, x_recon, all_dat_labs,
                                                                                    loss_function=self.discriminative_loss_fn,
                                                                                    device=self.device,
                                                                                    weight=self.discriminative_weight,
                                                                                    omega=self.omega)
                else:
                    sys.exit("Reconstruction Loss Not Defined")

                if self.hypothesis_test.optdir == 1:
                    full_loss = recon_loss + self.lambda_alpha * (-stat)
                else:
                    full_loss = recon_loss + self.lambda_alpha * stat

                # Average Statistics for Training Set
                temp_train_recon.append(recon_loss.data.item())
                temp_train_recon_bdna.append(bdna_recon_loss.data.item())
                temp_train_recon_nonb.append(nonb_recon_loss.data.item())
                temp_train_tstat.append(stat.data.item())
                temp_train_pval.append(pval)
                temp_train_total_loss.append(full_loss.data.item())

                encoder_p1.zero_grad()

                if self.Stiefel:
                    encoder_p2.zero_grad()
                decoder.zero_grad()

                full_loss.backward()

                enc_p1_optim.step()
                if self.Stiefel:
                    euclid_grad_norm, riem_grad_norm = replacegrad(encoder_p2)
                    enc_p2_optim.step()
                dec_optim.step()

                if self.Stiefel:
                    retract2manifold(encoder_p2)
                    enc_p2_scheduler.step()  # moved this to after retraction

                onebatch_end_time = time.time()
                onebatch_total_time = (onebatch_end_time - onebatch_start_time)

                if (step + 1) % 100 == 0:
                    hundbatch_end_time = time.time()
                    hundbatch_total_time = (hundbatch_end_time - hundbatch_start_time)

                    print(
                        "Epoch: [%d/%d], Step: [%d/%d], Recon Loss: %.4f, BDNA Recon Loss: %.4f, NonB Recon Loss: %.4f, %s: %.4f, Pval: %.4f, Single Batch Time: %.4f, 100 Batch Time: %.4f" %
                        (epoch + 1, self.epochs, step + 1, len(self.train_loader), recon_loss.data.item(),
                         bdna_recon_loss.data.item(), nonb_recon_loss.data.item(),
                         self.test.upper(), stat.data.item(), pval, onebatch_total_time, hundbatch_total_time), flush=True)

                step += 1

            track_train_recon.append(np.mean(np.array(temp_train_recon)))
            track_train_recon_bdna.append(np.mean(np.array(temp_train_recon_bdna)))
            track_train_recon_nonb.append(np.mean(np.array(temp_train_recon_nonb)))
            track_train_tstat.append(np.mean(np.array(temp_train_tstat)))
            track_train_pval.append(np.mean(np.array(temp_train_pval)))
            track_train_total_loss.append(np.mean(np.array(temp_train_total_loss)))

            # train_loader_fixed
            with torch.no_grad():
                encoder_p1.eval()
                decoder.eval()
                if self.Stiefel:
                    encoder_p2.eval()
                    train_rec_E, train_tstat_E, train_pval_E, train_total_loss_E = self.evaluate(
                        nets=(encoder_p1, encoder_p2, decoder), \
                        evaluate_loader=self.train_loader_fixed, n_z=self.n_z, use_proj=self.use_proj, \
                        latent_dim_proj=self.latent_dim_proj, num_projections=self.num_projections, loss_fn_eval=self.loss_fn_eval, \
                        htest=self.hypothesis_test, batch_size=self.batch_size, device=self.device)
                else:
                    train_rec_E, train_tstat_E, train_pval_E, train_total_loss_E = self.evaluate(nets=(encoder_p1, decoder), \
                                                                                            evaluate_loader=self.train_loader_fixed,
                                                                                            n_z=n_z, use_proj=self.use_proj, \
                                                                                            latent_dim_proj=self.latent_dim_proj,
                                                                                            num_projections=self.num_projections,
                                                                                            loss_fn_eval=self.loss_fn_eval, \
                                                                                            htest=self.hypothesis_test,
                                                                                            batch_size=self.batch_size,
                                                                                                 device=self.device)

            print('Training Set:  Recon: {:.4f}, Tstat: {:.4f}, P-value: {:.4f}\n'.format(np.mean(train_rec_E),
                                                                                          train_tstat_E, train_pval_E),
                  flush=True)

            ############################# Evaluate on Validation ####################################################

            if (epoch + 1) % self.ncv == 0:

                print('Evaluating Validation\n', flush=True)
                with torch.no_grad():
                    encoder_p1.eval()
                    decoder.eval()
                    if self.Stiefel:
                        encoder_p2.eval()
                        # Validation Set (Used for hyperparameter tuning and for Learning Scheduler)
                        val_rec, val_tstat, val_pval, val_total_loss = self.evaluate(nets=(encoder_p1, encoder_p2, decoder),
                                                                                evaluate_loader=self.val_loader, n_z=self.n_z,
                                                                                use_proj=self.use_proj, \
                                                                                latent_dim_proj=self.latent_dim_proj,
                                                                                num_projections=self.num_projections,
                                                                                loss_fn_eval=self.loss_fn_eval, \
                                                                                htest=self.hypothesis_test,
                                                                                batch_size=self.batch_size,
                                                                                device=self.device)
                    else:
                        val_rec, val_tstat, val_pval, val_total_loss = self.evaluate(nets=(encoder_p1, decoder),
                                                                                evaluate_loader=self.val_loader, n_z=self.n_z,
                                                                                use_proj=self.use_proj, \
                                                                                latent_dim_proj=self.latent_dim_proj,
                                                                                num_projections=self.num_projections,
                                                                                loss_fn_eval=self.loss_fn_eval, \
                                                                                htest=self.hypothesis_test,
                                                                                batch_size=self.batch_size,
                                                                                device=self.device)

                    track_val_recon.append(val_rec)
                    track_val_tstat.append(val_tstat)
                    track_val_pval.append(val_pval)
                    track_val_total_loss.append(val_total_loss)
                    print('Validation Set:  Recon: {:.4f}, Tstat: {:.4f}, P-value: {:.4f}\n'.format(val_rec, val_tstat,
                                                                                                    val_pval),
                          flush=True)

                encoder_p1.train()
                if self.Stiefel:
                    encoder_p2.train()
                decoder.train()

                # Use validation loss to modify learning rate on TRAINING SET
                enc_p1_scheduler.step(val_rec)
                # enc_p2_scheduler.step(val_rec)
                dec_scheduler.step(val_rec)

        end_time_train = time.time()
        total_time_train = end_time_train - start_time_train

        # Save Last Model
        if self.Stiefel:
            savemodel(dirpath=os.path.join(self.path, 'last_model.pt'), nets=(encoder_p1, encoder_p2, decoder),
                      optimizers=(enc_p1_optim, enc_p2_optim, dec_optim),
                      epoch=epoch + 1, stats=(running_mean, running_cov))
        else:
            savemodel(dirpath=os.path.join(self.path, 'last_model.pt'), nets=(encoder_p1, decoder), optimizers=(enc_p1_optim, dec_optim),
                      epoch=epoch + 1, stats=(running_mean, running_cov))

        eprint('Total time for {} epochs is {:.2f} minutes'.format(self.epochs, total_time_train / 60.), flush=True)

        full_train_data = np.hstack(
            (np.array(track_train_recon).reshape(-1, 1), np.array(track_train_recon_bdna).reshape(-1, 1),
             np.array(track_train_recon_nonb).reshape(-1, 1), np.array(track_train_tstat).reshape(-1, 1),
             np.array(track_train_pval).reshape(-1, 1), np.array(track_train_total_loss).reshape(-1, 1)))
        pd.DataFrame(full_train_data).to_csv(os.path.join(self.path, 'training.csv'),
                                             header=['recon', 'recon_bdna', 'recon_nonb', 'tstat', 'pval',
                                                     'total_loss'], index=False)

        # Saved information from VALIDATION
        full_val_data = np.hstack((np.array(track_val_recon).reshape(-1, 1), np.array(track_val_tstat).reshape(-1, 1),
                                   np.array(track_val_pval).reshape(-1, 1),
                                   np.array(track_val_total_loss).reshape(-1, 1)))
        pd.DataFrame(full_val_data).to_csv(os.path.join(self.path, 'validation.csv'), header=['recon', 'tstat', 'pval', 'total_loss'],
                                           index=False)

        bdna_test_latent_code, bdna_test_rec, bdna_test_full, bdna_MH_test, bdna_MH2_test = self.fit_test_novelty_stats(encoder_p1, encoder_p2, decoder)

        bdna_val_latent_code, bdna_val_rec, bdna_val_full, bdna_MH_val, \
        bdna_MH2_val = self.transform_with_novelty_stats(encoder_p1, encoder_p2, decoder, self.val_dataset)

        nonb_val_latent_code, nonb_val_rec, nonb_val_full, nonb_MH_val, \
        nonb_MH2_val = self.transform_with_novelty_stats(encoder_p1, encoder_p2, decoder, self.outlier_dataset_val)

        gstats_MH = NoveltyStats(bdna_MH_val, nonb_MH_val, self.X_val_outlier, fdr_level=self.fdr_level,
                                 poison_val_count=len(self.val_bdna_poison))
        emp_dist_MH_U, sorted_idx_MH_U, rejected_MH_U, disc_bdna_MH_U, disc_nonb_MH_U = gstats_MH.get_ns(tail='upper')

        # Maybe saving it out will be easier when alpha isn't set right
        pd.DataFrame(np.hstack((emp_dist_MH_U.reshape(-1, 1), sorted_idx_MH_U.reshape(-1, 1)))).to_csv(
            os.path.join(self.nonb_path, str(self.nonb_type) + '_MH_val_idx.csv'),
            header=['MH_U', 'MH_idx_U'], index=False)

        gstats_Rec = NoveltyStats(bdna_val_rec, nonb_val_rec, self.X_val_outlier, fdr_level=self.fdr_level,
                                  poison_val_count=len(self.val_bdna_poison))
        emp_dist_Rec_U, sorted_idx_Rec_U, rejected_Rec_U, disc_bdna_Rec_U, disc_nonb_Rec_U = gstats_Rec.get_ns(
            tail='upper')

        # Maybe saving it out will be easier when alpha isn't set right
        pd.DataFrame(np.hstack((emp_dist_Rec_U.reshape(-1, 1), sorted_idx_Rec_U.reshape(-1, 1)))).to_csv(
            os.path.join(self.nonb_path, str(self.nonb_type) + '_Rec_val_idx.csv'),
            header=['Rec_U', 'Rec_idx_U'], index=False)

        gstats_MH2 = NoveltyStats(bdna_MH2_val, nonb_MH2_val, self.X_val_outlier, fdr_level=self.fdr_level,
                                  poison_val_count=len(self.val_bdna_poison))
        emp_dist_MH2_U, sorted_idx_MH2_U, rejected_MH2_U, disc_bdna_MH2_U, disc_nonb_MH2_U = gstats_MH2.get_ns(
            tail='upper')

        pd.DataFrame(np.hstack((emp_dist_MH2_U.reshape(-1, 1), sorted_idx_MH2_U.reshape(-1, 1)))).to_csv(
            os.path.join(self.nonb_path, str(self.nonb_type) + '_MH2_val_idx.csv'),
            header=['MH2_U', 'MH2_idx_U'], index=False)

        # union upper
        union_nonb_U = get_union(disc_nonb_MH_U, disc_nonb_Rec_U)
        union_bdna_U = get_union(disc_bdna_MH_U, disc_bdna_Rec_U)
        union_all_U = get_union(sorted_idx_MH_U[rejected_MH_U], sorted_idx_Rec_U[rejected_Rec_U])

        # intersection upper
        intersection_nonb_U = get_intersection(disc_nonb_MH_U, disc_nonb_Rec_U)
        intersection_bdna_U = get_intersection(disc_bdna_MH_U, disc_bdna_Rec_U)
        intersection_all_U = get_intersection(sorted_idx_MH_U[rejected_MH_U], sorted_idx_Rec_U[rejected_Rec_U])

        # symmetric difference
        symdiff_nonb_U = get_symdif(disc_nonb_MH_U, disc_nonb_Rec_U)
        symdiff_bdna_U = get_symdif(disc_bdna_MH_U, disc_bdna_Rec_U)
        symdiff_all_U = get_symdif(sorted_idx_MH_U[rejected_MH_U], sorted_idx_Rec_U[rejected_Rec_U])

        # jaccard upper
        jaccard_nonb_U = get_jaccard(disc_nonb_MH_U, disc_nonb_Rec_U)
        jaccard_bdna_U = get_jaccard(disc_bdna_MH_U, disc_nonb_Rec_U)
        jaccard_all_U = get_jaccard(sorted_idx_MH_U[rejected_MH_U], sorted_idx_Rec_U[rejected_Rec_U])

        derekstat = len(intersection_nonb_U) / (len(symdiff_nonb_U) + len(intersection_bdna_U)) if (
                                                                                                               len(symdiff_nonb_U) + len(
                                                                                                           intersection_bdna_U)) > 0 else 0
        ratio_nonB_B = len(disc_nonb_MH2_U) / (1 + len(disc_bdna_MH2_U))

        savesum = np.hstack((len(self.X_val_outlier), len(self.val_nonb), len(self.val_bdna_poison), sum(rejected_MH_U),
                             len(disc_bdna_MH_U), len(disc_nonb_MH_U), \
                             sum(rejected_Rec_U), len(disc_bdna_Rec_U), len(disc_nonb_Rec_U), len(union_all_U),
                             len(union_bdna_U), len(union_nonb_U), \
                             len(intersection_all_U), len(intersection_bdna_U), len(intersection_nonb_U),
                             len(symdiff_all_U), len(symdiff_bdna_U), \
                             len(symdiff_nonb_U), jaccard_all_U, jaccard_bdna_U, jaccard_nonb_U, sum(rejected_MH2_U),
                             len(disc_bdna_MH2_U), len(disc_nonb_MH2_U), \
                             stats.pearsonr(bdna_MH_val.reshape(-1),bdna_val_rec.reshape(-1))[0], derekstat, ratio_nonB_B,
                             np.linalg.cond(self.MH2.covariance_)))

        columns = ['total', 'real_nonb', 'fake_nonb', 'disc_all_MH_U', 'disc_bdna_MH_U', 'disc_nonb_MH_U', \
                   'disc_all_Rec_U', 'disc_bdna_Rec_U', 'disc_nonb_Rec_U', 'union_all_U', 'union_bdna_U',
                   'union_nonb_U', \
                   'intersection_all_U', 'intersection_bdna_U', 'intersection_nonb_U', 'symdiff_all_U',
                   'symdiff_bdna_U', \
                   'symdiff_nonb_U', 'jaccard_all_U', 'jaccard_bdna_U', 'jaccard_nonb_U', 'disc_all_MH2_U',
                   'disc_bdna_MH2_U', 'disc_nonb_MH2_U', \
                   'Pearson_Cor', 'Derek_stat', 'ratio_nonB_B', 'MH2_cond_num']

        val_summary_upper = pd.DataFrame(savesum.reshape(1, -1), columns=columns)
        save_summary(nonb_path=self.nonb_path, name='val_summary_upper', data=val_summary_upper)

        if self.unblock_test:
            # for evaluating uniformity, but not really needed
            test_loader_DL = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, pin_memory=True,
                                        num_workers=self.num_workers, shuffle=False, drop_last=True)
            if self.Stiefel:
                # Validation Set (Used for hyperparameter tuning and for Learning Scheduler)
                test_rec_scalar, test_tstat, test_pval, test_total_loss = self.evaluate(
                    nets=(encoder_p1, encoder_p2, decoder),
                    evaluate_loader=test_loader_DL, n_z=self.n_z, use_proj=self.use_proj, \
                    latent_dim_proj=self.latent_dim_proj, num_projections=self.num_projections, loss_fn_eval=self.loss_fn_eval, \
                    htest=self.hypothesis_test, batch_size=self.batch_size, device=self.device)
            else:
                test_rec_scalar, test_tstat, test_pval, test_total_loss = self.evaluate(nets=(encoder_p1, decoder),
                                                                                   evaluate_loader=test_loader_DL,
                                                                                   n_z=self.n_z, use_proj=self.use_proj, \
                                                                                   latent_dim_proj=self.latent_dim_proj,
                                                                                   num_projections=self.num_projections,
                                                                                   loss_fn_eval=self.loss_fn_eval, \
                                                                                   htest=self.hypothesis_test, \
                                                                                   batch_size=self.batch_size, \
                                                                                   device=self.device)

            # Saved information from VALIDATION
            full_test_data = np.hstack((np.array(test_rec_scalar).reshape(-1, 1), np.array(test_tstat).reshape(-1, 1),
                                        np.array(test_pval).reshape(-1, 1),
                                        np.array(test_total_loss).reshape(-1, 1)))
            pd.DataFrame(full_test_data).to_csv(os.path.join(self.path, 'testing.csv'), header=['recon', 'tstat', 'pval', 'total_loss'],
                                                index=False)

            obs = 10

            # Forward
            trace_plots(x_true=bdna_test_full[0][obs][0], x_pred=bdna_test_full[1][obs][0], obs=obs, titlestring='Forward', \
                        path2save=os.path.join(self.img_path, 'test_sample_obs_' + str(obs) + '_Forward.png'))
            # Reverse
            trace_plots(x_true=bdna_test_full[0][obs][1], x_pred=bdna_test_full[1][obs][1], obs=obs, titlestring='Reverse', \
                        path2save=os.path.join(self.img_path,'test_sample_obs_' + str(obs) + '_Reverse.png'))

            nonb_test_latent_code, nonb_test_rec, nonb_test_full, nonb_MH_test, \
            nonb_MH2_test = self.transform_with_novelty_stats(encoder_p1, encoder_p2, decoder, self.outlier_dataset_test)

            emp_dist_test_U, sorted_idx_test_U = compute_empirical(bdna_MH2_test, nonb_MH2_test, tail='upper')
            rejected_test_U, _ = fdrcorrection(emp_dist_test_U, alpha=self.fdr_level, method='indep', is_sorted=True)

            pd.DataFrame(emp_dist_test_U).to_csv(os.path.join(self.nonb_path,str(self.nonb_type) + '_MH2_test_U.csv'), header=['MH2_U'],
                                                 index=False)
            pd.DataFrame(sorted_idx_test_U).to_csv(os.path.join(self.nonb_path, str(self.nonb_type) + '_MH2_idx_test_U.csv'),
                                                   header=['MH2_idx_U'], index=False)
            pd.DataFrame(np.hstack((emp_dist_test_U.reshape(-1, 1), sorted_idx_test_U.reshape(-1, 1)))).to_csv(
                os.path.join(self.nonb_path, str(self.nonb_type) + '_MH2_test_dist_idx_U.csv'),
                header=['MH2_U', 'MH2_idx_U'], index=False)

            if self.data_type != 'simulated':
                test_bed = pd.read_csv(os.path.join(self.exp_data_path, str(self.nonb_type) + '_test_bed.csv'))
                test_bed_filtered = test_bed.iloc[sorted_idx_test_U[rejected_test_U]]
                # test_bed_filtered.to_csv(nonb_path+str(nonb_type)+'_test_filtered.csv', sep='\t', index=False)
                test_bed_filtered.to_csv(os.path.join(self.nonb_path, str(self.nonb_type) + '_test_filtered_NEW.bed'), sep='\t', index=False)

            savesum_test = np.hstack((len(self.test_nonb), sum(rejected_test_U), self.fdr_level))

            columns_test = ['total', 'discoveries', 'fdr_level']

            test_summary_upper = pd.DataFrame(savesum_test.reshape(1, -1), columns=columns_test)
            save_summary(nonb_path=self.nonb_path, name='test_summary_upper', data=test_summary_upper)
            #bdna_test_latent_code, bdna_test_rec, bdna_test_full, bdna_MH_test, bdna_MH2_test

            fig = plt.figure(figsize=(8, 6))
            plt.hist(bdna_MH_test, range=(2, 20), bins=100, label='bdna', alpha=0.5, density=True)  # range=(2,12.5)
            plt.hist(nonb_MH_test, range=(2, 20), bins=100, label='nonb', alpha=0.5, density=True)  # range=(2,12.5)
            plt.legend(loc='upper right')
            plt.title('Mahalanobis Distance on Test Set')
            plt.savefig(os.path.join(self.img_path, 'testing_MH.tif'), dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig = plt.figure(figsize=(8, 6))
            plt.hist(bdna_test_rec, range=(2, 100), bins=100, label='bdna', alpha=0.5, density=True)
            plt.hist(nonb_test_rec, range=(2, 100), bins=100, label='nonb', alpha=0.5, density=True)
            plt.legend(loc='upper right')
            plt.title('Test Set Reconstruction Error', size=20)
            plt.savefig(os.path.join(self.img_path, 'testing_error.tif'), dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig = plt.figure(figsize=(8, 8))
            plt.scatter(bdna_MH_test, bdna_test_rec, s=2, alpha=.5, label='BDNA')
            plt.scatter(nonb_MH_test, nonb_test_rec, s=2, alpha=.5, label='NonB')
            plt.xlabel('Mahalanobis Distance', size=20)
            plt.ylabel('Reconstruction Error', size=20)
            plt.legend()
            plt.savefig(os.path.join(self.img_path, 'testing_MH_error.tif'), dpi=300, bbox_inches='tight')
            plt.close(fig)

            plot_test(bdna_MH_test, bdna_test_rec, nonb_MH_test, nonb_test_rec, idxs=sorted_idx_test_U[rejected_test_U],
                      img_path=self.img_path,
                      selection='MH2', fdr_level=self.fdr_level)

            if self.data_type =='simulated':
                tn_idxs, fp_idxs, fn_idxs, tp_idxs, real_idx = compute_true_sim_counts(test_sim=self.test_sim, nonb_type=self.nonb_type, \
                                                         idxs=sorted_idx_test_U[rejected_test_U])

                precision, recall, fscore = compute_accuracy_metrics(len(tn_idxs), len(fp_idxs), len(fn_idxs), len(tp_idxs))

                pd.DataFrame(np.array([self.experiment, len(tp_idxs), len(fp_idxs), len(fn_idxs), len(tn_idxs), precision, \
                                       recall, fscore]).reshape(1, -1)).to_csv(
                    os.path.join(self.nonb_path, 'sim_stats_test.csv'),
                    header=['experiment', 'tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'fscore'], index=False)
                txt_file = open(os.path.join(self.path, 'logfile_final_stats.txt'), "w")
                txt_file.write("TP: %d\n" % (len(tp_idxs)))
                txt_file.write("FP: %d\n" % (len(fp_idxs)))
                txt_file.write("FN: %d\n" % (len(fn_idxs)))
                txt_file.write("TN: %d\n" % (len(tn_idxs)))
                txt_file.write("Precision: %.8f\n" % (precision))
                txt_file.write("Recall: %.8f\n" % (recall))
                txt_file.write("Fscore: %.8f\n" % (fscore))
                txt_file.close()
                plot_test_sim(bdna_MH_test, bdna_test_rec, nonb_MH_test, nonb_test_rec, idxs_true_nonb=real_idx,
                              img_path=self.img_path)

                plot_test_sim_split(bdna_MH_test, bdna_test_rec, nonb_MH_test, nonb_test_rec, list(tp_idxs), list(fp_idxs),
                                    selection='MH2', img_path=self.img_path)











        # ks_unif_emp_dist = KolmogorovSmirnovUnif.get_ks_unif_info(self.diststat_path, self.test_set_samp_size)
        #
        # with open(os.path.join(self.path, str('evaluations.txt')), 'w') as filetowrite:
        #     filetowrite.write('Evaluations\n')
        #
        # pval_stat_test = []
        #
        # for test in htest_map:
        #
        #     hypothesis_test = htest_map[test]
        #
        #     print("Checking Test {} \n".format(test), flush=True)
        #
        #     #latent_dim_proj, num_projections = self.get_latent_dim_projections(test)
        #
        #     emp_dist, T_alpha = self.get_stat_info(path=self.diststat_path, use_proj=self.use_proj, test=test,
        #                                            n_z=self.n_z,
        #                                            batch_size=self.batch_size, latent_dim_proj=self.latent_dim_proj,
        #                                            num_projections=self.num_projections, new_stat=hypothesis_test.new_stat)
        #
        #     test_dictionary = self.get_test_param_dict(self.device, test, hypothesis_test.new_stat)
        #     self.hypothesis_test = testsetup(test, emp_dist, test_dictionary)
        #
        #     eval_stat, eval_pval = self.evaluate_all_tests(nets=(encoder_p1, encoder_p2), validate_loader=self.test_loader,
        #                                               n_z=self.n_z, use_proj=self.use_proj, latent_dim_proj=self.latent_dim_proj,
        #                                               num_projections=self.num_projections, htest=self.hypothesis_test,
        #                                               batch_size=self.batch_size, device=self.device)
        #     for i in range(len(eval_stat)):
        #         pval_stat_test.append(eval_pval[i])
        #         pval_stat_test.append(eval_stat[i])
        #         pval_stat_test.append(test)
        #
        #     ####### Uniformity Tests ############
        #     # Kolmogorov-Smirnov
        #     ks_htest = KolmogorovSmirnovUnif(self.device, ks_unif_emp_dist)
        #     ks_unif_tstat = ks_htest.teststat(torch.tensor(eval_pval, dtype=torch.float, device=self.device))  # this is a torch tensor
        #     ks_unif_pval = ks_htest.pval(ks_unif_tstat)  # numpy scalar
        #     print('KS Uniform Test Statistic: {:.4f}\n'.format(ks_unif_tstat.detach().cpu().numpy()), flush=True)
        #     print('KS Uniform P-value: {:.4f}\n'.format(ks_unif_pval), flush=True)
        #
        #     # Pearson's Chi-squared, not used in paper
        #     chi_htest = ChiSquareUnif()
        #     chisq_tstat = chi_htest.teststat(eval_pval)
        #     chisq_pval = chi_htest.chisq_pval
        #
        #     txt_file = open(os.path.join(self.path, str('evaluations.txt')), "a")
        #     txt_file.write(
        #         "Original Test: %s, Evaluating: %s - KS_unif stat: %0.4f, KS_unif pval: %0.4f, Chi-square stat: %0.4f, Chi-square pval: %0.4f\n" % (
        #             self.test_for_training, test, ks_unif_tstat, ks_unif_pval, chisq_tstat, chisq_pval))
        #     txt_file.close()
        # dt = np.dtype('float,float')
        # pd.DataFrame({'pval': pd.Series(pval_stat_test[0::3], dtype='float'),
        #               'teststat': pd.Series(pval_stat_test[1::3], dtype='float'),
        #                'test': pd.Series(pval_stat_test[2::3], dtype='object')}).to_csv(os.path.join(self.path, str('save_eval_stats.csv')),
        #                                       header=['pval', 'teststat', 'test'], index=False)
        #
        # print("Training and Evaluating Complete\n".format(self.test), flush=True)

    # def get_latent_dim_projections(self, test):
    #     if test in self.uvn:
    #         latent_dim_proj = self.latent_dim_proj # 1  # dimension of data the test is conducted on
    #         num_projections = self.num_projections # 64
    #     elif test in self.mvn:
    #         latent_dim_proj = 16
    #         num_projections = 8
    #     else:
    #         sys.exit('Test Not Specified')
    #     return latent_dim_proj, num_projections

    def generate_test_map(self):
        test_map={}

        # Only univariate tests considered for the paper
        tests = [('sw', True, 'min'), ('sf', True, 'min'), ('cvm', True, 'max'), ('ks', True, 'max'), ('ep1', True, 'min')]

        for test in tests:
            #latent_dim_proj, num_projections = self.get_latent_dim_projections(test[0])

            test_dictionary = self.get_test_param_dict(self.device, test[0], test[2])
            emp_dist, T_alpha = self.get_stat_info(path=self.diststat_path, use_proj=test[1], test=test[0],
                                                   n_z=self.n_z, batch_size=self.batch_size,
                                                   latent_dim_proj=self.latent_dim_proj,
                                                   num_projections=self.num_projections, new_stat=test[2])
            hypothesis_test = testsetup(test[0], emp_dist, test_dictionary)
            test_map[test[0]]=hypothesis_test

        return test_map

    def evaluate(self, nets, evaluate_loader, n_z, use_proj, latent_dim_proj, num_projections, loss_fn_eval, htest,
                 batch_size, device):
        with torch.no_grad():

            if self.Stiefel:
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

            val_recon_track = []
            val_stat_track = []
            val_pval_track = []
            val_total_loss_track = []

            for batch_idx, all_val_data in enumerate(evaluate_loader):

                images_eval = all_val_data[0].to(device)

                if not self.Stiefel:
                    p2_out = encoder_p1(images_eval)
                else:
                    code_s1 = encoder_p1(images_eval)
                    p2_out = encoder_p2(code_s1)

                if htest.is_univariate():
                    temp = []
                    stiefel_sample = sample_Stiefel(n_z, num_projections, device)
                    projected = torch.matmul(p2_out, stiefel_sample)
                    for direction in range(num_projections):
                        temp_stat = htest.teststat(x=projected[:, direction])
                        temp.append(temp_stat.view(-1, 1))
                # Option for multivariate tests
                else:  # is multivariate
                    # multivariate tests can be projected
                    if use_proj:
                        temp = []
                        for direction in range(num_projections):
                            stiefel_sample = sample_Stiefel(n_z, latent_dim_proj, device)
                            projected = torch.matmul(p2_out, stiefel_sample)
                            if isinstance(htest, Royston):
                                temp_stat, H_e = htest.teststat(x=projected)
                            else:
                                temp_stat = htest.teststat(x=projected)
                            temp.append(temp_stat.view(-1, 1))
                    # multivariate tests don't need to be projected
                    else:
                        if isinstance(htest, Royston):
                            temp_stat, H_e = htest.teststat(x=p2_out)
                        else:
                            temp_stat = htest.teststat(x=p2_out)

                #############################################
                # Calculate final test statistic #
                #############################################
                if htest.is_maximization() and self.new_stat == 'min':
                    out_tensor = torch.cat(temp, dim=1)
                    stat = torch.min(out_tensor)
                elif htest.is_maximization() and self.new_stat == 'avg':
                    out_tensor = torch.cat(temp, dim=1)
                    stat = torch.mean(out_tensor)
                elif not htest.is_maximization() and self.new_stat == 'max':
                    out_tensor = torch.cat(temp, dim=1)
                    stat = torch.max(out_tensor)
                elif not htest.is_maximization() and self.new_stat == 'avg':
                    out_tensor = torch.cat(temp, dim=1)
                    stat = torch.mean(out_tensor)
                elif not htest.is_univariate and self.new_stat == 'none':
                    stat = temp_stat
                else:
                    sys.exit("Configuration incorrect for test statistic")

                #############################################
                # Calculate P-values #
                #############################################
                pval = htest.pval(stat)

                #############################################
                # Decode #
                #############################################
                x_recon_eval = decoder(p2_out)

                # calculate reconstruction loss
                if loss_fn_eval == 'mse':
                    recon_loss = torch.sum(F.mse_loss(x_recon_eval, images_eval, reduction='none'),
                                           dim=1).mean()  # , reduction='sum').div(batch_size)
                elif loss_fn_eval == 'mae':
                    recon_loss = torch.sum(F.l1_loss(x_recon_eval, images_eval, reduction='none'), dim=(1, 2)).mean()
                elif loss_fn_eval == 'smae':
                    recon_loss = torch.sum(F.smooth_l1_loss(x_recon_eval, images_eval, reduction='none'), dim=1).mean()
                elif loss_fn_eval == 'max':
                    recon_loss = torch.mean(torch.amax(torch.abs(x_recon_eval - images_eval), dim=1))
                elif loss_fn_eval == 'rmse':
                    recon_loss = torch.mean(
                        torch.sqrt(torch.sum(F.mse_loss(x_recon_eval, images_eval, reduction='none'), dim=1)))
                else:
                    sys.exit("Reconstruction Loss Not Defined")

                # Version 1: No loop
                if htest.optdir == 1:
                    full_loss = recon_loss + self.lambda_alpha * (-stat)
                else:
                    full_loss = recon_loss + self.lambda_alpha * stat

                val_recon_track.append(recon_loss.data.item())
                val_stat_track.append(stat.detach().cpu().numpy())
                val_pval_track.append(pval)
                val_total_loss_track.append(full_loss.detach().cpu().numpy())

        return np.mean(np.array(val_recon_track)), np.mean(np.array(val_stat_track)), np.mean(
            np.array(val_pval_track)), np.mean(np.array(val_total_loss_track))

    def get_test_param_dict(self, device, test, new_stat):
        test_dictionary={}
        if test == 'sw':
            sw_wts, sw_mu, sw_sigma, _, _, _ = SWF_weights(self.batch_size, self.device)
            test_dictionary["sw_wts"]=sw_wts
            test_dictionary["sw_mu"]=sw_mu
            test_dictionary["sw_sigma"]=sw_sigma
        elif test == 'sf':
            _, _, _, sf_wts, sf_mu, sf_sigma = SWF_weights(self.batch_size, self.device)
            test_dictionary["sf_wts"]=sf_wts
            test_dictionary["sf_mu"]=sf_mu
            test_dictionary["sf_sigma"]=sf_sigma
        elif test == 'royston':
            sw_wts, sw_mu, sw_sigma, sf_wts, sf_mu, sf_sigma = SWF_weights(self.batch_size, self.device)
            test_dictionary["sw_wts"] = sw_wts
            test_dictionary["sw_mu"] = sw_mu
            test_dictionary["sw_sigma"] = sw_sigma
            test_dictionary["sf_wts"] = sf_wts
            test_dictionary["sf_mu"] = sf_mu
            test_dictionary["sf_sigma"] = sf_sigma
        else:
            pass
        test_dictionary["device"]=device
        test_dictionary["n_z"]=self.n_z
        test_dictionary["n"]=self.batch_size
        test_dictionary["use_emp"]=self.use_proj
        test_dictionary["new_stat"]=new_stat
        return test_dictionary

    def fit_test_novelty_stats(self, encoder_p1, encoder_p2, decoder):

        if self.use_test_rob_mean_cov:
            test_loader = DataLoader(dataset=self.test_dataset, batch_size=20000, pin_memory=True, \
                                     num_workers=self.num_workers, shuffle=False, drop_last=False)
            if self.Stiefel:
                latent_code, rec, full = get_novelty_info((encoder_p1, encoder_p2, decoder),
                                                                           test_loader, latent_dim=self.n_z,
                                                                           loss_fn_eval=self.loss_fn_eval,
                                                                         Stiefel=self.Stiefel, device=self.device)
            else:
                latent_code, rec, full = get_novelty_info((encoder_p1, decoder),
                                                                           test_loader, latent_dim=self.n_z,
                                                                           loss_fn_eval=self.loss_fn_eval,
                                                                         Stiefel=self.Stiefel, device=self.device)
            start_time_MCD1 = time.time()

            self.robust_cov_latent = MinCovDet().fit(latent_code)
            end_time_MCD1 = time.time()
            total_time_MCD1 = end_time_MCD1 - start_time_MCD1
            print('Total time for MCD1 computation on Test BDNA is {:.2f} minutes'.format(total_time_MCD1 / 60.),
                  flush=True)

            test_MH = self.robust_cov_latent.mahalanobis(latent_code) ** (self.s1_pow)
            test_bivdat = np.hstack((test_MH.reshape(-1, 1), rec.reshape(-1, 1)))
            start_time_MCD2 = time.time()
            self.MH2 = MinCovDet().fit(test_bivdat)
            end_time_MCD2 = time.time()
            total_time_MCD2 = end_time_MCD2 - start_time_MCD2
            print('Total time for MCD2 computation on Test BDNA is {:.2f} minutes'.format(total_time_MCD2 / 60.),
                  flush=True)
            MH2_test = self.MH2.mahalanobis(test_bivdat) ** (self.s2_pow)

            return latent_code, rec, full, test_MH, MH2_test

    def transform_with_novelty_stats(self, encoder_p1, encoder_p2, decoder, dataset):

        loader_fixed = DataLoader(dataset, batch_size=20000, pin_memory=True, num_workers=self.num_workers,shuffle=False,
                                  drop_last=False)

        if self.Stiefel:
            latent_code, rec, full = get_novelty_info((encoder_p1, encoder_p2, decoder),
                                                                                loader_fixed,
                                                                                latent_dim=self.n_z,
                                                                                loss_fn_eval=self.loss_fn_eval,
                                                      Stiefel=self.Stiefel, device=self.device)
        else:
            latent_code, rec, full = get_novelty_info((encoder_p1, decoder),
                                                                                loader_fixed,
                                                                                latent_dim=self.n_z,
                                                                                loss_fn_eval=self.loss_fn_eval,
                                                      Stiefel=self.Stiefel, device=self.device)

        MH = self.robust_cov_latent.mahalanobis(latent_code) ** (self.s1_pow)

        bivdat = np.hstack((MH.reshape(-1, 1), rec.reshape(-1, 1)))

        MH2_data = self.MH2.mahalanobis(bivdat) ** (self.s2_pow)

        return latent_code, rec, full, MH, MH2_data



