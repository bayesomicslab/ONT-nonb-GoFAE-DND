#!/usr/bin/env python
import sys
import argparse
import torch
import numpy as np
import random
from core.GoFAEDND import GoFAEDND
from utilities.IOUtilities import eprint


class Main(object):

    @classmethod
    def init(cls, cmd_args, tee):
        if tee is None:
            if sys.platform == "win32":
                tee = open("NUL:")
            else:
                #  should be fine for OSX/Linux
                tee = open("/dev/null")

        parser = Main.getParser()
        args = parser.parse_args(cmd_args[1:])

        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        cls.fQuiet = args.fQuiet

        if not cls.fQuiet:
            eprint("args =")
            for arg in cmd_args:
                eprint(" " + arg)
            eprint()
            eprint()

            print("Diagnostic info")
            print(torch.cuda.current_device(), flush=True)
            print(torch.cuda.device(0), flush=True)
            print(torch.cuda.device_count(), flush=True)
            print(torch.cuda.get_device_name(0), flush=True)
            print(torch.cuda.is_available(), flush=True)

        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not cls.fQuiet:
            print('Using device:', device, flush=True)
            print(flush=True)

            # Additional Info when using cuda
            if device.type == 'cuda':
                print(torch.cuda.get_device_name(0), flush=True)
                print('Memory Usage:', flush=True)
                print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB', flush=True)
                print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB', flush=True)

            print("Finished", flush=True)

        # need to add command line parameters here

        return GoFAEDND(device, args)

    @classmethod
    def getParser(cls):
        parser = argparse.ArgumentParser(
            description='Implementation of the Goodness of Fit Autoencoder.')
        parser.add_argument("--fQuiet", help="Flag indicates if we should output logging information. "
                            "(default: " + str(False) + ")", default=False, action='store_true')

        parser.add_argument('--data_type', type=str, help='experimental or simulated', default='simulated')
        parser.add_argument('--nonb_type', type=str, help='One of the non-B DNA types.', default='G_Quadruplex_Motif')
        parser.add_argument('--nonb_ratio', type=float, help='If simulated, ratio of B-DNA to non-B DNA', default=0.1)
        parser.add_argument('--test', type=str, help='Hypothesis test for training.', default='sw')
        parser.add_argument('--new_stat', type=str, help='Statistic to use after projection.', default='min')
        parser.add_argument('--use_proj', type=bool, help='Whether test requires projection to lower dimension.', default=True)
        parser.add_argument('--n_z', type=int, help='Size of the latent dimension.', default=64)
        parser.add_argument('--latent_dim_proj', type=int, help='Dimension of data the test is conducted on.', default=1)
        parser.add_argument('--num_projections', type=int, help='Number of times to project down to latent dimension.', default=64)
        parser.add_argument('--batch_size', type=int, help='Number of observations to include in B-DNA minibatch.', default=128)
        parser.add_argument('--outlier_batch_size', type=int, help='Number of observations to include in non-B minibatch.', default=128)
        parser.add_argument('--alpha', type=float, help='Used to produce p-value or critical value for hard-thresholding (optional).', default=0.3)
        parser.add_argument('--epochs', type=int, help='Number of epochs to train model.', default=1)
        parser.add_argument('--use_labels', type=bool, help='Whether model is discriminative or unsupervised', default=True)
        parser.add_argument('--poison_train', type=bool, help='Include B-DNA into the non-B DNA training dataset.', default=False)
        parser.add_argument('--poison_val', type=bool, help='Whether to include B-DNA into the non-B validation dataset.', default=True)
        parser.add_argument('--use_rec_MH', type=bool, help='Use vector of reconstruction residuals per observation or sum', default=False)
        parser.add_argument('--s1_pow', type=float, help='Power of Mahalanobis distance on encoding.', default=0.5)
        parser.add_argument('--s2_pow', type=float, help='Power of Mahalanobis distance on joint encoding and reconstruction.', default=0.5)
        parser.add_argument('--fdr_level', type=float, help='False discovery rate level.', default=0.25)
        parser.add_argument('--unblock_test', type=bool, help='Run test set after training.', default=True)
        parser.add_argument('--use_test_rob_mean_cov', type=bool, help='Use MCD on test set or train set (overfitting).', default=True)
        parser.add_argument('--inform_plots', type=bool, help='Generate figures explaining loss.', default=True)
        parser.add_argument('--loss_fn', type=str, help='Reconstruction loss function for training.', default='discriminative')
        parser.add_argument('--discriminative_loss_fn', type=str, help='Specific loss used in constrastive training.', default='mae')
        parser.add_argument('--loss_fn_eval', type=str, help='Loss used during evaluation.', default='mae')
        parser.add_argument('--discriminative_weight', type=float, help='Delta.', default=30.00)
        parser.add_argument('--lambda_alpha', type=float, help='Lambda', default=0.5)
        parser.add_argument('--omega', type=float, help='Omega', default=0.5)
        parser.add_argument('--Stiefel', type=bool, help='Whether to use the Stiefel manifold or not', default=True)
        parser.add_argument('--lr_e_d', type=float, help='Learning rate', default=1e-3)
        parser.add_argument('--beta1_enc_p1', type=float, help='First parameter for encoder_p1 for Adam.', default=0.9)
        parser.add_argument('--beta2_enc_p1', type=float, help='Second parameter for encoder_p1 for Adam.', default=0.999)
        parser.add_argument('--beta1_dec', type=float, help='First parameter for decoder for Adam.', default=0.9)
        parser.add_argument('--beta2_dec', type=float, help='Second parameter for decoder for Adam.', default=0.999)
        parser.add_argument('--h_dim', type=int, help='Dimension of latent space.', default=32)
        parser.add_argument('--momentum', type=float, help='Parameter for exponential moving average of mean and covariance statistics.', default=0.01)
        parser.add_argument('--ncv', type=int, help='Number of training epochs between each validation evaluation.', default=1)
        parser.add_argument('--num_workers', type=int, help='Number of workers for the dataloader.', default=0)
        parser.add_argument('--config', type=int, help='Where to store each run', default=10000)
        parser.add_argument('--seed', type=int, help='Random seed.', default=42)
        parser.add_argument('--diststat_path', type=str, help='Path for the empirical distribution directory.', default='../data/empirical_dist')
        parser.add_argument('--output_path', type=str, help='Path for output.', default='../../results')
        parser.add_argument('--exp_data_path', type=str, help='Path to experimental data.', default='D:/prepared_windows_req5_mean/final_dataset_splitted/')
        parser.add_argument('--sim_data_path', type=str, help='Path to simulated data.', default='D:/simulated_function_sin_random/')

        return parser

    @classmethod
    def main(cls, cmd_args):
        gofaednd = cls.init(cmd_args, sys.stdout)
        gofaednd.run()


if __name__ == '__main__':
    Main.main(sys.argv)