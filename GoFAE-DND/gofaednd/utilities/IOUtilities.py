import sys
import numpy as np
import pandas as pd
import torch
import argparse
import os


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def save_summary(nonb_path, name, data):
    '''
    data: should be a pandas dataframe
    '''
    data.to_csv(os.path.join(nonb_path, str(name)+'.csv'), index=False)


def savemodel(dirpath, nets, optimizers, epoch, stats=None):
    if len(nets) == 2:
        enc1 = nets[0]
        dec = nets[1]
        enc1_optim = optimizers[0]
        dec_optim = optimizers[1]

        torch.save({
            'epoch': epoch,
            'encoder_p1_state_dict': enc1.state_dict(),
            'decoder_state_dict': dec.state_dict(),
            'encoder_optimizer_p1_state_dict': enc1_optim.state_dict(),
            'decoder_optimizer_state_dict': dec_optim.state_dict(),
            'sampling_mean': stats[0] if stats is not None else None,
            'sampling_cov': stats[1] if stats is not None else None
        }, dirpath)

    elif len(nets) == 3:
        enc1 = nets[0]
        enc2 = nets[1]
        dec = nets[2]
        enc1_optim = optimizers[0]
        enc2_optim = optimizers[1]
        dec_optim = optimizers[2]

        torch.save({
            'epoch': epoch,
            'encoder_p1_state_dict': enc1.state_dict(),
            'encoder_p2_state_dict': enc2.state_dict(),
            'decoder_state_dict': dec.state_dict(),
            'encoder_optimizer_p1_state_dict': enc1_optim.state_dict(),
            'encoder_optimizer_p2_state_dict': enc2_optim.state_dict(),
            'decoder_optimizer_state_dict': dec_optim.state_dict(),
            'sampling_mean': stats[0] if stats is not None else None,
            'sampling_cov': stats[1] if stats is not None else None
        }, dirpath)


def save_data(path, tracked_recon, tracked_tstat, tracked_pval, tracked_total_loss):
    full_data = np.hstack((np.array(tracked_recon).reshape(-1, 1), np.array(tracked_tstat).reshape(-1, 1),
                                 np.array(tracked_pval).reshape(-1, 1),
                                 np.array(tracked_total_loss).reshape(-1, 1)))
    pd.DataFrame(full_data).to_csv(path, header=['recon', 'tstat', 'pval', 'total_loss'], index=False)


# def restore_model(path, encoder_p1, encoder_p2, decoder, Stiefel, device):
#
#     encoder_p1, encoder_p2, decoder = encoder_p1.to(device), encoder_p2.to(device), decoder.to(device)
#
#     choose_mod = 'last'  # best, last, fifty
#
#     if os.path.exists(path + 'best_recon_model.pt') and choose_mod == 'best':
#         checkpoint = torch.load(path + 'best_recon_model.pt')
#         encoder_p1.load_state_dict(checkpoint['encoder_p1_state_dict'])
#         if Stiefel:
#             encoder_p2.load_state_dict(checkpoint['encoder_p2_state_dict'])
#         decoder.load_state_dict(checkpoint['decoder_state_dict'])
#         sampling_mean = checkpoint['sampling_mean']
#         sampling_cov = checkpoint['sampling_cov']
#     elif os.path.exists(path + 'epoch_50_model.pt') and choose_mod == 'fifty':
#         checkpoint = torch.load(path + 'epoch_50_model.pt')
#         encoder_p1.load_state_dict(checkpoint['encoder_p1_state_dict'])
#         if Stiefel:
#             encoder_p2.load_state_dict(checkpoint['encoder_p2_state_dict'])
#         decoder.load_state_dict(checkpoint['decoder_state_dict'])
#         sampling_mean = checkpoint['sampling_mean']
#         sampling_cov = checkpoint['sampling_cov']
#     elif os.path.exists(path + 'last_model.pt') and choose_mod == 'last':
#         checkpoint = torch.load(path + 'last_model.pt')
#         encoder_p1.load_state_dict(checkpoint['encoder_p1_state_dict'])
#         if Stiefel:
#             encoder_p2.load_state_dict(checkpoint['encoder_p2_state_dict'])
#         decoder.load_state_dict(checkpoint['decoder_state_dict'])
#         sampling_mean = checkpoint['sampling_mean']
#         sampling_cov = checkpoint['sampling_cov']
#     else:
#         print('Model not available', flush=True)

def restore_model(path, encoder_p1, encoder_p2, decoder, Stiefel, device):

    encoder_p1, encoder_p2, decoder = encoder_p1.to(device), encoder_p2.to(device), decoder.to(device)

    if os.path.exists(os.path.join(path, 'last_model.pt')):
        checkpoint = torch.load(os.path.join(path, 'last_model.pt'))
        encoder_p1.load_state_dict(checkpoint['encoder_p1_state_dict'])
        if Stiefel:
            encoder_p2.load_state_dict(checkpoint['encoder_p2_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        sampling_mean = checkpoint['sampling_mean']
        sampling_cov = checkpoint['sampling_cov']
    else:
        print('Model not available', flush=True)
