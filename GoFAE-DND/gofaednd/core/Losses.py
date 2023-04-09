import torch
import torch.nn.functional as F


def discriminative_loss(x_true, x_recon, label, loss_function, device, weight=0.0, omega=0.5):
    '''
    omega: [0,1]
    '''

    if loss_function == 'mse':
        out = F.mse_loss(x_recon, x_true, reduction='none')
    elif loss_function == 'mae':
        out = F.l1_loss(x_recon, x_true, reduction='none')
    elif loss_function == 'smae':
        out = F.smooth_l1_loss(x_recon, x_true, reduction='none')
    else:
        print('bad loss')

    nonb_loss = torch.max(weight - torch.sum(out[label == -1], dim=(1, 2)),
                          torch.tensor(0.0, dtype=torch.float, device=device))
    bdna_loss = torch.sum(out[label == 1], dim=(1, 2))

    return bdna_loss.mean() + omega * nonb_loss.mean(), bdna_loss.mean(), nonb_loss.mean()

