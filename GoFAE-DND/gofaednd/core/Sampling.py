import torch
import numpy as np

def samples(nets, stats, num_samples=100, whiten=False, device=None):
    # no n_groups here as the covariance and mean have already taken this into account
    with torch.no_grad():
        encoder_p1 = nets[0]
        encoder_p1.eval()
        encoder_p2 = nets[1]
        encoder_p2.eval()
        decoder = nets[2]
        decoder.eval()

        if whiten:
            samp_mu = encoder_p2.dbn[0].running_sampling_mean[0]
            # this is absolutely necessary when Groups is used
            samp_Sigma = encoder_p2.dbn[0].running_sampling_Sigma

            dist = torch.distributions.multivariate_normal.MultivariateNormal(samp_mu, samp_Sigma)
            m = dist.sample(torch.Size([num_samples]))

            # if enc.dbn[0].affine:
            #    m = m * enc.dbn[0].weight + enc.dbn[0].bias
        else:
            # samp_mu = torch.tensor(stats[0], dtype=torch.float, device=device)
            # samp_Sigma = torch.tensor(stats[1], dtype=torch.float, device=device)
            m = torch.tensor(np.random.multivariate_normal(mean=stats[0].reshape(-1), cov=stats[1], size=num_samples),
                             dtype=torch.float, device=device)

        gen_ims = decoder(m).view(num_samples, encoder_p1.n_channel, encoder_p1.img_size, encoder_p1.img_size)
    return gen_ims





