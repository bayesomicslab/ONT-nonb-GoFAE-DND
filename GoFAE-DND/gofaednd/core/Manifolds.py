import numpy as np
import torch

# STIEFEL MANIFOLD FUNCTIONS
def replacegrad(encoder):
    # Replaces the original gradients to the gradient on manifold M
    Theta = encoder.fc.detach().cpu().numpy()
    Theta_grad = encoder.fc.grad.detach().cpu().numpy()
    N = np.matmul(Theta, Theta_grad.T)
    G = Theta_grad - np.matmul((N + N.T) / 2., Theta)
    encoder.fc.grad.copy_(torch.FloatTensor(G))
    # return Euclidean gradient, Riemannian Gradient
    return np.linalg.norm(Theta_grad), np.linalg.norm(G)


def retract2manifold(encoder):
    Theta = encoder.fc.detach().cpu().numpy()
    R, L, ST = np.linalg.svd(Theta.T, full_matrices=False)
    Theta_new = np.matmul(R, ST).T
    state_dict = encoder.state_dict()
    state_dict['fc'].copy_(torch.FloatTensor(Theta_new))

def sample_Stiefel(orig_dim, new_dim, device):
    x = np.random.multivariate_normal(np.zeros(new_dim), np.eye(new_dim), size=orig_dim)
    Q, _ = np.linalg.qr(x)  # Q^TQ = I
    return torch.from_numpy(Q).float().to(device)