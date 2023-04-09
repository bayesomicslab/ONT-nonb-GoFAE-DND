import torch
import torch.nn as nn
from torch.nn import Parameter


class DBN(nn.Module):
    def __init__(self, num_features, num_groups=32, num_channels=0, dim=4, eps=1e-8, momentum=0.1, momentum2 = 0.01, affine=True, mode=0,
                 *args, **kwargs):
        super(DBN, self).__init__()
        if num_channels > 0:
            num_groups = num_features // num_channels # integer division
        self.num_features = num_features
        self.num_groups = num_groups
        assert self.num_features % self.num_groups == 0 # % is modulus operator, assert must always be true or error
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.momentum2 = momentum2
        self.affine = affine
        self.mode = mode

        self.shape = [1] * dim # list that is dim elements long
        self.shape[1] = num_features
        # giving [1, 64]
        
        # create affine transformation by adding parameters to the module
        # these are extra learnable parameters
        if self.affine:
            self.weight = Parameter(torch.Tensor(*self.shape))
            self.bias = Parameter(torch.Tensor(*self.shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        # parameters in the model which should be saved and restored, but not trained by optimizer should be registered
        # as buffers
        self.register_buffer('running_mean', torch.zeros(num_groups, 1)) #initialize running mean
        self.register_buffer('running_projection', torch.eye(num_groups)) # initialize running projection
        #self.register_buffer('running_sampling_mean', torch.zeros(num_features,1)) # initialize running sampling mean
        self.register_buffer('running_sampling_mean', torch.zeros(num_features,num_features)) # initialize running sampling mean
        self.register_buffer('running_sampling_Sigma', torch.eye(num_features)) # initialize running sampling sigma
        self.reset_parameters()

    # def reset_running_stats(self):
    #     self.running_mean.zero_()
    #     self.running_var.eye_(1)

    # initialize parameters for extra learnable parameters
    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor):
        # .shape is an alias for .size()
        # shape is attribute, size is function
        size = input.size()
        # input.dim() gives number of dimensions
        #assertion must be true otherwise error
        assert input.dim() == self.dim and size[1] == self.num_features
        
        # view will reshape
        # size[0] = # obs
        # size[1] = # features
        # (obs x feats) // groups
        
        # the last input *size[2:] isn't used for the current matrix
        x = input.view(size[0] * size[1] // self.num_groups, self.num_groups, *size[2:])
        
        # For training, mode == 0 and encoder.training, decoder.training is true
        # Setting encoder.eval(), decoder.eval() sets it to False
        training = self.mode > 0 or (self.mode == 0 and self.training)
        #training = self.mode > 0 or (self.mode == 0 and self.training
        
        # transpose, contiguous memory, view to reshape but ours only has 2 dimensions anyway so probably not needed
        x = x.transpose(0, 1).contiguous().view(self.num_groups, -1)
        
        # if model.training == True
        if training:
        
            # Calculate the mean of each group, keeping dimensions (G x 1)
            mean = x.mean(1, keepdim=True)
            
            # Store running mean
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean.data
            
            # Center the groups
            x_mean = x - mean
            
            # Calculate the Sigma of groups (maybe move self.eps into Lambda
            sigma = x_mean.matmul(x_mean.t()) / x.size(1) + self.eps * torch.eye(self.num_groups, device=input.device)
            # print('sigma size {}'.format(sigma.size()))
            
            # SVD, drop u^T
            u, eig, _ = sigma.svd()
            
            # Inverse square root (consider adding self.eps here)
            scale = eig.rsqrt()
            
            # U @ Lambda^(-1/2) U^T (consider adding self.eps here as well)
            wm = u.matmul(scale.diag()).matmul(u.t())
            
            # Store running projection 
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * wm.data
            
            # Calculate whitening transformation
            y = wm.matmul(x_mean)
            
            # Reshape the output (G x group_size ) then transpose it
            output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
        
            # make the output contiguous in memory, then view_as takes a tensor whose shape is to be mimicked
            # i.e. tensor.view_as(other) is equivalent to tensor.view(other.size())
            output = output.contiguous().view_as(input)
            
            # if affine function is used, then this does elementwise multiplication and addition of a bias
            # this is the extra learnable parameters "gamma" and "beta"
            if self.affine:
                output = output * self.weight + self.bias        
            
            output_mean = output.mean(0, keepdim=True)
            self.running_sampling_mean = (1. - self.momentum2) * self.running_sampling_mean + self.momentum2*output_mean.data
            
            centered_output = output - output_mean
            
            output_Sigma = centered_output.t().matmul(centered_output) / (output.size(0)-1)# + self.eps * torch.eye(self.num_groups, device=input.device)
            #print(output_Sigma.shape)
            self.running_sampling_Sigma = (1. - self.momentum2) * self.running_sampling_Sigma + self.momentum2 * output_Sigma.data
            
        # if model.training == False i.e. model.eval == True
        else:
        
            # After grouping, center observation(s) groups with stored running_mean
            x_mean = x - self.running_mean
            
            # Calculate whitening transformation using calculated mean, and stored projection
            y = self.running_projection.matmul(x_mean)
            
            # Reshape the output (G x group_size ) then transpose it
            output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
        
            # make the output contiguous in memory, then view_as takes a tensor whose shape is to be mimicked
            # i.e. tensor.view_as(other) is equivalent to tensor.view(other.size())
            output = output.contiguous().view_as(input)
            
            # if affine function is used, then this does elementwise multiplication and addition of a bias
            # this is the extra learnable parameters "gamma" and "beta"
            if self.affine:
                output = output * self.weight + self.bias
                
        return output

    def extra_repr(self):
        return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'mode={mode}'.format(**self.__dict__)

class DBN2(DBN):
    """
    when evaluation phase, sigma using running average.
    """

    def forward(self, input: torch.Tensor):
    
        # same
        size = input.size()
        
        # same
        assert input.dim() == self.dim and size[1] == self.num_features
        
        # same
        x = input.view(size[0] * size[1] // self.num_groups, self.num_groups, *size[2:])
        
        # same
        training = self.mode > 0 or (self.mode == 0 and self.training)
        
        # same
        x = x.transpose(0, 1).contiguous().view(self.num_groups, -1)
        
        # which mean to use, training or eval
        # this is different, moved out of if statement
        mean = x.mean(1, keepdim=True) if training else self.running_mean
        
        # also moved out of if statement
        x_mean = x - mean
        
        # same
        if training:
        
            # same
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
            
            # same
            sigma = x_mean.matmul(x_mean.t()) / x.size(1) + self.eps * torch.eye(self.num_groups, device=input.device)
            
            # this running projection uses sigma instead of whitening matrix
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * sigma
            
        else:
            # running projection is sigma instead of whitening matrix
            sigma = self.running_projection
        
        # svd of either sigma or running_sigma
        u, eig, _ = sigma.svd()
        
        # same
        scale = eig.rsqrt()
        
        # same (consider moving eps here)
        wm = u.matmul(scale.diag()).matmul(u.t())
        
        # same
        y = wm.matmul(x_mean)
        
        # same
        output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
        
        # same
        output = output.contiguous().view_as(input)
        
        # Estimate final 
        
        # same
        if self.affine:
            output = output * self.weight + self.bias
        return output