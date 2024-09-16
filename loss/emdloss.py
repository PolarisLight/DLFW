import torch
import torch.nn.functional as F
import numpy as np
import platform

def base_emd_loss(x, y_true, dist_r=2):
    cdf_x = torch.cumsum(x, dim=-1)
    cdf_ytrue = torch.cumsum(y_true, dim=-1)
    if dist_r == 2:
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(cdf_ytrue - cdf_x, 2), dim=-1))
    else:
        samplewise_emd = torch.mean(torch.abs(cdf_ytrue - cdf_x), dim=-1)
    return samplewise_emd


class emd_loss(torch.nn.Module):
    """
    Earth Mover Distance loss
    """

    def __init__(self, dist_r=2,
                 use_l1loss=True, l1loss_coef=0.0):
        super(emd_loss, self).__init__()
        self.dist_r = dist_r
        self.use_l1loss = use_l1loss
        self.l1loss_coef = l1loss_coef

    def check_type_forward(self, in_types):
        assert len(in_types) == 2

        x_type, y_type = in_types
        assert x_type.size()[0] == y_type.shape[0]
        assert x_type.size()[0] > 0
        assert x_type.ndim == 2
        assert y_type.ndim == 2

    def forward(self, x, y_true):
        self.check_type_forward((x, y_true))

        cdf_x = torch.cumsum(x, dim=-1)
        cdf_ytrue = torch.cumsum(y_true, dim=-1)
        if self.dist_r == 2:
            samplewise_emd = torch.sqrt(torch.mean(torch.pow(cdf_ytrue - cdf_x, 2), dim=-1))
        else:
            samplewise_emd = torch.mean(torch.abs(cdf_ytrue - cdf_x), dim=-1)
        loss = torch.mean(samplewise_emd)
        if self.use_l1loss:
            rate_scale = torch.tensor([float(i + 1) for i in range(x.size()[1])], dtype=x.dtype, device=x.device)
            x_mean = torch.mean(x * rate_scale, dim=-1)
            y_true_mean = torch.mean(y_true * rate_scale, dim=-1)
            l1loss = torch.mean(torch.abs(x_mean - y_true_mean))
            loss += l1loss * self.l1loss_coef
        return loss


class MPEMDLoss(torch.nn.Module):
    def __init__(self, dist_r=2, eps=1e-6, beta=0.7, k=1.2,norm=False):
        super(MPEMDLoss, self).__init__()
        self.dist_r = dist_r
        self.eps = eps
        self.beta = beta
        self.k = k
        self.emd = base_emd_loss
        self.norm = norm
        # if system is linux, compile the emd loss
        if platform.system() == 'Linux':
            self.emd = torch.compile(base_emd_loss)

    def forward(self, x, y_true):
        patch_num = x.size(1)
        x_flatten = x.view(-1, x.size(-1))
        # copy y_true patch_num times at dim 1
        y_true_flatten = y_true.repeat(1, patch_num).view(-1, y_true.size(-1))
        loss = self.emd(x_flatten, y_true_flatten)
        loss = loss.contiguous().view(-1, patch_num)
        eps = torch.ones_like(loss) * self.eps
        emdc = torch.max(eps, 1 - self.k * loss)
        weight = 1 - torch.pow(emdc, self.beta)
        if self.norm:
            # normalize the weight
            weight = weight / torch.sum(weight, dim=1, keepdim=True)
        loss = torch.mean(loss * weight)
        return loss