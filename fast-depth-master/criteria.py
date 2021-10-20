import torch
import torch.nn as nn
from torch.autograd import Variable
from my_utils import *

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class CorrelationLoss(nn.Module):
    def __init__(self):
        super(CorrelationLoss, self).__init__()

    def forward(self, rgb, target, pred):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        BG_R = torch.max(rgb[:,1:, :,:], dim=1, keepdim=True)[0] - torch.unsqueeze(rgb[:,0,:,:], dim=1)
        valid_mask = (pred>0).detach()
        self.loss = 1 - (pred[valid_mask]-torch.mean(pred[valid_mask])*(BG_R[valid_mask]-torch.mean(BG_R[valid_mask]))) / (torch.sqrt(pred[valid_mask]-torch.mean(pred[valid_mask])**2) * torch.sqrt(BG_R[valid_mask]-torch.mean(BG_R[valid_mask])**2))
        self.loss = self.loss.mean()
        return self.loss
