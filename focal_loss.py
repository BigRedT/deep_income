import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self,gamma=1.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self,probs,labels):
        p0 = probs[:,0]
        p1 = probs[:,1]
        loss = \
            -p0.pow(self.gamma)*labels*torch.log(p1+1e-6) + \
            -p1.pow(self.gamma)*(1-labels)*torch.log(p0+1e-6)
        return loss.mean()