import torch
import torch.nn.functional as F
from torch import nn as nn


class GlobalDiscriminator(nn.Module):
    def __init__(self, M_channels=150, interm_size=512):
        super().__init__()

        self.c0 = torch.nn.Conv1d(768, 256, kernel_size=3)
        self.c1 = torch.nn.Conv1d(256, 64, kernel_size=3)
        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        # input of self.l0 is the concatenate of E and flattened output of self.c1 (C)
        in_feature = 64 * (150-2*2) + 256
        self.l0 = torch.nn.Linear(in_feature, interm_size)
        self.l1 = torch.nn.Linear(interm_size, interm_size)
        self.l2 = torch.nn.Linear(interm_size, 1)

    def forward(self, E, M):

        C = F.relu(self.c0(M))
        C = self.c1(C)
        C = C.view(E.shape[0], -1)
        out = torch.cat((E, C), dim=1)
        out = F.relu(self.l0(out))
        out = F.relu(self.l1(out))
        out = self.l2(out)
        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        # output of Table 5
        return out

class LocalDiscriminator(nn.Module):

    def __init__(self, interm_channels=512):
        super().__init__()
        
        self.c0 = torch.nn.Conv1d(256+768, interm_channels, kernel_size=1)
        self.c1 = torch.nn.Conv1d(interm_channels, interm_channels, kernel_size=1)
        self.c2 = torch.nn.Conv1d(interm_channels, 1, kernel_size=1)

    def forward(self, x):
        score = F.relu(self.c0(x))
        score = F.relu(self.c1(score))
        score = self.c2(score)
        return score
    
class PriorDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(256, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        h = torch.sigmoid(self.l2(h))
        h = torch.clamp(h, min=9e-6)
        return h
