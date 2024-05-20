import torch
import torch.nn as nn
import torch.nn.functional as F

from .mine import GlobalDiscriminator, LocalDiscriminator, PriorDiscriminator


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.global_d = GlobalDiscriminator()
        self.local_d = LocalDiscriminator()
        self.prior_d = PriorDiscriminator()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y, M, M_prime):

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        y_exp_b = y.unsqueeze(-1)
        y_exp = y_exp_b.expand(-1, -1, 150)

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)
        
        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej_global = -F.softplus(-self.global_d(y, M)).mean()
        Em_global = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em_global - Ej_global) * self.alpha

        prior = torch.rand_like(y)

        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
        
        return LOCAL + GLOBAL + PRIOR
