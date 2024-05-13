import torch
import torch.nn as nn
import torch.nn.functional as F
from verifyvoice.model.mine.mine import GlobalDiscriminator
from verifyvoice.model.mine.mine import LocalDiscriminator
import numpy as np
from verifyvoice.model.mine.mine import PriorDiscriminator

class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.global_d = GlobalDiscriminator()
        self.local_d = LocalDiscriminator()
        self.prior_d = PriorDiscriminator()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # print(
        #     "Initialised DeepInfoMaxLoss alpha %.4f beta %.4f gamma %.4f"
        #     % (self.alpha, self.beta, self.gamma)
        # )

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
        print(f"with alpha beta : LOCAL: {LOCAL}, GLOBAL: {GLOBAL} ,PRIOR: {PRIOR}, DIM; {LOCAL + GLOBAL + PRIOR}")
        
        return LOCAL + GLOBAL + PRIOR




















# class DeepInfoMaxLoss(nn.Module):
#     def __init__(self, alpha=0.5, beta=1.0, inputdim=512, x_dim=512):
#         super(DeepInfoMaxLoss, self).__init__()
#         self.global_d = GlobalDiscriminator()
#         self.local_d = LocalDiscriminator()
#         self.alpha = alpha
#         self.beta = beta

#         print(
#             "Initialised DeepInfoMaxLoss alpha %.3f beta %.3f" % (self.alpha, self.beta)
#         )

#     # @snoop
#     def forward(self, y, M, M_prime):

#         # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf

#         y_exp = y.unsqueeze(-1)
#         # # print(f"y_exp first unsqueeze = {y_exp.shape}")
#         # # print(f"y_exp first unsqueeze = {y_exp}")
#         # # y_exp = y_exp.unsqueeze(-1)
#         y_exp = y_exp.expand(-1, -1, 768)
#         # # print(f"y_exp expand = {y_exp.shape}")
#         # # print(f"M = {M.shape}")
#         # y_exp = F.pad(y.unsqueeze(2), (0, 767), "constant", 0).squeeze(2)

#         y_M = torch.cat((M, y_exp), dim=1)
#         # print(f"y_M.shape  = {y_M.shape}")
#         y_M_prime = torch.cat((M_prime, y_exp), dim=1)
#         # print(f"y_M_primne.shape  = {y_M_prime.shape}")

#         Ej = -F.softplus(-self.local_d(y_M)).mean()
#         Em = F.softplus(self.local_d(y_M_prime)).mean()
#         LOCAL = (Em - Ej) * self.beta

#         Ej = -F.softplus(-self.global_d(y, M)).mean()
#         Em = F.softplus(self.global_d(y, M_prime)).mean()
#         GLOBAL = (Em - Ej) * self.alpha

#         return LOCAL + GLOBAL
