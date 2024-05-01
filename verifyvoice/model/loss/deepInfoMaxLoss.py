import torch
import torch.nn as nn
import torch.nn.functional as F
from mine.mine import GlobalDiscriminator
from mine.mine import LocalDiscriminator
import numpy as np
# from mine.mine import PriorDiscriminator

class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=0.01, beta=0.05, gamma=0):
        super().__init__()
        self.global_d = GlobalDiscriminator()
        self.local_d = LocalDiscriminator()
        # self.prior_d = PriorDiscriminator()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y, M, M_prime):

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        # print(f"{y.shape} =")
        y_exp_b = y.unsqueeze(-1)
        # # print(f"{y_M_primey_exp.shape} =")
        # print(f"before exapnd{y_exp_b.shape=} ")
        y_exp = y_exp_b.expand(-1, -1, 768)
        # print(f"after exapnd{y_exp.shape=} ")
        # print(f"{y_exp_b=}")
        # print(f"{y_exp=}")
        # print(f"{M[0].shape=}")
        # print(f"{M[0][0].shape=}")
        # print(f"{M[0][0][0].shape=}")
        # # print(f"{M[0][0][0][0].shape=}")
        # print(f"{M[0]=}")
        # print(f"{M[0][0]=}")
        # print(f"{M[0][0][0]=}")

        # print(f"{M.shape=}")
        # print(f"{M_prime.shape=}")
        # height, width  = M.size()[2:]
        # y_exp2 = y.unsqueeze(2).unsqueeze(3).repeat(1, 1, height, width)

        # print(f"ggggggg {torch.eq(y_exp, y_exp2)}")

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)
        # print(f"{y_M.shape=}")
        # print(f"{y_M_prime.shape=}")

        # log2 = np.log(2.)
        # Ej = torch.mean(log2 - F.softplus(-self.local_d(y_M)))
        # E_prod = self.local_d(y_M_prime)
        # Em = torch.mean(F.softplus(-E_prod) + E_prod - log2)
        Ej = -F.softplus(-self.local_d(y_M)).mean()
        # Ej_scaled = ((Ej - Ej.min()) / (Ej.max() - Ej.min())).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        # Em_scaled = ((Em - Em.min()) / (Em.max() - Em.min())).mean()
        LOCAL = (Em - Ej) 
        # print(f"LOCAL: {LOCAL.item()}")
   
        # Ej_global = torch.mean(log2 - F.softplus(-self.global_d(y, M)))
        # E_prod_global = self.global_d(y, M_prime)
        # Em_global = torch.mean(F.softplus(-E_prod_global) + E_prod_global - log2)
        Ej_global = -F.softplus(-self.global_d(y, M)).mean()
        # Ej_global_scaled = ((Ej_global - Ej_global.min()) / (Ej_global.max() - Ej_global.min())).mean()
        Em_global = F.softplus(self.global_d(y, M_prime)).mean()
        # Em_global_scaled = ((Em_global - Em_global.min()) / (Em_global.max() - Em_global.min())).mean()
        GLOBAL = (Em_global - Ej_global)

        # prior = torch.rand_like(y)

        # term_a = torch.log(self.prior_d(prior)).mean()
        # term_b = torch.log(1.0 - self.prior_d(y)).mean()
        # PRIOR = - (term_a + term_b) * self.gamma
        print()
        print(f"not mul LOCAL: {LOCAL}, GLOBAL: {GLOBAL} , DIM; {LOCAL + GLOBAL}")
        LOCAL = LOCAL * self.beta
        GLOBAL = GLOBAL * self.alpha
        print(f"with alpha beta : LOCAL: {LOCAL}, GLOBAL: {GLOBAL} , DIM; {LOCAL + GLOBAL}")


        return LOCAL + GLOBAL




















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
