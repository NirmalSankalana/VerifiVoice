from loss.aamsoftmax import LossFunction as AMMLossFunction
from loss.deepInfoMaxLoss import DeepInfoMaxLoss
import torch.nn as nn


class LossFunction(nn.Module):
    def __init__(
        self,
        nOut,
        nClasses,
        alpha,
        beta,
        gamma,
        margin=0.3,
        scale=15,
        easy_margin=False,
        **kwargs,
    ):
        super(LossFunction, self).__init__()
        self.classificationLoss = AMMLossFunction(
            nOut, nClasses, margin, scale, easy_margin, **kwargs
        )
        self.DIMLoss = DeepInfoMaxLoss(alpha, beta, gamma)

    def forward(self, x, M, M_prime, label=None):
        loss, prec1 = self.classificationLoss.forward(x, label)
        dim = self.DIMLoss(x, M, M_prime)
        print(f"\nclassification Loss: {loss}, DIM Loss: {dim} total loss {dim+loss} \n")
        return loss + dim, prec1


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