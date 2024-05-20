#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

# from .DatasetLoader import test_dataset_loader
from .deepInfoMaxLoss import DeepInfoMaxLoss
from .Spk_Encoder import MainModel
from .aamsoftmax import LossFunction

class WrappedModel(nn.Module):

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None, l2_reg_dict=None):
        return self.module(x, label)
    

class SpeakerNet(nn.Module):
    """
    SpeakerNet class for speaker recognition neural network.

    forward:
    Forward pass of the SpeakerNet model.

    Args:
        data: Input data for the model.
        label: Labels for the input data (default is None).
        l2_reg_dict: Dictionary for L2 regularization (default is None).

    Returns:
        Total loss, precision, and classification loss.
    """


    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, device, **kwargs):
        super(SpeakerNet, self).__init__();

        self.__S__ = MainModel(device,**kwargs);

        self.__L__ = LossFunction(**kwargs);
        self.__DIM_L__ = DeepInfoMaxLoss(alpha=kwargs['alpha'], beta=kwargs['beta'], gamma=kwargs['gamma'])

        self.nPerSpeaker = nPerSpeaker
        self.weight_finetuning_reg = kwargs['weight_finetuning_reg']
        self.device = device

    def forward(self, data, label=None, l2_reg_dict=None):
        if label is None:
            data_reshape = data[0].to(self.device)
            outp, M = self.__S__.forward([data_reshape, data[1]])
            return outp
        else:
            data_reshape = data[0].reshape(-1, data[0].size()[-1]).to(self.device)
            outp, M = self.__S__.forward([data_reshape, data[1]])

            M = M.transpose(0, 1).transpose(1, 2)

            M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)

            nloss, prec1 = self.__L__.forward(outp, label)

            if l2_reg_dict is not None:
                Learned_dict = l2_reg_dict
                l2_reg = 0
                for name, param in self.__S__.model.named_parameters():
                    if name in Learned_dict:
                        l2_reg = l2_reg + torch.norm(param - Learned_dict[name].cuda(), 2)
                tloss = nloss / nloss.detach() + self.weight_finetuning_reg * l2_reg / (l2_reg.detach() + 1e-5)
            else:
                tloss = nloss
                # print("Without L2 Reg")

            dim_loss = self.__DIM_L__(outp, M, M_prime)

            # add loss
            t_loss = tloss + dim_loss

            print(f"classification Loss: {tloss}, skloss {nloss} , DIM Loss: {dim_loss} total loss {t_loss} \n")
            return t_loss, prec1, nloss
