import torch
import torch.nn as nn
import torch.nn.functional as F
from .WavLM import *

import torch
import torch.nn as nn

class MHFA(nn.Module):
    """
    MHFA class for Multi-Head Feature Aggregation.

    forward:
    Perform the forward pass of the MHFA model.

    Args:
        x: Input tensor.

    Returns:
        Output tensor after feature aggregation.
    """

    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256):
        super(MHFA, self).__init__()

        self.weights_k = nn.Parameter(data=torch.ones(13), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(13), requires_grad=True)

        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim

        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)

    def forward(self, x):

        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)

        att_k = self.att_head(k)
        v = v.unsqueeze(-2)

        pooling_outs = torch.sum(v.mul(nn.functional.softmax(att_k, dim=1).unsqueeze(-1)), dim=1)

        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)

        outs = self.pooling_fc(pooling_outs)

        return outs


class spk_extractor(nn.Module):
    """
    spk_extractor class for speaker feature extraction.

    forward:
    Perform the forward pass of the spk_extractor model.

    Args:
        wav_and_flag: Input tensor containing audio data and flag.

    Returns:
        Output tensor and last layer representation.
    """

    def __init__(self,device,**kwargs):
        super(spk_extractor, self).__init__()
        # print("Pre-trained Model: {}".format(kwargs['pretrained_model_path']))
        checkpoint = torch.load(kwargs['pretrained_model_path'], map_location=torch.device(device))
        cfg = WavLMConfig(checkpoint['cfg'])
        # cfg = WavLMConfig()
        self.model = WavLM(cfg)
        self.loadParameters(checkpoint['model'])
        head_nb = kwargs['attention_heads']
        self.backend = MHFA(head_nb=head_nb)
        # print(f"Number of Heads : {head_nb}")


    def forward(self,wav_and_flag):
        
        x = wav_and_flag[0]

        cnn_outs, layer_results, last_layer =  self.model.extract_features(x, output_layer=13)
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
       
        x = torch.stack(layer_reps).transpose(0,-1).transpose(0,1)
        out = self.backend(x)
        return out, last_layer

    def loadParameters(self, param):

        self_state = self.model.state_dict();
        loaded_state = param

        for name, param in loaded_state.items():
            origname = name;
            
            if name not in self_state:
                # print("%s is not in the model."%origname);
                continue;
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;
            self_state[name].copy_(param);

def MainModel(device,**kwargs):
    model = spk_extractor(device, **kwargs)
    return model