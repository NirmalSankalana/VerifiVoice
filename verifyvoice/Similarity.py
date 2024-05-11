import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Similarity():

    @staticmethod
    def cosine(audio1, audio2):
        if type(audio1) != np.ndarray or type(audio2) != np.ndarray:
            raise TypeError(
                "The two audio files are not of type numpy.ndarray")
        # if len(audio1) != len(audio2):
        #     raise ValueError("The two audio files are not of the same length")
        return np.dot(audio1, audio2.T)/(np.sqrt(np.sum(audio1**2))*np.sqrt(np.sum(audio2**2)))

    def eer(audio1, audio2):
        # ref_feat = F.normalize(ref_feat - 0, p=2, dim=1)
        score_1 = np.mean(np.matmul(audio1, audio2.T))
        return score_1

