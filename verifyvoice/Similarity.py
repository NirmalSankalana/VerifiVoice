import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Similarity():

    @staticmethod
    def cosine__(audio1, audio2):
        if type(audio1) != np.ndarray or type(audio2) != np.ndarray:
            raise TypeError(
                "The two audio files are not of type numpy.ndarray")
        # if len(audio1) != len(audio2):
        #     raise ValueError("The two audio files are not of the same length")
        return np.dot(audio1, audio2.T)/(np.sqrt(np.sum(audio1**2))*np.sqrt(np.sum(audio2**2)))

    def cosine(a, b):
        # ref_feat = F.normalize(ref_feat - 0, p=2, dim=1)
        dot_product = np.dot(a, b.T)
        magnitude_A = np.linalg.norm(a)
        magnitude_B = np.linalg.norm(b)

        if magnitude_A == 0 or magnitude_B == 0:
            return 0
        
        cosine_similarity = dot_product / (magnitude_A * magnitude_B)
        score_1 = np.mean(cosine_similarity)
        score = max(0, min(1, score_1))
        return score

