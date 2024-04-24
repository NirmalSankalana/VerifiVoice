import numpy as np


class Similarity():
    @staticmethod
    def cosine(audio1, audio2):
        if len(audio1) != len(audio2):
            raise ValueError("The two audio files are not of the same length")
        return np.dot(audio1, audio2)/(np.sqrt(np.sum(audio1**2))*np.sqrt(np.sum(audio2**2)))
