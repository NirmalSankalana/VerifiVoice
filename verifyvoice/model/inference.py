import torch
from DatasetLoader import loadWAV
import numpy as np


test_path = '/media/thejan/ubuntu_data/wav_test/'
filename = test_path + 'id10270/5r0dWxy17C8/00001.wav'


def cosine(audio1, audio2):
        if type(audio1) != np.ndarray or type(audio2) != np.ndarray:
            raise TypeError(
                "The two audio files are not of type numpy.ndarray")
        if len(audio1) != len(audio2):
            raise ValueError("The two audio files are not of the same length")
        return np.dot(audio1, audio2.T)/(np.sqrt(np.sum(audio1**2))*np.sqrt(np.sum(audio2**2)))


def load_model(model_path='/home/thejan/Downloads/model_final.pth'):
    model = torch.load(model_path)
    model.eval()
    # print(model)
    return model

# def get_model():
#     return model

def get_emb(filename):
    model = load_model()

    audio = loadWAV(filename,max_frames=300, evalmode=True, num_eval=1)
    audio = torch.FloatTensor(audio)
    inp1 = audio

    ref_feat = model([inp1, "test"])
    print(f"{ref_feat.shape=}")
    return ref_feat.cpu().detach()


emb1=get_emb('/media/thejan/ubuntu_data/wav_test/id10270/5r0dWxy17C8/00001.wav')
emb2=get_emb('/media/thejan/ubuntu_data/wav_test/id10271/1gtz-CUIygI/00001.wav')

print(f"cosine of different : {cosine(emb1.numpy(), emb2.numpy())}")

emb3=get_emb('/media/thejan/ubuntu_data/wav_test/id10270/zjwijMp0Qyw/00001.wav')

print(f"cosine of same : {cosine(emb1.numpy(), emb3.numpy())}")


