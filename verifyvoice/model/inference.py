import torch
import DatasetLoader

test_path = '/media/thejan/ubuntu_data/wav_test/'
filename = test_path + 'id10270/5r0dWxy17C8/00001.wav'


def load_model(model_path='/home/thejan/Downloads/model_final.pth'):
    model = torch.load(model_path)
    model.eval()
    print(model)
    return model

# def get_model():
#     return model

def get_emb(filename):
    model = load_model()

    audio = DatasetLoader.loadWAV(filename,max_frames=300, evalmode=True, num_eval=1)
    audio = torch.FloatTensor(audio)
    inp1 = audio

    ref_feat = model([inp1, "test"])
    print(f"{ref_feat.shape=}")
    return ref_feat


get_emb('/media/thejan/ubuntu_data/wav_test/id10270/5r0dWxy17C8/00001.wav')