import torch
import DatasetLoader

test_path = '/media/thejan/ubuntu_data/wav_test/'


def get_model(model_path='/home/thejan/Downloads/model_final.pth'):
    model = torch.load(model_path)
    model.eval()
    print(model)
    return model


def get_emb(filename, model):
    audio = DatasetLoader.loadWAV(filename,max_frames=300, evalmode=True, num_eval=1)
    audio = torch.FloatTensor(audio)
    inp1 = audio

    ref_feat = model([inp1, "test"])
    print(f"{ref_feat.shape=}")
    return ref_feat