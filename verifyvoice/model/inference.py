import torch
import os
import DatasetLoader

test_path = '/media/thejan/ubuntu_data/wav_test/'
filename = 'id10270/x6uYqmx31kE/00001.wav'
model = torch.load('/home/thejan/Downloads/model_final.pth')
model.eval()

# for param in model.parameters():
#     print(param)
print("ffffffffffffffffffff\n\n\n\n\n")
print(model)

print('After Reading')

PATH = os.path.join(test_path, filename)


audio = DatasetLoader.loadWAV(PATH ,max_frames=300, evalmode=True, num_eval=1)
audio = torch.FloatTensor(audio)
inp1 = audio
        
ref_feat = model([inp1, "test"])
print(torch.cuda.device_count())
print(f"{ref_feat.shape=}")
print(f"{ref_feat=}")