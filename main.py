import numpy as np
from verifyvoice.model.inference import  get_emb

test_path = '/media/thejan/ubuntu_data/wav_test/'
filename = test_path + 'id10270/5r0dWxy17C8/00001.wav'

# model = get_model()
get_emb(filename)
# s = DataLoader.load_audio(sample1, 160)
# print(len(s))
# eb =get_emb(sample1, model)

# e1 = np.array([1, 2, 3, 4, 5])
# e2 = np.array([6, 7, 8, 9, 10])

# si = Similarity.cosine(e1, e2)
# print(si)
