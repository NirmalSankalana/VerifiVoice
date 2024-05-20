from verifyvoice import ModelLoader
from verifyvoice import Similarity
import numpy as np
# model_path='/home/thejan/Downloads/model000000013.model'


model = ModelLoader(model_name="WavLM", attention_heads=8)

spk_1_audio_1_path = '/home/thejan/Music/thjn1.wav'
spk_1_audio_2_path = '/home/thejan/Music/thjn2.wav'
spk_1_audio_3_path = '/home/thejan/Music/empty.wav'
spk_2_audio_1_path = '/media/thejan/ubuntu_data/wav_test/id10282/37XQxZ5lBD8/00001.wav'

a1 = "/media/thejan/ubuntu_data/wav_test/id10282/37XQxZ5lBD8/00001.wav"
a2 = "/media/thejan/ubuntu_data/wav_test/id10282/37XQxZ5lBD8/00002.wav"
a3 = "/media/thejan/ubuntu_data/wav_test/id10292/0H91VC07Q3s/00003.wav"

# emb1 = model.get_embedding(spk_1_audio_1_path)
# emb2 = model.get_embedding(spk_1_audio_2_path)
# emb3 = model.get_embedding(spk_2_audio_1_path)
# emb4 = model.get_embedding(spk_1_audio_3_path)

ae1 = model.get_embedding(a1)
ae2 = model.get_embedding(a2)
ae3 = model.get_embedding(a3)
# print(f"{emb1.shape} {emb2.shape}")

embe =  np.zeros((10, 256), dtype=np.float64)
# print(emb4)


# print(Similarity.cosine_similarity(emb1, emb2))
# print(Similarity.cosine_similarity(emb2, emb3))

print(Similarity.cosine_similarity(ae1, ae2))
print(Similarity.cosine_similarity(ae1, ae3))
print(Similarity.cosine_similarity(ae2, ae3))
