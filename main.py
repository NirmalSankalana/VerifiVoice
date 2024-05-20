from verifyvoice import ModelLoader
from verifyvoice import Similarity
import numpy as np
# model_path='/home/thejan/Downloads/model000000013.model'


model = ModelLoader(model_name="WavLM", attention_heads=8)

spk_1_audio_1_path = '/home/thejan/Music/thjn1.wav'
spk_1_audio_2_path = '/home/thejan/Music/thjn2.wav'
spk_1_audio_3_path = '/home/thejan/Music/empty.wav'
spk_2_audio_1_path = '/media/thejan/ubuntu_data/wav_test/id10282/37XQxZ5lBD8/00001.wav'

emb1 = model.get_embedding(spk_1_audio_1_path)
emb2 = model.get_embedding(spk_1_audio_2_path)
emb3 = model.get_embedding(spk_2_audio_1_path)
emb4 = model.get_embedding(spk_1_audio_3_path)

# print(f"{emb1.shape} {emb2.shape}")

embe =  np.zeros((10, 256), dtype=np.float64)
# print(emb4)


print(Similarity.cosine_similarity(emb1, emb2).mean())
print(Similarity.cosine_similarity(emb2, emb3).mean())
