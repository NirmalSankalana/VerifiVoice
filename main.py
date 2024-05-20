from verifyvoice import ModelLoader
from verifyvoice.Similarity import cosine_similarity
import numpy as np
# model_path='/home/thejan/Downloads/model000000013.model'
"""
TODO: need to make ModelLoader function take attention heads and model like Wavlm and param
and in side load required model weights
and if not saved in cashe folder, download from hugging face
"""

model = ModelLoader(model_name="WavLM", attention_heads=8)

spk_1_audio_1_path = '/home/thejan/Music/thjn1.wav'
spk_1_audio_2_path = '/home/thejan/Music/thjn2.wav'
spk_1_audio_3_path = '/home/thejan/Music/empty.wav'
spk_2_audio_1_path = '/media/thejan/ubuntu_data/wav_test/id10282/37XQxZ5lBD8/00001.wav'

emb1 = model.get_embedding(spk_1_audio_1_path)
emb2 = model.get_embedding(spk_1_audio_2_path)
emb3 = model.get_embedding(spk_2_audio_1_path)
emb4 = model.get_embedding(spk_1_audio_3_path)

print(f"{emb1.shape} {emb2.shape}")

embe =  np.zeros((10, 256), dtype=np.float64)
print(emb4)
# print(emb1.shape)
# print(f'Similarity of Same Speaker : {Similarity.cosine(emb1, emb2)}')
# print(f'Similarity of Different Speaker ; {Similarity.cosine(emb2, emb4)}')
print(cosine_similarity(emb1, emb2).mean())
print(cosine_similarity(emb2, emb3).mean())
# print(cosine_similarity(emb4, emb1).mean())
# print(cosine_similarity(emb4, embe).mean())

# ref_feat = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
# ref_feat_2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
# score_1 = np.mean(np.matmul(ref_feat, ref_feat_2.T))
# print(score_1)

# ref_feat = emb1 / np.linalg.norm(emb3, axis=1, keepdims=True)
# ref_feat_2 = emb2 / np.linalg.norm(emb4, axis=1, keepdims=True)
# score_1 = np.mean(np.matmul(ref_feat, ref_feat_2.T))
# print(score_1)
