from verifyvoice import ModelLoader
from verifyvoice import Similarity

# model_path='/home/thejan/Downloads/model000000013.model'
"""
TODO: need to make ModelLoader function take attention heads and model like Wavlm and param
and in side load required model weights
and if not saved in cashe folder, download from hugging face
"""

model = ModelLoader(model_name="WavLM", attention_heads=8)

spk_1_audio_1_path = '/media/thejan/ubuntu_data/wav_test/id10270/OmSWVqpb-N0/00001.wav'
spk_1_audio_2_path = '/media/thejan/ubuntu_data/wav_test/id10270/5r0dWxy17C8/00002.wav'
spk_1_audio_3_path = '/media/thejan/ubuntu_data/wav_test/id10270/5r0dWxy17C8/00005.wav'
spk_2_audio_1_path = '/media/thejan/ubuntu_data/wav_test/id10282/37XQxZ5lBD8/00001.wav'

emb1 = model.get_embedding(spk_1_audio_1_path)
emb2 = model.get_embedding(spk_1_audio_2_path)
emb3 = model.get_embedding(spk_2_audio_1_path)
emb4 = model.get_embedding(spk_1_audio_3_path)

print()
print(f'Similarity of Same Speaker : {Similarity.cosine(emb1, emb2)}')
print(f'Similarity of Different Speaker ; {Similarity.cosine(emb2, emb4)}')
