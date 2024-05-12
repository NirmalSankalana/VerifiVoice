from verifyvoice import ModelLoader
from verifyvoice import Similarity

model_path='/home/thejan/Downloads/model000000013.model'

model = ModelLoader(model_path=model_path)

spk_1_audio_1_path = '/media/thejan/ubuntu_data/wav_test/id10270/OmSWVqpb-N0/00001.wav'
spk_1_audio_2_path = '/media/thejan/ubuntu_data/wav_test/id10270/5r0dWxy17C8/00001.wav'
spk_2_audio_1_path = '/media/thejan/ubuntu_data/wav_test/id10282/37XQxZ5lBD8/00001.wav'

emb1 = model.get_embedding(spk_1_audio_1_path)
emb2 = model.get_embedding(spk_1_audio_2_path)
emb3 = model.get_embedding(spk_2_audio_1_path)

# print(emb)
# print(emb.shape)
print(f'Similarity of Same Speaker : {Similarity.cosine(emb1, emb2)}')
print(f'Similarity of Different Speaker ; {Similarity.cosine(emb1, emb3)}')
