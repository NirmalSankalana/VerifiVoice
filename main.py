from verifyvoice import DataLoader
from verifyvoice import Similarity
import numpy as np
from verifyvoice.model.inference import get_model, get_emb
sample1 = "./samples/dr-uthaya-e1.mp3"


model = get_model()
s = DataLoader.load_audio(sample1, 160)
print(len(s))
eb = get_emb(sample1, model)

e1 = np.array([1, 2, 3, 4, 5])
e2 = np.array([6, 7, 8, 9, 10])

si = Similarity.cosine(e1, e2)
print(si)
