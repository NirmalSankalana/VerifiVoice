import onnx

onnx_model = onnx.load("/home/thejan/Music/model_onnx.onnx")
onnx.checker.check_model(onnx_model)


import onnxruntime as ort
import numpy as np


from DataLoader import DataLoader

feats = DataLoader.loadWAV("/media/thejan/ubuntu_data/wav_test/id10270/5r0dWxy17C8/00001.wav")
x = [feats, 'test']
ort_sess = ort.InferenceSession('/home/thejan/Music/model_onnx.onnx')
outputs = ort_sess.run(None, {'': x})
print(outputs)
print(outputs.shape)