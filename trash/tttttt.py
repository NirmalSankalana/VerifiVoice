from huggingface_hub import hf_hub_download

file = hf_hub_download(repo_id="thejan-fonseka/DeepSpeakerVerifier", filename="model_h8_dim.model")
print(file)