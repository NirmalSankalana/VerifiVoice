import numpy as np
from .DataLoader import DataLoader
import torch
import os
from huggingface_hub import hf_hub_download
from .model_args import get_default_param
import requests
from .SpeakerNet import SpeakerNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

args = get_default_param()

model_path = '/home/$USER/Downloads/models'
cache_dir = '/home/$USER/.cache/huggingface/hub/models--thejan-fonseka--DeepSpeakerVerifier'

model_head_mapping = {
    4: "model_h4_dim.model",
    8: "model_h8_dim.model",
    16: "model_h16_dim.model"
}

model_name_download_mapping = {
    "model_h4_dim.model": "https://huggingface.co/thejan-fonseka/DeepSpeakerVerifier/blob/main/model_h4_dim.model",
    "model_h8_dim.model": "https://huggingface.co/thejan-fonseka/DeepSpeakerVerifier/blob/main/model_h8_dim.model",
    "model_h16_dim.model": "https://huggingface.co/thejan-fonseka/DeepSpeakerVerifier/blob/main/model_h16_dim.model",
}


class ModelLoader:
    def __init__(self, model_name, attention_heads=8):
        args["attention_heads"] = attention_heads
        self.model_name = model_head_mapping[attention_heads]
        self.model_path = model_path
        self.model =SpeakerNet(**args).cuda(0)
        self.load_model(self.model, self.model_name, self.model_path)

    def get_embedding(self, audio_path):
        feats = DataLoader.load_audio(audio_path)
        feats = torch.FloatTensor(feats)
        embedding = self.model([feats, 'test'])
        embedding = embedding.detach().cpu().numpy()
        return embedding

    def load_model(self, model, model_name, model_path, cache_dir=cache_dir):
        model_file = os.path.join(model_path, model_name)

        if not os.path.exists(model_file):
            # Download the model from Hugging Face if not available locally
            # model_url = hf_hub_download(repo_id=model_name, filename=f"{model_name}.pt", cache_dir=cache_dir)
            # os.makedirs(model_path, exist_ok=True)
            # print("downloading model")
            model_file = hf_hub_download(repo_id="thejan-fonseka/DeepSpeakerVerifier", filename=model_name)
            # print(f"downloaded model {model_name} from hugging face saved to {model_file}")
            # Save the downloaded model to the desired folder
            # torch.hub.download_url_to_file(model_url, model_file)
        # Load the model from the local file
        model.load_state_dict(torch.load(model_file))        
        model.eval()
