import numpy as np
from .DataLoader import DataLoader
import torch
from verifyvoice.model.SpeakerNet import SpeakerNet
import os
import argparse    
import os
from huggingface_hub import hf_hub_download
from .model_args import get_default_param

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

args = get_default_param()

model_head_mapping = {
    4: "/home/thejan/Downloads/model000000013.model",
    8: "/home/thejan/Downloads/model000000013.model",
    16: "/home/thejan/Downloads/model000000013.model",
    32: "/home/thejan/Downloads/model000000013.model"
}



class ModelLoader:
    def __init__(self, model_name, attention_heads=8):
        args.attention_heads = attention_heads
        self.model_path = model_head_mapping[attention_heads]
        self.model =SpeakerNet(**vars(args)).cuda(args.gpu)
        self.load_model(self.model, self.model_path)

    def get_embedding(self, audio_path):
        feats = DataLoader.load_audio(audio_path)
        feats = torch.FloatTensor(feats)
        embedding = self.model([feats, 'test'])
        embedding = embedding.detach().cpu().numpy()
        return embedding

    def load_model(self, model, model_path, cache_dir=None):
        # model_file = os.path.join(model_path, f"{model_name}.pt")

        if not os.path.exists(model_path):
            # Download the model from Hugging Face if not available locally
            # model_url = hf_hub_download(repo_id=model_name, filename=f"{model_name}.pt", cache_dir=cache_dir)

            # Save the downloaded model to the desired folder
            os.makedirs(model_path, exist_ok=True)
            torch.hub.download_url_to_file(model_url, model_file)

        # Load the model from the local file
        model.load_state_dict(torch.load(self.model_path))        
        model.eval()
        