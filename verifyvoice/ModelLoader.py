from .DataLoader import DataLoader
import torch
import os
from huggingface_hub import hf_hub_download
from .model_args import get_default_param
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

model_threshold = {
    4: 0.24,
    8: 0.23,
    16: 0.21
}


class ModelLoader:
    """
    ModelLoader class for loading and working with deep learning models.

    get_embedding:
    Get the embedding for the audio data provided.

    Args:
        audio_path: Path to the audio file.
        evamode: Flag indicating evaluation mode (default is True).
        num_eval: Number of evaluations to perform (default is 10).
        vad: Flag indicating whether to use VAD mode (default is True).

    Returns:
        Embedding of the audio data.

    load_model:
    Load a model from a specified path or download it if not available locally.

    Args:
        model: The model object to load.
        model_name: Name of the model.
        model_path: Path to the directory where the model is stored.
        cache_dir: Directory for caching (default is cache_dir).

    Returns:
        None.

    get_threshold:
    Get the threshold for the model.

    loadParameters:
    Load parameters into the model.
    """

    def __init__(self, model_name, attention_heads=8):
        args["attention_heads"] = attention_heads
        self.model_name = model_head_mapping[attention_heads]
        self.model_path = model_path
        self.model =SpeakerNet(**args).to(args['device'])
        self.load_model(self.model, self.model_name, self.model_path)

    def get_embedding(self, audio_path, evamode=True, num_eval=10, vad=True):
        """
        Get the embedding for the audio data provided.

        Args:
            audio_path: Path to the audio file.
            evamode: Flag indicating evaluation mode (default is True).
            num_eval: Number of evaluations to perform (default is 10).
            vad: Flag indicating whether to use VAD mode (default is True).

        Returns:
            Embedding of the audio data.
        """
        feats = DataLoader.load_audio(filename=audio_path, evalmode=evamode, num_eval=num_eval, vad_mode=vad)
        if feats.size == 0:
            return None
        feats = torch.FloatTensor(feats)
        embedding = self.model([feats, 'test'])
        embedding = embedding.detach().cpu().numpy()
        return embedding

    def load_model(self, model, model_name, model_path, cache_dir=cache_dir):

        """
        Load a model from a specified path or download it if not available locally.

        Args:
            model: The model object to load.
            model_name: Name of the model.
            model_path: Path to the directory where the model is stored.
            cache_dir: Directory for caching (default is cache_dir).

        Returns:
            None.
        """
        model_file = os.path.join(model_path, model_name)

        if not os.path.exists(model_file):
            # Download the model from Hugging Face if not available locally
            model_file = hf_hub_download(repo_id="thejan-fonseka/DeepSpeakerVerifier", filename=model_name)
            
        # Load the model from the local file
        model.load_state_dict(torch.load(model_file, map_location=torch.device(args['device'])))        
        # self.loadParameters(torch.load(model_file, map_location=torch.device(args['device'])))
        model.eval()
    
    def get_threshold(self):
        """
        Get the threshold for the model.
        """
        return model_threshold[args["attention_heads"]]
    
    def loadParameters(self, param):
        """
        Load parameters into the model.
        """
        self_state = self.model.state_dict();
        loaded_state = param

        for name, param in loaded_state.items():
            origname = name;
            
            if name not in self_state:
                # print("%s is not in the model."%origname);
                continue;
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;
            self_state[name].copy_(param);