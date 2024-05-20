import os
from huggingface_hub import hf_hub_download
import torch

def get_default_param():
    """
    Get the default parameters for the model.

    Returns:
        Dictionary containing default parameters for the model.
    """

    args = {'config': None, 
            'max_frames': 300, 
            'eval_frames': 400, 
            'batch_size': 32, 
            'max_seg_per_spk': 500, 
            'nDataLoaderThread': 10, 
            'augment': False, 
            'seed': 20211202, 
            'test_interval': 1, 
            'max_epoch': 15, 
            'trainfunc': 'aamsoftmax', 
            'optimizer': 'adamw', 
            'scheduler': 'steplr', 
            'lr': 0.001, 
            'lr_decay': 0.95, 
            'pretrained_model_path': '/home/$USER/.cache/huggingface/hub/models--thejan-fonseka--DeepSpeakerVerifier/snapshots/*/WavLM-Base+.pt', 
            'weight_finetuning_reg': 0.01, 
            'LLRD_factor': 1.0, 
            'LR_Transformer': 2e-05, 
            'LR_MHFA': 0.005, 
            'hard_prob': 0.5, 
            'hard_rank': 10, 
            'margin': 0.2, 
            'scale': 30, 
            'nPerSpeaker': 1, 
            'nClasses': 1211, 
            'alpha': 0.01, 
            'beta': 0.05, 
            'gamma': 0.005, 
            'dcf_p_target': 0.05, 
            'dcf_c_miss': 1, 
            'dcf_c_fa': 1, 
            'initial_model': '', 
            'save_path': 'exps/exp1', 
            'train_list': '/mnt/proj3/open-24-5/pengjy_new/WavLM/vox_list/train_list.txt', 
            'test_list': '/mnt/proj3/open-24-5/pengjy_new/WavLM/vox_list/veri_test.txt', 
            'train_list_percentage': 1.0, 
            'train_path': '/mnt/proj3/open-24-5/plchot/data_wav/16kHz/voxceleb2/dev/aac/', 
            'test_path': '/mnt/proj3/open-24-5/plchot/data_wav/16kHz/voxceleb_1.1/', 
            'musan_path': '/mnt/proj3/open-24-5/pengjy_new/musan_split/', 
            'rir_path': '/mnt/proj3/open-24-5/plchot/data_augment/16kHz/simulated_rirs/', 
            'n_mels': 80, 
            'log_input': False, 
            'model': '', 
            'encoder_type': 
            'SAP', 'nOut': 256, 
            'eval': False, 
            'save_model_pt': False, 
            'port': '7888', 
            'distributed': False, 
            'mixedprec': False, 
            'attention_heads': 8, 
            'gpu': 0}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args['device'] = device

    if not os.path.exists(args["pretrained_model_path"]):
        model_file = hf_hub_download(repo_id="thejan-fonseka/DeepSpeakerVerifier", filename="WavLM-Base+.pt")
        args["pretrained_model_path"] = model_file

    return args

if __name__ == "__main__":
    get_default_param()