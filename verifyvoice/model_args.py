import os
from huggingface_hub import hf_hub_download
import torch

def get_default_param():
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
            'pretrained_model_path': '/home/$USER/.cache/huggingface/hub/models--thejan-fonseka--DeepSpeakerVerifier/snapshots/8d215ef073ee00a5637af677a78e37a0858ad0b1/WavLM-Base+.pt', 
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

    # parser = argparse.ArgumentParser(description="SpeakerNet");

    # parser.add_argument('--config', type=str, default=None, help='Config YAML file');

    # ## Data loader
    # parser.add_argument('--max_frames', type=int, default=300, help='Input length to the network for training');
    # parser.add_argument('--eval_frames', type=int, default=400,
    #                     help='Input length to the network for testing; 0 uses the whole files');
    # parser.add_argument('--batch_size', type=int, default=32, help='Batch size, number of speakers per batch');
    # parser.add_argument('--max_seg_per_spk', type=int, default=500,
    #                     help='Maximum number of utterances per speaker per epoch');
    # parser.add_argument('--nDataLoaderThread', type=int, default=10, help='Number of loader threads');
    # parser.add_argument('--augment', type=bool, default=False, help='Augment input')
    # parser.add_argument('--seed', type=int, default=20211202, help='Seed for the random number generator');

    # # Training details
    # parser.add_argument('--test_interval', type=int, default=1, help='Test and save every [test_interval] epochs');
    # parser.add_argument('--max_epoch', type=int, default=15, help='Maximum number of epochs');
    # parser.add_argument('--trainfunc', type=str, default="aamsoftmax", help='Loss function');

    # # Optimizer
    # parser.add_argument('--optimizer', type=str, default="adamw", help='sgd or adam');
    # parser.add_argument('--scheduler', type=str, default="steplr", help='Learning rate scheduler');
    # parser.add_argument('--lr', type=float, default=0.001, help='Learning rate');
    # parser.add_argument("--lr_decay", type=float, default=0.95, help='Learning rate decay every [test_interval] epochs');

    # # Pre-trained Transformer Model
    # parser.add_argument('--pretrained_model_path', type=str, default="/home/$USER/.cache/huggingface/hub/models--thejan-fonseka--DeepSpeakerVerifier/snapshots/8d215ef073ee00a5637af677a78e37a0858ad0b1/WavLM-Base+.pt", help='Absolute path to the pre-trained model');
    # parser.add_argument('--weight_finetuning_reg', type=float, default=0.01,
    #                     help='L2 regularization towards the initial pre-trained model');
    # parser.add_argument('--LLRD_factor', type=float, default=1.0, help='Layer-wise Learning Rate Decay (LLRD) factor');
    # parser.add_argument('--LR_Transformer', type=float, default=2e-5, help='Learning rate of pre-trained model');
    # parser.add_argument('--LR_MHFA', type=float, default=5e-3, help='Learning rate of back-end attentive pooling model');

    # # Loss functions
    # parser.add_argument("--hard_prob", type=float, default=0.5,
    #                     help='Hard negative mining probability, otherwise random, only for some loss functions');
    # parser.add_argument("--hard_rank", type=int, default=10,
    #                     help='Hard negative mining rank in the batch, only for some loss functions');
    # parser.add_argument('--margin', type=float, default=0.2, help='Loss margin, only for some loss functions');
    # parser.add_argument('--scale', type=float, default=30, help='Loss scale, only for some loss functions');
    # parser.add_argument('--nPerSpeaker', type=int, default=1,
    #                     help='Number of utterances per speaker per batch, only for metric learning based losses');
    # parser.add_argument('--nClasses', type=int, default=1211,
    #                     help='Number of speakers in the softmax layer, only for softmax-based losses');
    # parser.add_argument('--alpha', type=float, default=0.01, help='Global InfoMax Hyperparameter');
    # parser.add_argument('--beta', type=float, default=0.05, help='Local InfoMax Hyperparameter');
    # parser.add_argument('--gamma', type=float, default=0.005, help='Local InfoMax Hyperparameter');


    # # Evaluation parameters
    # parser.add_argument('--dcf_p_target', type=float, default=0.05,
    #                     help='A priori probability of the specified target speaker');
    # parser.add_argument('--dcf_c_miss', type=float, default=1, help='Cost of a missed detection');
    # parser.add_argument('--dcf_c_fa', type=float, default=1, help='Cost of a spurious detection');

    # # Load and save
    # parser.add_argument('--initial_model', type=str, default="", help='Initial model weights');
    # parser.add_argument('--save_path', type=str, default="exps/exp1", help='Path for model and logs');

    # # Training and test data
    # # Training and test data
    # parser.add_argument('--train_list', type=str, default="/mnt/proj3/open-24-5/pengjy_new/WavLM/vox_list/train_list.txt",
    #                     help='Train list');
    # parser.add_argument('--test_list', type=str, default="/mnt/proj3/open-24-5/pengjy_new/WavLM/vox_list/veri_test.txt",
    #                     help='Evaluation list');
    # parser.add_argument('--train_list_percentage', type=float, default=1.0, help='percentage of train dataset to consider in training');


    # parser.add_argument('--train_path', type=str, default="/mnt/proj3/open-24-5/plchot/data_wav/16kHz/voxceleb2/dev/aac/",
    #                     help='Absolute path to the train set');
    # parser.add_argument('--test_path', type=str, default="/mnt/proj3/open-24-5/plchot/data_wav/16kHz/voxceleb_1.1/",
    #                     help='Absolute path to the test set');
    # parser.add_argument('--musan_path', type=str, default="/mnt/proj3/open-24-5/pengjy_new/musan_split/",
    #                     help='Absolute path to the test set');
    # parser.add_argument('--rir_path', type=str, default="/mnt/proj3/open-24-5/plchot/data_augment/16kHz/simulated_rirs/",
    #                     help='Absolute path to the test set');

    # ## Model definition
    # parser.add_argument('--n_mels', type=int, default=80, help='Number of mel filterbanks');
    # parser.add_argument('--log_input', type=bool, default=False, help='Log input features')
    # parser.add_argument('--model', type=str, default="", help='Name of model definition');
    # parser.add_argument('--encoder_type', type=str, default="SAP", help='Type of encoder');
    # parser.add_argument('--nOut', type=int, default=256, help='Embedding size in the last FC layer');

    # ## For test only
    # parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')
    # parser.add_argument('--save_model_pt', dest='save_model_pt', action='store_true', help='Eval only')

    # ## Distributed and mixed precision training
    # parser.add_argument('--port', type=str, default="7888", help='Port for distributed training, input as text');
    # parser.add_argument('--distributed', dest='distributed', action='store_true', help='Enable distributed training')
    # parser.add_argument('--mixedprec', dest='mixedprec', action='store_true', help='Enable mixed precision training')

    # parser.add_argument('--attention_heads', type=int, default=8, help='Number of attention heads');

    # args = parser.parse_args();
    # # print(f"args    {args}\n\n\n\n")

    # args.gpu = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args['device'] = device

    if not os.path.exists(args["pretrained_model_path"]):
            # Download the model from Hugging Face if not available locally
            # model_url = hf_hub_download(repo_id=model_name, filename=f"{model_name}.pt", cache_dir=cache_dir)
            # os.makedirs(model_path, exist_ok=True)
        model_file = hf_hub_download(repo_id="thejan-fonseka/DeepSpeakerVerifier", filename="WavLM-Base+.pt")
        # print(f"downloaded model from hugging face saved to {model_file}")
        args["pretrained_model_path"] = model_file
            # Save the downloaded model to the desired folder
            # torch.hub.download_url_to_file(model_url, model_file)
    return args

if __name__ == "__main__":
    get_default_param()