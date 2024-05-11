import torch
from verifyvoice.model.SpeakerNet import SpeakerNet, WrappedModel, ModelTrainer
import os
import argparse
from verifyvoice.model.DatasetLoader import loadWAV

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

parser = argparse.ArgumentParser(description="SpeakerNet");

parser.add_argument('--config', type=str, default=None, help='Config YAML file');

## Data loader
parser.add_argument('--max_frames', type=int, default=300, help='Input length to the network for training');
parser.add_argument('--eval_frames', type=int, default=400,
                    help='Input length to the network for testing; 0 uses the whole files');
parser.add_argument('--batch_size', type=int, default=32, help='Batch size, number of speakers per batch');
parser.add_argument('--max_seg_per_spk', type=int, default=500,
                    help='Maximum number of utterances per speaker per epoch');
parser.add_argument('--nDataLoaderThread', type=int, default=10, help='Number of loader threads');
parser.add_argument('--augment', type=bool, default=False, help='Augment input')
parser.add_argument('--seed', type=int, default=20211202, help='Seed for the random number generator');

# Training details
parser.add_argument('--test_interval', type=int, default=1, help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch', type=int, default=15, help='Maximum number of epochs');
parser.add_argument('--trainfunc', type=str, default="aamsoftmax", help='Loss function');

# Optimizer
parser.add_argument('--optimizer', type=str, default="adamw", help='sgd or adam');
parser.add_argument('--scheduler', type=str, default="steplr", help='Learning rate scheduler');
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate');
parser.add_argument("--lr_decay", type=float, default=0.95, help='Learning rate decay every [test_interval] epochs');

# Pre-trained Transformer Model
parser.add_argument('--pretrained_model_path', type=str, default="/media/thejan/ubuntu_data/WavLM-Base+.pt", help='Absolute path to the pre-trained model');
parser.add_argument('--weight_finetuning_reg', type=float, default=0.01,
                    help='L2 regularization towards the initial pre-trained model');
parser.add_argument('--LLRD_factor', type=float, default=1.0, help='Layer-wise Learning Rate Decay (LLRD) factor');
parser.add_argument('--LR_Transformer', type=float, default=2e-5, help='Learning rate of pre-trained model');
parser.add_argument('--LR_MHFA', type=float, default=5e-3, help='Learning rate of back-end attentive pooling model');

# Loss functions
parser.add_argument("--hard_prob", type=float, default=0.5,
                    help='Hard negative mining probability, otherwise random, only for some loss functions');
parser.add_argument("--hard_rank", type=int, default=10,
                    help='Hard negative mining rank in the batch, only for some loss functions');
parser.add_argument('--margin', type=float, default=0.2, help='Loss margin, only for some loss functions');
parser.add_argument('--scale', type=float, default=30, help='Loss scale, only for some loss functions');
parser.add_argument('--nPerSpeaker', type=int, default=1,
                    help='Number of utterances per speaker per batch, only for metric learning based losses');
parser.add_argument('--nClasses', type=int, default=1211,
                    help='Number of speakers in the softmax layer, only for softmax-based losses');
parser.add_argument('--alpha', type=float, default=0.01, help='Global InfoMax Hyperparameter');
parser.add_argument('--beta', type=float, default=0.05, help='Local InfoMax Hyperparameter');
parser.add_argument('--gamma', type=float, default=0.005, help='Local InfoMax Hyperparameter');


# Evaluation parameters
parser.add_argument('--dcf_p_target', type=float, default=0.05,
                    help='A priori probability of the specified target speaker');
parser.add_argument('--dcf_c_miss', type=float, default=1, help='Cost of a missed detection');
parser.add_argument('--dcf_c_fa', type=float, default=1, help='Cost of a spurious detection');

# Load and save
parser.add_argument('--initial_model', type=str, default="", help='Initial model weights');
parser.add_argument('--save_path', type=str, default="exps/exp1", help='Path for model and logs');

# Training and test data
# Training and test data
parser.add_argument('--train_list', type=str, default="/mnt/proj3/open-24-5/pengjy_new/WavLM/vox_list/train_list.txt",
                    help='Train list');
parser.add_argument('--test_list', type=str, default="/mnt/proj3/open-24-5/pengjy_new/WavLM/vox_list/veri_test.txt",
                    help='Evaluation list');
parser.add_argument('--train_list_percentage', type=float, default=1.0, help='percentage of train dataset to consider in training');


parser.add_argument('--train_path', type=str, default="/mnt/proj3/open-24-5/plchot/data_wav/16kHz/voxceleb2/dev/aac/",
                    help='Absolute path to the train set');
parser.add_argument('--test_path', type=str, default="/mnt/proj3/open-24-5/plchot/data_wav/16kHz/voxceleb_1.1/",
                    help='Absolute path to the test set');
parser.add_argument('--musan_path', type=str, default="/mnt/proj3/open-24-5/pengjy_new/musan_split/",
                    help='Absolute path to the test set');
parser.add_argument('--rir_path', type=str, default="/mnt/proj3/open-24-5/plchot/data_augment/16kHz/simulated_rirs/",
                    help='Absolute path to the test set');

## Model definition
parser.add_argument('--n_mels', type=int, default=80, help='Number of mel filterbanks');
parser.add_argument('--log_input', type=bool, default=False, help='Log input features')
parser.add_argument('--model', type=str, default="", help='Name of model definition');
parser.add_argument('--encoder_type', type=str, default="SAP", help='Type of encoder');
parser.add_argument('--nOut', type=int, default=256, help='Embedding size in the last FC layer');

## For test only
parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')
parser.add_argument('--save_model_pt', dest='save_model_pt', action='store_true', help='Eval only')

## Distributed and mixed precision training
parser.add_argument('--port', type=str, default="7888", help='Port for distributed training, input as text');
parser.add_argument('--distributed', dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec', dest='mixedprec', action='store_true', help='Enable mixed precision training')

args = parser.parse_args();
args.gpu = 0

model_path='/home/thejan/Downloads/model000000013.model'
model = torch.load(model_path)
for i in model.keys():
    print(i)

s = SpeakerNet(**vars(args)).cuda(args.gpu)
# s = WrappedModel(s)

#     # print(model)
s.load_state_dict(model)

# # model = torch.load(model_path)
# # print(model.keys())
s.eval()
audio_path = '/media/thejan/ubuntu_data/wav_test/id10270/5r0dWxy17C8/00001.wav'
inp1 = loadWAV(filename=audio_path, max_frames=300, evalmode=True, num_eval=10)
inp1=torch.FloatTensor(inp1)
print(inp1)
print(inp1.shape)

ref_feat = s([inp1, "test"]).cuda()
print(ref_feat.shape)
print(ref_feat)