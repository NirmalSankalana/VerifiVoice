#!/usr/bin/python
# -*- coding: utf-8 -*-

import importlib
import random
import sys
import time
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from verifyvoice.model.DatasetLoader import test_dataset_loader
import verifyvoice.model.DatasetLoader
from verifyvoice.model.loss.deepInfoMaxLoss import DeepInfoMaxLoss
from verifyvoice.model.models.Baseline.Spk_Encoder import MainModel
from verifyvoice.model.loss.aamsoftmax import LossFunction

class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None, l2_reg_dict=None):
        return self.module(x, label)
    

class SpeakerNet(nn.Module):

    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, **kwargs):
        super(SpeakerNet, self).__init__();

        self.__S__ = MainModel(**kwargs);

        self.__L__ = LossFunction(**kwargs);
        self.__DIM_L__ = DeepInfoMaxLoss(alpha=kwargs['alpha'], beta=kwargs['beta'], gamma=kwargs['gamma'])

        self.nPerSpeaker = nPerSpeaker
        self.weight_finetuning_reg = kwargs['weight_finetuning_reg']

    def forward(self, data, label=None, l2_reg_dict=None):
        if label is None:
            data_reshape = data[0].cuda()
            outp, M = self.__S__.forward([data_reshape, data[1]])
            return outp
        else:
            data_reshape = data[0].reshape(-1, data[0].size()[-1]).cuda()
            outp, M = self.__S__.forward([data_reshape, data[1]])

            # print(f"data = {data}")
            # print(f"data shape= {len(data)}")
            # print(f"data reshape= {data[0].shape}")
            # print(f"label shape= {len(label)}")
            # print(f"M trans before == {M.shape}")

            M = M.transpose(0, 1).transpose(1, 2)
    
            # print(f"M == {M.shape}")
            # print(f"{M=}")

            # M = M.unsqueeze(1)
            # print(f"{M=}")

            M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
            # print(f"y = {outp.shape}")
            # print()
            # # print(outp[0])
            # print()            # M_e = M.unsqueeze(-1)
            # print(f"M e== {M_e.shape}")

            # print(f"M_prime == {M_prime[0].shape}")
            # # print(M)
            # print()
            # print(M_prime)
            # print(f"ffffffffffffffff {torch.eq(M[0], M_prime[-1])}\n")
            # print(f"ffffffffffffffff2222 {torch.eq(M[1], M_prime[0])}\n")
            nloss, prec1 = self.__L__.forward(outp, label)



            if l2_reg_dict is not None:
                Learned_dict = l2_reg_dict
                l2_reg = 0
                for name, param in self.__S__.model.named_parameters():
                    if name in Learned_dict:
                        l2_reg = l2_reg + torch.norm(param - Learned_dict[name].cuda(), 2)
                tloss = nloss / nloss.detach() + self.weight_finetuning_reg * l2_reg / (l2_reg.detach() + 1e-5)
            else:
                tloss = nloss
                # print("Without L2 Reg")

            dim_loss = self.__DIM_L__(outp, M, M_prime)

            # add loss
            t_loss = tloss + dim_loss


            print(f"classification Loss: {tloss}, skloss {nloss} , DIM Loss: {dim_loss} total loss {t_loss} \n")

            return t_loss, prec1, nloss


class ModelTrainer(object):

    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):

        self.__model__ = speaker_model
        # print(f'{self.__model__=}')
        # print(f'{len(list(self.__model__.parameters()))=}')


        # print(f"\n\n\n")
        WavLM_params = list(map(id, self.__model__.module.__S__.model.parameters()))
        DIM_params = list(map(id, self.__model__.module.__DIM_L__.parameters()))
        # print(f"self.__model__.module.__S__.model.parameters() = {self.__model__.module.__S__.model}")
        # Backend_params = filter(lambda p: id(p) not in WavLM_params, self.__model__.module.parameters())
        Backend_params = filter(lambda p: id(p) not in WavLM_params and id(p) not in DIM_params, self.__model__.module.parameters())

        # print(f"Backend_params = {len(list(Backend_params))}")
        # print(f"WavLM_params = {len(list(WavLM_params))}")
        # print(f"DIM_params = {len(list(DIM_params))}")

        # print(f"param self.__model__.module {sum(p.numel() for p in self.__model__.module.parameters())}")
        # print(f"param self.__model__.module.__S__ {sum(p.numel() for p in self.__model__.module.__S__.parameters())}")
        # print(f"param self.__model__.module.__S__.model {sum(p.numel() for p in self.__model__.module.__S__.model.parameters())}")
        # print(f"param self.__model__.module.__S__.backend {sum(p.numel() for p in self.__model__.module.__S__.backend.parameters())}")
        # print(f"param self.__model__.module.__DIM_L__ {sum(p.numel() for p in self.__model__.module.__DIM_L__.parameters())}")
        # print(f"param self.__model__.module.__L__ {sum(p.numel() for p in self.__model__.module.__L__.parameters())}")

        # quit()
        # print(f"{len(list(map(id, self.__model__.module.parameters())))=}")
        # print(f"\n\n\n")
        # # print(f"self.__model__.module.parameters() = {self.__model__.module}")
        # print(f"\n\n\n")
        # # print(f"{self.__model__.module.__L__}")
        # print(f"{len(list(map(id, self.__model__.module.__L__.DIMLoss.parameters())))=}")
        

        # for i in Backend_params:
        #     print(i)
        # print(f"{self.__model__.module}")
        # print()
        # print()
        # print(f"{self.__model__.module.__S__.model}")
        # print(f"{list(Backend_params)=}")
        # print(f"{WavLM_params=}")

        self.path = kwargs['pretrained_model_path']

        Optimizer = importlib.import_module(f'optimizer.{optimizer}').__getattribute__('Optimizer')

        # Define the initial param groups
        param_groups = [{'params': Backend_params, 'lr': kwargs['LR_MHFA']}]
        # print(f"params: Backend_params = {Backend_params} \nkwargs : {kwargs['LR_MHFA']}")
        # quit()
        # Extract the encoder layers
        encoder_layers = self.__model__.module.__S__.model.encoder.layers

        # Iterate over the encoder layers to create param groups
        for i in range(12):  # Assuming 12 layers from 0 to 11 (for BASE model, when it comes to LARGE model, 12->24)
            lr = kwargs['LR_Transformer'] * (kwargs['LLRD_factor'] ** i)
            param_groups.append({'params': encoder_layers[i].parameters(), 'lr': lr})

        # Initialize the optimizer with these param groups
        self.__optimizer__ = Optimizer(param_groups, **kwargs)
        self.__optimizer__dim__ = Optimizer(self.__model__.module.__DIM_L__.parameters(), lr=1e-4)


        # self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        Scheduler = importlib.import_module(f'scheduler.{scheduler}').__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)
        # self.__scheduler__dim__, self.lr_step_dim = Scheduler(self.__optimizer__dim__, **kwargs)

        # self.scaler = GradScaler() 

        self.gpu = gpu

        self.mixedprec = mixedprec
        print(f"Mix prec: {self.mixedprec}")

        assert self.lr_step in ['epoch', 'iteration']

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====
    def train_network(self, loader, verbose):

        self.__model__.train();

        stepsize = loader.batch_size;

        counter = 0;
        index = 0;
        loss = 0;
        top1 = 0  # EER or accuracy

        tstart = time.time()
        Learned_dict = {}
        checkpoint = torch.load(self.path)
        for name, param in checkpoint['model'].items():
            if 'w2v_encoder.w2v_model.' in name:
                newname = name.replace('w2v_encoder.w2v_model.', '')
            else:
                newname = name
            Learned_dict[newname] = param;

        for data, data_label in loader:
            data = data.transpose(1, 0)
            self.__model__.zero_grad()
            label = torch.LongTensor(data_label).cuda()

            nloss, prec1, spkloss = self.__model__([data, "train"], label, Learned_dict)

            nloss.backward();

            self.__optimizer__.step();
            self.__optimizer__dim__.step();

            loss += spkloss.detach().cpu()
            top1 += prec1.detach().cpu()

            counter += 1;
            index += stepsize;

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing (%d) " % (index));
                sys.stdout.write(
                    "Loss %f TEER/TAcc %2.3f%% - %.2f Hz " % (loss / counter, top1 / counter, stepsize / telapsed));
                sys.stdout.flush();
                # print(f"\nIteration: {counter}, loss not divide: {loss}, Loss: {loss / counter:.4f}, top not divide {top1}, Top1: {top1 / counter}%")

            if self.lr_step == 'iteration': self.__scheduler__.step()

        if self.lr_step == 'epoch': self.__scheduler__.step()

        sys.stdout.write("\n");

        return (loss / counter, top1 / counter);

    # ===== ===== ===== ===== ===== ===== ===== =====
    # Evaluate from list
    # ===== ===== ===== ===== ===== ===== ===== =====
    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, print_interval=10, num_eval=5, test_list_percentage=1.0,**kwargs):

        self.__model__.eval();

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        # Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        num_samples = int(len(lines) * test_list_percentage)
        print(f"test datset : {len(lines)}")
        print(f"num_samples: {num_samples}")

        print('After Reading')

        # Get a list of unique file names
        files = sum([x.strip().split()[-2:] for x in lines[: num_samples]], [])
        setfiles = list(set(files))
        setfiles.sort()
        print('After sorting')

        # Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
        )
        ref_feat_list = []
        ref_feat_2_list = []
        max_len = 0
        forward = 0
        # Extract features for every image
        for idx, data in enumerate(test_loader):

            inp1 = data[0][0].cuda()
            inp2 = data[1][0].cuda()
            # print(f"{inp1=}")
            # print(f"{inp2=}")
            # print(f"{inp1.shape=}")
            # print(f"{inp2.shape=}")
            # print(f"{len(data)=}")
            # print(f"{len(data[0])=}")
            # print(f"{len(data[0][0])=}")
            # print(f"{data[0][0].shape=}")

            telapsed_2 = time.time()
            b, utt_l = inp2.shape
            if utt_l > max_len:
                max_len = utt_l
            ref_feat = self.__model__([inp1, "test"]).cuda()
            # print(f"{ref_feat.shape=}")
            # print(f"{ref_feat=}")

            ref_feat = ref_feat.detach().cpu()

            ref_feat_2 = self.__model__(
                [inp2[:, :700000], "test"]).cuda()  # The reason why here is set to 700000 is due to GPU memory size.
            ref_feat_2 = ref_feat_2.detach().cpu()
            # print(f"{ref_feat_2.shape=}")
            # print(f"{ref_feat_2=}")
            # print(f"{data[2][0]=}")
            # print(f"{data=}")

            feats[data[2][0]] = [ref_feat, ref_feat_2]

            ref_feat_list.extend(ref_feat.numpy())
            ref_feat_2_list.extend(ref_feat_2.numpy())

            telapsed = time.time() - tstart
            forward = forward + time.time() - telapsed_2

            if idx % print_interval == 0:
                sys.stdout.write(
                    "\rReading %d of %d: %.2f Hz, forward speed: %.2f Hz, embedding size %d, max_len %d" % (
                        idx, len(setfiles), idx / telapsed, idx / forward, ref_feat.size()[-1], max_len));

        print('')
        all_scores = [];
        all_labels = [];
        all_trials = [];
        all_scores_1 = [];
        all_scores_2 = [];

        tstart = time.time()

        ref_feat_list = numpy.array(ref_feat_list)
        ref_feat_2_list = numpy.array(ref_feat_2_list)

        ref_feat_list_mean = 0
        ref_feat_2_list_mean = 0

        # Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split();

            # Append random label if missing
            if len(data) == 2: data = [random.randint(0, 1)] + data

            ref_feat, ref_feat_2 = feats[data[1]]
            com_feat, com_feat_2 = feats[data[2]]

            # if self.__model__.module.__L__.test_normalize:
            ref_feat = F.normalize(ref_feat - ref_feat_list_mean, p=2, dim=1)  # B, D
            com_feat = F.normalize(com_feat - ref_feat_list_mean, p=2, dim=1)
            ref_feat_2 = F.normalize(ref_feat_2 - ref_feat_2_list_mean, p=2, dim=1)  # B, D
            com_feat_2 = F.normalize(com_feat_2 - ref_feat_2_list_mean, p=2, dim=1)

            score_1 = torch.mean(torch.matmul(ref_feat, com_feat.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(ref_feat_2, com_feat_2.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()

            all_scores.append(score);
            all_scores_1.append(score_1);
            all_scores_2.append(score_2);

            all_labels.append(int(data[0]));
            all_trials.append(data[1] + " " + data[2])

            if idx % (10 * print_interval) == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz" % (idx, len(lines), idx / telapsed));
                sys.stdout.flush();

        print('')

        return all_scores, all_labels, all_trials, all_scores_1, all_scores_2;

    def saveParameters(self, path):
        torch.save(self.__model__.module.state_dict(), path);
    
    def save_model(self, path):
        torch.save(self.__model__, path);

    # ===== ===== ===== ===== ===== ===== ===== =====
    # Load parameters
    # ===== ===== ===== ===== ===== ===== ===== =====
    def loadParameters(self, path):

        self_state = self.__model__.module.state_dict();
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu);
        # loaded_state = torch.load(path, map_location="cpu");
        # print(f"gggggggggggb {loaded_state.keys()=}")

        for name, param in loaded_state.items():
            origname = name;

            if name not in self_state:
                name = name.replace("module.", "");

            if name not in self_state:
                print(f"{origname} is not in the model.");
                continue;

            if self_state[name].size() != loaded_state[origname].size():
                print(
                    f"Wrong parameter length: {origname}, model: {self_state[name].size()}, loaded: {loaded_state[origname].size()}");
                continue;

            self_state[name].copy_(param);

    def infer(self, test_list, test_path, nDataLoaderThread, print_interval=1, num_eval=1, test_list_percentage=1.0,**kwargs):

        self.__model__.eval();

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        # Read all lines
        with open(test_list) as f:
            lines = f.readlines()
        print(lines)

        num_samples = int(len(lines) * test_list_percentage)
        print(f"test datset : {len(lines)}")
        print(f"num_samples: {num_samples}")

        print('After Reading')

        # Get a list of unique file names
        # files = sum([x.strip().split()[-1] for x in lines[: num_samples]], [])
        # setfiles = list(set(files))
        # setfiles.sort()
        filename = os.path.join(test_path, lines[0])
        print('After sorting')

        # Define test data loader
        # test_dataset = test_dataset_loader(setfiles, test_path, num_eval=1, **kwargs)
        # test_loader = torch.utils.data.DataLoader(
        #     test_dataset,
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=nDataLoaderThread,
        #     drop_last=False,
        # )
        # ref_feat_list = []
        # ref_feat_2_list = []
        # max_len = 0
        # forward = 0

        audio = DatasetLoader.loadWAV(filename,max_frames=300, evalmode=True, num_eval=1)
        audio = torch.FloatTensor(audio)
        inp1 = audio.cuda()
        
        ref_feat = self.__model__([inp1, "test"]).cuda()
        print(f"{ref_feat.shape=}")
        print(f"{ref_feat=}")

        return ref_feat
        # Extract features for every image
        # for idx, data in enumerate(test_loader):

        #     inp1 = data[0][0].cuda()
        #     inp2 = data[1][0].cuda()
        #     print(f"{inp1=}")
        #     print(f"{inp2=}")
        #     print(f"{inp1.shape=}")
        #     print(f"{inp2.shape=}")
        #     print(f"{len(data)=}")
        #     print(f"{len(data[0])=}")
        #     print(f"{len(data[0][0])=}")
        #     print(f"{data[0][0].shape=}")

        #     telapsed_2 = time.time()
        #     b, utt_l = inp2.shape
        #     if utt_l > max_len:
        #         max_len = utt_l
        #     ref_feat = self.__model__([inp1, "test"]).cuda()
        #     print(f"{ref_feat.shape=}")
        #     print(f"{ref_feat=}")

        #     ref_feat = ref_feat.detach().cpu()

        #     ref_feat_2 = self.__model__(
        #         [inp2[:, :700000], "test"]).cuda()  # The reason why here is set to 700000 is due to GPU memory size.
        #     ref_feat_2 = ref_feat_2.detach().cpu()
        #     print(f"{ref_feat_2.shape=}")
        #     print(f"{ref_feat_2=}")
        #     print(f"{data[2][0]=}")
        #     print(f"{data=}")

        #     feats[data[2][0]] = [ref_feat, ref_feat_2]

        #     ref_feat_list.extend(ref_feat.numpy())
        #     ref_feat_2_list.extend(ref_feat_2.numpy())

        #     telapsed = time.time() - tstart
        #     forward = forward + time.time() - telapsed_2

        #     if idx % print_interval == 0:
        #         sys.stdout.write(
        #             "\rReading %d of %d: %.2f Hz, forward speed: %.2f Hz, embedding size %d, max_len %d" % (
        #                 idx, len(setfiles), idx / telapsed, idx / forward, ref_feat.size()[-1], max_len));

        # print('')
        # all_scores = [];
        # all_labels = [];
        # all_trials = [];
        # all_scores_1 = [];
        # all_scores_2 = [];

        # tstart = time.time()

        # ref_feat_list = numpy.array(ref_feat_list)
        # ref_feat_2_list = numpy.array(ref_feat_2_list)

        # ref_feat_list_mean = 0
        # ref_feat_2_list_mean = 0

        # # Read files and compute all scores
        # for idx, line in enumerate(lines):

        #     data = line.split();

        #     # Append random label if missing
        #     if len(data) == 2: data = [random.randint(0, 1)] + data

        #     ref_feat, ref_feat_2 = feats[data[1]]
        #     com_feat, com_feat_2 = feats[data[2]]

        #     # if self.__model__.module.__L__.test_normalize:
        #     ref_feat = F.normalize(ref_feat - ref_feat_list_mean, p=2, dim=1)  # B, D
        #     com_feat = F.normalize(com_feat - ref_feat_list_mean, p=2, dim=1)
        #     ref_feat_2 = F.normalize(ref_feat_2 - ref_feat_2_list_mean, p=2, dim=1)  # B, D
        #     com_feat_2 = F.normalize(com_feat_2 - ref_feat_2_list_mean, p=2, dim=1)

        #     score_1 = torch.mean(torch.matmul(ref_feat, com_feat.T))  # higher is positive
        #     score_2 = torch.mean(torch.matmul(ref_feat_2, com_feat_2.T))
        #     score = (score_1 + score_2) / 2
        #     score = score.detach().cpu().numpy()

        #     all_scores.append(score);
        #     all_scores_1.append(score_1);
        #     all_scores_2.append(score_2);

        #     all_labels.append(int(data[0]));
        #     all_trials.append(data[1] + " " + data[2])

        #     if idx % (10 * print_interval) == 0:
        #         telapsed = time.time() - tstart
        #         sys.stdout.write("\rComputing %d of %d: %.2f Hz" % (idx, len(lines), idx / telapsed));
        #         sys.stdout.flush();

        # print('')

        # return all_scores, all_labels, all_trials, all_scores_1, all_scores_2;
