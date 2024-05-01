import torch
import torch.nn.functional as F
from torch import nn as nn


class GlobalDiscriminator(nn.Module):
    def __init__(self, M_channels=150, interm_size=512):
        super().__init__()

        self.c0 = torch.nn.Conv1d(150, 75, kernel_size=3)
        self.c1 = torch.nn.Conv1d(75, 37, kernel_size=3)

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        # input of self.l0 is the concatenate of E and flattened output of self.c1 (C)
        in_feature = 37 * (768-2*2) + 256
        self.l0 = torch.nn.Linear(in_feature, interm_size)
        self.l1 = torch.nn.Linear(interm_size, interm_size)
        self.l2 = torch.nn.Linear(interm_size, 1)

    def forward(self, E, M):

        C = F.relu(self.c0(M))
        C = self.c1(C)
        C = C.view(E.shape[0], -1)
        out = torch.cat((E, C), dim=1)
        out = F.relu(self.l0(out))
        out = F.relu(self.l1(out))
        out = self.l2(out)

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        # output of Table 5
        return out

class LocalDiscriminator(nn.Module):

    def __init__(self, interm_channels=512):
        super().__init__()
        self.c0 = torch.nn.Conv1d(256+150, interm_channels, kernel_size=1)
        self.c1 = torch.nn.Conv1d(interm_channels, interm_channels, kernel_size=1)
        self.c2 = torch.nn.Conv1d(interm_channels, 1, kernel_size=1)

    def forward(self, x):
        score = F.relu(self.c0(x))
        score = F.relu(self.c1(score))
        score = self.c2(score)
        return score
    
# class GlobalDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.c0 = nn.Conv2d(1, 2, kernel_size=2)
#         # self.c1 = nn.Conv2d(2, 1, kernel_size=2)
#         # self.l0 = nn.Linear(1 * 752 + 768, 512)
#         # self.l1 = nn.Linear(512, 512)
#         # self.l2 = nn.Linear(512, 1)

#         self.c0 = nn.Conv1d(768, 1024, kernel_size=1)
#         self.c1 = nn.Conv1d(1024, 256, kernel_size=1)
#         self.l0 = nn.Linear(256 * 760 + 768, 512)
#         self.l1 = nn.Linear(512, 512)
#         self.l2 = nn.Linear(512, 1)

#     def forward(self, y, M):
#         h = F.relu(self.c0(M))
#         h = self.c1(h)
#         h = h.view(y.shape[0], -1)
#         h = torch.cat((y, h), dim=1)
#         h = F.relu(self.l0(h))
#         h = F.relu(self.l1(h))
#         return self.l2(h)

# class GlobalDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.c0 = nn.Conv1d(150, 64, kernel_size=3)
#         self.c1 = nn.Conv1d(64, 32, kernel_size=3)
#         self.l0 = nn.Linear(32 * (768 - 2**2) + 256, 512)
#         self.l1 = nn.Linear(512, 512)
#         self.l2 = nn.Linear(512, 1)

#     def forward(self, y, M):
#         h = F.relu(self.c0(M))
#         h = self.c1(h)
#         h = h.view(y.shape[0], -1)
#         h = torch.cat((y, h), dim=1)
#         h = F.relu(self.l0(h))
#         h = F.relu(self.l1(h))
#         return self.l2(h)

# class LocalDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.c0 = nn.Conv1d(150+256, 512, kernel_size=1)
#         self.c1 = nn.Conv1d(512, 512, kernel_size=1)
#         self.c2 = nn.Conv1d(512, 1, kernel_size=1)

#     def forward(self, y_M):
#         h = F.relu(self.c0(y_M))
#         h = F.relu(self.c1(h))
#         return self.c2(h)


# class GlobalDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc0 = nn.Linear(150 * 768, 64 * 768)
#         self.fc1 = nn.Linear(64 * 768, 32 * 768)
#         self.fc2 = nn.Linear(32 * 768 , 1024)
#         self.fc3 = nn.Linear(1024, 512)
#         self.fc4 = nn.Linear(512, 1)

#     def forward(self, y, M):
#         M = M.view(M.shape[0], -1)
#         M = F.relu(self.fc0(M))
#         M = self.fc1(M)
#         M = M.view(M.shape[0], -1)

#         h = torch.cat((y.view(y.shape[0], -1), M), dim=1)
#         h = F.relu(self.fc2(h))
#         h = F.relu(self.fc3(h))
#         return self.fc4(h)

# class LocalDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc0 = nn.Linear((150 + 256) * 768, 512 * 768)
#         self.fc1 = nn.Linear(512 * 768, 512 * 768)
#         self.fc2 = nn.Linear(512 * 768, 768)

#     def forward(self, y_M):
#         y_M = y_M.view(y_M.shape[0], -1)
#         h = F.relu(self.fc0(y_M))
#         h = F.relu(self.fc1(h))
#         return self.fc2(h).view(-1, 768)
    

# class LocalDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.c0 = nn.Conv2d(257, 128, kernel_size=1)
#         self.c1 = nn.Conv2d(128, 64, kernel_size=1)
#         self.c2 = nn.Conv2d(64, 1, kernel_size=1)

#     def forward(self, x):
#         h = F.relu(self.c0(x))
#         h = F.relu(self.c1(h))
#         return self.c2(h)

# class PriorDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.l0 = nn.Linear(256, 512)
#         self.l1 = nn.Linear(512, 128)
#         self.l2 = nn.Linear(128, 1)

#     def forward(self, x):
#         h = F.relu(self.l0(x))
#         h = F.relu(self.l1(h))
#         return torch.sigmoid(self.l2(h))


# class Classifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.l1 = nn.Linear(64, 15)
#         self.bn1 = nn.BatchNorm1d(15)
#         self.l2 = nn.Linear(15, 10)
#         self.bn2 = nn.BatchNorm1d(10)
#         self.l3 = nn.Linear(10, 10)
#         self.bn3 = nn.BatchNorm1d(10)

#     def forward(self, x):
#         encoded, _ = x[0], x[1]
#         clazz = F.relu(self.bn1(self.l1(encoded)))
#         clazz = F.relu(self.bn2(self.l2(clazz)))
#         clazz = F.softmax(self.bn3(self.l3(clazz)), dim=1)
#         return clazz


# class DeepInfoAsLatent(nn.Module):
#     def __init__(self, run, epoch):
#         super().__init__()
#         model_path = Path(r'c:/data/deepinfomax/models') / Path(str(run)) / Path('encoder' + str(epoch) + '.wgt')
#         self.encoder = Encoder()
#         self.encoder.load_state_dict(torch.load(str(model_path)))
#         self.classifier = Classifier()

#     def forward(self, x):
#         z, features = self.encoder(x)
#         z = z.detach()
#         return self.classifier((z, features))

# class GlobalDiscriminator(nn.Module):
#     def __init__(self, input_dim=115456):
#         super().__init__()
#         self.l0 = nn.Linear(input_dim, 512)
#         self.l1 = nn.Linear(512, 512)
#         self.l2 = nn.Linear(512, 1)

#     def forward(self, y, M):
#         print(f"Inside GIM")
#         print(f"{M.shape=}")
#         M_flat = M.flatten(start_dim=1)
#         print(f"{M_flat.shape=}")

#         h = torch.cat((y, M_flat), dim=1)
#         print(f"{h.shape=}") 
#         # h = torch.cat((y, M), dim=1)
#         h = F.relu(self.l0(h))
#         print(f"after l0 : {h.shape=}") 

#         h = F.relu(self.l1(h))
#         print(f"after l1: {h.shape=}") 

#         h = self.l2(h)
#         print(f"after l2: {h.shape=}") 

#         return h


# class LocalDiscriminator(nn.Module):
#     def __init__(self, x_dim=768):
#         super().__init__()
#         self.l0 = nn.Linear(x_dim, 512)  # Adjust input size based on x
#         self.l1 = nn.Linear(512, 512)
#         self.l2 = nn.Linear(512, 1)

#     def forward(self, x):
#         print(f"x : {x.shape=}")
#         h = F.relu(self.l0(x))
#         print(f"after l0 :{h.shape=}")
#         h = F.relu(self.l1(h))
#         print(f"after l1 :{h.shape=}")
#         o = self.l2(h)
#         print(f"after l2 : {o.shape=}")
#         return o
    
# class LocalDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc0 = nn.Linear((150 + 256) * 768, 512 * 768)
#         self.fc1 = nn.Linear(512 * 768, 512 * 768)
#         self.fc2 = nn.Linear(512 * 768, 768)

#     def forward(self, y_M):
#         y_M = y_M.view(y_M.shape[0], -1)
#         h = F.relu(self.fc0(y_M))
#         h = F.relu(self.fc1(h))
#         return self.fc2(h).view(-1, 768)
    