import torch
from torch import nn


class Model(torch.nn.Module):
    def __init__(self, obs_dim,act_dim):
        super().__init__()
        hid1_size = 128
        hid2_size = 128
        # 3层全连接网络
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(in_features=obs_dim,out_features=256)
        self.fc2 = torch.nn.Linear(in_features=256,out_features=128)
        self.fc3 = torch.nn.Linear(in_features=128,out_features=act_dim)

    def forward(self, obs):
        h1 = self.fc1(obs)
        h1 = self.relu(h1)
        h2 = self.fc2(h1)
        h2 = self.relu(h2)
        Q = self.fc3(h2)
        return Q

class DeepQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(DeepQNetwork, self).__init__()
        self.hidden_size = 256
        self.num_layers = 2

        self.gru = nn.GRU(obs_dim, self.hidden_size, act_dim, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_size,out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=act_dim)

    def forward(self, x):
        #h0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        #c0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        out, hidden = self.gru(x)
        h1 = self.fc1(out)
        h1 = torch.nn.functional.relu(h1)
        out = self.fc2(h1)
        #print(out.shape)
        return out