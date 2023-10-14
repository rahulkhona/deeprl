import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from constants import NOISE_MU, NOISE_SCALE, NOISE_SCALE_DECAY, NOISE_SIGMA, NOISE_THETA

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorNetwork(nn.Module):
    def __init__(self, obs_size:int, action_size:int):
        super().__init__()
        #self.layers = nn.Sequential(
        #    nn.Linear(obs_size, 400),
        #    nn.LeakyReLU(),
        #    nn.Linear(400, 2)
        #)
        self.fc1 = nn.Linear(obs_size, 16)
        self.fc3 = nn.Linear(16, 2)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        #self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x:torch.tensor)->torch.tensor:
        out = F.relu(self.fc1(x))
        #out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return F.tanh(out)


class CriticNetwork(nn.Module):
    def __init__(self, state_size:int, all_actions_size:int):
        super().__init__()
        self.fcs1 = nn.Linear(state_size, 16)
        self.fc1 = nn.Linear(16 + all_actions_size, 64)
        self.fc2 = nn.Linear(64, 20)
        self.fc3 = nn.Linear(20, 1)
        self.reset_parameters()
        #self.layers = nn.Sequential(nn.Linear(state_size + all_actions_size, 800),
        #                            nn.LeakyReLU(),
        #                            nn.Linear(800, 400),
        #                            nn.LeakyReLU(),
        #                            nn.Linear(400, 1)
        #                            )

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state:torch.tensor, all_actions:torch.tensor)->torch.tensor:
        x = self.fcs1(state)
        input = torch.cat((x, all_actions), dim=1)
        #return self.layers(input)
        out = F.relu(self.fc1(input))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class OUNoise:

    def __init__(self, action_dimension:int, scale:float=NOISE_SCALE, mu=NOISE_MU,
                 theta:float=NOISE_THETA, sigma:float=NOISE_SIGMA, decay:float=NOISE_SCALE_DECAY):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.decay = decay
        self.reset()

    def reset(self):
        self.scale = self.scale * self.decay
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()
