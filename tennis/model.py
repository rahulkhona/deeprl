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
        self.fc1 = nn.Linear(obs_size, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 2)
    
    def reset_parameters(self):
        pass

    def forward(self, x:torch.tensor)->torch.tensor:
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return F.tanh(out)


class CriticNetwork(nn.Module):
    def __init__(self, state_size:int, all_actions_size:int):
        super().__init__()
        self.fc1 = nn.Linear(state_size + all_actions_size, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
        #self.layers = nn.Sequential(nn.Linear(state_size + all_actions_size, 800),
        #                            nn.LeakyReLU(),
        #                            nn.Linear(800, 400),
        #                            nn.LeakyReLU(),
        #                            nn.Linear(400, 1)
        #                            )

    def forward(self, state:torch.tensor, all_actions:torch.tensor)->torch.tensor:
        input = torch.cat(state, all_actions, dim=1)
        #return self.layers(input)
        out = F.relu(self.fc(input))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
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
