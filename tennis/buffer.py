import torch
import numpy as np
from collections import deque
from typing import List, Tuple, NamedTuple, Optional
from constants import DEFAULT_BATCH_SIZE, DEFAULT_BUFFER_SIZE
from dataclasses import dataclass
import random
from enum import Enum


@dataclass
class MultiAgentExperience :
    obs : np.ndarray
    actions : np.ndarray
    rewards : np.ndarray
    next_obs : np.ndarray
    dones: np.ndarray

class MultiAgentReplayBuffer:
    def __init__(self, buffer_size:int=DEFAULT_BUFFER_SIZE, batch_size:int=DEFAULT_BATCH_SIZE, device:str="cpu"):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device

    def __len__(self)->int:
        return len(self.memory)

    def add(self, obs:List[np.ndarray], actions:List[np.ndarray], rewards:List[float], next_obs:List[np.ndarray], dones:List[bool]):
        self.memory.append(MultiAgentExperience(obs, actions, rewards, next_obs, dones))
        if np.all(rewards > 0.09):
            for i in range(5):
                self.memory.append(MultiAgentExperience(obs, actions, rewards, next_obs, dones))
        elif np.any(rewards > 0.09):
            for i in range(2):
                self.memory.append(MultiAgentExperience(obs, actions, rewards, next_obs, dones))
        elif np.all(rewards < 0.0):
            for i in range(5):
                self.memory.append(MultiAgentExperience(obs, actions, rewards, next_obs, dones))

    def sample(self)->List[MultiAgentExperience]:
        assert len(self) >= self.batch_size
        return random.choices(self.memory, k=self.batch_size)


class ExperienceSampler:
    def __init__(self, memory:MultiAgentReplayBuffer, device:str="cpu"):
        self.experiences = memory.sample()
        self.device = device
    
    def getObs(self, agent:Optional[int]=None)->torch.tensor:
        assert agent == None or (agent >=0 and agent <= len(self.experiences[0].obs))
        if agent is not None:
            return torch.from_numpy(np.vstack([e.obs[agent] for e in self.experiences])).float().to(self.device)
        else:
            return torch.from_numpy(np.vstack([e.obs.reshape(1, -1) for e in self.experiences])).float().to(self.device)

    def getNextObs(self, agent:Optional[int]=None)->torch.tensor:
        assert agent == None or (agent >=0 and agent <= len(self.experiences[0].next_obs))
        if agent is not None:
            return torch.from_numpy(np.vstack([e.next_obs[agent] for e in self.experiences])).float().to(self.device)
        else:
            return torch.from_numpy(np.vstack([e.next_obs.reshape(1, -1) for e in self.experiences])).float().to(self.device)

    def getActions(self, agent:Optional[int]=None)->torch.tensor:
        assert agent == None or (agent >=0 and agent <= len(self.experiences[0].obs))
        if agent is not None:
            return torch.from_numpy(np.vstack([e.actions[agent] for e in self.experiences])).float().to(self.device)
        else:
            return torch.from_numpy(np.vstack([e.actions.reshape(1, -1) for e in self.experiences])).float().to(self.device)

    def getRewards(self, agent:Optional[int]=None)->torch.tensor:
        assert agent == None or (agent >=0 and agent <= len(self.experiences[0].obs))
        if agent is not None:
            return torch.from_numpy(np.vstack([e.rewards[agent] for e in self.experiences])).float().to(self.device)
        else:
            return torch.from_numpy(np.vstack([e.rewards.reshape(1, -1) for e in self.experiences])).float().to(self.device)

    def getDones(self, agent:Optional[int]=None)->torch.tensor:
        assert agent == None or (agent >=0 and agent <= len(self.experiences[0].obs))
        if agent is not None:
            return torch.from_numpy(np.vstack([e.dones[agent] for e in self.experiences])).float().to(self.device)
        else:
            return torch.from_numpy(np.vstack([e.dones.reshape(1, -1) for e in self.experiences])).float().to(self.device)

    def getStates(self)->torch.tensor:
        return self.getObs(None)

    def getNextStates(self)->torch.tensor:
        return self.getNextObs(None)