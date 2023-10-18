import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import ActorNetwork, CriticNetwork, OUNoise
from buffer import ExperienceSampler, MultiAgentReplayBuffer
from constants import DEFAULT_BATCH_SIZE, DEFAULT_BUFFER_SIZE, GAMMA, ACTOR_LR, CRIITC_LR, TAU, CRITIC_WEIGHT_DECAY, LEARN_EVERY
from typing import Optional, List
import os

class DDPGAgent:
    def __init__(self, id:int, num_agents:int, obs_dim:int, action_dim:int, noise:OUNoise, device:str="cpu", gamma:float=GAMMA, tau:float=TAU,
                 actor_lr:float=ACTOR_LR, critic_lr:float=CRIITC_LR, critic_weight_decay:float=CRITIC_WEIGHT_DECAY):
        self.id=id
        self.obs_dim=obs_dim
        self.action_dim=action_dim
        self.device = device
        self.discount = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.num_agents = num_agents
        self.critic_weight_decay=critic_weight_decay

        self.actor = ActorNetwork(self.obs_dim, self.action_dim).to(device)
        self.actor_target = ActorNetwork(self.obs_dim, self.action_dim).to(device)
        #for p in self.actor_target.parameters():
        #    p.requires_grad_ = False
        self.critic = CriticNetwork(self.obs_dim * self.num_agents, self.action_dim * self.num_agents).to(device)
        self.critic_target = CriticNetwork(self.obs_dim * self.num_agents, self.action_dim * self.num_agents).to(device)
        #for p in self.critic_target.parameters():
        #    p.requires_grad_ = False
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, weight_decay=self.critic_weight_decay)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), self.actor_lr)
        self.soft_update_both(1)

        self.noise = noise
    
    def soft_update(self, target_model:nn.Module, source_model:nn.Module, tau:Optional[float]):
        if tau is None:
            tau = self.tau
        with torch.no_grad():
            for t, s in zip(target_model.parameters(), source_model.parameters()):
                t.data.copy_(t.data * (1-tau) + s.data * tau)

    def soft_update_actor(self, tau:Optional[float]):
        self.soft_update(self.actor_target, self.actor, tau)

    def soft_update_critic(self, tau:Optional[float]):
        self.soft_update(self.critic_target, self.critic, tau)

    def soft_update_both(self, tau:Optional[float]):
        self.soft_update_actor(tau)
        self.soft_update_critic(tau)

    def choose_action(self, obs:torch.tensor, add_noise:bool)->np.ndarray:
        actions = self.actor(obs)
        #actions = actions.detach().cpu().numpy()
        if add_noise:
            actions += self.noise.noise()
            #actions += np.random.uniform(-1, 1, size=len(actions))
        actions = actions.detach().cpu().numpy()
        return np.clip(actions, -1, 1)

    def predict(self, obs:torch.tensor, local:bool)->torch.tensor:
        return self.actor(obs) if local else self.actor_target(obs)
    
    def learn(self, sampler:ExperienceSampler, current_action_predictions:List[torch.tensor],
              next_action_predictions:List[torch.tensor]):
        states = sampler.getStates()
        next_states = sampler.getNextStates()
        rewards = sampler.getRewards(self.id)
        dones = sampler.getDones(self.id)
        actions_taken = sampler.getActions()

        # Train critic
        critic_values = self.critic(states, actions_taken)
        q_next = self.critic_target(next_states, torch.cat(next_action_predictions, dim=1)).detach()

        #TODO how do we handle if some actors are done and others are not done
        # in our case done would be true for all agents or none, so we are good with
        # dones for current agent
        #critic_loss = F.mse_loss(critic_values, rewards + self.discount * q_next * (1 - dones))
        critic_loss = F.smooth_l1_loss(critic_values, rewards + self.discount * q_next * (1 - dones))
        with torch.autograd.set_detect_anomaly(True):
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

        # Train actor
        # Note we will take derivatives w.r.t all actors actions but
        # only this agents actor actions will udpate weights. We should detach other actions

        predictions = [current_action_predictions[i] if i == self.id else current_action_predictions[i].detach() \
                       for i in range(len(current_action_predictions))]
        #predictions = current_action_predictions

        actor_loss = -self.critic(states, torch.cat(predictions, dim=1))
        actor_loss = actor_loss.mean()
        with torch.autograd.set_detect_anomaly(True):
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

    def save(self, path:str):
        torch.save(self.critic.state_dict(), os.path.join(path, f"{self.id}-critic.pth"))
        torch.save(self.actor.state_dict(), os.path.join(path, f"{self.id}-actor.pth"))

    def load(self, path:str):
        self.critic.load_state_dict(torch.load(os.path.join(path, f"{self.id}-critic.pth")))
        self.critic_target.load_state_dict(torch.load(os.path.join(path, f"{self.id}-critic.pth")))
        self.actor.state_dict(torch.load(os.path.join(path, f"{self.id}-actor.pth")))
        self.actor_target.load_state_dict(torch.load(os.path.join(path, f"{self.id}-actor.pth")))

    def play(self, obs:torch.tensor):
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs)
        self.actor.train()
        return action


class MADDPGAgent:
    def __init__(self, num_agents:int, obs_dim:int, action_dim:int, device:str="cpu", gamma:float=GAMMA, tau:float=TAU,
                 actor_lr:float=ACTOR_LR, critic_lr:float=CRIITC_LR, critic_weight_decay:float=CRITIC_WEIGHT_DECAY,
                 buffer_size:int=DEFAULT_BUFFER_SIZE, batch_size:int=DEFAULT_BATCH_SIZE, learn_every:int=LEARN_EVERY):

        self.device = device
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.noise = OUNoise(self.action_dim)
        self.agents = [DDPGAgent(i, num_agents, obs_dim, action_dim, self.noise, device,
                                 gamma, tau, actor_lr, critic_lr, critic_weight_decay) for i in range(num_agents)]
        self.memory = MultiAgentReplayBuffer(buffer_size, batch_size, device)
        self.iteration = 0
        self.tau = tau
    
    def reset(self):
        self.noise.reset()

    def choose_actions(self, obs:torch.tensor, add_noise:bool=True)->np.ndarray:
        actions = np.asarray([self.agents[i].choose_action(obs[i], add_noise) for i in range(self.num_agents)])
        assert(actions.shape[0] == self.num_agents)
        assert(actions.shape[1] == self.action_dim)
        return actions

    def step(self, obs, actions, rewards, next_obs, dones):
        self.iteration += 1
        self.memory.add(obs, actions, rewards, next_obs, dones)
        if len(self.memory) > self.memory.batch_size and self.iteration % LEARN_EVERY == 0:
            epocs = 4 #len(self.memory) // self.memory.batch_size
            for i in range(4):
                self.learn()

    def learn(self):
        for i in range(self.num_agents):
            sampler = ExperienceSampler(self.memory, self.device)
            # get actions that we would have predicted given current policy. We use these to maximize
            # returns for current state and learn the policy
            current_action_predictions = [self.agents[i].actor(sampler.getObs(i)).to(self.device) for i in range(self.num_agents)]

            # to learn critic
            next_action_predictions = [self.agents[i].actor_target(sampler.getNextObs(i)).to(self.device).detach() for i in range(self.num_agents)]
            self.agents[i].learn(sampler, current_action_predictions, next_action_predictions)
        for i in range(self.num_agents):
            self.agents[i].soft_update_both(self.tau)

    def save(self, folder:str):
        for agent in self.agents:
            agent.save(folder)

    def pretrain(self, obs, actions, rewards, next_obs, dones):
        errors = []
        for i in range(1000000):
            sampler = ExperienceSampler(self.memory, self.device)
            for i in range(self.num_agents):
                # get actions that we would have predicted given current policy. We use these to maximize
                # returns for current state and learn the policy
                errors.append(self.agents[i].pretrain(self, sampler))
        print(f"completed {i} iterations")
        return errors
