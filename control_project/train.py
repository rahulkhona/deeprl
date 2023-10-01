import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from typing import List, Tuple
from unityagents import UnityEnvironment
from agent import Agent
import os
import random

def ddpg(state_size:int, action_size:int, env:UnityEnvironment, brain_name:str, num_agents:int, n_episodes:int=5000, max_t:int=1200, print_every:int=100, random_seed:int=139595):
    agent = Agent(state_size, action_size, random_seed)
    scores_window = deque(maxlen=100)
    scores = []
    for episode in range(1, n_episodes):
        env_info = env.reset()[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros(num_agents)
        for i in range(max_t):

            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            for r in env_info.rewards:
                if np.isclose(r,0.1):
                    print("found 0.1 reward")
            score += env_info.rewards
            rewards = [r * 10 for r in env_info.rewards]
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            if any(dones):
                break
        scores_window.append(np.mean(score))
        scores.append(np.mean(score))
        if len(scores_window) >= 100 and np.mean(scores_window) >= 30:
            print("fsolved environment in {episode} episodes")
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            return scores, scores_window


        if episode % print_every == 0:
            print(f"\rEpisode {episode} average score {np.mean(scores_window)}")
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

    return scores, scores_window

def plot(scores:List[float], scores_window:deque):
    fig, axes = plt.subplots(2, 1)
    axes[0, 0] = plot(np.arange(len(scores)), scores)
    axes[0, 0].set_title("Scores")
    axes[0, 0].set_xlabel("timesteps")
    axes[0, 0].set_ylabel("episodes score")

    axes[1, 0] = plot(np.arange(len(scores_window)), list(scores_window))
    axes[1, 0].set_title("Last 100 scores")
    axes[1, 0].set_xlabel("timesteps")
    axes[1, 0].set_ylabel("episodes score")
    plt.show()
    plt.savefig("./plot.png")


if __name__ == '__main__':
    print("starting training")
    env = UnityEnvironment(file_name='./Reacher_multiagent.app', no_graphics=True)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]
    scores, scores_window = ddpg(state_size, action_size, env, brain_name, num_agents, print_every=10)
    plot(scores, scores_window)
    