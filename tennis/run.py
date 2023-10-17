from unityagents import UnityEnvironment
from agent import MADDPGAgent
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
import constants as C
import random
import math

def plot(scores, image_folder):
    x = [i for i in range(len(scores))]
    fig, ax = plt.subplots()
    ax.set_ylabel("episode scores")
    ax.set_xlabel("epidsode number")
    plt.plot(x, scores)
    if image_folder:
        plt.savefig(os.path.join(image_folder, "plot.png"))
    plt.show()

def run(seed:int=0x10020303):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    env = UnityEnvironment(file_name="./Tennis.app", no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    device = 'cpu' if not torch.cuda.is_available() else "cuda"

    agent = MADDPGAgent(num_agents, state_size, action_size)

    scores = []
    scores_window = deque(maxlen=C.SCORES_WINDOW_LENGTH)
    print("Starting training")
    steps = 0
    for i in range(1, C.NUM_EPISDOES + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        obs = env_info.vector_observations
        score = np.zeros(num_agents)

        while True:
            actions = agent.choose_actions(torch.from_numpy(obs).float().to(device), True if i < 100000 else False)
            #actions = np.asarray([np.random.uniform(-1, 1, 2), np.random.uniform(-1, 1, 2)])
            env_info = env.step(actions)[brain_name]
            rewards = np.asarray(env_info.rewards)
            rewards *= 10
            dones = np.array(env_info.local_done).astype(np.uint8)
            next_obs = env_info.vector_observations
            agent.step(obs, actions, rewards , next_obs, dones)
            obs = next_obs
            score += rewards
            steps += 1
            if np.any(dones):
                break
        scores.append(np.max(score))
        scores_window.append(np.max(score))
        if len(scores_window) == C.SCORES_WINDOW_LENGTH and np.mean(scores_window) > C.WINNING_AVG:
            print(f"solved environment in {i} episodes with avg score of {np.mean(scores_window)}")
            agent.save("./")
            env.close()
            return scores, scores_window, True, i
        if i % C.PRINT_EVERY == 0:
            print(f"completed {i} episodes and average score is {np.mean(scores_window)} average steps per episode {steps/i}", end="\n")
    
    env.close()
    return scores, scores_window, False, i

if __name__ == '__main__':
    scores, score_window, solved, iterations = run()
    if solved:
        plot(scores, "./")
        plot(score_window)
    print("\ndone")
    exit(0)
    