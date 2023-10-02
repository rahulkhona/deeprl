[//]: # (Image References)

[image1]: https://github.com/rahulkhona/deeprl/blob/main/control_project/solved_plot.png

# High level Report
## Algorithm used
The learnging algorithm used for this environment was DDPG. The actor model uses 3 linear layers, with last layer having tanh activation and prior layers using relu activation.  The critic uses 3 linear layers. The models are defined in <code>models.py</code> file.

Agent uses epsilon greedy algorithm to explore actions. I tried using OUNoise, but that performed signifanctly worse than epsilon-greedy and did not solve the environment even after running for several hours. I also stop exploring after 200 episodes as at this point actor seems to have learned to predict optimal action for the states and further exploration worsens the performacne a bit.

The algorithm also scales rewards for training so that there is a bigger difference between different reward values. This facilitated agent converging significantly faster and solve the environment in <b>234 episodes</b>.

## Trained agent Plot
![image1] You can find the plof the trained agent here (https://github.com/rahulkhona/deeprl/blob/main/control_project/solved_plot.png)

## Model file names
1. checkpoint_actor.pth
2. checkout_critic.pth
   
## Hyper parameters
- BUFFER_SIZE = int(1e5)  # replay buffer size
- BATCH_SIZE = 128        # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 1e-3              # for soft update of target parameters
- LR_ACTOR = 1e-4         # learning rate of the actor 
- LR_CRITIC = 1e-3        # learning rate of the critic
- WEIGHT_DECAY = 0        # L2 weight decay
- #SIGMA_MIN=0.0001
- SIGMA_MIN=0.05
- SIGMA_DECAY=0.99
- EPS=0.5 # starting point of epsilong greedy, it decays with every episode till it reaches SIGMA_MIN


## Ideas for future work
Would like to explore D4PG, PPO and A2C algorithms to see if they can solve the environment faster.