[//]: # (Image References)

[image1]: https://github.com/rahulkhona/deeprl/blob/4491cb061c6a16434fff1a1d7ac0225985771208/tennis/plot.png

# High level Report
## Algorithm used
The learnging algorithm used for this environment was MADDPG. The actor model uses 5 linear layers, with last layer having tanh activation and prior layers using relu activation.  The critic uses 5 linear layers with batch normalization after 1st layer. The models are defined in <code>model.py</code> file.

Agents use OUNoise for exploration and in addition rewards are scaled by 10 to create significant separation between 0 and -0.01 rewards. This seems to be helping in training.

The environment was solved in 1122 episodes.


## Trained agent Plot
![image1] You can find the plof the trained agent here (https://github.com/rahulkhona/deeprl/blob/4491cb061c6a16434fff1a1d7ac0225985771208/tennis/plot.png)

## Model file names
1. 0-actor.pth
2. 1-actor.pth
3. 0-critic.pth
4. 1-critic.pth
   
## Hyper parameters
- DEFAULT_BUFFER_SIZE=int(1e5)
- DEFAULT_BATCH_SIZE=64#128
- GAMMA= 0.99 #0.99
- ACTOR_LR=1e-4
- CRIITC_LR=1e-3
- CRITIC_WEIGHT_DECAY=0
- TAU=1e-3
- NOISE_SCALE=5.0
- NOISE_SCALE_DECAY=0.95
- NOISE_SCALE_MIN=0.01
- NOISE_SIGMA=0.2
- NOISE_MU=0
- NOISE_THETA=0.15
- LEARN_EVERY=2
- NUM_EPISDOES=5000
- WINNING_AVG=0.5
- SCORES_WINDOW_LENGTH=100
- PRINT_EVERY=10


## Ideas for future work
In this environment there are only 3 reward values possibel 0.1, 0 and -0.01 and also best outcome is that at every step each agent
gets a reward of 0.1. There might be a possibility to use GAN with critic using cross entropy loss against the loss for discrimination and
KL divergence that minmizes distance from 0.1 reward for generator loss function. This might train faster.

Also for MAADPG environment, we might consider discretizing state space as states near each other might do equally well with the same action. This might also help with training the model faster.