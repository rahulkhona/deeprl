[//]: # (Image References)
[Image1]: https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif
# Project 2 : Continous Control

## Introduction
In this project we work with [Reacher environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher). In this environment a double jointed-arm can move to target locations. A reward of +0.1 is provided for each step that agent's hand is in goal location. The goal of the agent is to maintain hand's position in goal location for as many time steps as possible.

The observation consists of 33 variables and the action is made of 4 numbers corresponding to the torque of each joint. Each of these action numbers can take values between -1 and 1.

Two different environments were available for this project. The first version contains single agent and 2nd version contains two agents. These agents are checked in this repo as Reacher_1agent.app and Reacher_multiagent.app

## Success criteria
For 1 agent environment, the genet must average score of +30 over 100 episodes. For 2 agent version, the agent must average score of +30 over 20 agents over 100 episodes.

##Getting started
Please follow instructions [https://github.com/udacity/deep-reinforcement-learning#dependencies] to setup the environment. 
- <i>Note:</i> replace requirements.txt under python folder with following as that env is old and dependencies dont install well. Essentially relax the version constraints and also remove tensorflow.
    - pillow
    - matplotlib
    - numpy
    - jupyter
    - pytest
    - docopt
    - pyyaml
    - protobuf==3.5.2
    - grpcio
    - torch
    - pandas
    - scipy
    - ipykernel

## Instructions to train the model
1. reate a virtual environment using conda or tool of your choice with python 3.6
2. activate the new environment
3. install dependencies using `python -m pip install requirements.txt`
4. Read report.md to learn more about the implementation
5. start the Jupyter Notebook in your environment. For macos use following command `~/opt/anaconda3/bin/jupyter_mac.command`
6. Run all the cells below **4. It's your turn** cell
