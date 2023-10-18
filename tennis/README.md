[//]: # (Image References)
[Image1]: https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif
# Project 2 : Continous Control

## Introduction
In this project we work with [Tennis environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis). This is a collabortive environment where the goal is to get a score of 0.5 averaged over 100 episodes. The environment consists of 2 agents playing tennis and if the the ball is hit outside the bounds then the game ends and a reward
of -0.01 is received. If the ball is hit in the net reward of 0 is received and if the ball is hit to be in the play on other side then
the reward of 0.1 is received. Since goal is to keep the ball in play such that at least 0.5 is scored in 100 consecutive games, both the
agents need to learn to play in such a manner that maximizes the chance of other agent being able to keep the ball in play.

The observation consists of 8 variables and 3 observations are stacked togather to create a state size of 24.The action space is of size 2. Both the spaces are contigous and for each of the action dimension the values can range from -1 to 1.

## Success criteria
Score 0.5 for 100 consecutve episodes where for each step max score of either agent is considered as timestep score.

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
1. create a virtual environment using conda or tool of your choice with python 3.6
2. activate the new environment
3. install dependencies using `python -m pip install requirements.txt`
4. Read report.md to learn more about the implementation
5. start the Jupyter Notebook in your environment. For macos use following command `~/opt/anaconda3/bin/jupyter_mac.command`
6. Run all the cells below **4. It's your turn** cell
