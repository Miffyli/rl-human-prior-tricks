# Reinforcement Learning Tricks, GFootball experiments
This branch contains the Google football experiments using DQN with various tricks. The 11 versus 11 easy stochastic environment is used.

Tricks used:
    
    Curriculum learning: gradualy increasing the games difficulty from 0.0 to 0.05 over the training.
    Manual hierarchy: two DQN agents, one for attacking and another for deffending. The attacking agent plays if we are in possesion of the ball, the deffending agent plays otherwise

## Requirements
The requirements are specified in requirements.txt
For the Google research football requirements we refer to the official repository: https://github.com/google-research/football


## Reproducing experiments

To reproduce the results reported in the paper simply run `python gfootball_agent.py`

You can activate or deactivate CL and MH by setting `CURRICULUM` and `HIERARCHICAL` to `True` or `False` respectively.

The results are logged into `experiments` directory. Copy this directory to `football-runs` in the `main` branch to plot the results.