# Reinforcement Learning Tricks, ViZDoom experiments

This branch contains the ViZDoom experiments using DQN with various tricks. The challenge is the Deatmatch scenario from ViZDoom repository.

**Note:** The reward is different from the original scenario file. In original file, agent was rewarded for kills it did not make. This repository uses a fixed version, where agent only gets reward if it kills an enemy.

Tricks used:

- Reward shaping: agent is given reward for hitting enemies and picking up items
- Manual actions: if an enemy is under crosshair, script picks an action that shoots (and this is added to DQN's replay memory)
- Manual hierarchy: two DQN agents, one for shooting and another for navigation. Shooting agent plays if any enemy is visible on the screen, navigation plays otherwise

## Major Requirements

```
pip install stable-baselines3 vizdoom
```

## Running experiments

In the root directory, run `./scripts/run_vizdoom_experiments.sh`. This will run all the experiments in a sequence.

The resulting logs are put into `experiments` directory. Copy this directory to `vizdoom-runs` in the `main` branch to plot the results.

