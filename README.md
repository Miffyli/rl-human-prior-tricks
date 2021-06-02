# Reinforcement Learning Tricks, Index

This repository contains the code for the paper "Distilling Reinforcement Learning Tricks for Video Games".

Code authors: [Anssi Kanervisto](https://github.com/Miffyli), [Christian Scheller](https://github.com/metataro/) and [Yanick Schraner](https://github.com/YanickSchraner).

The experiments in the three environments are split into three git branches:

- `vizdoom` for ViZDoom Deathmatch experiments
- `minerl` for MineRL ObtainDiamond experiments
- `gfootball` for Football environment experiments

To run the experiments, checkout the repository you want to run experiments for with `git checkout [branch name]`,
and follow the instructions in the README file there.

After running all the experiments, collect the results as described the respective branches. You should have
three directories

- `vizdoom-runs`
- `minerl-runs`
- `football-runs`

After this, running `python plot_paper.py` should create a `figures/learning_curves.pdf` file which summarizes
the results.
