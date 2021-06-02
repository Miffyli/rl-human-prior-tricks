# Reinforcement Learning Tricks, MineRL ObtainDiamond experiments

This branch contains the MineRL ObtainDiamond experiments using DQN with various tricks. 

Tricks used:

- *Reward shaping (RS)*: 
  The agent receives a reward each time a needed item is obtained, until the required amount is reached. 
  This is opposed to the default reward function that only awards an item once, when it is collected for the first time.
- *Manual hierarchy (MH)*: 
  The agent is split into one agent per each intermediate item.
  The transitions are scripted and happen after a agent achieves its goal.
- *Scripted actions (SA)*: 
  Some items require a specific sequence of crafting actions, which can be easily scripted.
  With this trick, we replace agents for crafting items with fixed policies that follow a pre-defined action sequence.

## Major Requirements

```
pip install -r requirements.txt
```

## Running experiments

You can run experiments as follows:

```
python minerl_agent.py --method rl_rs_mh_sa
``` 

For headless systems, you need to tun the above command with `xvfb-run`:

```
xvfb-run -a python minerl_agent.py --method rl_rs_mh_sa
```

The `--method` flag defines what tricks are applied. You can choose between 'rl, 'rl_rs_mh' or 'rl_rs_mh_sa'.

The resulting logs are put into `./minerl-runs` directory. 
Copy this directory to the main branch to plot the results.

Setting the `--wandb_logging` flag turns on logging to [Weights & Biases](https://wandb.ai/site). 
This assumes that you have [set up wandb](https://docs.wandb.ai/quickstart#1-set-up-wandb) before-hand.
