#!/bin/bash
# Run all experiments

./scripts/run_vizdoom_vanilla_dqn.sh
./scripts/run_vizdoom_reward_shaping.sh
./scripts/run_vizdoom_manual_actions.sh
./scripts/run_vizdoom_manual_actions-reward-shaping.sh
./scripts/run_vizdoom_manual_hierarchy.sh
