#!/bin/bash
# Run experiment with manual-actions-reward-shaping in vizdoom

repetitions=5

for repetition in $(seq 1 ${repetitions})
do
    # Standard deathmatch
    ./scripts/run_vizdoom_experiment.sh manual-actions-reward-shaping --agent-type manual-actions-reward-shaping
done
