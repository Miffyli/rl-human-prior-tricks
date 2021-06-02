#!/bin/bash
# Run experiment with vanilla-dqn in vizdoom

repetitions=5

for repetition in $(seq 1 ${repetitions})
do
    # Standard deathmatch
    ./scripts/run_vizdoom_experiment.sh vanilla-dqn --agent-type vanilla-dqn
done
