#!/bin/bash
# Run experiment with reward-shaping in vizdoom

repetitions=5

for repetition in $(seq 1 ${repetitions})
do
    # Standard deathmatch
    ./scripts/run_vizdoom_experiment.sh reward-shaping --agent-type reward-shaping 
done
