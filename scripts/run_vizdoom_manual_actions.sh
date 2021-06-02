#!/bin/bash
# Run experiment with manual-actions in vizdoom

repetitions=5

for repetition in $(seq 1 ${repetitions})
do
    # Standard deathmatch
    ./scripts/run_vizdoom_experiment.sh manual-actions --agent-type manual-actions
done
