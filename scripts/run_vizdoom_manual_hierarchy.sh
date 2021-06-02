#!/bin/bash
# Run experiment with manual-hierarchy in vizdoom

repetitions=5

for repetition in $(seq 1 ${repetitions})
do
    # Standard deathmatch
    ./scripts/run_vizdoom_experiment.sh manual-hierarchy --agent-type manual-hierarchy
done
