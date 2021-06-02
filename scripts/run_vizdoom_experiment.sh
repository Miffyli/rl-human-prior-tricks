#!/bin/bash
# Run experiment with stable-baselines

if test -z "$1"
then
    echo "Usage: run_vizdoom_experiment experiment_name [arguments]"
    exit
fi

experiment_dir=experiments/vizdoom_${1}_$(date -Iseconds)
# Prepare directory
./scripts/prepare_experiment_dir.sh ${experiment_dir}
# Store the launch parameters there
echo ${@:0} > ${experiment_dir}/launch_arguments.txt

# Run code
python3 vizdoom_deathmatch_agent.py \
  ${@:2} \
  | tee ${experiment_dir}/stdout.txt
