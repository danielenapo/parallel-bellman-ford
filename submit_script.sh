#!/bin/bash
#SBATCH --job-name=parallel_bf
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniele.napolitano4@studio.unibo.it
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=out
#SBATCH --gres=gpu:1

#compile the code
nvcc cuda_version.cu -o cuda

# Iterate over each file in the directory
for file in graphs/*
do
    vertices=$(basename "$file" .txt | cut -d'_' -f2)
    # Run the ./cuda command with the current file (vertices number) as an argument
    timeout 10m ./cuda "$vertices"
    exit_status=$?
    if [ $exit_status -eq 124 ]
    then
        echo "Timeout has been reached"
done


# sed -i 's/\x0D$//' submit_script.sh