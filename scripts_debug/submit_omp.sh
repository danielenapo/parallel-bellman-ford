#!/bin/bash
#SBATCH --job-name=parallel_bf
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniele.napolitano4@studio.unibo.it
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=outputs/log_omp
#SBATCH --gres=gpu:1

#compile the code
gcc -fopenmp omp_version.c -o omp
# generate graphs
chmod +x generate_graphs.sh
./generate_graphs.sh 10 50 100 250 500 750 1000 5000
# create output csv file (delete the file if it already exists)
rm -f outputs/omp.csv
touch outputs/omp.csv

# two iterations: one for parallel version, one for sequential version
for i in {0..1}
do
    if [ $i -eq 1 ]
    then
        echo "SEQUENTIAL"
    else
        echo "PARALLEL"
    fi
    # Iterate over each file in the directory
    for file in graphs/*
    do
        vertices=$(basename "$file" .txt | cut -d'_' -f2)
        # Run the ./cuda command with the current file (vertices number) as an argument
        timeout 5m ./omp "$vertices" 8 $i "omp.csv"
        exit_status=$?
        if [ $exit_status -eq 124 ]
        then
            echo "Timeout has been reached"
        fi
    done
done

# sed -i 's/\x0D$//' *.sh