#!/bin/bash
#SBATCH --job-name=parallel_bf
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniele.napolitano4@studio.unibo.it
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=outputs/log
#SBATCH --gres=gpu:1

# sed -i 's/\x0D$//' *.sh

echo "CUDA SPEEDUP"
#compile the code
nvcc cuda_version.cu -o cuda
# generate graphs
chmod +x generate_graphs.sh
./generate_graphs.sh 10 50 100 250 500 750 1000 5000
# create output csv file(delete the file if it already exists)
rm -f outputs/cuda.csv
touch outputs/cuda.csv

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
        timeout 5m ./cuda "$vertices" $i "cuda.csv"
        exit_status=$?
        if [ $exit_status -eq 124 ]
        then
            echo "Timeout has been reached"
        fi
    done
done

echo "----------------------------------"
echo "OMP SPEEDUP"

#compile the code
gcc -fopenmp omp_version.c -o omp
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

echo "----------------------------------"
echo "OMP STRONG SCALING EFFICIENCY"


thread_vector=(1 2 4 8 16 32 64 128)
vertices=2000

#compile the code
gcc -fopenmp omp_version.c -o omp
# generate a single graph instance
chmod +x generate_graphs.sh
./generate_graphs.sh $vertices 
# create output csv file (delete the file if it already exists)
rm -f outputs/omp_efficiency.csv
touch outputs/omp_efficiency.csv

# iterate over the thread_vector
for i in "${thread_vector[@]}"
do
    # Run omp with increasing number of threads (i) in parallel mode (0)
    timeout 5m ./omp $vertices $i 0 "omp_efficiency.csv"
    exit_status=$?
    if [ $exit_status -eq 124 ]
    then
        echo "Timeout has been reached"
    fi
done

