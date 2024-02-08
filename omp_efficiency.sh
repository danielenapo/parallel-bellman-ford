#!/bin/bash
#SBATCH --job-name=parallel_bf
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniele.napolitano4@studio.unibo.it
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=outputs/log_omp
#SBATCH --gres=gpu:1

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

# sed -i 's/\x0D$//' *.sh