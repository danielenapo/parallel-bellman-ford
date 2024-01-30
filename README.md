# Parallel Bellman-Ford
Project exam for the course _"Architectures and Platforms for Artificial Infelligence" (Module 1)_, of the Master's degree in Artificial Intelligence, University of Bologna.

## Instructions
### Graph generator
The program _graphGenerator.c_ can generate random graphs compatible with the algorithm, given the number of vertices. Run:
```
gcc graphGenerator.c -o graph
```
and
```
./graph <number of vertices>
```
To generate a text file containing the graph info in the folder _graphs_.
### OMP
To compile the OMP version, run:
```
gcc -fopenmp omp_version.c -o omp
```
then to run it:
```
./omp <number of vertices>
```
### CUDA
To run the CUDA version (inside the Unibo SLURM environment), just run this script:
```
sbatch submit_script.sh
```
The results will be printed in the file _"out"_.<br>
