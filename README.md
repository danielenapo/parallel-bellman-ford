# Parallel Bellman-Ford
Project exam for the course _"Architectures and Platforms for Artificial Infelligence" (Module 1)_, of the Master's degree in Artificial Intelligence, University of Bologna.<br>
The scope of this project is to write two parallel version of the **Bellman-Ford Algorithm** using:
- OpenMP
- CUDA
  
Their performance are evaluate and compared. The work and is discussed in detail in the <a  href="https://github.com/danielenapo/parallel-bellman-ford/blob/main/report.pdf"> report.pdf </a> file.
## Instructions
This project was thought to run in the SLURM environment of Unibo's GPU cluster. After accessing the server and copying the folder in it, just run:
```
sbatch project.sbatch
```
### Graph generator
The program _graphGenerator.c_ can generate random graphs compatible with the algorithm, given the number of vertices:
```
./generate_graphs.sh <number of edges>
```
To generate a text file containing the graph info in the folder _graphs_. (This process is already done by project.sbatch)
