// Bellman Ford Algorithm in C
// taken originally from https://www.programiz.com/dsa/bellman-ford-algorithm#google_vignette
// nvcc cuda_version.cu -o cuda

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <string.h>

#define INF 99999
#define BLKDIM 256

//struct for the edges of the graph
typedef struct Edge {
  int u;  //start vertex of the edge
  int v;  //end vertex of the edge
  int w;  //weight of the edge (u,v)
}Edge;

//Graph - it consists of edges
typedef struct Graph {
  int V;        //total number of vertices in the graph
  int E;        //total number of edges in the graph
  struct Edge *edge;  //array of edges
}Graph;


// ------------------------ CREATE GRAPH -------------------------- //
Graph* createGraph(int V, int E) {
    Graph* graph = (Graph*) malloc(sizeof(Graph));
    graph->V = V;
    graph->E = E;
    graph->edge = (Edge*) malloc(graph->E * sizeof(Edge));

    return graph;
}

Graph* readGraph( char* filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL){
        printf("Could not open file for reading\n");
        return NULL;
    }

    int V, E;
    fscanf(file, "%d\n", &E);
    fscanf(file, "%d\n", &V);
    printf("Edges: %d, Vertices: %d\n", E,V);

    int u, v, w;
    Graph* graph = createGraph(V, E);
    int i = 0;
    char line[2048];
    char* token;

    while(fgets(line, sizeof(line), file) != NULL){
      token = strtok(line, ":");
      u = atoi(token);

      while((token = strtok(NULL, ";")) != NULL){
        sscanf(token, "%d,%d", &v, &w);
        if (i!=0 && (v == graph->edge[i-1].v && u== graph->edge[i-1].u)){
          continue;
        }
        graph->edge[i].u = u;
        graph->edge[i].v = v;
        graph->edge[i].w = w;
        i++;
      }
    }
    fclose(file);
    return graph;
}


void bellmanford(struct Graph *g, int source);
void display(int arr[], int size);

// ----------------------- MAIN --------------------------//
int main(int argc, char *argv[]) {
  //read vertices num from cmd call
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <number of vertices>\n", argv[0]);
    return 1;
  }
  char filename[50];
  int arg = atoi(argv[1]);
  sprintf(filename, "graphs/graph_%d.txt", arg);
  Graph* g = readGraph(filename);

  float elapsed_time;
  cudaEvent_t start, stop;

  // Create CUDA events for timing
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //run algorithm
  cudaEventRecord(start, 0);
  bellmanford(g, 0);  //0 is the source vertex
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf("CUDA version: \n");
  printf("Elapsed time %f seconds\n", elapsed_time/1000); //show in seconds
  printf("-------------------\n");

  // Destroy CUDA events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}

__global__ void initialize(int *d, int *p, int tV, int source) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < tV) {
        d[i] = (i == source) ? 0 : INF;
        p[i] = 0;
    }
}

//CUDA parallel version of the relaxation phase
__global__ void relax(struct Edge *edges, int *d, int *p, int tE) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < tE) {
        int u = edges[j].u;
        int v = edges[j].v;
        int w = edges[j].w;
        //printf("u: %d, v: %d, w: %d d[u]: %d\n", u, v, w, d[u]);
        if (d[u] != INF && d[v] > d[u] + w) {
            d[v] = d[u] + w;
            p[v] = u;
        }
    }
}

// ------------------- ALGORITHM ---------------------//
void bellmanford(struct Graph *g, int source) {
    int tV = g->V;
    int tE = g->E;

    int *d, *p;
    cudaMalloc(&d, tV * sizeof(int));
    cudaMalloc(&p, tV * sizeof(int));

    struct Edge *edges;
    cudaMalloc(&edges, tE * sizeof(struct Edge));
    cudaMemcpy(edges, g->edge, tE * sizeof(struct Edge), cudaMemcpyHostToDevice);

    initialize<<<(tV+BLKDIM-1)/BLKDIM, BLKDIM>>>(d, p, tV, source);
    cudaDeviceSynchronize();

    //relaxation phase
    for (int i = 1; i <= tV - 1; i++) {
        relax<<<(tE+BLKDIM-1)/BLKDIM,BLKDIM>>>(edges, d, p, tE); //CUDA kernel call from host
        cudaDeviceSynchronize();
    }

    //copying d and p from device to host
    int *h_d = (int*)malloc(tV * sizeof(int));
    int *h_p = (int*)malloc(tV * sizeof(int));
    cudaMemcpy(h_d, d, tV * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p, p, tV * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < tE; i++) {
        int u = g->edge[i].u;
        int v = g->edge[i].v;
        int w = g->edge[i].w;
        if (h_d[u] != INF && h_d[v] > h_d[u] + w) {
            printf("Negative weight cycle detected!\n");
            break;
        }
    }

    // Call display to show the values of d and p
    printf("Values of d:\n");
    display(h_d, tV);
    printf("Values of p:\n");
    display(h_p, tV);



    cudaFree(d);
    cudaFree(p);
    cudaFree(edges);
    free(h_d);
    free(h_p);
}

void display(int arr[], int size) {
  int i;
  for (i = 0; i < size; i++) {
    if(arr[i] == INF){
      printf("INF ");
    }
    else{
      printf("%d ", arr[i]);
    }
  }
  printf("\n");
}