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

double gettime( void )
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts );
    return (ts.tv_sec + (double)ts.tv_nsec / 1e9);
}

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


void bellmanford(struct Graph *g, int source, bool is_seq) ;
void display(int arr[], int size);

// ----------------------- MAIN --------------------------//
int main(int argc, char *argv[]) {
//read vertices num from cmd call
if (argc != 3) {
  fprintf(stderr, "Usage: %s <number of vertices> <is sequential? (bool)>\n", argv[0]);
  return 1;
}
char filename[50];
int arg = atoi(argv[1]);
sprintf(filename, "graphs/graph_%d.txt", arg);
Graph* g = readGraph(filename);
bool is_seq = atoi(argv[2]);
double elapsed_time, tstart, tstop;

if (is_seq){printf("max cuda threads: %d\n", ((g->E+BLKDIM-1)/BLKDIM)*BLKDIM);}

//run algorithm
tstart=gettime();
bellmanford(g, 0, is_seq);  //0 is the source vertex
tstop=gettime();
elapsed_time = tstop - tstart;
printf("Elapsed time %f seconds\n", elapsed_time); //show in seconds
printf("-------------------\n");

return 0;
}

// --------- MULTIPLE THREADS AND BLOCKS (parallel) ---------
__global__ void initialize(int *d, int *p, int tV, int source) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < tV) {
      d[i] = (i == source) ? 0 : INF;
      p[i] = 0;
  }
}
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
__global__ void checkNegativeCycles(Edge* edge, int* h_d, int tE) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < tE) {
        int u = edge[i].u;
        int v = edge[i].v;
        int w = edge[i].w;
        if (h_d[u] != INF && h_d[v] > h_d[u] + w) {
            printf("Negative cycle detected!\n");
        }
    }
}
// ---------------- ONE THREAD AND BLOCK (sequential) -----------------//
__global__ void initialize_seq(int *d, int *p, int tV, int source) {
    for (int i = 0; i < tV; i++) {
        d[i] = (i == source) ? 0 : INF;
        p[i] = 0;
    }
}

__global__ void relax_seq(struct Edge *edges, int *d, int *p, int tE) {
    for (int j = 0; j < tE; j++) {
        int u = edges[j].u;
        int v = edges[j].v;
        int w = edges[j].w;
        if (d[u] != INF && d[v] > d[u] + w) {
            d[v] = d[u] + w;
            p[v] = u;
        }
    }
}

__global__ void checkNegativeCycles_seq(Edge* edge, int* h_d, int tE) {
    for (int i = 0; i < tE; i++) {
        int u = edge[i].u;
        int v = edge[i].v;
        int w = edge[i].w;
        if (h_d[u] != INF && h_d[v] > h_d[u] + w) {
            printf("Negative cycle detected!\n");
        }
    }
}

// ------------------- ALGORITHM ---------------------//
void bellmanford(struct Graph *g, int source, bool is_seq) {
  int tV = g->V;
  int tE = g->E;

  int *d, *p;
  cudaMalloc(&d, tV * sizeof(int));
  cudaMalloc(&p, tV * sizeof(int));

  struct Edge *edges;
  cudaMalloc(&edges, tE * sizeof(struct Edge));
  cudaMemcpy(edges, g->edge, tE * sizeof(struct Edge), cudaMemcpyHostToDevice);

  if (is_seq){ initialize_seq<<<1,1>>>(d, p, tV, source);}
  else{ initialize<<<(tV+BLKDIM-1)/BLKDIM, BLKDIM>>>(d, p, tV, source);}
  cudaDeviceSynchronize(); 

  //relaxation phase
  for (int i = 1; i <= tV - 1; i++) {
      if (is_seq){ relax_seq<<<1,1>>>(edges, d, p, tE);}
      else{relax<<<(tE+BLKDIM-1)/BLKDIM,BLKDIM>>>(edges, d, p, tE);} 
      cudaDeviceSynchronize();
  }

  //copying d and p from device to host
  int *h_d = (int*)malloc(tV * sizeof(int));
  int *h_p = (int*)malloc(tV * sizeof(int));
  cudaMemcpy(h_d, d, tV * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_p, p, tV * sizeof(int), cudaMemcpyDeviceToHost);

  if (is_seq){ checkNegativeCycles_seq<<<1,1>>>(edges, d, tE);}
  else{checkNegativeCycles<<<(tE+BLKDIM-1)/BLKDIM,BLKDIM>>>(edges, d, tE);}
  cudaDeviceSynchronize();

  // DEBUG: all display to show the values of d and p
  /*
  printf("Values of d:\n");
  display(h_d, tV);
  printf("Values of p:\n");
  display(h_p, tV);
  */

  cudaFree(d);
  cudaFree(p);
  cudaFree(edges);
  free(h_d);
  free(h_p);
}

// display function for debugging purposes
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