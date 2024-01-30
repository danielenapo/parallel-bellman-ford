// Bellman Ford Algorithm in C
// taken originally from https://www.programiz.com/dsa/bellman-ford-algorithm#google_vignette
// gcc -fopenmp omp_version.c -o omp
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>


#define INFINITY 99999
#define OMP_NUM_THREADS 12

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
    printf("Edges: %d, Weights: %d\n", E,V);

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

  double elapsed_time[2];

  //run algorithm
  for (int c=0; c<2; c++){
    if (c==0){ //first loop is parallel
      omp_set_num_threads(OMP_NUM_THREADS); 
    }
    else{ //second loop is sequential
      omp_set_num_threads(1);
    }
    double tstart, tstop;
    tstart = omp_get_wtime();
    bellmanford(g, 0);  //0 is the source vertex
    tstop = omp_get_wtime();
    elapsed_time[c] = tstop - tstart;
    printf("%d THREAD: \n", omp_get_max_threads());
    printf("Elapsed time %f\n", elapsed_time[c]);
    printf("-------------------\n");
  }
  printf("SPEEDUP: %f\n", elapsed_time[1]/elapsed_time[0]);
  return 0;
}


// ------------------- ALGORITHM ---------------------//
void bellmanford(struct Graph *g, int source) {
  //variables
  int i, j, u, v, w;

  //total vertex in the graph g
  int tV = g->V;

  //total edge in the graph g
  int tE = g->E;

  //distance array
  //size equal to the number of vertices of the graph g
  int d[tV];

  //predecessor array
  //size equal to the number of vertices of the graph g
  int p[tV];

  //step 1: fill the distance array and predecessor array
  #pragma omp parallel for private(i)
  for (i = 0; i < tV; i++) {
    d[i] = INFINITY;
    p[i] = 0;
  }

  //mark the source vertex
  d[source] = 0;

  //PART TO PARALLELIZE!!
  //step 2: relax edges |V| - 1 times
  for (i = 1; i <= tV - 1; i++) { 
    #pragma omp parallel for private(u, v, w, j) shared(d, p)
    for (j = 0; j < tE; j++) {
      //get the edge data
      u = g->edge[j].u; //start
      v = g->edge[j].v; //end
      w = g->edge[j].w; //weight

      if (d[u] != INFINITY && d[v] > d[u] + w) {
        d[v] = d[u] + w;
        p[v] = u;
      }
    }
  }

  //step 3: detect negative cycle
  //if value changes then we have a negative cycle in the graph
  //and we cannot find the shortest distances
  int negative_cycle_detected = 0;

  #pragma omp parallel for private(u, v, w, i)
  for (i = 0; i < tE; i++) {
    u = g->edge[i].u;
    v = g->edge[i].v;
    w = g->edge[i].w;
    if (d[u] != INFINITY && d[v] > d[u] + w) {
      printf("Negative weight cycle detected!\n");
      negative_cycle_detected = 1;
    }
  }

  if (negative_cycle_detected) {
  return;
}

  //No negative weight cycle found!
  //print the distance and predecessor array
  printf("Distance array: ");
  display(d, tV);
  printf("Predecessor array: ");
  display(p, tV);
}

void display(int arr[], int size) {
  int i;
  for (i = 0; i < size; i++) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}