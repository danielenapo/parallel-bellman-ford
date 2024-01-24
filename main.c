// Bellman Ford Algorithm in C
// taken originally from https://www.programiz.com/dsa/bellman-ford-algorithm#google_vignette
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>

#define INFINITY 99999

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

    //print graph for debugging
    printf("\n\nGRAPH:\n");
    for (i = 0; i < E; i++) {
      printf("%d -> %d (%d)\n", graph->edge[i].u, graph->edge[i].v, graph->edge[i].w);
    }

    return graph;
}

void bellmanford(struct Graph *g, int source);
void display(int arr[], int size);

// ----------------------- MAIN --------------------------//
int main(void) {
    //create random graph
    Graph* g = readGraph("graph.txt");
    //run algorithm
    double tstart, tstop;
    //tstart = omp_get_wtime();
    bellmanford(g, 0);  //0 is the source vertex
    //tstop = omp_get_wtime();
    //printf("Elapsed time %f\n", tstop - tstart);
    return 0;
}
// ------------------------- ALGORITHM --------------------//
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
  for (i = 0; i < tV; i++) {
    d[i] = INFINITY;
    p[i] = 0;
  }

  //mark the source vertex
  d[source] = 0;

  //PART TO PARALLELIZE!!
  //step 2: relax edges |V| - 1 times
  for (i = 1; i <= tV - 1; i++) {
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
  for (i = 0; i < tE; i++) {
    u = g->edge[i].u;
    v = g->edge[i].v;
    w = g->edge[i].w;
    if (d[u] != INFINITY && d[v] > d[u] + w) {
      printf("Negative weight cycle detected!\n");
      return;
    }
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
    if(arr[i] == INFINITY){
      printf("INF ");
    }
    else{
      printf("%d ", arr[i]);
    }
  }
  printf("\n");
}