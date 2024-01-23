// Bellman Ford Algorithm in C
// taken originally from https://www.programiz.com/dsa/bellman-ford-algorithm#google_vignette

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>


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


// ----------------- GENERATE RANDOM GRAPH ---------------------- //
Graph* createGraph(int V) {
    Graph* graph = (Graph*) malloc(sizeof(Graph));
    graph->V = V;
    graph->E = V * (V - 1);
    graph->edge = (Edge*) malloc(graph->E * sizeof(Edge));

    return graph;
}

void generateRandomGraph(Graph* graph) {
    // Create a matrix to keep track of the generated edges
    int** edges = (int**)malloc(graph->V * sizeof(int*));
    for (int i = 0; i < graph->V; i++) {
        edges[i] = (int*)calloc(graph->V, sizeof(int));
    }

    for (int i = 0; i < graph->E; i++) {
        do {
            graph->edge[i].u = rand() % graph->V;
            graph->edge[i].v = rand() % graph->V;
        } while(graph->edge[i].u == graph->edge[i].v || edges[graph->edge[i].u][graph->edge[i].v]); // Ensure u != v and the edge has not been generated before

        // Mark the edge as generated
        edges[graph->edge[i].u][graph->edge[i].v] = 1;

        graph->edge[i].w = rand() % 101 - 15; 
    }

    // Free the memory allocated for the matrix
    for (int i = 0; i < graph->V; i++) {
        free(edges[i]);
    }
    free(edges);
}

void bellmanford(struct Graph *g, int source);
void display(int arr[], int size);

// ----------------------- MAIN --------------------------//
int main(void) {
    //create random graph
    int totalVertices = 5; // Set the total number of vertices
    // set seed for rand() to get same graph every time
    srand(42);
    Graph* g = createGraph(totalVertices);
    generateRandomGraph(g);
    //print the graph
    printf("Graph:\n");
    for (int i = 0; i < g->E; i++) {
        printf("%d -> %d (weight: %d)\n", g->edge[i].u, g->edge[i].v, g->edge[i].w);
    }


    omp_set_num_threads(12);

    //run algorithm
    double tstart, tstop;
    tstart = omp_get_wtime();
    bellmanford(g, 0);  //0 is the source vertex
    tstop = omp_get_wtime();

    printf("Elapsed time %f\n", tstop - tstart);
    printf("Number of threads %d\n", omp_get_max_threads());
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
  #pragma omp parallel for
  for (i = 0; i < tV; i++) {
    d[i] = INFINITY;
    p[i] = 0;
  }

  //mark the source vertex
  d[source] = 0;

  //PART TO PARALLELIZE!!
  //step 2: relax edges |V| - 1 times
  for (i = 1; i <= tV - 1; i++) { 
    #pragma omp parallel for private(j, u, v, w) shared(d, p) schedule(dynamic)
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

  #pragma omp parallel for private(u, v, w)
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