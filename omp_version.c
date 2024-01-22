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


// ------------------------ CREATE GRAPH -------------------------- //
Graph* createGraph(int V) {
    Graph* graph = (Graph*) malloc(sizeof(Graph));
    graph->V = V;
    graph->E = V * (V - 1);
    graph->edge = (Edge*) malloc(graph->E * sizeof(Edge));

    return graph;
}

void generateRandomGraph(Graph* graph) {
    srand(time(0)); // Use current time as seed for random generator

    for (int i = 0; i < graph->E; i++) {
        do {
            graph->edge[i].u = rand() % graph->V;
            graph->edge[i].v = rand() % graph->V;
        } while(graph->edge[i].u == graph->edge[i].v); // Ensure u != v to prevent self-loops

        graph->edge[i].w = rand() % 100 + 1; // Random weight between 1 and 100
    }
}

void bellmanford(struct Graph *g, int source);
void display(int arr[], int size);

// ----------------------- MAIN --------------------------//
int main(void) {
    //create random graph
    //int totalVertices = 4; // Set the total number of vertices
    //Graph* g = createGraph(totalVertices);
    //generateRandomGraph(g);
    //create graph
    struct Graph *g = (struct Graph *)malloc(sizeof(struct Graph));
    g->V = 4;  //total vertices
    g->E = 5;  //total edges
    //array of edges for graph
    g->edge = (struct Edge *)malloc(g->E * sizeof(struct Edge));

    //------- adding the edges of the graph
    /*
      edge(u, v)
      where 	u = start vertex of the edge (u,v)
          v = end vertex of the edge (u,v)
      
      w is the weight of the edge (u,v)
    */


    //edge 0 --> 1
    g->edge[0].u = 0;
    g->edge[0].v = 1;
    g->edge[0].w = 5;

    //edge 0 --> 2
    g->edge[1].u = 0;
    g->edge[1].v = 2;
    g->edge[1].w = 4;

    //edge 1 --> 3
    g->edge[2].u = 1;
    g->edge[2].v = 3;
    g->edge[2].w = 3;

    //edge 2 --> 1
    g->edge[3].u = 2;
    g->edge[3].v = 1;
    g->edge[3].w = 6;

    //edge 3 --> 2
    g->edge[4].u = 3;
    g->edge[4].v = 2;
    g->edge[4].w = 2;


    //run algorithm
    double tstart, tstop;
    tstart = omp_get_wtime();
    bellmanford(g, 0);  //0 is the source vertex
    tstop = omp_get_wtime();

    printf("Elapsed time %f\n", tstop - tstart);
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
    #pragma omp parallel for private(j, u, v, w) shared(d, p)
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
  #pragma omp parallel for private(u, v, w)
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
    printf("%d ", arr[i]);
  }
  printf("\n");
}