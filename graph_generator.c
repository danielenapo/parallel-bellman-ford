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