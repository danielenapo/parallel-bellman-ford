#include<stdio.h>
#include<stdlib.h>
#include <time.h>
 
#define MAX_VERTICES 7
#define MAX_EDGES 5
#define SEED 42
#define MAX_WEIGHT 50 //max POSITIVE weight value of an edge
#define MIN_WEIGHT 5 //min NEGATIVE weight value of an edge
 
typedef unsigned char vertex;
 
int main(){
    srand(SEED);
    int numberOfVertices = MAX_VERTICES;
    int maxNumberOfEdges = MAX_EDGES;
    if( numberOfVertices == 0)
        numberOfVertices++;
    vertex ***graph;
    int **weights;
 
    if ((graph = (vertex ***) malloc(sizeof(vertex **) * numberOfVertices)) == NULL){
        printf("Could not allocate memory for graph\n");
        exit(1);
    }

    if ((weights = (int **) malloc(sizeof(int *) * numberOfVertices)) == NULL){
        printf("Could not allocate memory for weights\n");
        exit(1);
    }
 
    int vertexCounter = 0;
    int edgeCounter = 0;
 
    for (vertexCounter = 0; vertexCounter < numberOfVertices; vertexCounter++){
        if ((graph[vertexCounter] = (vertex **) malloc(sizeof(vertex *) * maxNumberOfEdges)) == NULL){
            printf("Could not allocate memory for edges\n");
            exit(1);
        }
        if ((weights[vertexCounter] = (int *) malloc(sizeof(int) * maxNumberOfEdges)) == NULL){
            printf("Could not allocate memory for weights\n");
            exit(1);
        }
        for (edgeCounter = 0; edgeCounter < maxNumberOfEdges; edgeCounter++){
            if ((graph[vertexCounter][edgeCounter] = (vertex *) malloc(sizeof(vertex))) == NULL){
                printf("Could not allocate memory for vertex\n");
                exit(1);
            }
        }
    }
 
    FILE *file = fopen("graph.txt", "w");
    if (file == NULL){
        printf("Could not open file for writing\n");
        exit(1);
    }
    //first line: number of vertices
    fprintf(file, "%d\n", numberOfVertices);
 
    vertexCounter = 0;edgeCounter = 0;
    for (vertexCounter = 0; vertexCounter < numberOfVertices; vertexCounter++){
        fprintf(file, "%d:",vertexCounter);
        printf("%d:\t",vertexCounter);
        
        int *connected = calloc(numberOfVertices, sizeof(int)); // Initialize all to 0

        for (edgeCounter=0; edgeCounter < maxNumberOfEdges; edgeCounter++){

            if (rand()%2 == 1){
                int linkedVertex;
                do {
                    linkedVertex = rand() % numberOfVertices;
                } while (linkedVertex == vertexCounter || connected[linkedVertex]); // Avoid self-loop and multiple connections

                connected[linkedVertex] = 1; // Mark as connected

                graph[vertexCounter][edgeCounter] = *graph[linkedVertex];
                weights[vertexCounter][edgeCounter] = rand() % MAX_WEIGHT - MIN_WEIGHT;
                fprintf(file, "%d,%d;", linkedVertex, weights[vertexCounter][edgeCounter]);
                printf("%d,%d; ", linkedVertex, weights[vertexCounter][edgeCounter]);
            }
            else{
                graph[vertexCounter][edgeCounter] = NULL;
                weights[vertexCounter][edgeCounter] = 0;
            }
        }
        free(connected); // Free the memory allocated for the array


        fprintf(file, "\n");
        printf("\n");
    }
 
    fclose(file);
    return 1;
}