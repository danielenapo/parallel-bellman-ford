// originally taken from: https://www.sanfoundry.com/c-program-generate-random-undirected-graph-given-number-edges/
// modified to also generate weights

#include<stdio.h>
#include<stdlib.h>
#include <time.h>
 
#define SEED 42
#define MAX_WEIGHT 50 //max POSITIVE weight value of an edge
#define MIN_WEIGHT 1 //min weight value of an edge
 
typedef unsigned char vertex;
 
int main(int argc, char *argv[]) {
    //read vertices num from cmd call
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number of vertices>\n", argv[0]);
        return 1;
    }
    int numberOfVertices = atoi(argv[1]);
    int maxNumberOfEdges = numberOfVertices/2;
    srand(SEED);
    if( numberOfVertices == 0)
        numberOfVertices++;
    vertex ***graph;
    int **weights;
    int totEdges=0;
 
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
    for (vertexCounter = 0; vertexCounter < numberOfVertices; vertexCounter++){ //loop over vertices
        fprintf(file, "%d:",vertexCounter);
        int *connected = calloc(numberOfVertices, sizeof(int)); // Initialize all to 0

        for (edgeCounter=0; edgeCounter < maxNumberOfEdges; edgeCounter++){ //loop over edges

            if (rand()%2 == 1){
                int linkedVertex;
                do {
                    linkedVertex = rand() % numberOfVertices;
                } while (linkedVertex == vertexCounter || connected[linkedVertex]); // Avoid self-loop and multiple connections
                connected[linkedVertex] = 1; // Mark as connected

                graph[vertexCounter][edgeCounter] = *graph[linkedVertex];
                weights[vertexCounter][edgeCounter] = rand() % MAX_WEIGHT + MIN_WEIGHT;
                fprintf(file, "%d,%d;", linkedVertex, weights[vertexCounter][edgeCounter]);
                totEdges++;
            }
            else{
                graph[vertexCounter][edgeCounter] = NULL;
                weights[vertexCounter][edgeCounter] = 0;
            }
        }
        free(connected); // Free the memory allocated for the array

        fprintf(file, "\n");
    }
    fclose(file);
    printf("Total number of edges: %d\n", totEdges);
    //write totEdges at the start of the file withouth overwriting the first line
    //using a temp file
    FILE *temp = fopen("temp.txt", "w");
    if (temp == NULL){
        printf("Could not open file for writing\n");
        exit(1);
    }
    fprintf(temp, "%d\n", totEdges);
    file = fopen("graph.txt", "r");
    if (file == NULL){
        printf("Could not open file for reading\n");
        exit(1);
    }
    //copy the rest of the file into temp
    char c;
    while ((c = fgetc(file)) != EOF)
        fputc(c, temp);

    fclose(file);
    fclose(temp);

    // Delete the original file and rename the new file to the original file's name
    remove("graph.txt");
    char filename[50]; 
    sprintf(filename, "graphs/graph_%d.txt", numberOfVertices);
    rename("temp.txt", filename);
    return 1;
}