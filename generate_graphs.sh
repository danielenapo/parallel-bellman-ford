# generate graphs if the foder /graphs is empty
if [ -z "$(ls -A graphs)" ]; then
    echo generating graphs
    gcc graphGenerator.c -o graph
    ./graph 10
    ./graph 100
    ./graph 500
    ./graph 1000
    ./graph 5000
fi