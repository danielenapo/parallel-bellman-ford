# create graphs folder if it does not exist
if [ ! -d "graphs" ]; then
    mkdir graphs
fi
rm -f graphs/* # clear graphs folder before generating new graphs
gcc graphGenerator.c -o graph #compile

#iterate over the call parameters of the cmd
for i in "$@"
do
    # generate graphs
    ./graph $i
done

