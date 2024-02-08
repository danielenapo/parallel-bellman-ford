import csv
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    print("Usage: python ./plot_outputs.py <filename>")
    sys.exit(1)
filename = "outputs/"+sys.argv[1]+".csv"

# Create an empty dictionary
times_dict = {}

# Open the CSV file
with open(filename, 'r') as file:
    # Create a CSV reader
    reader = csv.reader(file)
    
    # Iterate over each row in the CSV 
    for row in reader:
        # Extract the values from the row
        vertices = int(row[0])
        time = float(row[1])
        is_seq = bool(int(row[3]))
        
        # If the vertices key is not in the dictionary, add it with an empty list as the value
        if vertices not in times_dict:
            times_dict[vertices] = {'seq': None, 'parallel': None, 'speedup': None}               
                
        # Add the time to the appropriate key in the inner dictionary
        if is_seq:
            times_dict[vertices]['seq'] = time
        else:
            times_dict[vertices]['parallel'] = time
# Print the dictionary
sorted_times_dict = dict(sorted(times_dict.items()))

for vertices, times in sorted_times_dict.items():
    if times['seq'] != None:
        # compute the speedup and add it as a third element in the list
        sorted_times_dict[vertices]['speedup'] = times['seq'] / times['parallel']
        #print(f"Vertices: {vertices}, Seq Time: {times['seq']}, Parallel Time: {times['parallel']}, Speedup: {times['speedup']}")
        #debug print to be imported in latex table (for the report)
        print(f"{vertices} &  & {times['parallel']} & {times['seq']} & {times['speedup']} \\\\")
#sort the dictionary by key   
sorted_items = sorted(times_dict.items(), key=lambda item: item[0])
fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns, and optional figure size

# Get the items where the sequential time is not null
filtered_items = [(item[0], item[1]) for item in sorted_items if item[1]['seq'] is not None]

# Plot the times on y, the number of vertices on x
axs[0].plot([item[0] for item in filtered_items], [item[1]['seq'] for item in filtered_items], label='Sequential')
axs[0].plot([item[0] for item in filtered_items], [item[1]['parallel'] for item in filtered_items], label='Parallel')
axs[0].set_title('Execution Time')
axs[0].set_xlabel('Vertices')
axs[0].set_ylabel('Time (s)')

axs[0].legend()

# Plot the speedup on y, the number of vertices on x
axs[1].plot([item[0] for item in sorted_items], [item[1]['speedup'] for item in sorted_items], label='Speedup')
#plot a dashed red line at y=1
if (sys.argv[1] != "cuda"):
    axs[1].axhline(y=1, color='r', linestyle='--')
axs[1].set_title('Speedup') 
axs[1].set_xlabel('Vertices')
axs[1].set_ylabel('Speedup')

axs[1].legend()


plt.tight_layout()  # Adjusts subplot params so that subplots are nicely fit in the figure
plt.show()  # Displays the figure
