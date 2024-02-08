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
        threads =int(row[2])
        
        # If the vertices key is not in the dictionary, add it with an empty list as the value
        if threads not in times_dict:
            times_dict[threads] = {'time': time, 'efficiency': None, 'speedup': None}      


        
# Print the dictionary
sorted_times_dict = dict(sorted(times_dict.items()))
#compute efficiency for all elements, as times[threads=1]/times[threads]
for key in sorted_times_dict:
    sorted_times_dict[key]['speedup'] = sorted_times_dict[1]['time']/(sorted_times_dict[key]['time'])
    sorted_times_dict[key]['efficiency'] = sorted_times_dict[key]['speedup']/key

    print("Efficiency for", key, "threads is", sorted_times_dict[key]['efficiency'], "Speedup is", sorted_times_dict[key]['speedup'])
        
#sort the dictionary by key   
sorted_items = sorted(sorted_times_dict.items())


import matplotlib.ticker as ticker

# Create 1 row and 2 columns of subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # Adjust figure size as needed

# Plot the efficiency
axs[0].plot([key for key in sorted_times_dict], [sorted_times_dict[key]['efficiency'] for key in sorted_times_dict], marker='o')
axs[0].set(xlabel='Threads', ylabel='Efficiency', title=f'Strong Efficiency plot for Omp ({vertices} vertices)')
axs[0].set_xticks([key for key in sorted_times_dict])
axs[0].set_xscale('log', base=2)
axs[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.0f}'))

# Plot the speedup
axs[1].plot([key for key in sorted_times_dict], [sorted_times_dict[key]['speedup'] for key in sorted_times_dict], marker='o')
axs[1].set(xlabel='Threads', ylabel='Speedup', title=f'Speedup plot for Omp ({vertices} vertices)')
axs[1].set_xticks([key for key in sorted_times_dict])
axs[1].set_xscale('log', base=2)
axs[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.0f}'))

plt.tight_layout()  # Adjusts subplot params so that subplots are nicely fit in the figure
plt.show()  # Displays the figure

