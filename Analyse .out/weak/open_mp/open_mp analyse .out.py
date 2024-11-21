import pandas as pd
import re

def analyze_execution_times(file_path):
    # Regular expressions to capture the important lines
    thread_bind_regex = re.compile(r'Running with (\d+) threads and BIND=(spread|close)\.\.\.')
    time_regex = re.compile(r'Done: (\d+\.\d+) seconds')
    run_regex = re.compile(r'Run (\d+) \.\.\.')

    # Data storage
    data = []

    # Processing the file
    with open(file_path, 'r') as file:
        current_threads = None
        current_bind = None
        current_run = None
        for line in file:
            thread_bind_match = thread_bind_regex.match(line)
            if thread_bind_match:
                current_threads = int(thread_bind_match.group(1))
                current_bind = thread_bind_match.group(2)
                current_run = 1  # Reset run counter each time a new configuration starts
            elif run_regex.match(line):
                current_run = int(run_regex.match(line).group(1))
            time_match = time_regex.search(line)
            if time_match and current_threads and current_bind and current_run is not None:
                execution_time = float(time_match.group(1))
                data.append((current_threads, current_bind, current_run, execution_time))

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Threads', 'Bind Type', 'Run Number', 'Execution Time (s)'])

    # Group by Threads and Bind Type and calculate the mean execution time
    mean_df = df.groupby(['Threads', 'Bind Type'])['Execution Time (s)'].mean().reset_index()

    # Print the DataFrame with mean execution times
    print(mean_df)

    return mean_df

# Example usage
file_path = r'C:\École\BAC - 3\Q1\High Perf. Sci. Computing\analyse .out\weak_omp.out'
mean_df = analyze_execution_times(file_path)
