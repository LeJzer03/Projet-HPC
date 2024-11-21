import re

def analyze_mpi_output(file_path):
    data = []
    current_processes = None
    run_number = 0
    run_times = []

    with open(file_path, 'r') as file:
        for line in file:
            # Check for process count line
            proc_match = re.search(r'Running with (\d+) processes', line)
            if proc_match:
                if current_processes is not None and run_times:  # Save previous run's average before starting new
                    avg_time = sum(run_times) / len(run_times)
                    data.append((current_processes, run_number, avg_time))
                    print(f"Processes: {current_processes}, Run: {run_number}, Average Execution Time: {avg_time:.4f} seconds")
                
                current_processes = int(proc_match.group(1))
                run_number += 1  # Increment run number for new configuration
                run_times = []  # Reset times for new run
                print(f"Found new process configuration: {current_processes} processes, Run number: {run_number}")

            # Check for execution time line
            time_match = re.search(r'Done: ([\d.]+) seconds', line)
            if time_match:
                run_times.append(float(time_match.group(1)))
                print(f"Found execution time: {time_match.group(1)} seconds for run {run_number}")

            # End of run detection not required here as it is processed above

        # Processing last recorded run after file ends
        if current_processes and run_times:
            avg_time = sum(run_times) / len(run_times)
            data.append((current_processes, run_number, avg_time))
            print(f"Final run complete. Processes: {current_processes}, Run: {run_number}, Average Time: {avg_time:.4f} seconds")

    # Compute overall average time for each process configuration
    process_dict = {}
    for processes, run, avg_time in data:
        if processes not in process_dict:
            process_dict[processes] = []
        process_dict[processes].append(avg_time)

    print("\nOverall Average Execution Times by Process Count:")
    for processes, times in process_dict.items():
        overall_avg = sum(times) / len(times)
        print(f"Processes: {processes}, Overall Average Execution Time: {overall_avg:.4f} seconds")



# Usage example
file_path = r'C:\École\BAC - 3\Q1\High Perf. Sci. Computing\Scalability\weak_mpi\weak_mpi(64_32)_2.out'  # Adapt this to your file path
analyze_mpi_output(file_path)


