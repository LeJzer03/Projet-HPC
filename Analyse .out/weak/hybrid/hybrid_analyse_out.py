import re

def analyze_mpi_output(file_path):
    data = {}
    current_config = None
    run_times = []

    with open(file_path, 'r') as file:
        for line in file:
            # Check for new run and configuration line
            config_match = re.search(r'Run \d+ with (\d+) MPI ranks and (\d+) threads', line)
            if config_match:
                if current_config and run_times:  # Previous config and its runs
                    avg_time = sum(run_times) / len(run_times)
                    if current_config in data:
                        data[current_config].append(avg_time)
                    else:
                        data[current_config] = [avg_time]
                    print(f"Configuration: {current_config}, Average Execution Time: {avg_time:.4f} seconds")

                # Reset for new configuration
                current_config = (int(config_match.group(1)), int(config_match.group(2)))
                run_times = []  # Clear previous times
                print(f"Found new configuration: {current_config}")

            # Check for execution time line
            time_match = re.search(r'Done: ([\d.]+) seconds', line)
            if time_match:
                run_times.append(float(time_match.group(1)))

        # Last configuration processing
        if current_config and run_times:
            avg_time = sum(run_times) / len(run_times)
            if current_config in data:
                data[current_config].append(avg_time)
            else:
                data[current_config] = [avg_time]
            print(f"Final Configuration: {current_config}, Average Execution Time: {avg_time:.4f} seconds")

    # Compute overall averages for each configuration
    print("\nOverall Average Execution Times by Configuration:")
    for config, times in data.items():
        overall_avg = sum(times) / len(times)
        print(f"Configuration: {config[0]} ranks, {config[1]} threads, Overall Average Execution Time: {overall_avg:.4f} seconds")

# Usage example
file_path = r'C:\École\BAC - 3\Q1\High Perf. Sci. Computing\Scalability\weak_hybrid\hybrid_weak_1node.out'  # Adapt this to your actual file path
analyze_mpi_output(file_path)
