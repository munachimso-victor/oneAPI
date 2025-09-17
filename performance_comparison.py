# Imports from your analysis files
import time
import numpy as np
import matplotlib.pyplot as plt

# It is assumed that your two files, oneAPI_analysis.py and cpu_analysis.py,
# are in the same directory as this file.
from oneAPI_analysis import oneapi_anomaly_detection
from cpu_analysis import standard_python_covariance_analysis

def run_performance_test(function, filename, num_runs=100):
    """
    Runs a given function multiple times and returns a list of execution times.
    """
    run_times = []
    print(f"Running {function.__name__} for {num_runs} times...")
    for _ in range(num_runs):
        time_taken = function(filename)
        if time_taken is not None:
            run_times.append(time_taken)
            time.sleep(10)
    return run_times

def plot_performance_graph(oneapi_times, cpu_times):
    """
    Analyzes the run times and plots a line graph to compare performance.
    """
    # Calculate the average time for each method
    avg_oneapi = np.mean(oneapi_times)
    avg_cpu = np.mean(cpu_times)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    run_numbers = np.arange(1, len(oneapi_times) + 1)

    # Plot the line and scatter points for oneAPI
    ax.plot(run_numbers, oneapi_times, color='skyblue', marker='o', linestyle='-', label='oneAPI (daal4py)')
    ax.scatter(run_numbers, oneapi_times, color='darkblue', zorder=2)

    # Plot the line and scatter points for standard CPU
    ax.plot(run_numbers, cpu_times, color='salmon', marker='o', linestyle='-', label='Standard NumPy')
    ax.scatter(run_numbers, cpu_times, color='darkred', zorder=2)

    # Add horizontal lines for the average times for easy comparison
    ax.axhline(avg_oneapi, color='skyblue', linestyle='--', label=f'Avg oneAPI: {avg_oneapi:.4f}s')
    ax.axhline(avg_cpu, color='salmon', linestyle='--', label=f'Avg Standard CPU: {avg_cpu:.4f}s')

    ax.set_title('Performance Comparison of Covariance Analysis')
    ax.set_ylabel('Time Taken (seconds)')
    ax.set_xlabel('Run Number')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    print(f"\nAverage oneAPI time: {avg_oneapi:.4f} seconds")
    print(f"Average Standard CPU time: {avg_cpu:.4f} seconds")
    print(f"oneAPI is approximately {avg_cpu / avg_oneapi:.2f}x faster on average.")

    plt.show()

if __name__ == "__main__":
    file_path = "200722_win_scale_examples_anon.pcapng"
    num_runs = 10 # You can increase this for a more stable average

    # Get the run times for each method
    oneapi_times = run_performance_test(oneapi_anomaly_detection, file_path, num_runs)
    cpu_times = run_performance_test(standard_python_covariance_analysis, file_path, num_runs)

    # Plot the results
    if oneapi_times and cpu_times:
        plot_performance_graph(oneapi_times, cpu_times)
    else:
        print("Could not get a complete set of run times. Please check your file path and code.")
