# Imports for core data processing
from scapy.all import rdpcap # Imports the rdpcap function from Scapy to read pcap files.
import time # Used for timing the execution of the anomaly detection process.
import numpy as np # Used for numerical operations, especially for handling arrays and matrices.
import daal4py as d4p # Imports the daal4py library for oneAPI-accelerated data analytics.

# Imports for visualization
import matplotlib.pyplot as plt # A plotting library for creating static, animated, and interactive visualizations.
from matplotlib.patches import Ellipse # Used to draw ellipse shapes on a plot.
import matplotlib.transforms as transforms # Provides a framework for defining transformations in Matplotlib.

def plot_covariance_ellipse(covariance_matrix, mean_vector, ax, n_std=2.0, **kwargs):
    """
    Plots a confidence ellipse based on a covariance matrix.
    This function helps to visualize the spread and correlation of the data.

    The ellipse's shape and orientation are determined by the eigenvalues and
    eigenvectors of the covariance matrix. The size of the ellipse is defined by
    the number of standard deviations (n_std).
    """
    # Use np.linalg.eigh to compute eigenvalues and eigenvectors of a symmetric matrix.
    # The covariance matrix is symmetric, making this function the ideal choice.
    eigvals, eigvecs = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues in descending order to find the major and minor axes of the ellipse.
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # Calculate the angle of the major axis of the ellipse in degrees.
    # This determines the ellipse's rotation.
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    # Calculate the width and height of the ellipse based on eigenvalues and n_std.
    # The square root of eigenvalues corresponds to the standard deviation.
    width, height = 2 * n_std * np.sqrt(eigvals)

    # Create the Ellipse patch with the calculated parameters.
    ellipse = Ellipse(xy=mean_vector, width=width, height=height, angle=angle, **kwargs)

    # Add the ellipse to the plot.
    ax.add_patch(ellipse)

def visualize_covariance(data, covariance_matrix, mean_vector):
    """
    Creates a scatter plot with a covariance ellipse to visualize the data.

    The plot shows the original data points (packet length vs. time delta)
    and a green ellipse representing the spread of the data, centered at the mean.
    """
    # Create a new figure and axes for the plot.
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the individual data points (packet length and time delta).
    ax.scatter(data[:, 0], data[:, 1], color='blue', s=10, alpha=0.6, label='Packet Data')

    # Plot the mean of the data as a distinct marker.
    ax.scatter(mean_vector[0], mean_vector[1], color='black', marker='x', s=100, label='Mean')

    # Plot the covariance ellipse using the helper function.
    # The ellipse is configured with a certain number of standard deviations (n_std).
    plot_covariance_ellipse(
        covariance_matrix,
        mean_vector,
        ax,
        n_std=2,
        facecolor='green',
        alpha=0.2,
        label='2-Standard Deviation Ellipse'
    )

    # Set the title and labels for clarity.
    ax.set_title('Network Traffic Analysis via Covariance')
    ax.set_xlabel('Packet Length (bytes)')
    ax.set_ylabel('Time Delta (seconds)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def oneapi_anomaly_detection(filename):
    """
    Reads a pcapng file and performs a rudimentary anomaly detection
    using oneAPI's covariance algorithm.

    The function calculates the covariance matrix of packet length and time delta.
    Anomalies could then be detected by identifying data points that fall outside
    the normal distribution (e.g., outside the covariance ellipse).

    Returns:
        float: The time taken for the analysis in seconds.
    """
    try:
        # Initialize the daal4py covariance algorithm in streaming mode.
        # Streaming mode allows processing data in chunks, which is efficient for large datasets.
        algo = d4p.covariance(streaming=True)
        start_time = time.time()

        # Read packets from the specified pcap file.
        packets = rdpcap(filename)

        packet_data = []
        # Iterate through the packets to extract relevant features for analysis.
        for i, packet in enumerate(packets):
            if i > 0:
                # Calculate the time difference between the current and previous packet.
                time_delta = packet.time - packets[i-1].time
                # Store the packet length and time delta as a pair of features.
                packet_data.append([len(packet), time_delta])

        # Convert the list of features into a NumPy array with float32 data type.
        # daal4py algorithms work efficiently with NumPy arrays.
        data = np.array(packet_data, dtype=np.float32)

        # Process the data in smaller chunks to simulate a stream.
        # This is particularly useful for very large datasets that don't fit in memory.
        chunk_size = 100
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            # Use the daal4py algorithm to compute covariance on each chunk.
            # The streaming algorithm accumulates the results internally.
            algo.compute(chunk)

        # Finalize the streaming algorithm to get the final results.
        results = algo.finalize()

        print("oneAPI-accelerated Covariance Matrix:\n", results.covariance)
        end_time = time.time()
        print(f"\noneAPI-accelerated analysis took: {end_time - start_time} which is approximately {end_time - start_time:4f} seconds\n")

        # --- Calling the external visualization function ---
        # The mean vector is not directly returned by the daal4py covariance algorithm in this case,
        # so we calculate it manually from the original data.
        mean_vector = np.mean(data, axis=0)
        visualize_covariance(data, results.covariance, mean_vector)

        # Return the elapsed time for external use or comparison.
        return end_time - start_time

    except FileNotFoundError:
        # Handle the case where the specified file doesn't exist.
        print(f"Error: The file '{filename}' was not found.")
        return None
    except Exception as e:
        # Provide a general error message for any unexpected exceptions.
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    # Define the path to the network capture file.
    file_path = "dof-short-capture.pcapng"

    # Execute the anomaly detection function and store the returned time.
    analysis_time = oneapi_anomaly_detection(file_path)

    # Print the final result.
    if analysis_time is not None:
        print(f"Total analysis time: {analysis_time:.4f} seconds")
