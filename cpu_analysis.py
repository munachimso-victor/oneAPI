from scapy.all import rdpcap
import time
import numpy as np

# Imports for visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import logging
# Suppress the 'scapy.runtime' logger's messages that are not errors
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

def plot_covariance_ellipse(covariance_matrix, mean_vector, ax, n_std=2.0, **kwargs):
    """
    Plots a confidence ellipse based on a covariance matrix.
    This function helps to visualize the spread and correlation of the data.
    """
    eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse(xy=mean_vector, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

def visualize_covariance(data, covariance_matrix, mean_vector):
    """
    Creates a scatter plot with a covariance ellipse to visualize the data.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the data points
    ax.scatter(data[:, 0], data[:, 1], color='blue', s=10, alpha=0.6, label='Packet Data')

    # Plot the mean of the data
    ax.scatter(mean_vector[0], mean_vector[1], color='black', marker='x', s=100, label='Mean')

    # Plot the covariance ellipse
    plot_covariance_ellipse(
        covariance_matrix,
        mean_vector,
        ax,
        n_std=2,
        facecolor='green',
        alpha=0.2,
        label='2-Standard Deviation Ellipse'
    )

    ax.set_title('Network Traffic Analysis via Covariance')
    ax.set_xlabel('Packet Length (bytes)')
    ax.set_ylabel('Time Delta (seconds)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def standard_python_covariance_analysis(filename):
    """
    Reads a pcapng file and calculates the covariance matrix using NumPy.
    This is the CPU-bound 'before' scenario.

    Returns:
        float: The time taken for the analysis in seconds.
    """
    try:
        print("Starting standard Python covariance analysis...")
        start_time = time.time()
        packets = rdpcap(filename)

        # Extract features
        packet_data = []
        for i, packet in enumerate(packets):
            if i > 0:
                time_delta = packet.time - packets[i-1].time
                packet_data.append([len(packet), time_delta])

        if not packet_data:
            print("No packets with a preceding packet to calculate time delta.")
            return

        # Convert to a NumPy array
        data = np.array(packet_data, dtype=np.float32)

        # Calculate the covariance matrix using numpy.cov
        # The 'rowvar=False' argument is used because each row represents an observation
        # and each column represents a variable (packet length, time delta).
        covariance_matrix = np.cov(data, rowvar=False)

        # Print the covariance matrix
        print("\nStandard Python Covariance Matrix:\n", covariance_matrix)

        end_time = time.time()
        print(f"\nStandard Python analysis took: {end_time - start_time} which is approximately {end_time - start_time:4f} seconds\n")

        mean_vector = np.mean(data, axis=0)
        visualize_covariance(data, covariance_matrix, mean_vector)

        return end_time - start_time

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    file_path = "dof-short-capture.pcapng"
    time = standard_python_covariance_analysis(file_path)
    print(time)
