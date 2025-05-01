# PLACEHOLDER CODE
import numpy as np
import matplotlib.pyplot as plt

class SignalVisualizer:
    def __init__(self, signal, sampling_rate):
        """
        Initialize the SignalVisualizer.

        :param signal: The signal data as a NumPy array or list.
        :param sampling_rate: The sampling rate of the signal in Hz.
        """
        self.signal = np.array(signal)
        self.sampling_rate = sampling_rate
        self.time = np.linspace(0, len(self.signal) / self.sampling_rate, len(self.signal))

    def plot_signal(self, title="Signal Output", xlabel="Time (s)", ylabel="Amplitude"):
        """
        Plot the signal.

        :param title: Title of the plot.
        :param xlabel: Label for the x-axis.
        :param ylabel: Label for the y-axis.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(self.time, self.signal, label="Signal")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_spectrum(self, title="Signal Spectrum", xlabel="Frequency (Hz)", ylabel="Amplitude"):
        """
        Plot the frequency spectrum of the signal.

        :param title: Title of the plot.
        :param xlabel: Label for the x-axis.
        :param ylabel: Label for the y-axis.
        """
        freq = np.fft.rfftfreq(len(self.signal), d=1/self.sampling_rate)
        spectrum = np.abs(np.fft.rfft(self.signal))

        plt.figure(figsize=(10, 4))
        plt.plot(freq, spectrum, label="Spectrum")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

# Example usage:
# signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 500))  # Example sine wave
# visualizer = SignalVisualizer(signal, sampling_rate=500)
# visualizer.plot_signal()
# visualizer.plot_spectrum()