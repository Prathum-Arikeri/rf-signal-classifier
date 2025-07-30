import numpy as np

def normalize_signal(signal):
    """Normalize complex RF signal to max amplitude 1."""
    max_val = np.max(np.abs(signal))
    return signal / max_val if max_val != 0 else signal

def add_noise(signal, noise_level):
    """Add Gaussian noise to signal."""
    noise = noise_level * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise
