import numpy as np

def sythesize_data(
    num_samples=1000, time_steps=50, noise_level=0.1, seed=42
):
    np.random.seed(seed)
    X = []
    y = []

    for _ in range(num_samples):
        label = np.random.randint(0, 2)

        if label == 0:
            # Class 0: Sinusoidal pattern
            freq = np.random.uniform(0.1, 0.5)
            phase = np.random.uniform(0, np.pi)
            series = np.sin(np.linspace(0, 2 * np.pi * freq, time_steps) + phase)
        else:
            # Class 1: Linear + noise
            slope = np.random.uniform(0.05, 0.2)
            intercept = np.random.uniform(-1, 1)
            series = slope * np.arange(time_steps) + intercept

        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, size=time_steps)
        series += noise

        X.append(series)
        y.append(label)

    X = np.array(X)
    X = X.reshape(num_samples, 1, time_steps)
    y = np.array(y)
    return X, y