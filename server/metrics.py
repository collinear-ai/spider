import numpy as np
import scipy.signal

def discounted_future_sum_vectorized(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute discounted sum of future values for each position using a vectorized approach.

    Args:
        x (np.ndarray): 1D array of rewards.
        gamma (float): Discount factor.

    Returns:
        np.ndarray: discounted sum of future values.
    """
    # Reverse x so lfilter processes from end to start

    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1].astype(x.dtype)  # type: ignore
