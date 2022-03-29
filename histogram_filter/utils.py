import numpy as np


def bayes(prior: np.ndarray, conditional: np.ndarray) -> np.ndarray:
    """Applies bias to update current believes based on measurement.

    Define:
        A: belief
        B: measurement
        P(A): Prior
        P(B|A): Conditional
        P(A|B): Poserior
    Then Bayes is defined as:
        P(A|B) ~ P(B|A) * P(A)
    where we additionally need to normalize over all values in A such that we have a proper prob. distribution that sums
    up to one.

    Args:
        prior (np.ndarray): 1D array of shape (N,) of prior probabilities
        conditional (np.ndarray): 1D array of shape (N,) of conditional probabilities

    Returns:
        np.ndarray: 1D array of shape (N,) of posterior probabilities
    """
    posterior_raw = conditional * prior
    return posterior_raw / np.sum(posterior_raw)


def sense(curr_belief: np.ndarray, measurement_idx: int, emission_matrix: np.ndarray) -> np.ndarray:
    measurement_conditional = emission_matrix[measurement_idx, :]
    return bayes(conditional=measurement_conditional, prior=curr_belief)


def propagate(curr_belief: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
    return transition_matrix @ curr_belief
