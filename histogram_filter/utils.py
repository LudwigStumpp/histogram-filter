from typing import Iterable

import numpy as np
import termplotlib as tpl

from histogram_filter.world import (
    Agent,
    Color,
    Environment,
    Grid,
    TransitionSpecification,
)


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


def setup(
    field: list[Color] = [Color.BLUE, Color.ORANGE, Color.BLUE, Color.BLUE, Color.ORANGE],
    correct_measurement_prob: float = 0.9,
    transition: TransitionSpecification = TransitionSpecification(0, [0.05, 0.9, 0.05]),
    start_position: int = 0,
) -> Environment:
    """Sets up the environment for the histogram filter.

    Args:
        correct_measurement_prob (float, optional): Sensor accuracy. Defaults to 0.9.
        transition (TransitionSpecification, optional): Transition probabilities. Defaults to
            TransitionSpecification(0, [0.05, 0.9, 0.05]).
        start_position (int, optional): Start index on the grid. Defaults to 0.

    Returns:
        Environment: Environment object.
    """
    my_grid = Grid(field)
    my_agent = Agent(transition=transition, correct_measurement_prob=correct_measurement_prob)
    my_env = Environment(my_grid, my_agent, start_position=start_position)
    return my_env


def visualize_belief(belief: np.ndarray):
    """Visualizes the belief as a line plot.

    Args:
        belief (np.ndarray): 1D array of shape (N,) of belief probabilities.
            Number of entries equals the number of states.
    """
    fig = tpl.figure()
    fig.plot(np.arange(1, len(belief) + 1), belief, width=50, height=15)
    fig.show()


def format_list(list_to_format: Iterable, element_formatter: str) -> list:
    """Formats a list.

    Args:
        list_to_format (list): List to format.
        element_formatter (str, optional): Format for each element. Defaults to "{:.2f}".
        delimiter (str, optional): Delimiter between elements. Defaults to ", ".

    Returns:
        str: Formatted list.
    """
    return [element_formatter.format(element) for element in list_to_format]


def format_belief(belief: np.ndarray) -> list:
    """Formats a belief.

    Args:
        belief (np.ndarray): 1D array of shape (N,) of belief probabilities.
            Number of entries equals the number of states.

    Returns:
        list: Formatted belief.
    """
    return format_list(belief, element_formatter="{:.2f}")
