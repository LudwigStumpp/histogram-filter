from enum import Enum
import numpy as np
from typing import NamedTuple


class Color(Enum):
    BLUE = 0
    ORANGE = 1


# Character codes for visualizing the world
chrs = {
    Color.BLUE: "\x1b[6;38;44m",
    Color.ORANGE: "\x1b[6;30;43m",
    "CEND": "\x1b[0m",
}


class TransitionSpecification(NamedTuple):
    """
    Specification of a transition between two states in a Markov chain.

    Example
    -------
    >>> TransitionSpecification(0, [0.05, 0.9, 0.05])
    - 0.05 prob of staying in the same state
    - 0.9 prob of transitioning one to the right
    - 0.05 prob of transitioning two to the right
    >>> TransitionSpecification(-1, [0.05, 0.9, 0.05])
    - 0.05 prob of transitioning one to the left
    - 0.9 prob of staying in the same state
    - 0.05 prob of transitioning one to the right
    >>> TransitionSpecification(1, [0.9])
    - 0.9 prob of transitioning one to the right

    Attributes:
        start_index (int): Indicates to what relative state the first entry in probs is related to.
        probs (list[float]): List of probabilities of transitioning to each relative state.
            The sum of the probabilities must be 1.0.
    """

    start_index: int
    probs: list[float]


def get_transition_matrix(num_states: int, transition_spec: TransitionSpecification) -> np.ndarray:
    """Get transition matrix.

    Example
    -------
    >>> get_transition_matrix(3, TransitionSpecification(0, [0.2, 0.8]))
    array([ [0.2, 0. , 0.8],
            [0.8, 0.2, 0. ],
            [0. , 0.8, 0.2]])
    with entries:
    array([ [1->1, 2->1, 3->1],
            [1->2, 2->2, 3->2],
            [1->3, 2->3, 3->3]])


    Args:
        num_states (int): Number of states in the cyclic Markov chain.
        transition_spec (TransitionSpecification): Specification of the transition between states.

    Returns:
        np.ndarray: Transition matrix of shape (num_states, num_states) with probs. P_ij = P(j -> i).
            Sum of columns must be 1.0.
    """
    transition_matrix = np.zeros((num_states, num_states))
    for col in range(num_states):
        for row in range(len(transition_spec.probs)):
            transition_matrix[(row + col + transition_spec.start_index) % num_states, col] = transition_spec.probs[row]
    return transition_matrix


def get_sensor_matrix(field: list[Color], sensor_acc: float) -> np.ndarray:
    """Get sensor matrix with emission probabilities.

    Example
    -------
    >>> get_sensor_matrix([Color.BLUE, Color.BLUE, Color.ORANGE], 0.9)
    array([[0.9, 0.9, 0.1],
           [0.1, 0.1, 0.9]])
    with entries:
    array([[P(BLUE|1)   , P(BLUE|2)     , P(BLUE|3)    ],
           [P(ORANGE|1) , P(ORANGE|2)   , P(ORANGE|3)  ]])

    Args:
        field (list[Color]): List of colors in the field.
        sensor_acc (float): Sensor accuracy.

    Returns:
        np.ndarray: Sensor matrix of shape (len(Color), len(field)) with emission probabilities
            P_ij = P(color=i|state=j), where i is the index of the color and j is the index of the state.
    """
    sensor_matrix = np.zeros((len(Color), len(field)))
    for row_index, color in enumerate(Color):
        for col_index, field_color in enumerate(field):
            if field_color == color:
                sensor_matrix[row_index, col_index] = sensor_acc
            else:
                sensor_matrix[row_index, col_index] = 1 - sensor_acc
    return sensor_matrix


class Grid:
    """Grid of colors.

    Attributes:
        field (list[Color]): List of colors in the grid.
    """

    def __init__(self, field: list[Color]) -> None:
        self.field = field

    def __len__(self) -> int:
        return len(self.field)

    def __str__(self) -> str:
        # prints squares with colors next to each other in one line
        return " ".join(chrs[c] + "   " + chrs["CEND"] for c in self.field)

    @staticmethod
    def create_random_field(length: int, p_blue: float) -> list[Color]:
        """Create random list of colors with blue and orange.

        Args:
            length (int): Length of the list.
            p_blue (float): Probability of a blue square.

        Returns:
            list[Color]: List of colors.
        """
        color_values = np.random.choice([e.value for e in Color], length, p=[p_blue, 1 - p_blue])
        return [Color(value) for value in color_values]


class Agent:
    """Agent that moves in a grid.

    Attributes:
        transition (TransitionSpecification): Specification of the transition between states.
        correct_measurement_prob (float): Probability of correct measurement.
    """

    def __init__(self, transition: TransitionSpecification, correct_measurement_prob: float) -> None:
        self.transition = transition
        self.correct_measurement_prob = correct_measurement_prob


class Environment:
    """Environment that contains a grid and an agent.

    Attributes:
        grid (Grid): Grid of colors.
        agent (Agent): Agent that moves in the grid.
        position (int): Position of the agent in the grid.
    """

    def __init__(self, grid: Grid, agent: Agent, start_position: int = 0) -> None:
        self.grid = grid
        self.agent = agent
        self.position = start_position

    def move(self) -> None:
        """Move the agent in the grid according to the transition specification."""
        transition = self.agent.transition
        move = np.random.choice(
            np.arange(transition.start_index, transition.start_index + len(transition.probs)),
            p=transition.probs,
        )
        self.position = (self.position + move) % len(self.grid)

    def sense(self) -> Color:
        """Sense the color of the square the agent is on based on the agent's sensor accuracy."""
        true_color = self.grid.field[self.position]
        correct_measurement_prob = self.agent.correct_measurement_prob
        is_correct_measurement = np.random.choice(2, p=[1 - correct_measurement_prob, correct_measurement_prob])
        if is_correct_measurement:
            return true_color
        return Color(int(not (true_color.value)))

    def get_transition_matrix(self) -> np.ndarray:
        """Get transition matrix."""
        return get_transition_matrix(len(self.grid), self.agent.transition)

    def get_sensor_matrix(self) -> np.ndarray:
        """Get sensor matrix."""
        return get_sensor_matrix(self.grid.field, self.agent.correct_measurement_prob)

    def __str__(self) -> str:
        """Print the grid with the agent on it."""
        chars = [" X " if self.position == i else "   " for i in range(len(self.grid))]
        return " ".join(chrs[color] + char + chrs["CEND"] for color, char in zip(self.grid.field, chars))
