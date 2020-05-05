"""
DiscreteAV Class
By ALlen Diao (axdiao@umich.edu)

Takes the continuous state space of the cart pole problem and
discretizes it for approximation of the action-value function
Q(s, a)

"""
import numpy as np


class DiscreteAV:
    UPPER_BOUND = 1e3
    LOWER_BOUND = -1e3

    def __init__(self, state_action_dims, state_high, state_low):
        self.state_action_dims = state_action_dims
        self.state_high = state_high
        self.state_low = state_low
        self.tile_size = np.zeros(state_high.shape)
        self.Q = np.zeros(state_action_dims)

        for i in range(len(state_high)):
            if self.state_low[i] < self.LOWER_BOUND or self.state_high[i] > self.UPPER_BOUND:
                # preparation for rescaling using tanh
                self.state_low[i] = -1
                self.state_high[i] = 1

            self.tile_size[i] = (self.state_high[i] - self.state_low[i]) / self.state_action_dims[i]

    def num_state_actions(self):
        return np.prod(self.state_action_dims)

    def value(self, state):
        """
        Given a state as input, returns a n-dimensional numpy array containing the expected
        rewards for all possible actions in that state

        :returns
            n-dimensional numpy array representing action-values for the state
        """
        return NotImplementedError

    def __discretize(self, state):
        """
        Takes a n-dimensional continuous state and determines its index within Q array using
        Tiling. For state dimensions with infinite range, hyperbolic tangent function is
        used to rescale.

        :returns
            n-dimensional numpy array representing index into Q array for state
        """

        state_indices = []
        for i in range(len(self.state_high)):
            if state[i] < self.state_low[i] or state[i] > self.state_high[i]:
                # rescale dim using hyperbolic tangent function
                state[i] = np.tanh(state[i])

            state_indices.append(int(np.floor(state[i] / self.tile_size[i] + self.state_action_dims[i] / 2)))

        return np.array(state_indices)