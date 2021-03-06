"""
DiscreteAV Class
By ALlen Diao (axdiao@umich.edu)

Takes the continuous state space of the cart pole problem and
discretizes it for approximation of the action-value function
Q(s, a)

"""
import numpy as np
from utility import config

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
        indices = self.__discretize(state)
        return self.Q[indices]

    def action_value(self, state, action):
        """
        Given a state and action as input, returns the float value associated with the state
        action pair; Q(s, a).

        :returns
            float value representing Q(s, a)
        """
        indices = self.__discretize(state)
        indices = tuple(list(indices).append(action))
        return self.Q[indices]

    def update(self, state, action, reward, next_state, next_action):
        """
        Given a state and action as input, updates the value of the Q for that state action pair with
        the reward that was received using
        """

        indices = self.__discretize(state)
        next_indices = self.__discretize(next_state)
        # append action to state for indexing
        idx_list = list(indices)
        idx_list.append(action)
        indices = tuple(idx_list)
        next_idx_list = list(next_indices)
        next_idx_list.append(next_action)
        next_indices = tuple(next_idx_list)

        # load hyper-parameters
        alpha = config('Discretized.step_size')
        gamma = config('Discretized.discount_rate')

        # perform update
        self.Q[indices] = self.Q[indices] + alpha * (reward + gamma * self.Q[next_indices] - self.Q[indices])



    def __discretize(self, state):
        """
        Takes a n-dimensional continuous state and determines its index within Q array using
        Tiling. For state dimensions with infinite range, hyperbolic tangent function is
        used to rescale.

        :returns
            n-dimensional tuple representing index into Q array for state
        """

        state_indices = []
        for i in range(len(self.state_high)):
            if state[i] < self.state_low[i] or state[i] > self.state_high[i]:
                # rescale dim using hyperbolic tangent function
                state[i] = np.tanh(state[i])

            state_indices.append(int(np.floor(state[i] / self.tile_size[i] + self.state_action_dims[i] / 2)))

        return tuple(state_indices)
