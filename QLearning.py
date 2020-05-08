"""
Q-Learning ALgorithm Class
By ALlen Diao (axdiao)

Implementation of the Q-Learning algorithm
"""

class QLearning:
    def __init__(self, env, epsilon=0.1, step_size=0.9, discount_rate=0.9):
        self.env = env
        self.epsilon = epsilon
        self.step_size = step_size
        self.discount_rate = discount_rate

