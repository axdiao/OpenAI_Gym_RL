"""
Q-Learning Algorithm Class
By Allen Diao (axdiao)

Implementation of the Q-Learning algorithm
"""

from DiscreteAV import DiscreteAV
import numpy as np
import gym

class QLearning:
    def __init__(self, env, policy='epsilon_greedy', num_episodes=1000, epsilon=0.1, step_size=0.9, discount_rate=0.9):
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.step_size = step_size
        self.discount_rate = discount_rate

    def run(self):
        # specific to DiscreteAV implementation
        state_action_dims = np.array([20, 20, 20, 20, 2])
        discretizer = DiscreteAV(state_action_dims, self.env.observation_space.high, self.env.observation_space.low)

        for i in range(self.num_episodes):
            print('Episode', i)
            state = self.env.reset()
            done = False
            reward_sum = 0
            while not done:
                self.env.render()
                action_values = discretizer.value(state)
                policy_action = self.__select_action(action_values)
                # interact with environment
                next_state, reward, done, _ = self.env.step(policy_action)
                reward_sum += reward
                # determine greedy action in next_state
                next_action = np.argmax(discretizer.value(next_state))
                discretizer.update(state, policy_action, reward, next_state, next_action)
                state = next_state

                # DECAY EPSILON
                # self.epsilon = 0.999 * self.epsilon

            print('\tReward:', reward_sum)
        self.env.close()

    def __select_action(self, actions):
        if self.policy == 'greedy':
            return np.argmax(actions)
        elif self.policy == 'epsilon_greedy':
            action = -1
            rand_num = np.random.rand()
            greedy_action = np.argmax(actions)
            if rand_num < 1 - self.epsilon:
                # take the greedy action (exploit)
                return greedy_action
            else:
                # choose an action randomly that isn't greedy (explore)
                action = np.random.randint(0, len(actions))
                while action == greedy_action:
                    action = np.random.randint(0, len(actions))
                return action
        else:
            return NotImplementedError
