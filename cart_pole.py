import gym
import numpy as np
from QLearning import QLearning
from utility import config

def main():
    print('Cart Pole')
    env = gym.make('CartPole-v1')

    q_learn = QLearning(env, num_episodes=3000)
    q_learn.run()


if __name__ == '__main__':
    main()