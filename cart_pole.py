import gym
import numpy as np
import DiscreteAV

def main():
    print('Cart Pole')
    env = gym.make('CartPole-v1')

    for i in range(20):
        print('Episode', i)
        env.reset()
        reward = 0
        done = False
        while not done:
            env.render()
            _, r, done, _ = env.step(env.action_space.sample())
            reward += r
        print('Reward:', reward)
        print('')

    env.close()


if __name__ == '__main__':
    main()