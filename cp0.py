import numpy as np

import gym

import qlearn

class cart_state(qlearn.state):
    def __init__(self, arr):
        super(cart_state, self).__init__(arr)

    def __str__(self):
        return str(super(cart_state, self).value())

class cart_pole:
    def __init__(self, num_episodes):
        self.num_episodes = num_episodes
        self.step = 0

    def run(self):
        env = gym.make('CartPole-v1')
        q = qlearn.qlearn(env.observation_space.shape, env.action_space.n)
        #env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1')

        last_rewards = []
        last_rewards_size = 100

        for i_episode in range(self.num_episodes):
            observation = env.reset()
            s = cart_state(observation)

            cr = 0
            while True:
                env.render()

                a = q.get_action(s)
                new_observation, reward, done, info = env.step(a)
                self.step += 1
                sn = cart_state(new_observation)

                q.store(s, a, sn, reward, done)
                q.learn()

                cr += reward

                observation = new_observation
                s = sn

                if done:
                    q.random_action_alpha_cap = q.ra_range_begin + (q.ra_range_end - q.ra_range_begin) * (1. - cr/500.)

                    if len(last_rewards) >= last_rewards_size:
                        last_rewards = last_rewards[1:]

                    last_rewards.append(cr)

                    print "%d episode, its reward: %d, total steps: %d, mean reward over last %d episodes: %.1f, std: %.1f" % (
                            i_episode, cr, self.step, len(last_rewards), np.mean(last_rewards), np.std(last_rewards))
                    break


        env.close()

import tensorflow as tf
with tf.device('/cpu:0'):
    cp = cart_pole(10000)
    cp.run()
