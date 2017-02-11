import numpy as np

import gym

import qlearn

class cart_state(qlearn.state):
    def __init__(self, arr):
        super(cart_state, self).__init__(arr)

    def __str__(self):
        return str(super(cart_state, self).value())

class cart_pole:
    def __init__(self, num_eposides, lr):
        self.num_episodes = num_episodes
        self.lr = lr

    def run(self):
        env = gym.make('CartPole-v0')
        q = qlearn.qlearn(env.observation_space.shape, env.action_space.n)
        #env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1')

        history = []
        learned_episodes = 0
        for i_episode in range(num_episodes):
            env.seed(1)
            observation = env.reset()
            s = cart_state(observation)

            cr = 0
            history = []
            while True:
                env.render()

                a, qvals = q.get_action(s)
                new_observation, reward, done, info = env.step(a)
                sn = cart_state(new_observation)

                #q.update(s, a, sn, cr)

                cr += reward
                history.append((s, a, sn, reward))

                if len(history) > 1024:
                    history = history[1:]

                observation = new_observation
                s = sn

                if done:
                    print "%d episode finished after %d time steps" % (i_episode, cr)
                    break

            if True:
                q.history = []
                for h in history:
                    s = h[0]
                    a = h[1]
                    sn = h[2]
                    reward = h[3]

                    q.update(s, a, sn, cr)

            for i in range(60):
                q.Q.learn(q.history)

            learned_episodes += 1

            if learned_episodes % 10 == 0:
                ra = q.random_action_alpha * 0.99
                print "changing random action alpha: %.2f -> %.2f" % (q.random_action_alpha, ra)
                q.random_action_alpha = ra

        env.close()

import tensorflow as tf
with tf.device('/cpu:0'):
    num_episodes = 10000
    learning_rates = [5e-2]
    for lr in learning_rates:
        cp = cart_pole(num_episodes, lr)
        cp.run()
