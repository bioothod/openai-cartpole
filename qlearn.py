import time
import math

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l1_l2_regularizer

class state(object):
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value

class action(object):
    def __init__(self, n):
        self.action = np.array([n])

    def value(self):
        return self.action

    def __hash__(self):
        return hash(self.action)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return str(self.action)

class qnn(object):
    def __init__(self, input_size, output_size):
        self.learning_rate = 0.0025
        self.reg_beta = 0.001
        self.l1_neurons = 128
        self.train_num = 0

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 10000, 0.5, staircase=True)
        reg_beta = tf.train.exponential_decay(self.reg_beta, global_step, 10000, 0.5, staircase=True)

        x = tf.placeholder(tf.float32, [None, input_size], name='x')
        y = tf.placeholder(tf.float32, [None, output_size], name='y')

        w1 = tf.Variable(tf.random_uniform([input_size, self.l1_neurons], dtype=tf.float32), name='w1')
        b1 = tf.Variable(tf.random_uniform([self.l1_neurons], dtype=tf.float32), name='b1')
        tf.summary.histogram('w1', w1)
        h1 = tf.add(tf.matmul(x, w1), b1, name='h1')
        nl_h1 = tf.nn.tanh(h1, name='nonlinear_h1')
        tf.summary.histogram('nonlinear_h1', nl_h1)

        w2 = tf.Variable(tf.random_uniform([self.l1_neurons, output_size], dtype=tf.float32), name='w2')
        b2 = tf.Variable(tf.random_uniform([output_size], dtype=tf.float32), name='b2')
        tf.summary.histogram('w2', w2)

        #self.model =  tf.nn.relu(tf.add(tf.matmul(nl_h1, w2), b2, name='model'))
        self.model =  tf.add(tf.matmul(nl_h1, w2), b2, name='model')
        tf.summary.histogram('model', self.model)

        reg_val = apply_regularization(l1_l2_regularizer(scale_l1=1.0, scale_l2=1.0), [w1, w2]) * reg_beta
        tf.summary.scalar('reg_beta', reg_beta)
        tf.summary.scalar('reg_val', reg_val)
        tf.summary.scalar('learning_rate', learning_rate)

        self.error = tf.reduce_mean(tf.square(self.model - y) + reg_val, name='error')
        tf.summary.scalar('error', self.error)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9, name='optimizer').minimize(self.error, global_step=global_step)
        self.sess = tf.Session()
        self.merged = tf.summary.merge_all()
        self.summary_writter = tf.summary.FileWriter("/tmp/cp0/run.%d" % (time.time()), self.sess.graph)

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self, states, qvals):
        self.train_num += 1

        summary, _, error = self.sess.run([self.merged, self.optimizer, self.error], feed_dict={
                'x:0': states,
                'y:0': qvals,
            })
        self.summary_writter.add_summary(summary, self.train_num)

        return error

    def q(self, states):
        p = self.sess.run([self.model], feed_dict={
                'x:0': states,
            })
        return p[0]

class qlearn(object):
    def __init__(self, state_shape, actions):
        self.alpha = 1
        self.gamma = 0.99
        self.random_action_alpha = 1
        self.random_action_alpha_cap = 1
        self.ra_range_begin = 0.1
        self.ra_range_end = 0.99
        self.total_actions = 0
        self.lam = 0.9
        self.history_size = 100000
        self.batch_size = 128

        self.actions = actions

        self.Q = qnn(state_shape[0], actions)
        self.history = []
        self.E = {}

    def weighted_choice(self, ch):
        return np.random.choice()

    def get_action(self, state):
        self.total_actions += 1
        self.random_action_alpha = self.ra_range_begin + (self.random_action_alpha_cap - self.ra_range_begin) * math.exp(-0.0001 * self.total_actions)

        self.random_action_alpha = 0.1
        random_choice = np.random.choice([True, False], p=[self.random_action_alpha, 1-self.random_action_alpha])

        ch = 0
        if random_choice:
            ch = np.random.randint(0, self.actions)
        else:
            v = state.value()
            q = self.Q.q(v.reshape(1, v.shape[0]))
            ch = np.argmax(q[0])
            #print "state: %s, q: %s, action: %s" % (state, q, ch)

        return ch

    def max_action(self, state):
        q = self.Q.q(state)
        a = np.argmax(q)
        return a, q

    def et_increment(self, s, a):
        et = self.E.get(s)
        if not et:
            et = {}
        eta = et.get(a)
        if not eta:
            eta = 1
        else:
            eta += 1

        et[a] = eta
        self.E[s] = et
        return eta

    def et_discount(self):
        for s, et in self.E.iteritems():
            for a, eta in et.iteritems():
                eta *= self.lam * self.gamma
                et[a] = eta
            self.E[s] = et

    def store(self, s, a, sn, r, done):
        if len(self.history) > self.history_size:
            self.history = self.history[1:]

        self.history.append((s, a, sn, r, done))

    def learn(self):
        hsize = len(self.history)
        indexes = np.random.randint(hsize, size=min(self.batch_size, hsize))
        batch = []
        for i in indexes:
            batch.append(self.history[i])

        assert len(batch) != 0
        assert len(batch[0]) != 0
        assert len(batch[0][0].value()) != 0

        states_shape = (len(batch), len(batch[0][0].value()))
        states = np.ndarray(shape=states_shape)
        next_states = np.ndarray(shape=states_shape)

        q_shape = (len(batch), self.actions)
        qvals = np.ndarray(shape=q_shape)
        next_qvals = np.ndarray(shape=q_shape)

        idx = 0
        for e in batch:
            s, a, sn, r, done = e

            states[idx] = s.value()
            next_states[idx] = sn.value()
            idx += 1

        qvals = self.Q.q(states)
        next_qvals = self.Q.q(next_states)

        qvals_orig = qvals.copy()
        for idx in range(len(batch)):
            e = batch[idx]
            s, a, sn, r, done = e

            qmax_next = np.amax(next_qvals[idx])
            if done:
                r = -10
                qmax_next = 0

            current_qa = qvals[idx][a]
            qsa = current_qa + self.alpha * (r + self.gamma * qmax_next - current_qa)
            qvals[idx][a] = qsa

        self.Q.train(states, qvals)
