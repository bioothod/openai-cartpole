import random

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Merge, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad, RMSprop
from keras.regularizers import l1l2, activity_l2

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
    def __init__(self, state_shape, actions):
        self.batch_size = 128
        self.fit_epochs = 128

        self.learn_num = 0

        self.m = Sequential()
        self.m.add(Dense(64, input_shape=state_shape, activation='tanh', W_regularizer=l1l2(0.01), activity_regularizer=activity_l2(0.01)))
        self.m.add(Dropout(0.5))
        self.m.add(Dense(64, activation='relu', W_regularizer=l1l2(0.01), activity_regularizer=activity_l2(0.01)))
        self.m.add(Dropout(0.5))
        self.m.add(Dense(actions, activation='relu', W_regularizer=l1l2(0.01), activity_regularizer=activity_l2(0.01)))

        self.m.compile(optimizer='rmsprop', loss='mse')

    def update(self, state, actions):
        self.m.fit(state.value(), actions.value(), nb_epoch=self.fit_epochs, verbose=0)

    def learn(self, history):
        #batch = history
        batch = random.sample(history, min(self.batch_size, len(history)))
        #batch[-1] = history[-1]

        assert len(batch) != 0
        assert len(batch[0]) != 0
        assert len(batch[0][0].value()) != 0


        xs = np.ndarray(shape=(len(batch), len(batch[0][0].value())))
        y = np.ndarray(shape=(len(batch), len(batch[0][1])))
        df = pd.DataFrame(index=range(xs.shape[0]), columns=['State', 'Q'])
        idx = 0
        for e in batch:
            xs[idx] = e[0].value()
            y[idx] = e[1]

            df.ix[idx, ['State', 'Q']] = e[0].value(), e[1]

            idx += 1
            #print "learn: state: %s -> %s" % (e[0].value(), e[1])

        #print df.head()
        out = 'json/learn.%d.json' % (self.learn_num)
        df.to_json(out, orient='index')
        self.learn_num += 1


        s = batch[-1][0]
        q = batch[-1][1]
        #print "learn: before: %s -> %s, need: %s" % (s.value(), self.q(s), q)

        self.m.fit(xs, y, nb_epoch=self.fit_epochs, verbose=0)

        #print "learn: after : %s -> %s, need: %s" % (s.value(), self.q(s), q)

    def q(self, state):
        p = self.m.predict(np.array([state.value()]), verbose=0)
        return p[0]

    def qa(self, state, action):
        p = self.q(state)

        #print "qa: state: %s -> %s" % (state.value(), p)
        return p[action]

class qlearn(object):
    def __init__(self, state_shape, actions):
        self.alpha = 0.9
        self.gamma = 0.1
        self.random_action_alpha = 0.8
        self.lam = 0.9

        self.Q = qnn(state_shape, actions)
        self.history = []
        self.E = {}

    def weighted_choice(self, ch):
        return np.random.choice()

    def get_action(self, state):
        random_alpha_choice = np.random.choice([True, False], p=[self.random_action_alpha, 1-self.random_action_alpha])

        q = self.Q.q(state)
        ch = 0

        if random_alpha_choice:
            ch = np.random.randint(0, len(q))
        else:
            ch = np.argmax(q)

        return ch, q

    def qa(self, state, action):
        return self.Q.qa(state, action)

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

    def update(self, s, a, sn, r):
        #eta = self.et_increment(s, a)
        eta = 1.

        amax_next, qvals_next = self.max_action(sn)
        qmax_next = qvals_next[amax_next]
        qvals = self.Q.q(s)
        current_qa = qvals[a]
        qsa = current_qa + self.alpha * (r + self.gamma * qmax_next - current_qa) * eta
        qvals[a] = qsa

        self.history.append((s, qvals))
        #self.et_discount()
        return qsa
