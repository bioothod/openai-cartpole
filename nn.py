import os

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Merge, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad, RMSprop

class qnn(object):
    def __init__(self, state_shape):
        self.batch_size = 30

        self.learn_num = 0

        state_branch = Sequential()
        state_branch.add(Dense(40, input_shape=state_shape, activation='relu'))

        action_branch = Sequential()
        action_branch.add(Dense(1, input_shape=(1,), activation='relu'))

        merge = Merge([state_branch, action_branch], mode='concat', concat_axis=1)

        self.m = Sequential()
        self.m.add(merge)
        #self.m.add(Dropout(0.5))
        self.m.add(Dense(20, activation='relu'))
        #self.m.add(Dropout(0.5))
        self.m.add(Dense(1, activation='relu'))

        self.m.compile(optimizer='rmsprop', loss='mse')

    def update(self, state, action, q):
        comb = [np.array([state.value()]), np.array([action.value()])]
        self.m.fit(comb, q, nb_epoch=30, verbose=0)

    def test(self):
        state = np.array([ 0.03073904 , 0.00145001, -0.03088818, -0.03131252])
        for a in [0, 1]:
            action = np.array([a])
            print "test: %s/%s -> %s" % (state, action, self.qa(state, action))

    def learn(self, path):
        df = pd.read_json(path, orient='index').astype({'State': np.ndarray, 'Action': np.ndarray}, raise_on_error=True, copy=True)
        print df.info()
        print df.head()

        s = df.ix[0, 'State']
        print "%s, type: %s" % (s, type(s))
        self.m.fit([df.State[0], df.Action[0]], df.Q, nb_epoch=128, verbose=0)
        self.test()

    def qa(self, state, action):
        comb = [np.array([state]), np.array([action])]
        p = self.m.predict(comb, verbose=0)

        #print "qa: state: %s, action: %s, predict: %s" % (state.value(), action.value(), p)

        return p[0][0]

q = qnn((4,))
q.learn('json/learn.0.json')
exit(0)

dir='json/'
for f in os.listdir(dir):
    path = dir + f
    q.learn(path)
