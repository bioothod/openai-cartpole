import math

import numpy as np
import gym
import tensorflow as tf

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.models import Sequential
from keras.layers import Dense, Activation, Merge, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad, RMSprop
from keras.regularizers import l2

"""
1. Q learning
2. Experience Replay
3. e greedy
4. learning rate

1. We need place to store experience
2. model to predict q values for given state and action
3. Agent which will take actions for given state. perform greddy exploration
4. Environment stimulation
5. figure out better way to change epsilon
"""
EXPERIENCE_REPLAY_BATCH = 64
EXPERIENCE_BUFFER_SIZE = 100000
START_EPSILON = 0.99
END_EPSILON = 0.1
EPSILON_STEP_LIMIT = 100
LEARNING_RATE = 0.0025
DISCOUNT_FACTORE = 0.99
ALPHA = 1

class Model(object):
  def __init__(self, input_size, output_size):
      self.count = 0

      self.m = Sequential()
      self.m.add(Dense(64, input_dim=input_size, activation='tanh', bias=True, init='uniform'))
      #self.m.add(Dense(64, activation='relu', W_regularizer=l2(0.01), bias=True))
      self.m.add(Dense(output_size, activation='linear', bias=True))

      self.m.compile(optimizer='rmsprop', loss='mse', learning_rate=0.0025)

  
  def get_action(self, state):
      state = np.array(state, dtype=np.float64)
      p = self.m.predict(state, verbose=0)
      return p[0][0]
  
  def train(self, states, actions):
      #states = np.array(states, dtype=np.float64)
      #actions = np.array(actions, dtype=np.float64)
      self.m.train_on_batch(states, actions)
  
  def get_action_multiple(self, states):
      states = np.array(states, dtype=np.float64)
      p = self.m.predict_on_batch(states)
      return p

  def close(self):
      pass


class Memory(object):
  def __init__(self):
      self.experience = []
      self.visited = {}
  
  def remember(self, state, next_state, action, reward, is_done):
      state = np.array(state, dtype=np.float64)
      next_state = np.array(next_state, dtype=np.float64)
      experience = (state, next_state, action, reward, is_done)
      if len(self.experience) > EXPERIENCE_BUFFER_SIZE:
        self.experience = self.experience[1:]
      
      self.experience.append(experience)

  def recall(self):
      experience_size = len(self.experience)
      _EXPERIENCE_REPLAY_BATCH = EXPERIENCE_REPLAY_BATCH
      if experience_size < EXPERIENCE_REPLAY_BATCH:
        _EXPERIENCE_REPLAY_BATCH = experience_size

      indexes = np.random.randint(experience_size, size=_EXPERIENCE_REPLAY_BATCH)
      experiences = []
      for index in indexes:
        experiences.append(self.experience[index])
      
      return experiences
  
  
class Agent(object):
    def __init__(self, epsilons):
      self.epsilons = epsilons
      self.epsilons_index = 0
      self.epsilon = START_EPSILON
      self.total_actions = 0
      self.total_greedy_actions = 0
      self.model = Model(4, 2)
      self.memory = Memory()

    @staticmethod
    def is_greddy(epsilon):
      return np.random.choice([0, 1], 1, p=[epsilon, 1 - epsilon])[0]
    
    def update_epsilon(self):
      index = int(self.total_actions / EPSILON_STEP_LIMIT)
      if index > len(self.epsilons - 1):
        index = len(self.epsilons) - 1
      
      self.epsilons_index = index
    
    def take_action(self, state):
      """
      actions are whether you want to go right or left
      """
      self.total_actions += 1
      q_values = self.model.get_action(state.reshape(1, 4))
      is_greedy = Agent.is_greddy(self.epsilon)
      msg = ''
      if is_greedy:
        action = np.argmax(q_values)
      else:
        action = np.random.choice([0, 1], 1)[0]
        msg = 'explorer'

      self.epsilon = END_EPSILON + (START_EPSILON - END_EPSILON) * math.exp(-0.001 * self.total_actions)
      return action

    def observe_results(self, state, next_state, action, reward, is_done):
      """
      after taking action environment return result of it, store (state, action, reward, is_done) in memory 
      for experience replay
      """
      self.memory.remember(state, next_state, action, reward, is_done)
      self.update()
    
    def close(self):
      return self.model.close()

    def update(self):
      experiences = self.memory.recall()
      current_states = None
      next_states = None
      for experience in experiences:
        current_state, next_state, action, reward, is_done = experience
        current_state = np.array(current_state).reshape(1, 4)
        next_state = np.array(next_state).reshape(1, 4)
        if current_states is None:
          current_states = current_state
          next_states = next_state
        else:
          current_states = np.vstack((current_states, current_state))
          next_states = np.vstack((next_states, next_state))
      
      current_state_q_values = self.model.get_action_multiple(current_states)
      next_state_q_values = self.model.get_action_multiple(next_states)
      #print "current: states: %s, values: %s" % (current_states, current_state_q_values)
      x = None
      y = None
      for i in range(len(experiences)):
        current_state, next_state, action, reward, is_done = experiences[i]
        current_state_q_value = np.array(current_state_q_values[i], dtype=np.float64)
        next_state_q_value = np.array(next_state_q_values[i], dtype=np.float64)
        if is_done:
          reward = -10
          next_state_q_value = [0.0, 0.0]
        
        current_state_q_value[action] = ALPHA * (reward + DISCOUNT_FACTORE * np.amax(next_state_q_value))
        
        current_state = np.array(current_state).reshape(1, 4)
        current_state_q_value = np.array(current_state_q_value).reshape(1, 2)

        if x is None:
          x = current_state
          y = current_state_q_value
        else:
          #print "x: %s, current_state: %s" % (x, current_state)
          x = np.vstack((x, current_state))
          y = np.vstack((y, current_state_q_value))
      
      self.model.train(x, y)
  
class Environment(object):
  def __init__(self, env_name, total_episodes):
    self.env = gym.make(env_name)
    self.total_episodes = total_episodes
    self.epsilons = np.linspace(START_EPSILON, END_EPSILON)
    self.agent = Agent(self.epsilons)
    self.step = 0
    self.avg = []
  
  def add_rewards(self, total_rewards):
    self.avg.append(total_rewards)
    l = len(self.avg)
    if l < 100:
      return False

    _avg = float(sum(self.avg[l - 100: l])) / max(len(self.avg[l - 100: l]), 1)
    print 'avg rewards: %s' % str(_avg)
    if _avg > 195:
      return True
    
    return False


  def run(self):
    episodes = 0
    #self.env.monitor.start('results/cartpole',force=True)
    while episodes < self.total_episodes:
      print 'running episode: %s' % str(episodes + 1)
      state = self.env.reset()
      is_done = False
      total_reward = 0
      while not is_done:
        self.env.render()
        action = self.agent.take_action(state)
        next_state, reward, is_done, info = self.env.step(action)
        self.step += 1
        total_reward += reward
        self.agent.observe_results(state, next_state, action, reward, is_done)
        state = next_state
      
      print 'rewards: %s, step: %s' % (str(total_reward), str(self.step))
      if self.add_rewards(total_reward):
        print 'done with episods %s and steps: %s' % (str(episodes), str(self.step))
        self.env.monitor.close()
        self.agent.close()
        # self._plot()
        return 

      episodes += 1
    
  def _plot(self):
    plt.plot(self.avg)
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.savefig('rewards.png')
    plt.show()

with tf.device('/cpu:0'):
    env = Environment('CartPole-v0', 250)
    env.run()
