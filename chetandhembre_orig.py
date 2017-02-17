import time
import math

import numpy as np
import gym
import tensorflow as tf

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


      x = tf.placeholder(tf.float64, [None, input_size], name='x')
      y = tf.placeholder(tf.float64, [None, output_size], name='y')

      w1 = tf.Variable(tf.random_uniform([input_size, 64], dtype=tf.float64), name='w1')
      b1 = tf.Variable(tf.random_uniform([64], dtype=tf.float64), name='b1')
      tf.summary.histogram('w1', w1)
      
      w2 = tf.Variable(tf.random_uniform([64, output_size], dtype=tf.float64), name='w2')
      b2 = tf.Variable(tf.random_uniform([output_size], dtype=tf.float64), name='b2')
      tf.summary.histogram('w2', w2)

      h1 = tf.add(tf.matmul(x, w1), b1, name='h1')
      relu_h1 = tf.nn.tanh(h1, name='relu_h1')
      tf.summary.histogram('relu_h1', relu_h1)

      self.model =  tf.add(tf.matmul(relu_h1, w2), b2, name='model')
      tf.summary.histogram('model', self.model)

      self.error = tf.reduce_mean(tf.square(self.model - y), name='error')
      tf.summary.scalar('error', self.error)

      self.optimzer = tf.train.RMSPropOptimizer(LEARNING_RATE, name='Optimizer').minimize(self.error)
      self.sess = tf.Session()
      self.merged = tf.summary.merge_all()
      self.summary_writter = tf.summary.FileWriter("/tmp/cart_pole/run.%d" % (time.time()), self.sess.graph)

      self.init = tf.global_variables_initializer()
      self.sess.run(self.init)
  
  def get_action(self, state):
      output = self.sess.run([self.model], feed_dict={
        'x:0': state
      })
      return output[0][0]
  
  def train(self, states, actions):
      summary, _, error = self.sess.run([self.merged, self.optimzer, self.error], feed_dict={
        'x:0': states,
        'y:0': actions
      })
      self.summary_writter.add_summary(summary, self.count)
      self.count += 1
      return error
  
  def get_action_multiple(self, states):
      output = self.sess.run([self.model], feed_dict={
        'x:0': states
      })
      # sess.close()
      return output[0]

  def close(self):
      self.sess.close()


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
        #print "state: %s, q: %s, action: %s" % (state, q_values, action)
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
      current_states = np.ndarray(shape=(len(experiences), 4))
      next_states = np.ndarray(shape=(len(experiences), 4))

      idx = 0
      for experience in experiences:
        current_state, next_state, action, reward, is_done = experience
        current_states[idx] = current_state
        next_states[idx] = next_state

        idx += 1
      
      current_state_q_values = self.model.get_action_multiple(current_states)
      next_state_q_values = self.model.get_action_multiple(next_states)

      for i in range(len(experiences)):
        current_state, next_state, action, reward, is_done = experiences[i]

        next_max = np.amax(next_state_q_values[i])
        if is_done:
          reward = -10
          next_max = 0
        
        cur = current_state_q_values[i][action]
        current_state_q_values[i][action] = cur + ALPHA * (reward + DISCOUNT_FACTORE * next_max - cur)
      
      self.model.train(current_states, current_state_q_values)
  
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
    #print 'avg rewards: %s' % str(_avg)
    if _avg > 195 and False:
      return True
    
    return False


  def run(self):
    episodes = 0
    #self.env.monitor.start('results/cartpole',force=True)
    while episodes < self.total_episodes:
      #print 'running episode: %s' % str(episodes + 1)
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
      
      print 'episode: %d, rewards: %s, step: %s' % (episodes+1, str(total_reward), str(self.step))
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

env = Environment('CartPole-v0', 2500)
env.run()
