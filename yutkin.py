import gym
from gym import wrappers
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier

def generate_session(env, agent, t_max=1000):
    
    states,actions = [],[]
    total_reward = 0
    
    s = env.reset()
    
    for t in range(t_max):
        
        #predict array of action probabilities
        probs = agent.predict_proba(np.array([s]))[0] 
        
        a = np.random.choice(n_actions, p=probs)
        
        new_s,r,done,info = env.step(a)
        # env.render(close=True)
        
        states.append(s)
        actions.append(a)
        total_reward+=r
        
        s = new_s
        if done:
            break
    return states,actions,total_reward

if __name__ == '__main__':
  env = gym.make("CartPole-v0")
  # env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)

  env.reset()
  n_actions = env.action_space.n

  agent = MLPClassifier(hidden_layer_sizes=(40,40), activation='relu', warm_start=True, max_iter=1)
  #initialize agent to the dimension of state an amount of actions
  agent.fit([env.reset()]*n_actions,list(range(n_actions)))

  n_samples = 500
  percentile = 50
  smoothing = 0.01

  for i in range(100):
    sessions = [generate_session(env, agent) for _ in range(n_samples)]
    
    # sessions = Parallel(n_jobs=4)(delayed(generate_session)() for _ in range(n_samples))
    batch_states, batch_actions, batch_rewards = map(np.array, zip(*sessions))
    threshold = np.percentile(batch_rewards, percentile)
    
    elite_states = batch_states[batch_rewards > threshold]
    elite_actions = batch_actions[batch_rewards > threshold]
    
    if len(elite_states) > 0:
      elite_states, elite_actions = map(np.concatenate,[elite_states,elite_actions])
      agent.fit(elite_states, elite_actions)
      print("mean reward = %.5f\tthreshold = %.1f"%(np.mean(batch_rewards), threshold))
