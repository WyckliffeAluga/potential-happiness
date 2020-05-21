import gym
import os
import sys
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

from mountain_car_v1_q_learning import Transformer



# so you can test different architectures
class Layer:

  def __init__(self, m1, m2, f=T.nnet.relu, use_bias=True, zeros=False):

    if zeros:
      w = np.zeros((m1, m2))
    else:
      w = np.random.randn(m1, m2) * np.sqrt(2 / m1)

    self.w = theano.shared(w)
    self.params = [self.w]
    self.use_bias = use_bias

    if use_bias:
      self.b = theano.shared(np.zeros(m2))
      self.params += [self.b]

    self.f = f

  def forward(self, x):

    if self.use_bias:
      a = x.dot(self.w) + self.b
    else:
      a = x.dot(self.w)

    return self.f(a)


# approximates pi(a | s)
class Policy:


  def __init__(self, ft, D, layer_sizes_mean=[], layer_sizes_var=[]):
    # save inputs for copy
    self.ft = ft
    self.D = D
    self.layer_sizes_mean = layer_sizes_mean
    self.layer_sizes_var = layer_sizes_var

    ##### model the mean #####

    self.mean_layers = []
    m1 = D
    for m2 in layer_sizes_mean:
      layer = Layer(m1, m2)
      self.mean_layers.append(layer)
      m1 = m2

    # final layer
    layer = Layer(m1, 1, lambda x: x, use_bias=False, zeros=True)
    self.mean_layers.append(layer)


    ##### model the variance #####
    self.var_layers = []
    M1 = D
    for M2 in layer_sizes_var:
      layer = Layer(m1, m2)
      self.var_layers.append(layer)
      m1 = m2

    # final layer
    layer = Layer(m1, 1, T.nnet.softplus, use_bias=False, zeros=False)
    self.var_layers.append(layer)

    # get all params for gradient
    params = []
    for layer in (self.mean_layers + self.var_layers):
      params += layer.params
    self.params = params

    # inputs and targets
    x = T.matrix('x')
    actions = T.vector('actions')
    advantages = T.vector('advantages')

    # calculate output and cost
    def get_output(layers):
      z = x
      for layer in layers:
        z = layer.forward(z)
      return z.flatten()

    mean = get_output(self.mean_layers)
    var = get_output(self.var_layers) + 1e-4 # smoothing


    self.predict_ = theano.function(
      inputs=[x],
      outputs=[mean, var],
      allow_input_downcast=True
    )

  def predict(self, x):

    x = np.atleast_2d(x)
    x = self.ft.transform(x)

    return self.predict_(x)

  def sample_action(self, x):

    pred = self.predict(x)
    mu = pred[0][0]
    v = pred[1][0]
    a = np.random.randn()*np.sqrt(v) + mu

    return min(max(a, -1), 1)

  def copy(self):

    clone = Policy(self.ft, self.D, self.layer_sizes_mean, self.layer_sizes_mean)
    clone.copy_from(self)
    return clone

  def copy_from(self, other):
    # self is being copied from other
    for p, q in zip(self.params, other.params):
      v = q.get_value()
      p.set_value(v)

  def perturb_params(self):

    for p in self.params:
      v = p.get_value()
      noise = np.random.randn(*v.shape) / np.sqrt(v.shape[0]) * 5.0
      if np.random.random() < 0.1:
        # with probability 0.1 start completely from scratch
        p.set_value(noise)
      else:
        p.set_value(v + noise)


def episode(env, policy, gamma):

  observation = env.reset()
  done = False
  total_reward = 0
  iterations = 0

  while not done and iterations < 2000:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early
    action = policy.sample_action(observation)
    # oddly, the mountain car environment requires the action to be in
    # an object where the actual action is stored in object[0]
    observation, reward, done, info = env.step([action])

    total_reward += reward
    iterations += 1

  return total_reward


def series(env, T, policy, gamma, print_iters=False):

  total_rewards = np.empty(T)

  for i in range(T):
    total_rewards[i] = episode(env, policy, gamma)

    if print_iters:
      print(i, "Average so far:", total_rewards[:(i+1)].mean())

  avg_totalrewards = total_rewards.mean()
  print("Average total rewards:", avg_totalrewards)
  return avg_totalrewards


def random_search(env, policy, gamma):

  total_rewards = []
  best_avg_totalreward = float('-inf')
  best_policy = policy
  num_episodes_per_param_test = 3

  for t in range(100):
    tmp_model = best_policy.copy()

    tmp_model.perturb_params()

    avg_totalrewards = series(
      env,
      num_episodes_per_param_test,
      tmp_model,
      gamma
    )
    total_rewards.append(avg_totalrewards)

    if avg_totalrewards > best_avg_totalreward:
      best_policy = tmp_model
      best_avg_totalreward = avg_totalrewards

  return total_rewards, best_policy


def main():
  env = gym.make('MountainCarContinuous-v0')
  ft = Transformer(env, n_components=100)
  D = ft.dimensions
  model = Policy(ft, D, [], [])
  gamma = 0.99

  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)


  total_rewards, model = random_search(env, model, gamma)

  print("max reward:", np.max(total_rewards))

  # play 100 episodes and check the average
  avg_totalrewards = series(env, 100, model, gamma, print_iters=True)
  print("avg reward over 100 episodes with best models:", avg_totalrewards)

  plt.plot(total_rewards)
  plt.title("Rewards")
  plt.show()


if __name__ == '__main__':
  main()