# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:43:25 2020

@author: wyckliffe
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
## solves the cart pole environment with random search
def get_action(s, w) :
    # takes in the state s and returns the dot product of states with weights
    return 1 if s.dot(w) > 0 else 0

def episode(env, params) :
    # plays one episode
    observation = env.reset()
    done = False
    t = 0
    while not done and t < 10000:
      #  env.render() # uncomment to see the enviroment will take longer
        t += 1
        action = get_action(observation, params)
        observation, reward , done , info = env.step(action)
        if done :
            break
    return t

def series(env, T , params) :
    length_of_episode = np.empty(T)
    
    for i in range(T) :
        length_of_episode[i] = episode(env, params)
    average_length = length_of_episode.mean()
    print("Average length: ", average_length)
    return average_length

def random_search(env) :
    length_of_episode = []
    best = 0
    params = None
    
    for t in range(100) :
        new_params = np.random.random(4) * 2 - 1
        average_length = series(env, 100 , new_params)
        length_of_episode.append(average_length)
        if average_length > best :
            params = new_params
            best = average_length

    return length_of_episode , params

if __name__ == "__main__"  :

    env = gym.make("CartPole-v0")
    length_of_episode , params = random_search(env)
    plt.plot(length_of_episode)
    plt.show()

    # play a final set of episodes
    env = wrappers.Monitor(env, 'videos')
    print("*** Final run with final weights***", episode(env, params))
    series(env, 100, params)
