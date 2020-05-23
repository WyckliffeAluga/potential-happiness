# -*- coding: utf-8 -*-
"""
Created on Fri May 22 20:20:46 2020

@author: wyckliffe
"""

import numpy as np
# initialize the states so that they can represent all the positions

states = {"A" : 0,
          "B" : 1,
          "C" : 2,
          "D" : 3,
          "E" : 4,
          "F" : 5,
          "G" : 6,
          "H" : 7,
          "I" : 8,
          "J" : 9,
          "K" : 10,
          "L" : 11,}
# creat ea list of all possible actionss (we won't be using this )
actions = [0,1,2,3,4,5,6,7,8,9,10,11]
# create the rewards in form of rewards matrix 0's represents the positions the robot can't go
# from a specific position
# and 1's represent the positions the robot can go
# for instance the first row / column you can only go to one position from A
R = np.array([
              [0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,1,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])
# create route function which uses Q learning to find best route
def route(starting_location, ending_location):
    gamma = 0.75
    alpha = 0.9
    state_to_location = {state: location for location, state in states.items()}
    R_new = np.copy(R)
    ending_state = states[ending_location]
    R_new[ending_state, ending_state] = 100
    Q = np.array(np.zeros([12,12]))

    for i in range(1000):
        current_state = np.random.randint(0,12)
        playable_actions = []

        for j in range(12):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)

        TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD
    route = [starting_location]
    next_location = starting_location

    print(Q.astype(int))

    while (next_location != ending_location):
        starting_state = states[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location

    return route
# this function allows for the robot to got to top prioritity via other locations that may be in top 3 location.
def best_route(starting_location, intermediary_location, ending_location):
    return route(starting_location, intermediary_location) + route(intermediary_location, ending_location)[1:]

if __name__ == '__main__' :

      route= best_route('A', 'H', 'D')
      print(route)