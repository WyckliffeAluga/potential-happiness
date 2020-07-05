# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 19:19:14 2020

@author: wyckliffe
"""


import gym
import argparse
import numpy as np
#import atari_py
from models.ddqn_model import DDQNTrainer, DDQNSolver
from models.ge_model import GETrainer, GESolver
from wrappers import MainGymWrapper