# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:51:17 2020

@author: wyckliffe
"""


class Color :

    black = (0, 0, 0)
    red   = (255, 0, 0)
    green = (0, 150, 0)
    white = (255, 255, 255)
    gray  = (211, 211, 211)


class Constants:

    SLITHERIN_NAME = "Snake"
    FONT = "Arial"
    MODEL_FEATURE_COUNT = 5 #[action_vector, left_neighbor_accessible, top_neighbor_accessible, right_point_accessible, self.get_angle_from_fruit()]
    MODEL_NAME = "model.tf"
    DQN_MODEL_NAME = "model.h5"
    CHECKPOINT_NAME = "model.ckpt"
    MODEL_DIRECTORY = "./tf_models/"
    NAVIGATION_BAR_HEIGHT = 30
    FPS = 10
    PIXEL_SIZE = 25
    SCREEN_WIDTH = 300
    SCREEN_HEIGHT = 300
    FRAMES_TO_REMEMBER = 4
    SCREEN_DEPTH = 32
    ENV_HEIGHT = SCREEN_HEIGHT/PIXEL_SIZE
    ENV_WIDTH = SCREEN_WIDTH/PIXEL_SIZE

class Node:

    point = None
    previous_node = None
    action = None

    def __init__(self, point):
        self.point = point

    def __eq__(self, other):
        return self.point == other.point

    def __hash__(self):
        return hash(str(self.point.x)+str(self.point.y))

class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(str(self.x)+str(self.y))

class Queue:

    def __init__(self, initial_values):
        self.queue = initial_values

    def enqueue(self, val):
        self.queue.insert(0, val)

    def dequeue(self):
        if self.is_empty():
            return None
        else:
            return self.queue.pop()

    def size(self):
        return len(self.queue)

    def is_empty(self):
        return self.size() == 0

class Run:

    def __init__(self, action, score):
        self.action = action
        self.score = score

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Stack:

    def __init__(self, initial_values):
        self.stack = initial_values

    def pop(self):
        if self.is_empty():
            return None
        else:
            return self.stack.pop()

    def push(self, val):
        return self.stack.append(val)

    def peak(self):
        if self.is_empty():
            return None
        else:
            return self.stack[-1]

    def size(self):
        return len(self.stack)

    def is_empty(self):
        return self.size() == 0