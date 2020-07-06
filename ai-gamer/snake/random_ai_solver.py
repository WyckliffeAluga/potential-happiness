

import random
from base_model import BaseModel


class RandomSolver(BaseModel):

    def __init__(self):
        BaseModel.__init__(self, "Random", "random", "r")

    def move(self, environment):
        BaseModel.move(self, environment)
        return random.choice(environment.possible_actions_for_current_action(environment.snake_action))