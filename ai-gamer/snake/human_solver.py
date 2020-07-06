
from base_model import BaseModel
from action import Action
from pygame.locals import *

class HumanSolver(BaseModel):

    action = None

    def __init__(self):
        BaseModel.__init__(self, "Human", "human", "hu")

    def move(self, environment):
        BaseModel.move(self, environment)
        if self.action is None:
            return environment.snake_action
        backward_action = self.action[0] == environment.snake_action[0] * -1 or self.action[1] == environment.snake_action[1] * -1
        return environment.snake_action if backward_action else self.action

    def user_input(self, event):
        if event.key == K_UP:
            self.action = Action.up
        elif event.key == K_DOWN:
            self.action = Action.down
        elif event.key == K_LEFT:
            self.action = Action.left
        elif event.key == K_RIGHT:
            self.action = Action.right