import random
from Snake.SnakeLogic import SnakeLogic
from .BaseAgent import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, env):
        super().__init__(env)
        self.actions = [SnakeLogic.UP, SnakeLogic.DOWN, SnakeLogic.LEFT, SnakeLogic.RIGHT]

    def extract_state(self):
        pass
    
    def act(self):
        return random.choice(self.actions), False