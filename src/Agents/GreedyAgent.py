from Snake.SnakeLogic import SnakeLogic
from .BaseAgent import BaseAgent

class GreedyAgent(BaseAgent):
    def __init__(self, env: SnakeLogic):
        super().__init__(env)
        self.actions = self.env.get_actions()

    def extract_state(self):
        pass
    
    def _manhattan(self, node, food):
        x1 = node % self.env.width
        y1 = node // self.env.width
        x2 = food % self.env.width
        y2 = food // self.env.width
        return abs(x1 - x2) + abs(y1 - y2)

    def act(self):
        head = self.env.snake[0]
        food = self.env.food

        best_action = self.env.direction
        best_score = float("-inf")

        hx = head % self.env.width
        hy = head // self.env.width

        body = set(self.env.snake)

        for action in self.actions:
            if action == SnakeLogic.UP:
                nx, ny = hx, hy - 1
            elif action == SnakeLogic.DOWN:
                nx, ny = hx, hy + 1
            elif action == SnakeLogic.LEFT:
                nx, ny = hx - 1, hy
            elif action == SnakeLogic.RIGHT:
                nx, ny = hx + 1, hy

            # strict boundary check
            if not (0 <= nx < self.env.width and 0 <= ny < self.env.height):
                continue

            new_node = ny * self.env.width + nx

            # body collision check
            if new_node in body:
                continue

            score = -self._manhattan(new_node, food)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action, False
