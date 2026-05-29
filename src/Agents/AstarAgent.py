from queue import PriorityQueue
from .BaseAgent import BaseAgent
from Snake.SnakeLogic import SnakeLogic


class AstarAgent(BaseAgent):
    def __init__(self, env: SnakeLogic):
        super().__init__(env)
        self.actions = self.env.get_actions()
    
    def extract_state(self):
        pass
    
    def _heuristic(self, node, goal):
        x1 = node % self.env.width
        y1 = node // self.env.width
        x2 = goal % self.env.width
        y2 = goal // self.env.width
        return abs(x1 - x2) + abs(y1 - y2)

    def _neighbors(self, node):
        x = node % self.env.width
        y = node // self.env.width

        for action in self.actions:
            if action == SnakeLogic.UP:
                nx, ny = x, y - 1
            elif action == SnakeLogic.DOWN:
                nx, ny = x, y + 1
            elif action == SnakeLogic.LEFT:
                nx, ny = x - 1, y
            elif action == SnakeLogic.RIGHT:
                nx, ny = x + 1, y

            if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                yield action, ny * self.env.width + nx

    def _path_to_food(self):
        start = self.env.snake[0]
        goal = self.env.food

        open_set = PriorityQueue()
        open_set.put((0, start, []))

        closed = set()
        body = set(self.env.snake)

        while not open_set.empty():
            f, node, path = open_set.get()

            if node == goal:
                return path

            if node in closed:
                continue

            closed.add(node)

            for action, next_node in self._neighbors(node):
                if next_node in body:
                    continue

                new_path = path + [action]
                g = len(new_path)
                h = self._heuristic(next_node, goal)
                open_set.put((g + h, next_node, new_path))

        return None

    def act(self):
        path = self._path_to_food()
        if path:
            return path[0], False
        return self.env.direction, False
