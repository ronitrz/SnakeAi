import random
from collections import deque


class SnakeLogic:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    STRAIGHT = 0
    LEFT_TURN = 1
    RIGHT_TURN = 2

    _DIRECTIONS = {
        UP: (0, -1),
        DOWN: (0, 1),
        LEFT: (-1, 0),
        RIGHT: (1, 0),
    }

    _REL_TO_ABS = {
        UP: {STRAIGHT: UP, LEFT_TURN: LEFT, RIGHT_TURN: RIGHT},
        DOWN: {STRAIGHT: DOWN, LEFT_TURN: RIGHT, RIGHT_TURN: LEFT},
        LEFT: {STRAIGHT: LEFT, LEFT_TURN: DOWN, RIGHT_TURN: UP},
        RIGHT: {STRAIGHT: RIGHT, LEFT_TURN: UP, RIGHT_TURN: DOWN},
    }

    _REWARD_DEATH = -1
    _REWARD_PER_STEP = -0.01
    _REWARD_STARVATION = -1
    _REWARD_EAT = 10

    def __init__(self, width, height, starvation_timeout=None):
        self.width = width
        self.height = height
        self.size = width * height

        self.starvation_timeout = starvation_timeout

        self.opposite = {
            self.UP: self.DOWN,
            self.DOWN: self.UP,
            self.LEFT: self.RIGHT,
            self.RIGHT: self.LEFT,
        }

        self.reset()

    def reset(self):
        self.free_cells = list(range(self.size))
        self.free_index = {c: i for i, c in enumerate(self.free_cells)}

        cx = random.randint(0, self.width - 1)
        cy = random.randint(0, self.height - 1)
        start = cy * self.width + cx

        self.snake = deque([start])
        self._remove_free(start)

        self.direction = random.choice([self.UP, self.DOWN, self.LEFT, self.RIGHT])
        self.food = random.choice(self.free_cells)

        self.done = False
        self.score = 0

        self.steps_alive = 0
        self.steps_since_food = 0
        self.last_action = self.direction
        self.direction_changes = 0

    def _remove_free(self, cell):
        index = self.free_index[cell]
        last = self.free_cells[-1]

        self.free_cells[index] = last
        self.free_index[last] = index
        self.free_cells.pop()
        del self.free_index[cell]

    def _add_free(self, cell):
        self.free_index[cell] = len(self.free_cells)
        self.free_cells.append(cell)

    def index_to_cell(self, index):
        x = index % self.width
        y = index // self.width
        return (x, y)

    def cell_to_index(self, x, y):
        return y * self.width + x

    def relative_to_absolute(self, action):
        return self._REL_TO_ABS[self.direction][action]

    def step(self, action, relative=False):
        if self.done:
            return self._REWARD_DEATH, self.done

        head = self.snake[0]
        hx, hy = self.index_to_cell(head)

        if relative:
            action = self.relative_to_absolute(action)

        if action == self.opposite[self.direction]:
            action = self.direction
        elif action != self.direction:
            self.direction_changes += 1
            self.direction = action

        self.last_action = action
        self.steps_alive += 1
        self.steps_since_food += 1

        if (
            self.starvation_timeout != None
            and self.steps_since_food > self.starvation_timeout
        ):
            self.done = True
            return self._REWARD_STARVATION, self.done

        dx, dy = self._DIRECTIONS[self.direction]
        x, y = hx + dx, hy + dy

        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            self.done = True
            return self._REWARD_DEATH, self.done

        new_head = self.cell_to_index(x, y)

        if new_head not in self.free_index:
            self.done = True
            return self._REWARD_DEATH, self.done

        self.snake.appendleft(new_head)
        self._remove_free(new_head)

        reward = self._REWARD_PER_STEP

        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0

            reward = self._REWARD_EAT

            if self.free_cells:
                self.food = random.choice(self.free_cells)
            else:
                self.done = True
        else:
            tail = self.snake.pop()
            self._add_free(tail)

        return reward, self.done

    def get_actions(self, relative=False):
        if relative:
            return [self.STRAIGHT, self.LEFT_TURN, self.RIGHT_TURN]
        return [self.UP, self.DOWN, self.LEFT, self.RIGHT]

    def get_board(self):
        board = [[0] * self.width for _ in range(self.height)]
        board[self.food // self.width][self.food % self.width] = 1
        for i, cell in enumerate(self.snake):
            board[cell // self.width][cell % self.width] = i + 2

        return board
