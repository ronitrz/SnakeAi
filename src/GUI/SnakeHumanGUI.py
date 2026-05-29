import pygame
from Snake.SnakeLogic import SnakeLogic
from .SnakeBaseGUI import SnakeBaseGUI


class SnakeHumanGUI(SnakeBaseGUI):
    def __init__(
        self,
        game: SnakeLogic,
        recorder=None,
        window_w: int = 600,
        window_h: int = 600,
        fps: int = 8,
    ):
        super().__init__(game, recorder=recorder, window_w=window_w, window_h=window_h, fps=fps)
        pygame.display.set_caption("Snake (Human)")

    def _handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            self._handle_resize_event(event)

            if event.type == pygame.KEYDOWN:
                if self.state == self.PLAYING:
                    if event.key == pygame.K_UP:
                        self.action = self.game.UP
                    elif event.key == pygame.K_DOWN:
                        self.action = self.game.DOWN
                    elif event.key == pygame.K_LEFT:
                        self.action = self.game.LEFT
                    elif event.key == pygame.K_RIGHT:
                        self.action = self.game.RIGHT
                    elif event.key == pygame.K_SPACE:
                        self.state = self.PAUSED

                elif self.state == self.MENU:
                    if event.key == pygame.K_RETURN:
                        self._restart()
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False

                elif self.state == self.PAUSED:
                    if event.key == pygame.K_SPACE:
                        self.state = self.PLAYING
                    elif event.key == pygame.K_r:
                        self._restart()
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False

                elif self.state == self.GAME_OVER:
                    if event.key == pygame.K_RETURN:
                        self._restart()
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False

    def run_one_frame(self):
        self.clock.tick(self.fps)
        self._handle_input()

        if self.state == self.PLAYING:
            board    = self.game.get_board()
            curr_dir = self.game.direction

            reward, done = self.game.step(self.action)
            self._record_step(board, self.action, curr_dir, done)

            if done:
                self.state = self.GAME_OVER

            self._draw_game()

        elif self.state == self.MENU:
            self._draw_menu()
        elif self.state == self.PAUSED:
            self._draw_paused()
        elif self.state == self.GAME_OVER:
            self._draw_game_over()

        pygame.display.flip()