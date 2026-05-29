import pygame
from Snake.SnakeLogic import SnakeLogic
from .SnakeBaseGUI import SnakeBaseGUI


class SnakeAIGUI(SnakeBaseGUI):
    def __init__(
        self,
        game: SnakeLogic,
        trainer,
        recorder=None,
        window_w: int = 600,
        window_h: int = 600,
        fps: int = 24,
    ):
        super().__init__(game, recorder=recorder, window_w=window_w, window_h=window_h, fps=fps)
        pygame.display.set_caption("Snake (AI)")

        self.trainer = trainer

    def _draw_hud(self):
        lines = [
            f"Score:   {self.game.score}",
            f"Episode: {self.trainer.episodes}",
            f"Steps:   {self.trainer.total_steps}",
        ]

        if hasattr(self.trainer.agent, "epsilon"):
            lines.append(f"Epsilon: {self.trainer.agent.epsilon:.4f}")

        for i, text in enumerate(lines):
            surf = self.font.render(text, True, self.TEXT_COLOR)
            self.screen.blit(surf, (10, 10 + i * (self.font.get_height() + 4)))


    def _handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            self._handle_resize_event(event)

            if event.type == pygame.KEYDOWN:
                if self.state == self.PLAYING:
                    if event.key == pygame.K_SPACE:
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

            reward, done = self.trainer.step()
            self._record_step(board, self.game.last_action, curr_dir, done)

            if done:
                if not self.trainer.running:
                    self.running = False
                else:
                    self.state = self.PLAYING

            self._draw_game()

        elif self.state == self.MENU:
            self._draw_menu()
        elif self.state == self.PAUSED:
            self._draw_paused()
        elif self.state == self.GAME_OVER:
            self._draw_game_over()

        pygame.display.flip()