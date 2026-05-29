import pygame
from Snake.SnakeLogic import SnakeLogic


class SnakeBaseGUI:
    # ---------- COLORS ----------
    BG_COLOR = (12, 12, 18)
    TEXT_COLOR = (230, 230, 240)
    SNAKE_HEAD = (0, 255, 170)
    SNAKE_BODY = (0, 200, 140)
    FOOD_COLOR = (255, 80, 80)
    BOARD_BG = (22, 22, 32)
    UI_DARK = (45, 45, 60)
    MENU_PANEL_BG = (28, 28, 40)

    # ---------- FONT CONFIG ----------
    MIN_FONT_SIZE = 12
    BASE_FONT_SCALE = 28
    BIG_FONT_MULT = 2.0
    SMALL_FONT_MULT = 0.75

    # ---------- SNAKE DRAW CONFIG ----------
    SEGMENT_SCALE = 0.95
    MIN_SEGMENT_PX = 4

    # ---------- UI CONFIG ----------
    HUD_MARGIN_X = 10
    HUD_MARGIN_Y = 10

    TITLE_OFFSET_DIV = 8
    MENU_SPACING_DIV = 16

    OVERLAY_ALPHA = 140

    # ---------- WINDOW ----------
    MIN_WINDOW_W = 400
    MIN_WINDOW_H = 400

    MENU = 0
    PLAYING = 1
    PAUSED = 2
    GAME_OVER = 3

    _DIR_OFFSET = {
        SnakeLogic.UP: (0, -1),
        SnakeLogic.DOWN: (0, 1),
        SnakeLogic.LEFT: (-1, 0),
        SnakeLogic.RIGHT: (1, 0),
    }

    def __init__(
        self,
        game: SnakeLogic,
        recorder=None,
        window_w: int = 600,
        window_h: int = 600,
        fps: int = 8,
    ):
        self.game = game
        self.recorder = recorder
        self.fps = fps

        self.window_w = max(window_w, self.MIN_WINDOW_W)
        self.window_h = max(window_h, self.MIN_WINDOW_H)

        self.cell_size = min(
            self.window_w // self.game.width, self.window_h // self.game.height
        )

        self.cell_w = self.cell_size
        self.cell_h = self.cell_size

        self.board_w = self.cell_w * self.game.width
        self.board_h = self.cell_h * self.game.height

        self.offset_x = (self.window_w - self.board_w) // 2
        self.offset_y = (self.window_h - self.board_h) // 2

        pygame.init()

        self.screen = pygame.display.set_mode(
            (self.window_w, self.window_h), pygame.RESIZABLE
        )

        pygame.display.set_caption("Snake")

        self.clock = pygame.time.Clock()

        self._rebuild_fonts()

        self.state = self.MENU
        self.action = game.direction
        self.running = True

    def _rebuild_fonts(self):

        base = max(self.MIN_FONT_SIZE, self.window_h // self.BASE_FONT_SCALE)

        self.font = pygame.font.SysFont(None, base)
        self.big_font = pygame.font.SysFont(None, int(base * self.BIG_FONT_MULT))

        small_size = int(base * self.SMALL_FONT_MULT)
        self.small_font = pygame.font.SysFont(None, small_size)

    def _on_resize(self, new_w, new_h):

        self.window_w = max(new_w, self.MIN_WINDOW_W)
        self.window_h = max(new_h, self.MIN_WINDOW_H)

        self.cell_size = min(
            self.window_w // self.game.width, self.window_h // self.game.height
        )

        self.cell_w = self.cell_size
        self.cell_h = self.cell_size

        self.board_w = self.cell_w * self.game.width
        self.board_h = self.cell_h * self.game.height

        self.offset_x = (self.window_w - self.board_w) // 2
        self.offset_y = (self.window_h - self.board_h) // 2

        self._rebuild_fonts()

    def _get_connections(self, index):

        snake = self.game.snake
        cell = snake[index]

        cx, cy = cell % self.game.width, cell // self.game.width

        connections = set()

        for neighbor_idx in (index - 1, index + 1):

            if 0 <= neighbor_idx < len(snake):

                nc = snake[neighbor_idx]

                nx, ny = nc % self.game.width, nc // self.game.width

                dx, dy = nx - cx, ny - cy

                for d, (ddx, ddy) in self._DIR_OFFSET.items():

                    if ddx == dx and ddy == dy:

                        connections.add(d)
                        break

        return connections

    def _draw_segment(self, cell, connections, color):

        cw, ch = self.cell_w, self.cell_h

        cx = self.offset_x + (cell % self.game.width) * cw
        cy = self.offset_y + (cell // self.game.width) * ch

        tw = max(self.MIN_SEGMENT_PX, int(cw * self.SEGMENT_SCALE))
        th = max(self.MIN_SEGMENT_PX, int(ch * self.SEGMENT_SCALE))

        ox = cx + (cw - tw) // 2
        oy = cy + (ch - th) // 2

        pygame.draw.rect(self.screen, color, pygame.Rect(ox, oy, tw, th))

        if SnakeLogic.UP in connections:
            pygame.draw.rect(self.screen, color, pygame.Rect(ox, cy, tw, oy - cy + th))

        if SnakeLogic.DOWN in connections:
            pygame.draw.rect(
                self.screen, color, pygame.Rect(ox, oy, tw, (cy + ch) - oy)
            )

        if SnakeLogic.LEFT in connections:
            pygame.draw.rect(self.screen, color, pygame.Rect(cx, oy, ox - cx + tw, th))

        if SnakeLogic.RIGHT in connections:
            pygame.draw.rect(
                self.screen, color, pygame.Rect(ox, oy, (cx + cw) - ox, th)
            )

    def _draw_text(self, text, y, big=False, color=None):

        font = self.big_font if big else self.font
        color = color or self.TEXT_COLOR

        surface = font.render(text, True, color)

        rect = surface.get_rect(center=(self.window_w // 2, y))

        self.screen.blit(surface, rect)

    def _draw_game(self):

        self.screen.fill(self.BG_COLOR)

        pygame.draw.rect(
            self.screen,
            self.BOARD_BG,
            pygame.Rect(self.offset_x, self.offset_y, self.board_w, self.board_h),
        )

        cw, ch = self.cell_w, self.cell_h

        fc = self.game.food

        pygame.draw.rect(
            self.screen,
            self.FOOD_COLOR,
            pygame.Rect(
                self.offset_x + (fc % self.game.width) * cw,
                self.offset_y + (fc // self.game.width) * ch,
                cw,
                ch,
            ),
        )

        for i, cell in enumerate(self.game.snake):

            connections = self._get_connections(i)

            if i == 0:
                self._draw_segment(cell, connections, self.SNAKE_HEAD)
            else:
                self._draw_segment(cell, connections, self.SNAKE_BODY)

        self._draw_hud()

    def _draw_hud(self):

        surf = self.font.render(f"Score: {self.game.score}", True, self.TEXT_COLOR)

        self.screen.blit(surf, (self.HUD_MARGIN_X, self.HUD_MARGIN_Y))

    def _draw_menu(self):

        mid_y = self.window_h // 2

        self.screen.fill(self.MENU_PANEL_BG)

        self._draw_text(
            "SNAKE", mid_y - self.window_h // self.TITLE_OFFSET_DIV, big=True
        )

        self._draw_text("ENTER - Start", mid_y)

        self._draw_text("ESC   - Exit", mid_y + self.window_h // self.MENU_SPACING_DIV)

    def _draw_paused(self):

        mid_y = self.window_h // 2

        self._draw_game()

        overlay = pygame.Surface((self.window_w, self.window_h), pygame.SRCALPHA)

        overlay.fill((0, 0, 0, self.OVERLAY_ALPHA))

        self.screen.blit(overlay, (0, 0))

        self._draw_text(
            "PAUSED", mid_y - self.window_h // self.TITLE_OFFSET_DIV, big=True
        )

        self._draw_text("SPACE - Resume", mid_y)

        self._draw_text(
            "R     - Restart", mid_y + self.window_h // self.MENU_SPACING_DIV
        )

        self._draw_text("ESC   - Exit", mid_y + self.window_h // self.TITLE_OFFSET_DIV)

    def _draw_game_over(self):

        mid_y = self.window_h // 2

        self._draw_game()

        overlay = pygame.Surface((self.window_w, self.window_h), pygame.SRCALPHA)

        overlay.fill((0, 0, 0, self.OVERLAY_ALPHA))

        self.screen.blit(overlay, (0, 0))

        self._draw_text(
            "GAME OVER", mid_y - self.window_h // self.TITLE_OFFSET_DIV, big=True
        )

        self._draw_text(f"Score: {self.game.score}", mid_y)

        self._draw_text(
            "ENTER - Play Again", mid_y + self.window_h // self.MENU_SPACING_DIV
        )

        self._draw_text("ESC   - Exit", mid_y + self.window_h // self.TITLE_OFFSET_DIV)

    def _record_step(self, board, action, curr_dir, done):

        if self.recorder:

            self.recorder.record_step(
                board=board,
                action=action,
                curr_dir=curr_dir,
                done=done,
            )

    def _start_episode(self):

        if self.recorder:
            self.recorder.start_episode()

    def _restart(self):

        self.game.reset()

        self._start_episode()

        self.action = self.game.direction

        self.state = self.PLAYING

    def _handle_resize_event(self, event):

        if event.type == pygame.VIDEORESIZE:

            self._on_resize(event.w, event.h)

    def run(self):

        while self.running:

            self.run_one_frame()

        if self.recorder:
            self.recorder.close()

        pygame.quit()

    def run_one_frame(self):
        raise NotImplementedError
