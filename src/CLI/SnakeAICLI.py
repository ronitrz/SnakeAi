import time
import keyboard


class SnakeAICLI:
    PLAYING = 0
    PAUSED  = 1

    def __init__(self, trainer, verbose: bool = True):
        self.trainer = trainer
        self.verbose = verbose
        self.state   = self.PLAYING

    def _handle_keyboard(self):
        if keyboard.is_pressed("space"):
            time.sleep(0.3)
            if self.state == self.PLAYING:
                print("======== Paused ========")
                self.state = self.PAUSED
            else:
                print("======== Resumed ========")
                self.state = self.PLAYING

        if keyboard.is_pressed("esc"):
            time.sleep(0.3)
            print("======== Exiting ========")
            self.trainer.running = False

    def run(self):
        try:
            while self.trainer.running:
                self._handle_keyboard()

                if self.state == self.PLAYING:
                    reward, done = self.trainer.step()

                    if self.verbose and done:
                        print(
                            f"Episode {self.trainer.episodes:>6} | "
                            f"Score: {self.trainer.game.score:>4} | "
                            f"Steps: {self.trainer._episode_steps:>6}"
                        )

        except KeyboardInterrupt:
            print("\nInterrupted. Saving checkpoint...")
            self.trainer.agent.save_checkpoint()
        finally:
            self.trainer.close()
            