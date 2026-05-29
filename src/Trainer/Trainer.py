import os, csv
from datetime import datetime


class Trainer:
    def __init__(
        self,
        env,
        agent,
        stats_dir="Stats",
        stats_filename=None,
        headless=False,
        verbose=False,
    ):
        self.env = env
        self.agent = agent
        self.headless = headless
        self.verbose = verbose

        self.episodes = 0
        self.total_steps = 0
        self.running = True

        self._episode_reward = 0.0
        self._episode_steps = 0

        os.makedirs(stats_dir, exist_ok=True)
        filename = stats_filename or f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._csv_path = os.path.join(stats_dir, filename)
        self._csv_file = None
        self._csv_writer = None

        self._init_csv()

    def _init_csv(self):
        resuming = os.path.exists(self._csv_path)

        if resuming:
            try:
                with open(self._csv_path, "r") as f:
                    rows = list(csv.reader(f))
                if len(rows) > 1:
                    self.episodes = int(rows[-1][0]) + 1
            except (ValueError, IndexError):
                resuming = False

        mode = "a" if resuming else "w"
        self._csv_file = open(self._csv_path, mode, newline="")
        self._csv_writer = csv.writer(self._csv_file)

        if not resuming:
            self._csv_writer.writerow(["episode", "steps", "score", "reward"])

    def _log_episode(self, score):
        self._csv_writer.writerow([
            self.episodes,
            self._episode_steps,
            score,
            round(self._episode_reward, 4),
        ])
        self._csv_file.flush()

    def step(self):
        action, relative = self.agent.act()
        reward, done = self.env.step(action, relative=relative)
        self.agent.observe(reward, done)

        self._episode_reward += reward
        self._episode_steps += 1
        self.total_steps += 1

        if done:
            self._on_episode_end()

        return reward, done

    def _on_episode_end(self):
        self._log_episode(self.env.score)

        if self.verbose:
            print(
                f"Episode {self.episodes:>6} | "
                f"Score: {self.env.score:>4} | "
                f"Steps: {self._episode_steps:>6} | "
                f"Reward: {self._episode_reward:>8.2f}"
            )

        self.episodes += 1
        self._episode_reward = 0.0
        self._episode_steps = 0
        self.env.reset()

        if hasattr(self.agent, "training_finished") and self.agent.training_finished:
            self.running = False

    def run(self):
        if not self.headless:
            raise RuntimeError(
                "Trainer.run() is for headless mode only. "
                "Pass the Trainer to SnakeAIGUI or SnakeAICLI for visual runs."
            )

        try:
            while self.running:
                self.step()
        except KeyboardInterrupt:
            print("\nInterrupted. Saving checkpoint...")
            if hasattr(self.agent, "save_checkpoint"):
                self.agent.save_checkpoint()
        finally:
            self.close()

    def close(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            print(f"[Trainer] Stats saved -> {self._csv_path}")