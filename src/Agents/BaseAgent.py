import numpy as np


class BaseAgent:
    def __init__(self, env):
        self.env = env

    def extract_state(self):
        """Extract the state representation from the environment.
        Must return a numpy array. Every agent must implement this."""
        raise NotImplementedError

    def act(self):
        """Return an action for the current state."""
        raise NotImplementedError

    def observe(self, reward, done):
        """Called after every step(). Override to handle transitions."""
        pass

    def save_checkpoint(self):
        """Save agent state to disk. Override in agents that support it."""
        pass

    def load_checkpoint(self):
        """Load agent state from disk. Override in agents that support it."""
        pass
    