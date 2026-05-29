from .BaseAgent import BaseAgent
from .RandomAgent import RandomAgent
from .GreedyAgent import GreedyAgent
from .AstarAgent import AstarAgent

# Lazy imports for agents that require torch
def __getattr__(name):
    if name == "DQNAgent":
        from .DQNAgent import DQNAgent
        return DQNAgent
    elif name == "CNNDQNAgent":
        from .CNNDQNAgent import CNNDQNAgent
        return CNNDQNAgent
    elif name == "CNNDDQNAgent":
        from .CNNDDQNAgent import CNNDDQNAgent
        return CNNDDQNAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseAgent",
    "DQNAgent",
    "CNNDQNAgent",
    "CNNDDQNAgent",
    "RandomAgent",
    "GreedyAgent",
    "AstarAgent",
]