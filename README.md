# 🐍 Snake AI: Reinforcement Learning & Pathfinding Arena

An interactive, modular python platform designed to train, evaluate, and visualize pathfinding algorithms and Reinforcement Learning (RL) agents playing the classic game of Snake.

Developed with **Pygame** for beautiful interactive rendering, **PyTorch** for deep reinforcement learning architectures, and structured for research and comparison.

---

## 🚀 Features

*   🎮 **Multiple Game Modes**: Play as a human, watch an AI agent play in real-time, or run high-speed headless training.
*   🧠 **Advanced AI Agents**:
    *   **A\***: Optimal pathfinding search to compute the safest, shortest path to food.
    *   **Greedy**: A heuristic-driven agent that prioritizes immediate proximity to food.
    *   **Deep Q-Network (DQN)**: Reinforcement learning agent using fully connected layers.
    *   **CNN-DQN**: Grid-state representation parsed through a Convolutional Neural Network (CNN).
    *   **CNN-DDQN**: Double DQN utilizing a CNN architecture to minimize Q-value overestimation.
    *   **Random**: A basic baseline agent selecting random valid moves.
*   🎥 **Record & Replay**: Seamlessly record gameplay sessions (human or AI) to compressed binary files and replay them step-by-step.
*   📈 **Metrics & Training Logs**: Automatically log training steps, cumulative rewards, episode scores, and steps taken to a CSV file for analytical plotting.

---

## 📂 Codebase Architecture

```text
snake-ai/
├── src/
│   ├── Agents/            # AI Agent implementations (A*, DQN, Greedy, PPO, etc.)
│   ├── CLI/               # Command-line interface logic
│   ├── GUI/               # Pygame interface for human play and AI visualizer
│   ├── Hyperparameters/   # YAML configs for neural network agents
│   ├── Replay/            # Compressed game recorders and replay parsers
│   ├── Snake/             # Core Snake game rules and state transitions
│   ├── Trainer/           # High-performance training loops
│   └── main.py            # Unified application entry point
├── requirements.txt       # Project dependencies
└── LICENSE                # Repository License
```

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ronitrz/SnakeAi.git
cd SnakeAi
```

### 2. Set Up a Virtual Environment (Recommended)
On **Windows**:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

On **macOS/Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎮 How to Play & Run

The application uses a unified CLI runner. Here is how to run the game in different modes:

### 1. Play as a Human (GUI)
Control the snake using the arrow keys:
```bash
python src/main.py -hu
```
*Add `-r` to record your gameplay!*

### 2. Watch an AI Agent Play (GUI)
Visualize any of the registered agents playing the game in real-time:
```bash
# Watch the A* Pathfinding agent
python src/main.py -ai gui -a astar

# Watch the CNN-DDQN reinforcement learning agent
python src/main.py -ai gui -a cnn-ddqn
```
*Supported agents (`-a`): `random`, `greedy`, `astar`, `dqn`, `cnn-dqn`, `cnn-ddqn`*

### 3. Train an AI Agent (Headless Mode)
Train RL agents at maximum simulation speed without rendering graphics:
```bash
python src/main.py --headless -a cnn-ddqn -v
```
*Options:*
*   `-v`: Verbose training updates printed to the console.
*   `--stats-path <path>`: Customize where CSV training logs are saved (default: `../stats/stats.csv`).

### 4. Record & Replay Gameplay
To record any GUI session:
```bash
python src/main.py -ai gui -a astar -r --record-path Recordings/best_astar.bin.gz
```

To replay the recorded session through the CLI:
```bash
python src/main.py --replay Recordings/best_astar.bin.gz -s 15
```
*`-s 15` sets the speed/FPS of the replay visualization.*

---

## ⚙️ Command-Line Arguments Guide

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-hu` | Flag | `False` | Play as a human (GUI mode) |
| `-ai` | Choice | `None` | Run AI agent with rendering (`gui`) |
| `--headless` | Flag | `False` | Train an AI agent without rendering |
| `--replay` | String | `None` | Path to a recording file to replay |
| `-a` | Choice | `random` | Agent type (`random`, `greedy`, `astar`, `dqn`, `cnn-dqn`, `cnn-ddqn`) |
| `-W` | Int | `20` | Game board width (grid units) |
| `-H` | Int | `20` | Game board height (grid units) |
| `-s` | Int | `8` | Game speed / updates per second |
| `-ww` | Int | `600` | Pygame window width (pixels) |
| `-wh` | Int | `600` | Pygame window height (pixels) |
| `-r` | Flag | `False` | Enable recording of gameplay |
| `--record-path` | String | `None` | Custom path to save the binary recording |
| `--stats-path` | String | `../stats/stats.csv` | Path to save CSV training metrics |

---

## 🧠 Core Agent Implementations

### 1. A* Pathfinding Agent (`src/Agents/AstarAgent.py`)
Computes the shortest path to food at each step using a Manhattan Distance heuristic. If a safe path is found, the agent takes it. If no path is found (e.g., trapped), it gracefully falls back to survival moves to maximize steps survived.

### 2. Deep Q-Networks (`src/Agents/DQNAgent.py`, `CNNDQNAgent.py`, `CNNDDQNAgent.py`)
*   **State Space**: Custom spatial grid representation highlighting the snake body, head, wall boundary, and food position.
*   **Action Space**: Discrete actions: `0` (continue straight), `1` (turn right), `2` (turn left) relative to the current direction.
*   **Experience Replay**: Stores transitions to break correlation in sequential observations.
*   **Target Network Updates**: Periodically copies weights to stabilize Q-target calculations.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
