import torch
import numpy as np
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt
import os, random, yaml, logging

from datetime import datetime
from collections import deque
from .BaseAgent import BaseAgent
from Snake.SnakeLogic import SnakeLogic


class ReplayMemory:
    """Experience replay buffer for storing and sampling training transitions."""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a random batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def is_full(self):
        """Check if memory is at capacity."""
        return len(self.memory) == self.capacity


class CNNNetwork(nn.Module):
    """CNN architecture - configurable from YAML."""
    
    def __init__(self, board_size, conv_filters, kernel_sizes, fc_layers, output_size, dropout_rate=0.0):
        super().__init__()
        
        # conv_filters: [32, 64, 128] - filters per layer
        # kernel_sizes: [3, 3, 3] - kernel size per layer
        # fc_layers: [128, 64] - fully connected layer sizes
        
        self.board_size = board_size
        conv_layers = []
        in_channels = 4
        
        # Build conv layers dynamically
        for out_channels, kernel_size in zip(conv_filters, kernel_sizes):
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2))
            conv_layers.append(nn.ReLU())
            if dropout_rate > 0:
                conv_layers.append(nn.Dropout2d(dropout_rate))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate flattened size after conv layers
        # Board size stays same due to padding=kernel_size//2
        flattened_size = in_channels * board_size * board_size
        
        # Build FC layers dynamically
        fc_layer_list = []
        prev_size = flattened_size
        
        for fc_size in fc_layers:
            fc_layer_list.append(nn.Linear(prev_size, fc_size))
            fc_layer_list.append(nn.ReLU())
            if dropout_rate > 0:
                fc_layer_list.append(nn.Dropout(dropout_rate))
            prev_size = fc_size
        
        fc_layer_list.append(nn.Linear(prev_size, output_size))
        self.fc_layers = nn.Sequential(*fc_layer_list)
    
    def forward(self, x):
        # x shape: (batch_size, 4, board_size, board_size)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


class CNNDQNAgent(BaseAgent):
    """CNN-based Deep Q-Network agent for Snake game."""
    
    def __init__(self, env: SnakeLogic):
        super().__init__(env)

        # ============================================================================
        # SYSTEM OPTIMIZATION
        # ============================================================================
        torch.set_num_threads(os.cpu_count())
        os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
        os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
        torch.set_float32_matmul_precision('high')

        # ============================================================================
        # LOAD CONFIGURATION
        # ============================================================================
        with open("Hyperparameters/CNNDQN.yaml") as f:
            self.params = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ============================================================================
        # MODE CONFIGURATION
        # ============================================================================
        mode = self.params["mode"]
        self.training_mode = mode["training"]
        self.should_load_checkpoint = mode.get("load_checkpoint", True)
        self.model_path = mode["model_path"]
        self.logging_enabled = mode.get("logging", False)
        self.logs_dir = mode.get("logs_dir", "logs")

        # ============================================================================
        # ENVIRONMENT CONFIGURATION
        # ============================================================================
        env_cfg = self.params["environment"]
        self.board_size = env_cfg["board_size"]  # 5x5, 4x4, etc
        self.action_size = env_cfg["action_dim"]

        # Set starvation timeout if specified
        starvation = env_cfg.get("starvation_steps", None)
        if starvation is not None:
            self.env.starvation_timeout = starvation

        # ============================================================================
        # DQN HYPERPARAMETERS
        # ============================================================================
        dqn_cfg = self.params["dqn"]
        self.gamma = dqn_cfg["gamma"]
        self.batch_size = dqn_cfg["batch_size"]
        self.epsilon = dqn_cfg["epsilon_start"]
        self.epsilon_min = dqn_cfg["epsilon_min"]
        self.epsilon_decay = dqn_cfg["epsilon_decay"]
        self.target_update_freq = dqn_cfg["target_update_frequency"]
        self.learning_rate = dqn_cfg["learning_rate"]

        # ============================================================================
        # TRAINING CONFIGURATION
        # ============================================================================
        trn_cfg = self.params["training"]
        self.num_episodes = trn_cfg["num_episodes"]
        self.save_best_only = trn_cfg["save_best_only"]
        self.checkpoint_interval = trn_cfg["checkpoint_interval"]

        # ============================================================================
        # INITIALIZE REPLAY MEMORY
        # ============================================================================
        mem_cfg = self.params["replay_memory"]
        self.memory = ReplayMemory(mem_cfg["capacity"])

        # ============================================================================
        # BUILD CNN NETWORKS
        # ============================================================================
        net_cfg = self.params["network"]
        conv_filters = net_cfg["conv_filters"]      # [32, 64]
        kernel_sizes = net_cfg["kernel_sizes"]      # [3, 3]
        fc_layers = net_cfg["fc_layers"]            # [128]
        dropout = net_cfg.get("dropout", 0.0)

        self.policy_net = CNNNetwork(
            board_size=self.board_size,
            conv_filters=conv_filters,
            kernel_sizes=kernel_sizes,
            fc_layers=fc_layers,
            output_size=self.action_size,
            dropout_rate=dropout
        ).to(self.device)
        
        self.target_net = CNNNetwork(
            board_size=self.board_size,
            conv_filters=conv_filters,
            kernel_sizes=kernel_sizes,
            fc_layers=fc_layers,
            output_size=self.action_size,
            dropout_rate=dropout
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # ============================================================================
        # OPTIMIZER AND LOSS
        # ============================================================================
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # ============================================================================
        # TRAINING STATE
        # ============================================================================
        self.step_counter = 0
        self.training_finished = False

        # ============================================================================
        # EPISODE STATISTICS
        # ============================================================================
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        self.previous_episode_reward = None

        # ============================================================================
        # HISTORY TRACKING
        # ============================================================================
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_steps = []
        self.epsilon_history = []
        self.loss_history = []
        self.q_value_history = []
        self.best_reward = -float("inf")
        self.last_avg_q = None

        # ============================================================================
        # STATE AND ACTION
        # ============================================================================
        self.state = None
        self.last_action = None

        # ============================================================================
        # SETUP LOGGING
        # ============================================================================
        self._setup_logging()

        # ============================================================================
        # LOAD CHECKPOINT OR MODEL
        # ============================================================================
        if self.should_load_checkpoint and self.training_mode:
            self.load_checkpoint()
        elif not self.training_mode and self.model_path:
            self.load_model(self.model_path)
            
    def _setup_logging(self):
        """Initialize logging if enabled."""
        if not (self.logging_enabled and self.training_mode):
            return

        os.makedirs(self.logs_dir, exist_ok=True)
        
        checkpoint_path = self.model_path.replace(".pt", "_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            log_files = [f for f in os.listdir(self.logs_dir) if f.startswith("run_")]
            if log_files:
                log_files.sort()
                log_file = os.path.join(self.logs_dir, log_files[-1])
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(self.logs_dir, f"run_{timestamp}.log")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.logs_dir, f"run_{timestamp}.log")

        self.logger = logging.getLogger(f"CNNDQNAgent_{os.path.basename(log_file)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        handler = logging.FileHandler(log_file, mode='a')
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.logger.info("===== CNN DQN TRAINING SESSION STARTED =====")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Board Size: {self.board_size}x{self.board_size}")
        self.logger.info(f"Conv Filters: {self.params['network']['conv_filters']}")
        self.logger.info(f"FC Layers: {self.params['network']['fc_layers']}")
        self.logger.info(f"Learning Rate: {self.learning_rate}")

    def extract_state(self):
        board = self.env.get_board()
        board_array = np.array(board, dtype=np.int32)
        
        num_categories = 4  # 0: empty, 1: food, 2: head, 3: body
        # One-hot: (num_categories, board_size, board_size)
        one_hot = (np.arange(num_categories)[:, None, None] == board_array[None]).astype(np.float32)
        
        return one_hot  # shape: (4, 5, 5)

    def _to_tensor(self, state):
        # state is already (4, board_size, board_size)
        # just add batch dim → (1, 4, board_size, board_size)
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def act(self):
        """Select an action using epsilon-greedy policy."""
        self.state = self._to_tensor(self.extract_state())

        if self.training_mode and random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                action = torch.argmax(self.policy_net(self.state)).item()

        self.last_action = action
        return action, True if self.action_size == 3 else False

    def observe(self, reward, done):
        """Process environment feedback and update agent."""
        next_state = self._to_tensor(self.extract_state())

        if self.training_mode:
            self.memory.push(self.state, self.last_action, reward, next_state, done)
            self.step_counter += 1

            if len(self.memory) >= self.batch_size:
                self.learn()
            
            if done:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if self.step_counter % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        self._update_episode_stats(reward, done)
        self.state = next_state

    def learn(self):
        """Update Q-network using a batch of experiences."""
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)
        self.loss_history.append(loss.item())
        self.last_avg_q = current_q.detach().mean().item()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def _update_episode_stats(self, reward, done):
        """Update training statistics at the end of each episode."""
        self.current_episode_reward += reward
        self.current_episode_steps += 1

        if not done:
            return

        self.episode_scores.append(self.env.score)
        self.episode_steps.append(self.current_episode_steps)

        if self.last_avg_q is not None:
            self.q_value_history.append(self.last_avg_q)
            self.last_avg_q = None

        self.episode_count += 1
        reward_delta = 0

        if self.previous_episode_reward is not None:
            reward_delta = self.current_episode_reward - self.previous_episode_reward

        self.previous_episode_reward = self.current_episode_reward
        self.episode_rewards.append(self.current_episode_reward)

        if self.current_episode_reward > self.best_reward:
            self.best_reward = self.current_episode_reward
            if self.training_mode and self.save_best_only:
                self.save_model(self.model_path)

        if self.logging_enabled and self.training_mode:
            self.logger.info(
                f"Episode: {self.episode_count} | "
                f"Reward: {self.current_episode_reward:.2f} | "
                f"Delta: {reward_delta:+.2f} | "
                f"Steps: {self.current_episode_steps} | "
                f"Epsilon: {self.epsilon:.4f} | "
                f"Best: {self.best_reward:.2f}"
            )

        if self.training_mode:
            if self.episode_count % self.checkpoint_interval == 0:
                self.save_checkpoint()
            self.epsilon_history.append(self.epsilon)

        self.current_episode_reward = 0
        self.current_episode_steps = 0

        if self.training_mode and len(self.episode_rewards) >= self.num_episodes:
            self.training_finished = True
            self.finalize_training()

    def save_model(self, path):
        """Save the policy network to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
        print(f"[Model Saved] -> {path}")

    def load_model(self, path):
        """Load a pretrained policy network."""
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"[Model Loaded] -> {path}")
        else:
            print(f"[Model Not Found] -> {path}")

    def save_checkpoint(self):
        """Save a training checkpoint with full state."""
        checkpoint_path = self.model_path.replace(".pt", "_checkpoint.pt")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        checkpoint = {
            "model_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "episode_rewards": self.episode_rewards,
            "episode_scores": self.episode_scores,
            "episode_steps": self.episode_steps,
            "epsilon_history": self.epsilon_history,
            "loss_history": self.loss_history,
            "q_value_history": self.q_value_history,
            "best_reward": self.best_reward,
            "step_counter": self.step_counter,
            "current_episode_reward": self.current_episode_reward,
            "current_episode_steps": self.current_episode_steps,
            "previous_episode_reward": self.previous_episode_reward,
            "last_avg_q": self.last_avg_q,
            "replay_memory": list(self.memory.memory),
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"[Checkpoint Saved] Episodes: {len(self.episode_rewards)}")
        
        if self.logging_enabled and hasattr(self, 'logger'):
            self.logger.info(f"Checkpoint saved at episode {len(self.episode_rewards)}")

    def load_checkpoint(self):
        """Restore training from a checkpoint."""
        checkpoint_path = self.model_path.replace(".pt", "_checkpoint.pt")

        if not os.path.exists(checkpoint_path):
            print("No checkpoint found. Starting fresh training.")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.epsilon = checkpoint["epsilon"]
        self.episode_count = checkpoint["episode_count"]
        self.episode_rewards = checkpoint["episode_rewards"]
        self.episode_scores = checkpoint.get("episode_scores", [])
        self.episode_steps = checkpoint.get("episode_steps", [])
        self.epsilon_history = checkpoint["epsilon_history"]
        self.loss_history = checkpoint.get("loss_history", [])
        self.q_value_history = checkpoint.get("q_value_history", [])
        self.best_reward = checkpoint["best_reward"]
        self.step_counter = checkpoint.get("step_counter", 0)
        self.current_episode_reward = checkpoint.get("current_episode_reward", 0)
        self.current_episode_steps = checkpoint.get("current_episode_steps", 0)
        self.previous_episode_reward = checkpoint.get("previous_episode_reward", None)
        self.last_avg_q = checkpoint.get("last_avg_q", None)
        
        replay_memory_data = checkpoint.get("replay_memory", [])
        self.memory.memory = deque(replay_memory_data, maxlen=self.memory.capacity)

        print(f"[Checkpoint Loaded] Resuming from episode {len(self.episode_rewards)}")
        
        if self.logging_enabled and hasattr(self, 'logger'):
            self.logger.info(f"Checkpoint loaded. Resuming from episode {len(self.episode_rewards)}")

    def finalize_training(self):
        """Complete training and generate plots."""
        if not self.save_best_only:
            self.save_model(self.model_path)
        print(f"Training Finished. Best Reward: {self.best_reward}")
        self.plot_training()

    def plot_training(self):
        """Generate clean training plots with proper formatting."""
        if not self.episode_rewards:
            return

        sns.set_style("whitegrid")
        os.makedirs(self.logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        rewards = np.array(self.episode_rewards)
        scores = np.array(self.episode_scores)
        episodes = np.arange(1, len(rewards) + 1)
        window = max(20, min(500, len(rewards) // 20))

        def smooth(data):
            if len(data) >= window:
                return np.convolve(data, np.ones(window) / window, mode="valid")
            return None

        # Plot 1: Score
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(episodes, scores, alpha=0.3, linewidth=1, color='steelblue')
        s = smooth(scores)
        if s is not None:
            ax.plot(episodes[window - 1:], s, linewidth=2.5, color='darkblue', label='Smoothed')
        ax.fill_between(episodes, scores, alpha=0.1, color='steelblue')
        ax.set_xlabel("Episode", fontsize=11, fontweight='bold')
        ax.set_ylabel("Score (Food Eaten)", fontsize=11, fontweight='bold')
        ax.set_title("Score per Episode", fontsize=13, fontweight='bold', pad=15)
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(self.logs_dir, f"01_score_{timestamp}.png"), dpi=200, bbox_inches='tight')
        plt.close()

        # Plot 2: Reward
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(episodes, rewards, alpha=0.3, linewidth=1, color='coral')
        s = smooth(rewards)
        if s is not None:
            ax.plot(episodes[window - 1:], s, linewidth=2.5, color='darkred', label='Smoothed')
        ax.axhline(y=np.mean(rewards), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.1f}')
        ax.fill_between(episodes, rewards, alpha=0.1, color='coral')
        ax.set_xlabel("Episode", fontsize=11, fontweight='bold')
        ax.set_ylabel("Total Reward", fontsize=11, fontweight='bold')
        ax.set_title("Reward per Episode", fontsize=13, fontweight='bold', pad=15)
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(self.logs_dir, f"02_reward_{timestamp}.png"), dpi=200, bbox_inches='tight')
        plt.close()

        # Plot 3: Loss
        if self.loss_history:
            loss = np.array(self.loss_history)
            steps_axis = np.arange(1, len(loss) + 1)
            loss_window = max(100, len(loss) // 20)
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(steps_axis, loss, alpha=0.25, linewidth=0.8, color='tomato')
            if len(loss) >= loss_window:
                s = np.convolve(loss, np.ones(loss_window) / loss_window, mode="valid")
                ax.plot(steps_axis[loss_window - 1:], s, linewidth=2.5, color='darkred', label='Smoothed')
            ax.set_xlabel("Training Step", fontsize=11, fontweight='bold')
            ax.set_ylabel("Loss", fontsize=11, fontweight='bold')
            ax.set_title("Network Loss", fontsize=13, fontweight='bold', pad=15)
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.2, which='both')
            plt.tight_layout()
            plt.savefig(os.path.join(self.logs_dir, f"03_loss_{timestamp}.png"), dpi=200, bbox_inches='tight')
            plt.close()

        # Plot 4: Score Distribution
        fig, ax = plt.subplots(figsize=(11, 6))
        n, bins, patches = ax.hist(scores, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axvline(x=np.mean(scores), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {np.mean(scores):.2f}')
        ax.axvline(x=np.median(scores), color='green', linestyle='--', linewidth=2.5, label=f'Median: {np.median(scores):.0f}')
        stats_text = f'Min: {int(np.min(scores))}\nMax: {int(np.max(scores))}\nStd: {np.std(scores):.2f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', 
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_xlabel("Score", fontsize=11, fontweight='bold')
        ax.set_ylabel("Frequency", fontsize=11, fontweight='bold')
        ax.set_title("Score Distribution", fontsize=13, fontweight='bold', pad=15)
        ax.legend()
        ax.grid(True, alpha=0.2, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.logs_dir, f"04_score_dist_{timestamp}.png"), dpi=200, bbox_inches='tight')
        plt.close()

        # Plot 5: Epsilon
        if self.epsilon_history:
            epsilon_vals = np.array(self.epsilon_history)
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(range(len(epsilon_vals)), epsilon_vals, linewidth=2.5, color='orange')
            ax.fill_between(range(len(epsilon_vals)), epsilon_vals, alpha=0.2, color='orange')
            ax.axhline(y=self.epsilon_min, color='red', linestyle='--', linewidth=2, label=f'Min: {self.epsilon_min:.4f}')
            ax.set_xlabel("Episode", fontsize=11, fontweight='bold')
            ax.set_ylabel("Epsilon (Exploration Rate)", fontsize=11, fontweight='bold')
            ax.set_title("Exploration Decay", fontsize=13, fontweight='bold', pad=15)
            ax.legend()
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(self.logs_dir, f"05_epsilon_{timestamp}.png"), dpi=200, bbox_inches='tight')
            plt.close()

        # Plot 6: Q-Values
        if self.q_value_history:
            q_vals = np.array(self.q_value_history)
            q_eps = np.arange(1, len(q_vals) + 1)
            q_window = max(20, len(q_vals) // 20)
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(q_eps, q_vals, alpha=0.3, linewidth=1, color='mediumseagreen')
            if len(q_vals) >= q_window:
                s = np.convolve(q_vals, np.ones(q_window) / q_window, mode="valid")
                ax.plot(q_eps[q_window - 1:], s, linewidth=2.5, color='darkgreen', label='Smoothed')
            ax.axhline(y=np.mean(q_vals), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(q_vals):.2f}')
            ax.fill_between(q_eps, q_vals, alpha=0.1, color='mediumseagreen')
            ax.set_xlabel("Episode", fontsize=11, fontweight='bold')
            ax.set_ylabel("Avg Max Q-Value", fontsize=11, fontweight='bold')
            ax.set_title("Network Q-Value Estimation", fontsize=13, fontweight='bold', pad=15)
            ax.legend()
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(self.logs_dir, f"06_qvalues_{timestamp}.png"), dpi=200, bbox_inches='tight')
            plt.close()

        # Plot 7: Steps
        if self.episode_steps:
            steps = np.array(self.episode_steps)
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(episodes, steps, alpha=0.3, linewidth=1, color='mediumpurple')
            s = smooth(steps)
            if s is not None:
                ax.plot(episodes[window - 1:], s, linewidth=2.5, color='darkviolet', label='Smoothed')
            ax.axhline(y=np.mean(steps), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(steps):.0f}')
            ax.fill_between(episodes, steps, alpha=0.1, color='mediumpurple')
            ax.set_xlabel("Episode", fontsize=11, fontweight='bold')
            ax.set_ylabel("Steps Survived", fontsize=11, fontweight='bold')
            ax.set_title("Survival Time per Episode", fontsize=13, fontweight='bold', pad=15)
            ax.legend()
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(self.logs_dir, f"07_steps_{timestamp}.png"), dpi=200, bbox_inches='tight')
            plt.close()

        # Plot 8: Score vs Steps
        if self.episode_steps:
            steps = np.array(self.episode_steps)
            fig, ax = plt.subplots(figsize=(11, 6))
            scatter = ax.scatter(steps, scores, c=episodes, cmap='viridis', s=40, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax.set_xlabel("Steps per Episode", fontsize=11, fontweight='bold')
            ax.set_ylabel("Score (Food Eaten)", fontsize=11, fontweight='bold')
            ax.set_title("Efficiency: Score vs Survival Time", fontsize=13, fontweight='bold', pad=15)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Episode', fontsize=10)
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(self.logs_dir, f"08_score_vs_steps_{timestamp}.png"), dpi=200, bbox_inches='tight')
            plt.close()

        print(f"[Plots Saved] -> {self.logs_dir}")
        
        if self.logging_enabled and hasattr(self, 'logger'):
            self.logger.info(f"Plots generated successfully")