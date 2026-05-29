import os
import sys
import argparse
from datetime import datetime

from Snake.SnakeLogic import SnakeLogic
from Trainer.Trainer import Trainer

RECORDINGS_FOLDER = "Recordings"

AGENT_REGISTRY = {
    "random": ("Agents.RandomAgent", "RandomAgent"),
    "greedy": ("Agents.GreedyAgent", "GreedyAgent"),
    "astar": ("Agents.AstarAgent", "AstarAgent"),
    "dqn": ("Agents.DQNAgent", "DQNAgent"),
    "cnn-dqn": ("Agents.CNNDQNAgent", "CNNDQNAgent"),
    "cnn-ddqn": ("Agents.CNNDDQNAgent", "CNNDDQNAgent"),
}


def build_agent(name, game):
    if name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent: {name!r}. Choose from: {list(AGENT_REGISTRY)}"
        )
    module_path, class_name = AGENT_REGISTRY[name]
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)(game)


def build_recorder(args, game):
    if not args.r:
        return None

    os.makedirs(RECORDINGS_FOLDER, exist_ok=True)
    path = args.record_path or os.path.join(
        RECORDINGS_FOLDER,
        f"recording_{args.W}x{args.H}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.bin.gz",
    )

    from Replay.Recorder import Recorder

    recorder = Recorder(
        path=path,
        width=args.W,
        height=args.H,
        compress=path.endswith(".gz"),
    )
    recorder.start_episode()
    return recorder


def main():
    parser = argparse.ArgumentParser(description="Snake Game")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Mode
    parser.add_argument("-hu", action="store_true", help="Play as human (GUI)")
    parser.add_argument("-ai", choices=["gui"], help="Run AI agent (gui)")
    parser.add_argument(
        "--headless", action="store_true", help="Train headlessly, no rendering"
    )
    parser.add_argument("--replay", type=str, help="Replay a recorded game (CLI)")

    # Board
    parser.add_argument("-W", type=int, default=20, help="Board width")
    parser.add_argument("-H", type=int, default=20, help="Board height")
    parser.add_argument("-s", type=int, default=8, help="Speed / FPS")
    parser.add_argument("-ww", type=int, default=600, help="Window width (pixels)")
    parser.add_argument("-wh", type=int, default=600, help="Window height (pixels)")

    # AI options
    parser.add_argument(
        "-a", default="random", choices=list(AGENT_REGISTRY), help="Agent type"
    )
    parser.add_argument("-v", action="store_true", help="Verbose output")

    # Recording
    parser.add_argument("-r", action="store_true", help="Record gameplay")
    parser.add_argument(
        "--record-path", default=None, help="Custom recording file path"
    )
    
    # Stats
    parser.add_argument("--stats-path", default="../stats/stats.csv", help="Store game stats in a csv at ../stats/stats.csv by default.")

    args = parser.parse_args()

    # Replay mode
    if args.replay:
        from Replay.ReplayCLI import replay

        replay(args.replay, fps=args.s)
        return

    game = SnakeLogic(args.W, args.H)

    # Human play
    if args.hu:
        recorder = None
        if args.r:
            recorder = build_recorder(args, game)
        from GUI.SnakeHumanGUI import SnakeHumanGUI

        SnakeHumanGUI(
            game=game,
            recorder=recorder,
            window_w=args.ww,
            window_h=args.wh,
            fps=args.s,
        ).run()
        return

    # AI modes
    if args.ai or args.headless:    
        agent = build_agent(args.a, game)
        stats_dir = os.path.dirname(args.stats_path)
        stats_filename = os.path.basename(args.stats_path)
        trainer = Trainer(
            env=game,
            agent=agent,
            stats_dir=stats_dir,
            stats_filename=stats_filename,
            headless=args.headless,
            verbose=args.v,
        )

        if args.headless:
            trainer.run()
            return

        if args.ai == "gui":
            recorder = None
            if args.r:
                recorder = build_recorder(args, game)
            from GUI.SnakeAIGUI import SnakeAIGUI

            SnakeAIGUI(
                game=game,
                trainer=trainer,
                window_w=args.ww,
                window_h=args.wh,
                fps=args.s,
                recorder=recorder
            ).run()
            trainer.close()
            return

    print("No mode specified. Use --help")
    sys.exit(1)


if __name__ == "__main__":
    main()
