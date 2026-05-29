import os, time
from Replay.Reader import ReplayReader

DIR_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

def cell_char(value):
    if value == 0:
        return " . "   # empty
    if value == 1:
        return " @ "   # head
    if value > 1:
        return " o "   # body (any length value)
    return " F "       # food (negative value)

def print_board(board):
    for row in board:
        print("".join(cell_char(cell) for cell in row))

def replay(path, fps=10):
    delay  = 1.0 / fps
    reader = ReplayReader(path)
    last_episode = None

    try:
        while True:
            record = reader.read_record()
            if record is None:
                break

            episode_id, step_id, board, action, curr_dir, done, features = record

            if last_episode is None or episode_id != last_episode:
                last_episode = episode_id
                clear_terminal()
                print(f"=== EPISODE {episode_id} ===")
                time.sleep(0.5)

            clear_terminal()
            print(f"Episode:   {episode_id}")
            print(f"Step:      {step_id}")
            print(f"Action:    {action}")
            print(f"Direction: {DIR_NAMES[curr_dir]}")
            print(f"Done:      {done}")
            print()
            print_board(board)

            time.sleep(delay)
    finally:
        reader.close()