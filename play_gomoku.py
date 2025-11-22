#!/usr/bin/env python
"""
Interactive Gomoku game player.
Play against the AI opponent by entering moves in format like "A1", "E5", etc.
"""

import gym
import gym_gomoku
import numpy as np

# Workaround for NumPy 2.0 compatibility with Gym
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

def coord_to_action(coord_str, board_size):
    """
    Convert coordinate string like "A1" or "E5" to action integer.
    Args:
        coord_str: String like "A1", "E5", etc. (letter for column, number for row)
        board_size: Size of the board
    Returns:
        Action integer in [0, board_size**2)
    """
    coord_str = coord_str.strip().upper()
    if len(coord_str) < 2:
        raise ValueError("Invalid coordinate format. Use format like 'A1' or 'E5'")
    
    # Extract letter and number
    letter = coord_str[0]
    try:
        number = int(coord_str[1:])
    except ValueError:
        raise ValueError("Invalid coordinate format. Use format like 'A1' or 'E5'")
    
    # Convert letter to column index (A=0, B=1, etc.)
    col = ord(letter) - ord('A')
    if col < 0 or col >= board_size:
        raise ValueError(f"Column {letter} is out of range. Use A-{chr(ord('A') + board_size - 1)}")
    
    # Convert number to row index (1=0, 2=1, etc., but board is rendered bottom-to-top)
    if number < 1 or number > board_size:
        raise ValueError(f"Row {number} is out of range. Use 1-{board_size}")
    row = number - 1
    
    # Convert to action (row * board_size + col)
    action = row * board_size + col
    return action

def action_to_coord(action, board_size):
    """
    Convert action integer to coordinate string like "A1".
    """
    row = action // board_size
    col = action % board_size
    letter = chr(ord('A') + col)
    number = row + 1
    return f"{letter}{number}"

def play_interactive(board_size=9):
    """
    Play an interactive game of Gomoku.
    """
    # Create environment
    if board_size in [9, 19]:
        env_id = f'Gomoku{board_size}x{board_size}-v0'
        env = gym.make(env_id)
    else:
        # Create environment with custom board size
        from gym_gomoku.envs.gomoku import GomokuEnv
        env = GomokuEnv(player_color='black', opponent='random', board_size=board_size)
    
    print(f"\n{'='*60}")
    print(f"Welcome to Interactive Gomoku!")
    print(f"Board size: {board_size}x{board_size}")
    print(f"You are playing as BLACK (X)")
    print(f"Enter moves in format: A1, E5, etc.")
    print(f"Type 'quit' to exit, 'reset' to start a new game")
    print(f"{'='*60}\n")
    
    obs, info = env.reset()
    env.render()
    
    while True:
        try:
            # Get user input
            user_input = input("\nEnter your move (e.g., A1): ").strip()
            
            if user_input.lower() == 'quit':
                print("Thanks for playing!")
                break
            elif user_input.lower() == 'reset':
                print("\nStarting a new game...\n")
                obs, info = env.reset()
                env.render()
                continue
            
            # Convert coordinate to action
            try:
                action = coord_to_action(user_input, board_size)
            except ValueError as e:
                print(f"Error: {e}")
                continue
            
            # Check if action is valid
            if action not in env.action_space.valid_spaces:
                print(f"Error: Position {user_input} is already occupied!")
                continue
            
            # Take step
            observation, reward, done, info = env.step(action)
            env.render()
            
            if done:
                if reward == 1:
                    print("\n🎉 Congratulations! You won!")
                elif reward == -1:
                    print("\n😔 You lost! Better luck next time!")
                else:
                    print("\n🤝 It's a draw!")
                
                play_again = input("\nPlay again? (y/n): ").strip().lower()
                if play_again == 'y':
                    obs, info = env.reset()
                    env.render()
                else:
                    print("Thanks for playing!")
                    break
        
        except KeyboardInterrupt:
            print("\n\nThanks for playing!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    env.close()

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    board_size = 9
    
    if len(sys.argv) > 1:
        try:
            board_size = int(sys.argv[1])
            if board_size not in [9, 19]:
                print("Warning: Board size should be 9 or 19. Using 9.")
                board_size = 9
        except ValueError:
            print("Warning: Invalid board size. Using 9.")
    
    play_interactive(board_size)

