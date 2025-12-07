#!/usr/bin/env python
"""
Interactive Gomoku game player.
Play against the AI opponent by entering moves in format like "A1", "E5", etc.
"""

import gym
import gym_gomoku
import numpy as np
import torch
import os
import sys

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

def play_interactive(board_size=9, model_path=None, deterministic=True):
    """
    Play an interactive game of Gomoku.
    
    Args:
        board_size: Size of the board (default: 9)
        model_path: Path to trained model file. If None, plays against random opponent.
        deterministic: If True, model always picks best move. If False, samples from distribution.
    """
    # Create environment
    if board_size in [9, 19]:
        env_id = f'Gomoku{board_size}x{board_size}-v0'
        env = gym.make(env_id)
    else:
        # Create environment with custom board size
        from gym_gomoku.envs.gomoku import GomokuEnv
        env = GomokuEnv(player_color='black', opponent='random', board_size=board_size)
    
    # Load model if provided
    model = None
    if model_path:
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            print("Playing against random opponent instead.")
        else:
            try:
                from models import ActorCritic, load_model
                from gym_gomoku.envs.util import make_model_policy
                
                print(f"Loading model from {model_path}...")
                model = ActorCritic(
                    board_size=board_size,
                    action_size=board_size * board_size,
                    channels=128,
                    num_layers=4,
                    hidden_size=256,
                    input_channels=3,  # One-hot encoding: 3 channels (empty, black, white)
                    extra_feature_size=4  # Pattern features: 4 features
                )
                model, metadata = load_model(model, model_path)
                model.eval()
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = model.to(device)
                
                model_policy = make_model_policy(model, device=device, deterministic=deterministic)
                env.opponent_policy = model_policy
                
                print("Model loaded successfully!")
                if metadata:
                    print(f"Model metadata: {metadata}")
                print(f"Using device: {device}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Playing against random opponent instead.")
                model = None
    
    opponent_type = "trained model" if model else "random opponent"
    print(f"\n{'='*60}")
    print(f"Welcome to Interactive Gomoku!")
    print(f"Board size: {board_size}x{board_size}")
    print(f"Opponent: {opponent_type}")
    if model:
        print(f"Model mode: {'Deterministic (best move)' if deterministic else 'Stochastic (sampling)'}")
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
    # Parse command line arguments
    board_size = 9
    model_path = None
    deterministic = True
    
    if len(sys.argv) > 1:
        # First argument: board size or model path
        arg1 = sys.argv[1]
        if arg1.endswith('.pt'):
            # It's a model path
            model_path = arg1
        else:
            # It's board size
            try:
                board_size = int(arg1)
                if board_size not in [9, 19]:
                    print("Warning: Board size should be 9 or 19. Using 9.")
                    board_size = 9
            except ValueError:
                print("Warning: Invalid board size. Using 9.")
    
    if len(sys.argv) > 2:
        # Second argument: board size (if first was model) or deterministic
        arg2 = sys.argv[2]
        if arg2.endswith('.pt'):
            model_path = arg2
        elif arg2.isdigit():
            board_size = int(arg2)
        else:
            deterministic = arg2.lower() == 'true'
    
    if len(sys.argv) > 3:
        # Third argument: deterministic
        deterministic = sys.argv[3].lower() == 'true'
    
    play_interactive(board_size, model_path, deterministic)

