"""
Example: Play against a trained neural network model.

This script shows how to load a trained model and use it as an opponent
in the Gomoku environment.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gym
import gym_gomoku
import numpy as np
import torch

from models import ActorCritic, load_model
from gym_gomoku.envs.gomoku import GomokuEnv
from gym_gomoku.envs.util import make_model_policy

# Workaround for NumPy 2.0 compatibility
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_


def coord_to_action(coord_str, board_size):
    """Convert coordinate string like "A1" to action integer."""
    coord_str = coord_str.strip().upper()
    if len(coord_str) < 2:
        raise ValueError("Invalid coordinate format. Use format like 'A1' or 'E5'")
    
    letter = coord_str[0]
    try:
        number = int(coord_str[1:])
    except ValueError:
        raise ValueError("Invalid coordinate format. Use format like 'A1' or 'E5'")
    
    col = ord(letter) - ord('A')
    if col < 0 or col >= board_size:
        raise ValueError(f"Column {letter} is out of range. Use A-{chr(ord('A') + board_size - 1)}")
    
    if number < 1 or number > board_size:
        raise ValueError(f"Row {number} is out of range. Use 1-{board_size}")
    row = number - 1
    
    action = row * board_size + col
    return action


def play_against_model(model_path, board_size=9, deterministic=True):
    """
    Play an interactive game against a trained model.
    
    Args:
        model_path: Path to the saved model file (.pt)
        board_size: Size of the board (default: 9)
        deterministic: If True, model always picks best move. If False, samples from distribution.
    """
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = ActorCritic(
        board_size=board_size,
        action_size=board_size * board_size,
        channels=128,  # Adjust these to match your trained model
        num_layers=4,
        hidden_size=256
    )
    
    model, metadata = load_model(model, model_path)
    model.eval()  # Set to evaluation mode
    
    print("Model loaded successfully!")
    if metadata:
        print(f"Model metadata: {metadata}")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Create model policy function
    model_policy = make_model_policy(model, device=device, deterministic=deterministic)
    
    # Create environment
    env = GomokuEnv(player_color='black', opponent='random', board_size=board_size)
    
    # Replace the opponent policy with the model
    env.opponent_policy = model_policy
    
    print(f"\n{'='*60}")
    print(f"Playing against trained model!")
    print(f"Board size: {board_size}x{board_size}")
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
            import traceback
            traceback.print_exc()
            continue
    
    env.close()


if __name__ == "__main__":
    import sys
    
    # Default model path
    model_path = 'saved_models/test_model.pt'
    board_size = 9
    deterministic = True
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        board_size = int(sys.argv[2])
    
    if len(sys.argv) > 3:
        deterministic = sys.argv[3].lower() == 'true'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print(f"Usage: python play_against_model.py [model_path] [board_size] [deterministic]")
        print(f"Example: python play_against_model.py saved_models/my_model.pt 9 true")
        sys.exit(1)
    
    play_against_model(model_path, board_size, deterministic)

