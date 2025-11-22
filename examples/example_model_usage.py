"""
Example script demonstrating how to use the neural network models.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import gym
import gym_gomoku

from models import ActorCritic, get_action_mask_from_board, save_model, load_model

# Workaround for NumPy 2.0 compatibility
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_


def example_basic_usage():
    """Basic example of creating and using a model."""
    print("=" * 60)
    print("Example 1: Basic Model Usage")
    print("=" * 60)
    
    # Create environment
    env = gym.make('Gomoku9x9-v0')
    board_size = 9
    action_size = board_size * board_size
    
    # Create model
    model = ActorCritic(
        board_size=board_size,
        action_size=action_size,
        channels=64,  # Smaller for faster demo
        num_layers=3,
        hidden_size=128
    )
    
    print(f"Model created: {type(model).__name__}")
    print(f"  Board size: {board_size}")
    print(f"  Action size: {action_size}")
    print(f"  Feature extractor channels: {model.feature_extractor.channels}")
    
    # Reset environment
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    
    # Convert observation to tensor
    x = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension
    
    # Get action mask (valid moves)
    action_mask = get_action_mask_from_board(obs, board_size)
    action_mask = action_mask.reshape(1, -1)  # Add batch dimension
    
    print(f"Valid actions: {action_mask.sum().item()} out of {action_size}")
    
    # Get action and value estimate
    with torch.no_grad():
        action, log_prob, value = model.get_action(
            x, 
            action_mask=action_mask, 
            deterministic=False
        )
    
    print(f"\nSelected action: {action.item()}")
    print(f"Log probability: {log_prob.item():.4f}")
    print(f"State value estimate: {value.item():.4f}")
    
    env.close()


def example_action_masking():
    """Example demonstrating action masking."""
    print("\n" + "=" * 60)
    print("Example 2: Action Masking")
    print("=" * 60)
    
    board_size = 9
    action_size = board_size * board_size
    
    # Create a board with some occupied positions
    board_state = np.zeros((board_size, board_size))
    board_state[0, 0] = 1
    board_state[1, 1] = 2
    board_state[2, 2] = 1
    
    # Create action mask
    action_mask = get_action_mask_from_board(board_state, board_size)
    
    print(f"Board state (first 3x3):")
    print(board_state[:3, :3])
    print(f"\nValid actions: {action_mask.sum()} out of {action_size}")
    print(f"Invalid actions: {(~action_mask).sum()}")
    
    # Show which positions are valid
    valid_positions = np.where(action_mask)[0]
    print(f"\nFirst 10 valid action indices: {valid_positions[:10]}")


def example_model_save_load():
    """Example demonstrating model saving and loading."""
    print("\n" + "=" * 60)
    print("Example 3: Model Save/Load")
    print("=" * 60)
    
    board_size = 9
    action_size = board_size * board_size
    
    # Create model
    model = ActorCritic(
        board_size=board_size,
        action_size=action_size,
        channels=64,
        num_layers=3,
        hidden_size=128
    )
    
    # Create some dummy input
    x = torch.randn(1, board_size, board_size)
    
    # Get output before saving
    with torch.no_grad():
        _, _, value_before = model(x)
    
    # Save model with metadata to project directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    saved_models_dir = os.path.join(project_root, 'saved_models')
    os.makedirs(saved_models_dir, exist_ok=True)
    
    filepath = os.path.join(saved_models_dir, 'test_model.pt')
    metadata = {
        'board_size': board_size,
        'action_size': action_size,
        'channels': 64,
        'num_layers': 3,
        'hidden_size': 128,
        'training_step': 1000,
        'test_accuracy': 0.85
    }
    
    save_model(model, filepath, metadata=metadata)
    print(f"Model saved to: {filepath}")
    print(f"Metadata saved to: {filepath.replace('.pt', '.json')}")
    
    new_model = ActorCritic(
        board_size=board_size,
        action_size=action_size,
        channels=64,
        num_layers=3,
        hidden_size=128
    )
    
    loaded_model, loaded_metadata = load_model(new_model, filepath)
    
    # Verify outputs match
    with torch.no_grad():
        _, _, value_after = loaded_model(x)
    
    print(f"\nValue before save: {value_before.item():.4f}")
    print(f"Value after load: {value_after.item():.4f}")
    print(f"Values match: {torch.allclose(value_before, value_after)}")
    
    print(f"\nLoaded metadata:")
    for key, value in loaded_metadata.items():
        print(f"  {key}: {value}")


def example_training_step():
    """Example showing how to use models in a training loop."""
    print("\n" + "=" * 60)
    print("Example 4: Training Step (Forward Pass)")
    print("=" * 60)
    
    board_size = 9
    action_size = board_size * board_size
    
    # Create model
    model = ActorCritic(
        board_size=board_size,
        action_size=action_size,
        channels=64,
        num_layers=3,
        hidden_size=128
    )
    
    # Create environment
    env = gym.make('Gomoku9x9-v0')
    obs, info = env.reset()
    
    # Convert to tensor
    x = torch.FloatTensor(obs).unsqueeze(0)
    
    # Get action mask
    action_mask = get_action_mask_from_board(obs, board_size)
    action_mask = action_mask.reshape(1, -1)
    
    # Forward pass
    with torch.no_grad():
        logits, probs, value = model(x, action_mask=action_mask)
        action, log_prob, _ = model.get_action(x, action_mask=action_mask)
    
    print(f"Observation shape: {x.shape}")
    print(f"Action logits shape: {logits.shape}")
    print(f"Action probabilities shape: {probs.shape}")
    print(f"Value estimate shape: {value.shape}")
    print(f"Selected action: {action.item()}")
    print(f"Action probability: {probs[0, action.item()].item():.6f}")
    print(f"Log probability: {log_prob.item():.4f}")
    print(f"State value: {value.item():.4f}")
    
    # Verify probabilities sum to 1 (only for valid actions)
    valid_probs = probs[0, action_mask[0]]
    print(f"\nValid action probabilities sum: {valid_probs.sum().item():.6f}")
    
    env.close()


if __name__ == '__main__':
    # Run examples
    example_basic_usage()
    example_action_masking()
    example_model_save_load()
    example_training_step()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)

