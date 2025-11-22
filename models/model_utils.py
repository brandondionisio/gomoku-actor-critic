"""
Utility functions for model saving, loading, and action masking.
"""

import torch
import numpy as np
from pathlib import Path
import json


def get_action_mask(valid_actions, action_size):
    """
    Create an action mask from a list of valid action indices.
    
    Args:
        valid_actions: List or array of valid action indices
        action_size: Total number of possible actions
    
    Returns:
        mask: Boolean numpy array of shape (action_size,)
              True for valid actions, False for invalid actions
    """
    mask = np.zeros(action_size, dtype=bool)
    if len(valid_actions) > 0:
        mask[valid_actions] = True
    return mask


def get_action_mask_from_board(board_state, board_size):
    """
    Create an action mask from a board state.
    
    Args:
        board_state: 2D numpy array representing the board (0=empty, 1=black, 2=white)
        board_size: Size of the board
    
    Returns:
        mask: Boolean numpy array of shape (board_size^2,)
              True for empty positions (valid actions), False for occupied positions
    """
    action_size = board_size * board_size
    mask = np.zeros(action_size, dtype=bool)
    
    flat_board = board_state.flatten()
    empty_positions = np.where(flat_board == 0)[0]
    mask[empty_positions] = True
    
    return mask


def save_model(model, filepath, metadata=None):
    """
    Save a PyTorch model with optional metadata.
    
    Args:
        model: PyTorch model to save
        filepath: Path where to save the model (should end with .pt or .pth)
        metadata: Optional dictionary containing metadata (e.g., training config, metrics)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), filepath)
    
    # Save metadata if provided
    if metadata is not None:
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def load_model(model, filepath, device=None, strict=True):
    """
    Load a PyTorch model from a saved state dict.
    
    Args:
        model: PyTorch model instance (architecture must match)
        filepath: Path to the saved model file
        device: Device to load the model on (default: same as model)
        strict: Whether to strictly enforce that the keys in state_dict match
    
    Returns:
        model: Model with loaded weights
        metadata: Metadata dictionary if available, None otherwise
    """
    filepath = Path(filepath)
    
    if device is None:
        device = next(model.parameters()).device
    
    # Load state dict
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict, strict=strict)
    
    # Try to load metadata
    metadata = None
    metadata_path = filepath.with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata


def create_model_from_config(config):
    """
    Create an ActorCritic model from a configuration dictionary.
    
    Args:
        config: Dictionary containing:
            - board_size: Size of the board
            - channels: Number of channels in CNN (optional, default: 128)
            - num_layers: Number of CNN layers (optional, default: 4)
            - hidden_size: Size of hidden layers (optional, default: 256)
    
    Returns:
        model: ActorCritic model instance
    """
    from .networks import ActorCritic  # Import here to avoid circular dependency
    
    board_size = config['board_size']
    action_size = board_size * board_size
    channels = config.get('channels', 128)
    num_layers = config.get('num_layers', 4)
    hidden_size = config.get('hidden_size', 256)
    
    model = ActorCritic(
        board_size=board_size,
        action_size=action_size,
        channels=channels,
        num_layers=num_layers,
        hidden_size=hidden_size
    )
    
    return model

