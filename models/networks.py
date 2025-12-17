"""
Neural network architectures for Actor-Critic reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNNFeatureExtractor(nn.Module):
    """
    CNN feature extractor for processing Gomoku board states.
    
    Takes a board state (board_size x board_size) and extracts features
    using convolutional layers suitable for spatial pattern recognition.
    """
    
    def __init__(self, board_size=9, channels=128, num_layers=4, input_channels=3):
        """
        Args:
            board_size: Size of the board (e.g., 9 for 9x9 board)
            channels: Number of channels in convolutional layers
            num_layers: Number of convolutional blocks
            input_channels: Number of input channels (3 for one-hot encoding, 1 for legacy)
        """
        super(CNNFeatureExtractor, self).__init__()
        self.board_size = board_size
        self.channels = channels
        self.input_channels = input_channels
        
        # Input: (batch, input_channels, board_size, board_size)
        # First conv layer: input_channels -> channels
        layers = []
        layers.append(nn.Conv2d(input_channels, channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Additional conv layers: channels -> channels
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate feature size after convolutions
        # With padding=1, spatial dimensions remain the same
        self.feature_size = channels * board_size * board_size
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, input_channels, board_size, board_size) 
               or (batch, board_size, board_size) for legacy single-channel
        Returns:
            features: Flattened feature tensor of shape (batch, feature_size)
        """
        # Ensure input has channel dimension
        if x.dim() == 3:
            # Legacy single-channel input: add channel dimension
            x = x.unsqueeze(1)  # (batch, board_size, board_size) -> (batch, 1, board_size, board_size)
        elif x.dim() == 4 and x.size(1) != self.input_channels:
            # If input is 4D but channel dimension doesn't match, adjust
            # This handles cases where one-hot (3 channels) is passed but model expects 1 channel
            if x.size(1) == 3 and self.input_channels == 1:
                # Convert one-hot to single channel (use argmax)
                x = x.argmax(dim=1, keepdim=True).float() / 2.0
            elif x.size(1) == 1 and self.input_channels == 3:
                # Convert single channel to one-hot (not ideal, but for compatibility)
                raise ValueError("Model expects 3-channel input but received 1-channel. Please use one-hot encoding.")
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten: (batch, channels, board_size, board_size) -> (batch, feature_size)
        x = x.view(x.size(0), -1)
        
        return x
    
    def get_feature_size(self):
        """Return the size of the feature vector."""
        return self.feature_size


class ActorNetwork(nn.Module):
    """
    Actor network (policy head) that outputs action probabilities.
    
    Uses action masking to ensure only valid moves are considered.
    """
    
    def __init__(self, feature_extractor, action_size, hidden_size=256, extra_feature_size=0):
        """
        Args:
            feature_extractor: CNNFeatureExtractor instance
            action_size: Number of possible actions (board_size^2)
            hidden_size: Size of hidden layers
        """
        super(ActorNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.action_size = action_size
        self.extra_feature_size = extra_feature_size

        # Adjust input size if extra features exist
        input_size = feature_extractor.get_feature_size() + extra_feature_size
        
        # Policy head: features -> hidden -> action logits
        # Use input_size to account for extra features
        self.policy_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x, action_mask=None, extra_features=None):
        """
        Forward pass through the actor network.
        
        Args:
            x: Input tensor of shape (batch, board_size, board_size) or (batch, 1, board_size, board_size)
            action_mask: Boolean mask of shape (batch, action_size) indicating valid actions.
                        True = valid action, False = invalid action.
                        If None, all actions are considered valid.
        
        Returns:
            logits: Raw action logits of shape (batch, action_size)
            probs: Action probabilities after masking and softmax of shape (batch, action_size)
        """
        # Extract features
        features = self.feature_extractor(x)

        if extra_features is not None:
            features = torch.cat([features, extra_features], dim=1)
        elif self.extra_feature_size > 0:
            # If model expects extra features but none provided, use zeros
            batch_size = features.size(0)
            zero_features = torch.zeros(batch_size, self.extra_feature_size, device=features.device, dtype=features.dtype)
            features = torch.cat([features, zero_features], dim=1)
        
        # Get action logits
        logits = self.policy_head(features)
        
        # Apply action masking
        if action_mask is not None:
            # Convert mask to same device and dtype as logits
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask).to(logits.device)
            else:
                action_mask = action_mask.to(logits.device)
            
            # Set invalid actions to very negative value (before softmax)
            # This ensures they have near-zero probability after softmax
            logits = logits.masked_fill(~action_mask, float('-inf'))
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        return logits, probs
    
    def get_action(self, x, action_mask=None, deterministic=False, extra_features=None):
        """
        Sample an action from the policy distribution.
        
        Args:
            x: Input tensor of shape (batch, board_size, board_size) or (batch, 1, board_size, board_size)
            action_mask: Boolean mask indicating valid actions
            deterministic: If True, return the action with highest probability.
                          If False, sample from the distribution.
            extra_features: Optional extra features tensor to concatenate with CNN features
        
        Returns:
            action: Selected action index
            log_prob: Log probability of the selected action
        """
        _, probs = self.forward(x, action_mask, extra_features)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            # Sample from the distribution
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        # Calculate log probability
        log_prob = torch.log(probs.gather(1, action.unsqueeze(1)) + 1e-8).squeeze(1)
        
        return action, log_prob


class CriticNetwork(nn.Module):
    """
    Critic network (value head) that estimates state values.
    """
    
    def __init__(self, feature_extractor, hidden_size=256, extra_feature_size=0):
        """
        Args:
            feature_extractor: CNNFeatureExtractor instance
            hidden_size: Size of hidden layers
            extra_feature_size: Size of extra features
        """
        super(CriticNetwork, self).__init__()
        input_size = feature_extractor.get_feature_size() + extra_feature_size
        self.feature_extractor = feature_extractor
        self.extra_feature_size = extra_feature_size
        
        # Value head: features -> hidden -> scalar value
        self.value_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x, extra_features=None):
        """
        Forward pass through the critic network.
        
        Args:
            x: Input tensor of shape (batch, board_size, board_size) or (batch, 1, board_size, board_size)
        
        Returns:
            value: State value estimate of shape (batch, 1)
        """
        # Extract features
        features = self.feature_extractor(x)

        if extra_features is not None:
            features = torch.cat([features, extra_features], dim=1)
        elif hasattr(self, 'extra_feature_size') and self.extra_feature_size > 0:
            # If model expects extra features but none provided, use zeros
            batch_size = features.size(0)
            zero_features = torch.zeros(batch_size, self.extra_feature_size, device=features.device, dtype=features.dtype)
            features = torch.cat([features, zero_features], dim=1)
        
        # Get value estimate
        value = self.value_head(features)
        
        return value


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network sharing the same feature extractor.
    
    This is more efficient than separate networks as features are computed once.
    """
    
    def __init__(self, board_size=9, action_size=81, channels=128, num_layers=4, hidden_size=256, extra_feature_size=0, input_channels=3):
        """
        Args:
            board_size: Size of the board (e.g., 9 for 9x9 board)
            action_size: Number of possible actions (board_size^2)
            channels: Number of channels in convolutional layers
            num_layers: Number of convolutional blocks
            hidden_size: Size of hidden layers in actor/critic heads
            extra_feature_size: Size of extra features (e.g., pattern features)
            input_channels: Number of input channels (3 for one-hot encoding, 1 for legacy)
        """
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = CNNFeatureExtractor(
            board_size=board_size,
            channels=channels,
            num_layers=num_layers,
            input_channels=input_channels
        )
        
        # Actor and Critic networks
        self.actor = ActorNetwork(
            feature_extractor=self.feature_extractor,
            action_size=action_size,
            hidden_size=hidden_size,
            extra_feature_size=extra_feature_size
        )
        
        self.critic = CriticNetwork(
            feature_extractor=self.feature_extractor,
            hidden_size=hidden_size,
            extra_feature_size=extra_feature_size
        )

    def forward(self, x, action_mask=None, extra_features=None):
        """
        Forward pass through both actor and critic.
        
        Args:
            x: Input tensor of shape (batch, board_size, board_size) or (batch, 1, board_size, board_size)
            action_mask: Boolean mask indicating valid actions
        
        Returns:
            logits: Action logits from actor
            probs: Action probabilities from actor
            value: State value estimate from critic
        """
        logits, probs = self.actor(x, action_mask, extra_features)
        value = self.critic(x, extra_features)
        
        return logits, probs, value
    
    def get_action(self, x, action_mask=None, deterministic=False, extra_features=None):
        """
        Get action from actor network.
        
        Args:
            x: Input tensor
            action_mask: Boolean mask indicating valid actions
            deterministic: If True, return the action with highest probability
        
        Returns:
            action: Selected action index
            log_prob: Log probability of the selected action
            value: State value estimate
        """
        action, log_prob = self.actor.get_action(x, action_mask, deterministic, extra_features)
        value = self.critic(x, extra_features)
        
        return action, log_prob, value

