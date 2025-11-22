"""
Unit tests for neural network components.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add parent directory to path to import models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.networks import (
    CNNFeatureExtractor,
    ActorNetwork,
    CriticNetwork,
    ActorCritic
)
from models.model_utils import (
    get_action_mask,
    get_action_mask_from_board,
    save_model,
    load_model,
    create_model_from_config
)


class TestCNNFeatureExtractor(unittest.TestCase):
    """Test CNN feature extractor."""
    
    def setUp(self):
        self.board_size = 9
        self.batch_size = 4
        self.channels = 128
        self.num_layers = 4
        
    def test_initialization(self):
        """Test that feature extractor initializes correctly."""
        extractor = CNNFeatureExtractor(
            board_size=self.board_size,
            channels=self.channels,
            num_layers=self.num_layers
        )
        self.assertEqual(extractor.board_size, self.board_size)
        self.assertEqual(extractor.channels, self.channels)
        
    def test_forward_shape_2d(self):
        """Test forward pass with 2D input (batch, board_size, board_size)."""
        extractor = CNNFeatureExtractor(
            board_size=self.board_size,
            channels=self.channels,
            num_layers=self.num_layers
        )
        
        x = torch.randn(self.batch_size, self.board_size, self.board_size)
        features = extractor(x)
        
        expected_size = self.channels * self.board_size * self.board_size
        self.assertEqual(features.shape, (self.batch_size, expected_size))
        
    def test_forward_shape_3d(self):
        """Test forward pass with 3D input (batch, 1, board_size, board_size)."""
        extractor = CNNFeatureExtractor(
            board_size=self.board_size,
            channels=self.channels,
            num_layers=self.num_layers
        )
        
        x = torch.randn(self.batch_size, 1, self.board_size, self.board_size)
        features = extractor(x)
        
        expected_size = self.channels * self.board_size * self.board_size
        self.assertEqual(features.shape, (self.batch_size, expected_size))
        
    def test_get_feature_size(self):
        """Test feature size calculation."""
        extractor = CNNFeatureExtractor(
            board_size=self.board_size,
            channels=self.channels,
            num_layers=self.num_layers
        )
        
        feature_size = extractor.get_feature_size()
        expected_size = self.channels * self.board_size * self.board_size
        self.assertEqual(feature_size, expected_size)
        
    def test_different_board_sizes(self):
        """Test with different board sizes."""
        for board_size in [9, 15, 19]:
            extractor = CNNFeatureExtractor(
                board_size=board_size,
                channels=self.channels,
                num_layers=self.num_layers
            )
            x = torch.randn(2, board_size, board_size)
            features = extractor(x)
            expected_size = self.channels * board_size * board_size
            self.assertEqual(features.shape, (2, expected_size))


class TestActorNetwork(unittest.TestCase):
    """Test Actor network."""
    
    def setUp(self):
        self.board_size = 9
        self.action_size = 81
        self.batch_size = 4
        self.channels = 128
        self.hidden_size = 256
        
        self.feature_extractor = CNNFeatureExtractor(
            board_size=self.board_size,
            channels=self.channels
        )
        
    def test_initialization(self):
        """Test that actor network initializes correctly."""
        actor = ActorNetwork(
            feature_extractor=self.feature_extractor,
            action_size=self.action_size,
            hidden_size=self.hidden_size
        )
        self.assertEqual(actor.action_size, self.action_size)
        
    def test_forward_without_mask(self):
        """Test forward pass without action mask."""
        actor = ActorNetwork(
            feature_extractor=self.feature_extractor,
            action_size=self.action_size,
            hidden_size=self.hidden_size
        )
        
        x = torch.randn(self.batch_size, self.board_size, self.board_size)
        logits, probs = actor(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.action_size))
        self.assertEqual(probs.shape, (self.batch_size, self.action_size))
        
        # Check that probabilities sum to 1
        prob_sums = probs.sum(dim=1)
        torch.testing.assert_close(prob_sums, torch.ones(self.batch_size))
        
    def test_forward_with_mask(self):
        """Test forward pass with action mask."""
        actor = ActorNetwork(
            feature_extractor=self.feature_extractor,
            action_size=self.action_size,
            hidden_size=self.hidden_size
        )
        
        x = torch.randn(self.batch_size, self.board_size, self.board_size)
        
        # Create mask: only first 10 actions are valid
        action_mask = np.zeros((self.batch_size, self.action_size), dtype=bool)
        action_mask[:, :10] = True
        
        logits, probs = actor(x, action_mask=action_mask)
        
        self.assertEqual(logits.shape, (self.batch_size, self.action_size))
        self.assertEqual(probs.shape, (self.batch_size, self.action_size))
        
        # Check that masked actions have near-zero probability
        masked_probs = probs[:, 10:]
        self.assertTrue((masked_probs < 1e-6).all())
        
        # Check that valid actions have non-zero probability
        valid_probs = probs[:, :10]
        self.assertTrue((valid_probs > 0).all())
        
    def test_get_action_deterministic(self):
        """Test getting action deterministically."""
        actor = ActorNetwork(
            feature_extractor=self.feature_extractor,
            action_size=self.action_size,
            hidden_size=self.hidden_size
        )
        
        x = torch.randn(1, self.board_size, self.board_size)
        action_mask = np.ones((1, self.action_size), dtype=bool)
        
        action, log_prob = actor.get_action(x, action_mask=action_mask, deterministic=True)
        
        self.assertEqual(action.shape, (1,))
        self.assertEqual(log_prob.shape, (1,))
        self.assertTrue(0 <= action.item() < self.action_size)
        
    def test_get_action_stochastic(self):
        """Test getting action stochastically."""
        actor = ActorNetwork(
            feature_extractor=self.feature_extractor,
            action_size=self.action_size,
            hidden_size=self.hidden_size
        )
        
        x = torch.randn(1, self.board_size, self.board_size)
        action_mask = np.ones((1, self.action_size), dtype=bool)
        
        action, log_prob = actor.get_action(x, action_mask=action_mask, deterministic=False)
        
        self.assertEqual(action.shape, (1,))
        self.assertEqual(log_prob.shape, (1,))
        self.assertTrue(0 <= action.item() < self.action_size)


class TestCriticNetwork(unittest.TestCase):
    """Test Critic network."""
    
    def setUp(self):
        self.board_size = 9
        self.batch_size = 4
        self.channels = 128
        self.hidden_size = 256
        
        self.feature_extractor = CNNFeatureExtractor(
            board_size=self.board_size,
            channels=self.channels
        )
        
    def test_initialization(self):
        """Test that critic network initializes correctly."""
        critic = CriticNetwork(
            feature_extractor=self.feature_extractor,
            hidden_size=self.hidden_size
        )
        self.assertIsNotNone(critic)
        
    def test_forward(self):
        """Test forward pass."""
        critic = CriticNetwork(
            feature_extractor=self.feature_extractor,
            hidden_size=self.hidden_size
        )
        
        x = torch.randn(self.batch_size, self.board_size, self.board_size)
        value = critic(x)
        
        self.assertEqual(value.shape, (self.batch_size, 1))
        
    def test_value_range(self):
        """Test that values are reasonable (not NaN or Inf)."""
        critic = CriticNetwork(
            feature_extractor=self.feature_extractor,
            hidden_size=self.hidden_size
        )
        
        x = torch.randn(self.batch_size, self.board_size, self.board_size)
        value = critic(x)
        
        self.assertFalse(torch.isnan(value).any())
        self.assertFalse(torch.isinf(value).any())


class TestActorCritic(unittest.TestCase):
    """Test combined Actor-Critic network."""
    
    def setUp(self):
        self.board_size = 9
        self.action_size = 81
        self.batch_size = 4
        self.channels = 128
        self.hidden_size = 256
        
    def test_initialization(self):
        """Test that ActorCritic initializes correctly."""
        model = ActorCritic(
            board_size=self.board_size,
            action_size=self.action_size,
            channels=self.channels,
            hidden_size=self.hidden_size
        )
        self.assertIsNotNone(model.actor)
        self.assertIsNotNone(model.critic)
        self.assertIsNotNone(model.feature_extractor)
        
    def test_forward(self):
        """Test forward pass."""
        model = ActorCritic(
            board_size=self.board_size,
            action_size=self.action_size,
            channels=self.channels,
            hidden_size=self.hidden_size
        )
        
        x = torch.randn(self.batch_size, self.board_size, self.board_size)
        logits, probs, value = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.action_size))
        self.assertEqual(probs.shape, (self.batch_size, self.action_size))
        self.assertEqual(value.shape, (self.batch_size, 1))
        
    def test_get_action(self):
        """Test getting action."""
        model = ActorCritic(
            board_size=self.board_size,
            action_size=self.action_size,
            channels=self.channels,
            hidden_size=self.hidden_size
        )
        
        x = torch.randn(1, self.board_size, self.board_size)
        action_mask = np.ones((1, self.action_size), dtype=bool)
        
        action, log_prob, value = model.get_action(x, action_mask=action_mask)
        
        self.assertEqual(action.shape, (1,))
        self.assertEqual(log_prob.shape, (1,))
        self.assertEqual(value.shape, (1, 1))
        self.assertTrue(0 <= action.item() < self.action_size)


class TestModelUtils(unittest.TestCase):
    """Test model utility functions."""
    
    def setUp(self):
        self.board_size = 9
        self.action_size = 81
        
    def test_get_action_mask(self):
        """Test action mask creation from valid actions."""
        valid_actions = [0, 5, 10, 15, 20]
        mask = get_action_mask(valid_actions, self.action_size)
        
        self.assertEqual(len(mask), self.action_size)
        self.assertTrue(mask[0])
        self.assertTrue(mask[5])
        self.assertTrue(mask[10])
        self.assertFalse(mask[1])
        self.assertFalse(mask[2])
        
    def test_get_action_mask_from_board(self):
        """Test action mask creation from board state."""
        # Create a board with some occupied positions
        board_state = np.zeros((self.board_size, self.board_size))
        board_state[0, 0] = 1  # Occupied
        board_state[1, 1] = 2  # Occupied
        board_state[2, 2] = 1  # Occupied
        
        mask = get_action_mask_from_board(board_state, self.board_size)
        
        self.assertEqual(len(mask), self.action_size)
        # Check that occupied positions are masked
        self.assertFalse(mask[0])  # (0, 0) = action 0
        self.assertFalse(mask[10])  # (1, 1) = action 10
        self.assertFalse(mask[20])  # (2, 2) = action 20
        # Check that empty positions are valid
        self.assertTrue(mask[1])
        self.assertTrue(mask[2])
        
    def test_save_and_load_model(self):
        """Test saving and loading a model."""
        import tempfile
        import os
        
        model = ActorCritic(
            board_size=self.board_size,
            action_size=self.action_size
        )
        
        metadata = {
            'board_size': self.board_size,
            'action_size': self.action_size,
            'test_metric': 0.95
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.pt')
            
            # Save model
            save_model(model, filepath, metadata=metadata)
            
            # Verify files exist
            self.assertTrue(os.path.exists(filepath))
            self.assertTrue(os.path.exists(filepath.replace('.pt', '.json')))
            
            # Create new model and load
            new_model = ActorCritic(
                board_size=self.board_size,
                action_size=self.action_size
            )
            loaded_model, loaded_metadata = load_model(new_model, filepath)
            
            # Check that weights match
            for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
                torch.testing.assert_close(p1, p2)
            
            # Check metadata
            self.assertEqual(loaded_metadata['board_size'], self.board_size)
            self.assertEqual(loaded_metadata['test_metric'], 0.95)
            
    def test_create_model_from_config(self):
        """Test model creation from configuration."""
        config = {
            'board_size': self.board_size,
            'channels': 64,
            'num_layers': 3,
            'hidden_size': 128
        }
        
        model = create_model_from_config(config)
        
        self.assertIsInstance(model, ActorCritic)
        self.assertEqual(model.feature_extractor.board_size, self.board_size)
        self.assertEqual(model.feature_extractor.channels, 64)


if __name__ == '__main__':
    unittest.main()

