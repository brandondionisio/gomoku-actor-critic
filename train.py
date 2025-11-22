"""
Training script for Gomoku RL agent.

Supports two training modes:
1. Training against random opponent (simpler, good for initial training)
2. Self-play (agent plays against itself, leads to stronger play)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gym
import gym_gomoku

from models import ActorCritic, get_action_mask_from_board, save_model, load_model
from gym_gomoku.envs.gomoku import GomokuEnv
from gym_gomoku.envs.util import make_model_policy

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) trainer for Gomoku.
    """
    
    def __init__(
        self,
        model,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        value_coef=0.5,
        entropy_coef=0.001,
        max_grad_norm=0.5,
        device='cpu'
    ):
        """
        Args:
            model: ActorCritic model
            lr: Learning rate
            gamma: Discount factor
            eps_clip: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping threshold
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training statistics
        self.stats = {
            'episodes': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_reward': 0.0,
        }
    
    def collect_rollout(self, env, max_steps=200):
        """
        Collect a single game rollout (episode).
        
        Returns:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            log_probs: List of log probabilities
            values: List of value estimates
            dones: List of done flags
        """
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get action mask
            action_mask = get_action_mask_from_board(obs, env.board_size)
            action_mask = action_mask.reshape(1, -1)
            action_mask_tensor = torch.BoolTensor(action_mask).to(self.device)
            
            # Get action from model
            self.model.eval()
            with torch.no_grad():
                action, log_prob, value = self.model.get_action(
                    obs_tensor,
                    action_mask=action_mask_tensor,
                    deterministic=False
                )
            
            # Store data
            states.append(obs.copy())
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())
            
            # Take step
            next_obs, reward, done, info = env.step(action.item())
            
            rewards.append(reward)
            dones.append(done)
            
            obs = next_obs
            steps += 1
        
        # Update statistics
        self.stats['episodes'] += 1
        if rewards and rewards[-1] > 0:
            self.stats['wins'] += 1
        elif rewards and rewards[-1] < 0:
            self.stats['losses'] += 1
        else:
            self.stats['draws'] += 1
        self.stats['total_reward'] += sum(rewards)
        
        return states, actions, rewards, log_probs, values, dones
    
    def compute_returns(self, rewards, dones, values, next_value=0.0):
        """
        Compute discounted returns (advantages).
        
        Args:
            rewards: List of rewards
            dones: List of done flags
            values: List of value estimates
            next_value: Value estimate for next state (0 if terminal)
        
        Returns:
            returns: Discounted returns
            advantages: Advantages (returns - values)
        """
        returns = []
        advantages = []
        
        # Compute returns using GAE (Generalized Advantage Estimation)
        gae = 0
        for step in reversed(range(len(rewards))):
            if dones[step]:
                delta = rewards[step] - values[step]
                gae = delta
            else:
                delta = rewards[step] + self.gamma * next_value - values[step]
                gae = delta + self.gamma * 0.95 * gae  # lambda=0.95 for GAE
            
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)
            next_value = values[step]
        
        return np.array(returns), np.array(advantages)
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        """
        Update model using PPO algorithm.
        
        Returns:
            loss_dict: Dictionary of loss components
        """
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Get action masks
        action_masks = []
        for state in states:
            mask = get_action_mask_from_board(state, int(np.sqrt(len(state.flatten()))))
            action_masks.append(mask)
        action_masks_tensor = torch.BoolTensor(np.array(action_masks)).to(self.device)
        
        # Forward pass
        self.model.train()
        logits, probs, values = self.model(states_tensor, action_mask=action_masks_tensor)
        
        # Get log probabilities for taken actions
        log_probs = torch.log(probs.gather(1, actions_tensor.unsqueeze(1)) + 1e-8).squeeze(1)
        
        # Compute entropy
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        
        # Compute policy loss (PPO clipped objective)
        ratio = torch.exp(log_probs - old_log_probs_tensor)
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss
        value_loss = nn.MSELoss()(values.squeeze(), returns_tensor)
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
        }


def train_against_random(
    board_size=9,
    num_episodes=1000,
    save_interval=100,
    model_path='saved_models/trained_model.pt',
    device='cpu'
):
    """
    Train agent against random opponent.
    
    This is simpler and good for initial training.
    """
    print("=" * 60)
    print("Training against Random Opponent")
    print("=" * 60)
    
    # Create environment
    env = gym.make(f'Gomoku{board_size}x{board_size}-v0')
    
    # Create model
    action_size = board_size * board_size
    model = ActorCritic(
        board_size=board_size,
        action_size=action_size,
        channels=128,
        num_layers=4,
        hidden_size=256
    )
    
    # Create trainer
    trainer = PPOTrainer(model, device=device)
    
    # Training loop
    for episode in range(num_episodes):
        # Collect rollout
        states, actions, rewards, log_probs, values, dones = trainer.collect_rollout(env)
        
        if len(states) == 0:
            continue
        
        # Compute returns and advantages
        next_value = 0.0 if dones[-1] else values[-1]
        returns, advantages = trainer.compute_returns(rewards, dones, values, next_value)
        
        # Update model
        loss_dict = trainer.update(states, actions, log_probs, returns, advantages)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            win_rate = trainer.stats['wins'] / max(trainer.stats['episodes'], 1)
            avg_reward = trainer.stats['total_reward'] / max(trainer.stats['episodes'], 1)
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Win Rate: {win_rate:.2%} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Loss: {loss_dict['total_loss']:.4f}")
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            metadata = {
                'board_size': board_size,
                'episode': episode + 1,
                'win_rate': trainer.stats['wins'] / max(trainer.stats['episodes'], 1),
            }
            save_model(model, model_path, metadata=metadata)
            print(f"Saved checkpoint to {model_path}")
    
    env.close()
    print("Training complete!")


def train_self_play(
    board_size=9,
    num_episodes=1000,
    save_interval=100,
    update_opponent_interval=50,
    model_path='saved_models/selfplay_model.pt',
    device='cpu'
):
    """
    Train agent using self-play.
    
    The agent plays against itself, which leads to stronger play over time.
    Periodically updates the opponent to use the latest model.
    """
    print("=" * 60)
    print("Self-Play Training")
    print("=" * 60)
    
    # Create environment
    env = GomokuEnv(player_color='black', opponent='random', board_size=board_size)
    
    # Create model
    action_size = board_size * board_size
    model = ActorCritic(
        board_size=board_size,
        action_size=action_size,
        channels=128,
        num_layers=4,
        hidden_size=256
    )
    
    # Create trainer
    trainer = PPOTrainer(model, device=device)
    
    # Create opponent policy (starts as random, updates to model)
    opponent_policy = None
    
    # Training loop
    for episode in range(num_episodes):
        # Update opponent periodically
        if episode % update_opponent_interval == 0:
            opponent_policy = make_model_policy(model, device=device, deterministic=False)
            env.opponent_policy = opponent_policy
            print(f"Updated opponent to use current model (episode {episode})")
        
        # Collect rollout
        states, actions, rewards, log_probs, values, dones = trainer.collect_rollout(env)
        
        if len(states) == 0:
            continue
        
        # Compute returns and advantages
        next_value = 0.0 if dones[-1] else values[-1]
        returns, advantages = trainer.compute_returns(rewards, dones, values, next_value)
        
        # Update model
        loss_dict = trainer.update(states, actions, log_probs, returns, advantages)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            win_rate = trainer.stats['wins'] / max(trainer.stats['episodes'], 1)
            avg_reward = trainer.stats['total_reward'] / max(trainer.stats['episodes'], 1)
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Win Rate: {win_rate:.2%} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Loss: {loss_dict['total_loss']:.4f}")
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            metadata = {
                'board_size': board_size,
                'episode': episode + 1,
                'win_rate': trainer.stats['wins'] / max(trainer.stats['episodes'], 1),
                'training_mode': 'self-play',
            }
            save_model(model, model_path, metadata=metadata)
            print(f"Saved checkpoint to {model_path}")
    
    env.close()
    print("Training complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Gomoku RL agent')
    parser.add_argument('--mode', type=str, default='random', choices=['random', 'self-play'],
                        help='Training mode: random (vs random opponent) or self-play')
    parser.add_argument('--board-size', type=int, default=9, choices=[9, 19],
                        help='Board size (9 or 19)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Save model every N episodes')
    parser.add_argument('--update-opponent-interval', type=int, default=50,
                        help='Update opponent model every N episodes (self-play only)')
    parser.add_argument('--model-path', type=str, default='saved_models/trained_model.pt',
                        help='Path to save model')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Determine device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Train
    if args.mode == 'random':
        train_against_random(
            board_size=args.board_size,
            num_episodes=args.episodes,
            save_interval=args.save_interval,
            model_path=args.model_path,
            device=args.device
        )
    else:
        train_self_play(
            board_size=args.board_size,
            num_episodes=args.episodes,
            save_interval=args.save_interval,
            update_opponent_interval=args.update_opponent_interval,
            model_path=args.model_path,
            device=args.device
        )

