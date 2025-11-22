"""
Test script to verify the model is actually being used and not random.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import torch
import numpy as np
import gym
import gym_gomoku

from models import ActorCritic, load_model, get_action_mask_from_board
from gym_gomoku.envs.gomoku import GomokuEnv
from gym_gomoku.envs.util import make_model_policy

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_


def test_model_consistency(model_path, board_size=9):
    """
    Test if model makes consistent (non-random) moves.
    If deterministic, same board state should always produce same action.
    """
    print("=" * 60)
    print("Testing Model Consistency")
    print("=" * 60)
    
    # Load model
    model = ActorCritic(
        board_size=board_size,
        action_size=board_size * board_size,
        channels=128,
        num_layers=4,
        hidden_size=256
    )
    model, metadata = load_model(model, model_path)
    model.eval()
    
    print(f"Model loaded: {model_path}")
    if metadata:
        print(f"Metadata: {metadata}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Create a test board state
    env = gym.make(f'Gomoku{board_size}x{board_size}-v0')
    obs, info = env.reset()
    
    # Take a few moves to create an interesting position
    for _ in range(3):
        action = env.action_space.sample()
        obs, _, done, _ = env.step(action)
        if done:
            obs, info = env.reset()
    
    print(f"\nTest board state:")
    env.render()
    
    # Test deterministic mode - should get same action every time
    print("\nTesting deterministic mode (should be consistent):")
    actions_deterministic = []
    for i in range(5):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        action_mask = get_action_mask_from_board(obs, board_size)
        action_mask_tensor = torch.BoolTensor(action_mask.reshape(1, -1)).to(device)
        
        with torch.no_grad():
            action, _, value = model.get_action(
                obs_tensor,
                action_mask=action_mask_tensor,
                deterministic=True
            )
        actions_deterministic.append(action.item())
        print(f"  Run {i+1}: Action {action.item()}, Value: {value.item():.4f}")
    
    if len(set(actions_deterministic)) == 1:
        print("✓ Deterministic mode works - same action every time")
    else:
        print("✗ Deterministic mode inconsistent - model may have issues")
    
    # Test stochastic mode - should vary but not be completely random
    print("\nTesting stochastic mode (should vary but show preferences):")
    actions_stochastic = []
    for i in range(10):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        action_mask = get_action_mask_from_board(obs, board_size)
        action_mask_tensor = torch.BoolTensor(action_mask.reshape(1, -1)).to(device)
        
        with torch.no_grad():
            action, log_prob, value = model.get_action(
                obs_tensor,
                action_mask=action_mask_tensor,
                deterministic=False
            )
        actions_stochastic.append(action.item())
    
    unique_actions = len(set(actions_stochastic))
    print(f"  Unique actions in 10 runs: {unique_actions}/10")
    if unique_actions < 10:
        print("✓ Model shows preferences (not completely random)")
    else:
        print("? Model varies a lot - might be under-trained")
    
    # Check action probabilities
    print("\nChecking action probabilities:")
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    action_mask = get_action_mask_from_board(obs, board_size)
    action_mask_tensor = torch.BoolTensor(action_mask.reshape(1, -1)).to(device)
    
    with torch.no_grad():
        logits, probs, value = model(obs_tensor, action_mask=action_mask_tensor)
    
    # Get top 5 actions
    probs_np = probs[0].cpu().numpy()
    top_5_indices = np.argsort(probs_np)[-5:][::-1]
    top_5_probs = probs_np[top_5_indices]
    
    print("  Top 5 action probabilities:")
    for idx, prob in zip(top_5_indices, top_5_probs):
        row = idx // board_size
        col = idx % board_size
        print(f"    Action {idx} ({chr(ord('A')+col)}{row+1}): {prob:.4f}")
    
    if top_5_probs[0] > 0.3:
        print("✓ Model has strong preferences (good sign)")
    elif top_5_probs[0] > 0.1:
        print("? Model has moderate preferences")
    else:
        print("✗ Model probabilities are too uniform (may be under-trained)")
    
    env.close()


def test_model_vs_random(model_path, board_size=9, num_games=20):
    """
    Test if model actually beats random opponent.
    """
    print("\n" + "=" * 60)
    print("Testing Model vs Random Opponent")
    print("=" * 60)
    
    # Load model
    model = ActorCritic(
        board_size=board_size,
        action_size=board_size * board_size,
        channels=128,
        num_layers=4,
        hidden_size=256
    )
    model, _ = load_model(model, model_path)
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Create environment with model as opponent
    env = GomokuEnv(player_color='black', opponent='random', board_size=board_size)
    model_policy = make_model_policy(model, device=device, deterministic=True)
    env.opponent_policy = model_policy
    
    wins = 0
    losses = 0
    draws = 0
    
    print(f"\nPlaying {num_games} games (you=random, model=opponent)...")
    
    for game in range(num_games):
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 200:
            # Random action for "you"
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            steps += 1
        
        if reward == -1:  # Model won
            wins += 1
        elif reward == 1:  # Random won
            losses += 1
        else:
            draws += 1
        
        if (game + 1) % 5 == 0:
            print(f"  Games {game+1}/{num_games}: Model wins: {wins}, Losses: {losses}, Draws: {draws}")
    
    win_rate = wins / num_games
    print(f"\nResults:")
    print(f"  Model win rate: {win_rate:.2%}")
    print(f"  Model wins: {wins}, Random wins: {losses}, Draws: {draws}")
    
    if win_rate > 0.6:
        print("✓ Model is clearly better than random")
    elif win_rate > 0.5:
        print("? Model is slightly better than random")
    else:
        print("✗ Model is not better than random - may need more training")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='saved_models/trained_model.pt',
                        help='Path to model file')
    parser.add_argument('--board-size', type=int, default=9, choices=[9, 19])
    parser.add_argument('--test-consistency', action='store_true',
                        help='Test if model makes consistent moves')
    parser.add_argument('--test-vs-random', action='store_true',
                        help='Test if model beats random')
    
    args = parser.parse_args()
    
    if not args.test_consistency and not args.test_vs_random:
        # Run both by default
        args.test_consistency = True
        args.test_vs_random = True
    
    if args.test_consistency:
        test_model_consistency(args.model_path, args.board_size)
    
    if args.test_vs_random:
        test_model_vs_random(args.model_path, args.board_size)

