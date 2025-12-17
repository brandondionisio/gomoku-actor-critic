# Changelog

## Overview

Implemented one-hot encoding for board state representation, changed reward shaping and tuned some hyperparameters. Since I added some more rewards, the average reward will be higher than before. I added support for three channels for encodings, but we could definitely add more later; the option is available. Outside of changes to the model and environment, I made a small addition to the training script so it outputs some more metrics about the current training progress such as rolling average reward, etc. since I thought it'd be cool if we could see our bot's improvement in live time. Since there were a quite a few additions and minor changes here and there, I got Cursor to do a little before/after.

---

## File-by-File Changes

### 1. `gym_gomoku/envs/gomoku.py`

#### **Observation Encoding**
- **Before**: `Board.encode()` returned normalized single-channel array `(board_size, board_size)`
  - Values: 0.0 (empty), 0.5 (black), 1.0 (white)
- **After**: `Board.encode(use_onehot=True)` returns one-hot encoded `(3, board_size, board_size)`
  - Channel 0: Empty positions (1.0 = empty, 0.0 = occupied)
  - Channel 1: Black stones (1.0 = black, 0.0 = not black)
  - Channel 2: White stones (1.0 = white, 0.0 = not white)
  - Maintains backward compatibility with `use_onehot=False` parameter

#### **Observation Space**
- **Before**: `spaces.Box(low=0.0, high=1.0, shape=(board_size, board_size))`
- **After**: `spaces.Box(low=0.0, high=1.0, shape=(3, board_size, board_size))`

#### **Reward Shaping Enhancements**
Enhanced reward structure to provide better learning signals:

| Reward Type | Before | After | Change |
|------------|--------|-------|--------|
| Block opponent open four | +0.15 | +0.20 | Increased |
| Block opponent open three | N/A | +0.08 | **NEW** |
| Create open four | +0.25 | +0.30 | Increased |
| Create closed four | N/A | +0.15 | **NEW** |
| Create open three | +0.05 | +0.08 | Increased |
| Center control (within 2 moves) | N/A | +0.02 | **NEW** |
| Step penalty | -0.001 | -0.0005 | Reduced |

#### **New Methods**
- `_check_opponent_three_in_row()`: Detects and rewards blocking opponent's open three threats

---

### 2. `gym_gomoku/envs/util.py`

#### **New Functions Added**

**`encode_board_onehot(board_state)`**
- Converts 2D board state to 3-channel one-hot encoding
- Returns: `(3, board_size, board_size)` numpy array
- Each channel represents one state: empty, black, or white

**`extract_pattern_features(board_state, player_color)`**
- Extracts 4 hand-crafted pattern features:
  1. Player open three (binary: exists or not)
  2. Player open four (binary: exists or not)
  3. Opponent open three (binary: exists or not)
  4. Opponent open four (binary: exists or not)
- Returns: `(4,)` numpy array of float32

**`build_feature_vector(board_state, player_color)`**
- Combines one-hot encoded board with pattern features
- Returns: Flattened concatenation of both feature types
- Currently not used in main training loop (features passed separately)

#### **Updated Functions**

**`make_model_policy(model, device, deterministic)`**
- **Before**: Used normalized single-channel encoding
- **After**: 
  - Uses one-hot encoding (`board.encode(use_onehot=True)`)
  - Extracts pattern features using `extract_pattern_features()`
  - Passes pattern features as `extra_features` to model
  - Handles all model types (ActorCritic, ActorNetwork, etc.)

---

### 3. `models/networks.py`

#### **CNNFeatureExtractor**

**Changes:**
- Added `input_channels=3` parameter (default: 3 for one-hot encoding)
- **Before**: Expected 1-channel input `(batch, 1, board_size, board_size)`
- **After**: Handles both:
  - 3-channel one-hot input `(batch, 3, board_size, board_size)` - primary
  - 1-channel legacy input `(batch, 1, board_size, board_size)` - backward compatibility

**Forward Method:**
- Auto-detects input dimensions
- Converts 3D input to 4D if needed
- Handles dimension mismatches gracefully

#### **ActorNetwork**

**Changes:**
- Added `extra_feature_size=0` parameter
- **Before**: Only processed CNN features
- **After**: 
  - Accepts `extra_features` parameter in `forward()` and `get_action()`
  - Concatenates CNN features with pattern features: `torch.cat([cnn_features, pattern_features])`
  - Policy head input size accounts for extra features: `cnn_feature_size + extra_feature_size`
  - Handles missing `extra_features` by using zero-filled tensors

**Key Methods:**
- `forward(x, action_mask, extra_features)`: Now accepts and uses pattern features
- `get_action(x, action_mask, deterministic, extra_features)`: Passes features through

#### **CriticNetwork**

**Changes:**
- Added `extra_feature_size=0` parameter
- **Before**: Only processed CNN features
- **After**:
  - Accepts `extra_features` parameter in `forward()`
  - Same concatenation logic as ActorNetwork
  - Value head input size accounts for extra features

#### **ActorCritic**

**Changes:**
- Added `input_channels=3` parameter (default: 3)
- Added `extra_feature_size=4` parameter (default: 4 for pattern features)
- **Before**: Single-channel input, no extra features
- **After**: 
  - 3-channel one-hot input by default
  - Passes `extra_features` to both actor and critic networks
  - `forward()` and `get_action()` accept `extra_features` parameter

---

### 4. `models/model_utils.py`

#### **`get_action_mask_from_board()`**

**Changes:**
- **Before**: Only handled 2D board arrays `(board_size, board_size)`
- **After**: Handles both formats:
  - 2D array `(board_size, board_size)`: Legacy format, values 0/1/2
  - 3D array `(3, board_size, board_size)`: One-hot format, extracts channel 0 (empty channel)
- Backward compatible with existing code

**Logic:**
```python
if board_state.ndim == 3 and board_state.shape[0] == 3:
    # One-hot: extract empty channel
    empty_channel = board_state[0]
    empty_positions = np.where(empty_channel.flatten() == 1.0)[0]
else:
    # Legacy: direct check
    empty_positions = np.where(flat_board == 0)[0]
```

---

### 5. `train.py`

#### **Hyperparameter Changes**

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| Learning Rate (`lr`) | 3e-4 | 1.5e-4 | Better stability |
| Discount Factor (`gamma`) | 0.99 | 0.95 | Faster reward signal |
| Value Coefficient (`value_coef`) | 0.5 | 1.0 | Stabilize value function |
| Entropy Coefficient (`entropy_coef`) | 0.001 | 0.015 | More exploration |
| Opponent Update Interval | 50 | 200 | Reduce catastrophic forgetting |

#### **Statistics Tracking**

**Before:**
- Basic stats: `episodes`, `wins`, `losses`, `draws`, `total_reward`

**After:**
- Enhanced stats using `collections.deque`:
  - `episode_lengths`: Last 100 episode lengths
  - `recent_rewards`: Last 100 episode rewards (for rolling average)
  - `recent_losses`: Last 100 loss dictionaries
- Efficient O(1) operations with `deque(maxlen=100)`

#### **Progress Reporting**

**Before:**
```python
print(f"Episode {episode + 1}/{num_episodes} | "
      f"Win Rate: {win_rate:.2%} | "
      f"Avg Reward: {avg_reward:.3f} | "
      f"Loss: {loss_dict['total_loss']:.4f}")
```

**After:**
```python
print(f"\nEpisode {episode + 1}/{num_episodes}")
print(f"  Win Rate: {win_rate:.2%} | Avg Reward: {avg_reward:.3f} | Rolling Avg: {rolling_avg_reward:.3f}")
print(f"  Episode Length: {rolling_avg_length:.1f} moves | Advantages: {avg_advantage:.3f} ± {std_advantage:.3f}")
print(f"  Losses - Total: {total_loss:.4f} | Policy: {policy_loss:.4f} | Value: {value_loss:.4f} | Entropy: {entropy:.4f}")
```

**New Metrics:**
- Rolling average reward (last 20 episodes)
- Episode length (average moves per game)
- Advantages (mean ± standard deviation)
- Detailed loss breakdown (total, policy, value, entropy)

#### **`collect_rollout()` Method**

**Changes:**
- **Before**: Returned `(states, actions, rewards, log_probs, values, dones)`
- **After**: Returns `(states, actions, rewards, log_probs, values, dones, pattern_features, board_states)`

**New Functionality:**
- Extracts pattern features during rollout: `gomoku_util.extract_pattern_features()`
- Stores raw board states for efficient action mask generation
- Passes pattern features to model: `model.get_action(..., extra_features=pattern_features_tensor)`
- Tracks episode length and reward for statistics

#### **`update()` Method**

**Changes:**
- **Before**: Reconstructed board states from one-hot encoding (slow)
- **After**: Uses pre-stored `board_states` parameter (fast)

**Performance Optimization:**
```python
# Fast path: use pre-stored board states
if board_states is not None:
    action_masks = [get_action_mask_from_board(bs, board_size) for bs in board_states]
else:
    # Fallback: reconstruct from one-hot (slower)
    # ... reconstruction code ...
```

**New Functionality:**
- Accepts `pattern_features` parameter
- Converts pattern features to tensor and passes to model
- Tracks recent losses in statistics

#### **Model Initialization**

**Before:**
```python
model = ActorCritic(
    board_size=board_size,
    action_size=action_size,
    channels=128,
    num_layers=4,
    hidden_size=256
)
```

**After:**
```python
model = ActorCritic(
    board_size=board_size,
    action_size=action_size,
    channels=128,
    num_layers=4,
    hidden_size=256,
    input_channels=3,  # One-hot encoding: 3 channels
    extra_feature_size=4  # Pattern features: 4 features
)
```

---

### 6. `play_gomoku.py`

#### **Bug Fix: Model Architecture Mismatch**

**Issue:**
- Model was created without required `input_channels=3` and `extra_feature_size=4` parameters
- This caused architecture mismatch when loading trained models, leading to:
  - Incorrect weight loading
  - Wrong input/output dimensions
  - Poor or random gameplay behavior
  - Model unable to connect pieces or play strategically

**Fix:**
- **Before**: Model created with default parameters (missing `input_channels` and `extra_feature_size`)
```python
model = ActorCritic(
    board_size=board_size,
    action_size=board_size * board_size,
    channels=128,
    num_layers=4,
    hidden_size=256
    # Missing input_channels and extra_feature_size
)
```

- **After**: Model created with correct architecture parameters matching training
```python
model = ActorCritic(
    board_size=board_size,
    action_size=board_size * board_size,
    channels=128,
    num_layers=4,
    hidden_size=256,
    input_channels=3,  # One-hot encoding: 3 channels
    extra_feature_size=4  # Pattern features: 4 features
)
```

**Impact:**
- Models now load correctly with matching architecture
- Trained models can now play games properly
- Model can utilize pattern features for strategic play

---

## New Files

### `.gitignore`

Prevents committing (Since I had to use a venv to run the code):
- Virtual environments (`gomoku-env/`, `venv/`, etc.)
- Python cache files (`__pycache__/`, `*.pyc`)
- IDE files (`.vscode/`, `.idea/`)
- OS files (`.DS_Store`, `Thumbs.db`)
- Model checkpoints (optional, currently tracked)

---

## Architecture Summary

### **Before:**
```
Board State (2D) → Normalize → CNN (1 channel) → Policy/Value Heads
```

### **After:**
```
Board State (2D) → One-Hot Encode (3 channels) → CNN (3 channels) → Concatenate with Pattern Features (4) → Policy/Value Heads
```

### **Feature Flow:**
1. **Board Encoding**: `board_state` → `encode_board_onehot()` → `(3, board_size, board_size)`
2. **Pattern Extraction**: `board_state` + `player_color` → `extract_pattern_features()` → `(4,)`
3. **CNN Processing**: One-hot board → `CNNFeatureExtractor` → `(10368,)` for 9x9 board
4. **Feature Fusion**: CNN features + Pattern features → Concatenation → `(10372,)`
5. **Decision Making**: Fused features → Policy/Value heads → Action probabilities + Value estimate