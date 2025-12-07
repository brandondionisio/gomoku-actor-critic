'''Gomoku Rule and Policy Util
@author: Xichen Ding
@date: 2017/2/13
'''

import numpy as np
import gymnasium as gym
from gym import spaces
from gym import error
from gym.utils import seeding
from six import StringIO
import sys
import six

class GomokuUtil(object):
    
    def __init__(self):
        # default setting
        self.BLACK = 'black'
        self.WHITE = 'white'
        self.color = [self.BLACK, self.WHITE]
        self.color_dict = {'empty': 0, 'black': 1, 'white': 2}
        self.color_dict_rev = {v: k for k, v in self.color_dict.items()}
        self.color_shape = {0: '.', 1: 'X', 2: 'O'}
    
    def other_color(self, color):
        '''Return the opositive color of the current player's color
        '''
        assert color in self.color, 'Invalid player color'
        opposite_color = self.color[0] if color == self.color[1] else self.color[1]
        return opposite_color
    
    def iterator(self, board_state):
        ''' Iterator for 2D list board_state
            Return: Row, Column, diagnoal, list of coordinate tuples, [(x1, y1), (x2, y2), ...,()], (6n-2-16) lines
        '''
        list = []
        size = len(board_state)
        
        # row
        for i in range(size): # [(i,0), (i,1), ..., (i,n-1)]
            list.append([(i, j) for j in range(size)])
        
        # column
        for j in range(size):
            list.append([(i, j) for i in range(size)])
        
        # diagonal: left triangle
        for k in range(size):
            # lower_line consist items [k][0], [k-1][1],...,[0][k]
            # upper_line consist items [size-1][size-1-k], [size-1-1][size-1-k +1],...,[size-1-k][size-1]
            lower_line = [((k-k1), k1) for k1 in range(k+1)]
            upper_line = [((size-1-k2), (size-k-1+k2)) for k2 in range(k+1)]
            if (k == (size-1)): # one diagnoal, lower_line same as upper_line
                list.append(lower_line)
            else :
                if (len(lower_line)>=5):
                    list.append(lower_line)
                if (len(upper_line)>=5):
                    list.append(upper_line)
        
        # diagonal: right triangle
        for k in range(size):
            # lower_line consist items [0][k], [1][k+1],...,[size-1-k][size-1]
            # upper_line consist items [k][0], [k+1][1],...,[size-1][size-1-k]
            lower_line = [(k1, k + k1) for k1 in range(size-k)]
            upper_line = [(k + k2, k2) for k2 in range(size-k)]
            if (k == 0): # one diagnoal, lower_line same as upper_line
                list.append(lower_line)
            else :
                if (len(lower_line)>=5):
                    list.append(lower_line)
                if (len(upper_line)>=5):
                    list.append(upper_line)
        
        for line in list:
            yield line
    
    def value(self, board_state, coord_list):
        ''' Fetch Value from 2D list with coord_list
        '''
        val = []
        for (i,j) in coord_list:
            val.append(board_state[i][j])
        return val
    
    def check_five_in_row(self, board_state):
        ''' Args: board_state 2D list
            Return: exist, color
        '''
        size = len(board_state)
        black_pattern = [self.color_dict[self.BLACK] for _ in range(5)] # [1,1,1,1,1]
        white_pattern = [self.color_dict[self.WHITE] for _ in range(5)] # [2,2,2,2,2]
        
        exist_final = False
        color_final = "empty"
        black_win, _ = self.check_pattern(board_state, black_pattern)
        white_win, _ = self.check_pattern(board_state, white_pattern)
        
        if (black_win and white_win):
            raise error.Error('Both Black and White has 5-in-row, rules conflicts')
        # Check if there is any one party wins
        if not (black_win or white_win):
            return exist_final, "empty"
        else:
            exist_final = True
        if (black_win):
            return exist_final, self.BLACK
        if (white_win):
            return exist_final, self.WHITE
    
    def check_board_full(self, board_state):
        is_full = True
        size = len(board_state)
        for i in range(size):
            for j in range(size):
                if (board_state[i][j]==0):
                    is_full = False
                    break
        return is_full
    
    def check_pattern(self, board_state, pattern):
        ''' Check if pattern exist in the board_state lines,
            Return: exist: boolean
                    line: coordinates that contains the patterns
        '''
        exist = False
        pattern_found = [] # there maybe multiple patterns found
        for coord in self.iterator(board_state):
            line_value = self.value(board_state, coord)
            if (self.is_sublist(line_value, pattern)):
                exist = True
                pattern_found.append(coord)
        return exist, pattern_found
    
    def check_pattern_index(self, board_state, pattern):
        '''Return the line contains the pattern, and its start position index of the pattern
        '''
        start = -1
        startlist = []
        exist_patttern, lines = self.check_pattern(board_state, pattern)
        if (exist_patttern):
            for line in lines:
                start = self.index(self.value(board_state, line), pattern)
                startlist.append(start)
            return lines, startlist  # line: list[list[(x1, y1),...]], startlist: list[int]
        else: # pattern not found
            return None, startlist
    
    def is_sublist(self, list, sublist):
        l1 = len(list)
        l2 = len(sublist)
        is_sub = False
        for i in range(l1):
            curSub = list[i: min(i+l2, l1)]
            if (curSub == sublist): # check list equal
                is_sub = True
                break
        return is_sub
    
    def index(self, list, sublist):
        ''' Return the starting index of the sublist in the list
        '''
        idx = - 1
        l1 = len(list)
        l2 = len(sublist)
        
        for i in range(l1):
            curSub = list[i: min(i+l2, l1)]
            if (curSub == sublist):
                idx = i
                break
        return idx
    
    # One-hot board encoding (3 channels: empty, black, white)
    def encode_board_onehot(self, board_state):
        size = len(board_state)
        onehot = np.zeros((3, size, size), dtype=np.float32)
        for i in range(size):
            for j in range(size):
                cell = board_state[i][j]    # 0 empty, 1 black, 2 white
                onehot[cell, i, j] = 1.0
        return onehot

    # Count useful patterns as extra features
    def extract_pattern_features(self, board_state, player_color):
        """
        Create hand-crafted features:
        - open three for player
        - open four for player
        - same but for opponent
        """

        opp = self.other_color(player_color)

        player_val = self.color_dict[player_color]
        opp_val    = self.color_dict[opp]

        # patterns
        open_three        = [0, player_val, player_val, player_val, 0]
        open_four         = [player_val]*4
        opp_open_three    = [0, opp_val, opp_val, opp_val, 0]
        opp_open_four     = [opp_val]*4

        f = []
        f.append(int(self.check_pattern(board_state, open_three)[0]))
        f.append(int(self.check_pattern(board_state, open_four)[0]))
        f.append(int(self.check_pattern(board_state, opp_open_three)[0]))
        f.append(int(self.check_pattern(board_state, opp_open_four)[0]))

        return np.array(f, dtype=np.float32)

    # Combine everything into one feature vector
    def build_feature_vector(self, board_state, player_color):
        """
        Combine:
        - One-hot encoded board  → (3 × size × size)
        - Extra ANN features     → 4 binary features
        """
        onehot = self.encode_board_onehot(board_state)
        extra  = self.extract_pattern_features(board_state, player_color)

        onehot_flat = onehot.reshape(-1)
        return np.concatenate([onehot_flat, extra])

gomoku_util = GomokuUtil()

def make_random_policy(np_random):
    ''' Get the random action ID of all the empty legal moves, prev_state and prev_action not used
    '''
    def random_policy(curr_state, prev_state, prev_action):
        b = curr_state.board
        legal_moves = b.get_legal_move()
        next_move = legal_moves[np_random.choice(len(legal_moves))]
        return b.coord_to_action(next_move[0], next_move[1])
    return random_policy


def make_model_policy(model, device='cpu', deterministic=False):
    """
    Create an opponent policy function that uses a trained neural network model.
    
    Args:
        model: Trained ActorCritic model (or ActorNetwork)
        device: Device to run the model on ('cpu' or 'cuda')
        deterministic: If True, always selects the best action. If False, samples from distribution.
    
    Returns:
        A policy function that matches the signature: (curr_state, prev_state, prev_action) -> action
    """
    import torch
    from models.model_utils import get_action_mask_from_board
    
    def model_policy(curr_state, prev_state, prev_action):
        """
        Policy function that uses the neural network model to select actions.
        
        Args:
            curr_state: Current GomokuState (opponent's turn)
            prev_state: Previous GomokuState (not used by model)
            prev_action: Previous action (not used by model)
        
        Returns:
            action: Integer action index
        """
        # Get the board state
        board = curr_state.board
        board_size = board.size
        
        # Encode board to one-hot numpy array
        obs = board.encode(use_onehot=True)  # Shape: (3, board_size, board_size)
        
        # Convert to tensor and add batch dimension
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)  # Shape: (1, 3, board_size, board_size)
        
        # Get action mask (valid moves) - use raw board_state for mask
        from models.model_utils import get_action_mask_from_board
        board_state_2d = np.array(board.board_state, dtype=np.float32)
        action_mask = get_action_mask_from_board(board_state_2d, board_size)
        action_mask = action_mask.reshape(1, -1)  # Shape: (1, action_size)
        action_mask_tensor = torch.BoolTensor(action_mask).to(device)
        
        # Extract pattern features for extra_features
        from gym_gomoku.envs.util import gomoku_util
        # Determine player color from state
        player_color = curr_state.color
        pattern_features = gomoku_util.extract_pattern_features(board.board_state, player_color)
        pattern_features_tensor = torch.FloatTensor(pattern_features).unsqueeze(0).to(device)  # Shape: (1, 4)
        
        # Get action from model
        model.eval()  # Ensure model is in eval mode
        with torch.no_grad():
            if hasattr(model, 'get_action'):
                # ActorCritic model
                action, _, _ = model.get_action(
                    obs_tensor, 
                    action_mask=action_mask_tensor, 
                    deterministic=deterministic,
                    extra_features=pattern_features_tensor
                )
            elif hasattr(model, 'actor'):
                # ActorCritic model (alternative access)
                action, _ = model.actor.get_action(
                    obs_tensor,
                    action_mask=action_mask_tensor,
                    deterministic=deterministic,
                    extra_features=pattern_features_tensor
                )
            else:
                # Just ActorNetwork
                action, _ = model.get_action(
                    obs_tensor,
                    action_mask=action_mask_tensor,
                    deterministic=deterministic,
                    extra_features=pattern_features_tensor
                )
        
        # Convert tensor to Python int
        action_int = action.item() if isinstance(action, torch.Tensor) else action
        
        return action_int
    
    return model_policy
