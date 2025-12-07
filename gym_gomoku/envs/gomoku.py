import numpy as np
import gymnasium as gym
from gym import spaces
from gym import error
from gym.utils import seeding
from six import StringIO
import sys, os
import six

from gym_gomoku.envs.util import gomoku_util
from gym_gomoku.envs.util import make_random_policy

class GomokuState(object):
    '''
    Similar to Go game, Gomoku state consists of a current player and a board.
    Actions are exposed as integers in [0, num_actions), which is to place stone on empty intersection
    '''
    def __init__(self, board, color):
        '''
        Args:
            board: current board
            color: color of current player
        '''
        assert color in ['black', 'white'], 'Invalid player color'
        self.board, self.color = board, color
    
    def act(self, action):
        '''
        Executes an action for the current player
        
        Returns:
            a new GomokuState with the new board and the player switched
        '''
        return GomokuState(self.board.play(action, self.color), gomoku_util.other_color(self.color))
    
    def __repr__(self):
        '''stream of board shape output'''
        # To Do: Output shape * * * o o
        return 'To play: {}\n{}'.format(six.u(self.color), self.board.__repr__())

# Sampling without replacement Wrapper 
# sample() method will only sample from valid spaces
class DiscreteWrapper(spaces.Discrete):
    def __init__(self, n):
        super().__init__(n)  # Properly initialize parent class
        self.valid_spaces = list(range(n))
    
    def sample(self):
        '''Only sample from the remaining valid spaces
        '''
        if len(self.valid_spaces) == 0:
            print ("Space is empty")
            return None
        np_random, _ = seeding.np_random()
        # Use integers() for newer NumPy, fallback to randint() for older versions
        if hasattr(np_random, 'integers'):
            randint = np_random.integers(len(self.valid_spaces))
        else:
            randint = np_random.randint(len(self.valid_spaces))
        return self.valid_spaces[randint]
    
    def remove(self, s):
        '''Remove space s from the valid spaces
        '''
        if s is None:
            return
        if s in self.valid_spaces:
            self.valid_spaces.remove(s)
        else:
            print ("space %d is not in valid spaces" % s)


### Environment
class GomokuEnv(gym.Env):
    '''
    GomokuEnv environment. Play against a fixed opponent.
    '''
    metadata = {"render_modes": ["human", "ansi"], "render.modes": ["human", "ansi"]}
    
    def __init__(self, player_color, opponent='random', board_size=9):
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
            opponent: Opponent policy (defaults to 'random', other options ignored)
            board_size: board_size of the board to use
        """
        self.board_size = board_size
        self.player_color = player_color
        
        self._seed()
        
        # opponent
        self.opponent_policy = None
        self.opponent = opponent
        
        # Observation space on board
        # Using one-hot encoding: (3, board_size, board_size)
        # Channels: [empty, black, white] - each channel is binary (0 or 1)
        shape = (3, self.board_size, self.board_size)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=shape, dtype=np.float32
        )
        
        # One action for each board position
        self.action_space = DiscreteWrapper(self.board_size**2)
        
        # Keep track of the moves
        self.moves = []
        
        # Empty State
        self.state = None
        
        # reset the board during initialization
        self._reset()
    
    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed.
        # hash_seed was removed in newer Gym versions, use simple hash instead
        seed2 = hash(seed1 + 1) % 2**32
        return [seed1, seed2]
    
    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observation and info."""
        if seed is not None:
            self._seed(seed)
        obs = self._reset()
        info = {}
        return obs, info
    
    def _reset(self):
        self.state = GomokuState(Board(self.board_size), gomoku_util.BLACK) # Black Plays First
        # Only reset opponent if it hasn't been manually set (e.g., to a model)
        if self.opponent_policy is None:
            self._reset_opponent(self.state.board) # (re-initialize) the opponent,
        self.moves = []
        
        # Let the opponent play if it's not the agent's turn, there is no resign in Gomoku
        if self.state.color != self.player_color:
            self.state, _ = self._exec_opponent_play(self.state, None, None)
            opponent_action_coord = self.state.board.last_coord
            self.moves.append(opponent_action_coord)
        
        # We should be back to the agent color
        assert self.state.color == self.player_color
        
        # reset action_space
        self.action_space = DiscreteWrapper(self.board_size**2)
        
        self.done = self.state.board.is_terminal()
        return self.state.board.encode(use_onehot=True)
    
    def _close(self):
        self.opponent_policy = None
        self.state = None
    
    def render(self, mode="human"):
        """Render the environment."""
        return self._render(mode)
    
    def _render(self, mode="human", close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(repr(self.state) + '\n')
        return outfile
    
    def step(self, action):
        """Run one timestep of the environment's dynamics."""
        return self._step(action)
    
    def _step(self, action):
        assert self.state.color == self.player_color

        # ---------- ILLEGAL MOVE ----------
        if action not in self.action_space:
            self.done = True
            return self.state.board.encode(use_onehot=True), -1.0, True, {
                'state': self.state,
                'illegal_move': True
            }

        # ---------- TERMINAL CHECK ----------
        if self.done:
            return self.state.board.encode(use_onehot=True), 0.0, True, {'state': self.state}

        # ---------- PLAYER MOVE ----------
        prev_state = self.state
        self.state = self.state.act(action)
        self.moves.append(self.state.board.last_coord)
        self.action_space.remove(action)

        # ---------- OPPONENT MOVE ----------
        if not self.state.board.is_terminal():
            self.state, opponent_action = self._exec_opponent_play(
                self.state, prev_state, action
            )
            self.moves.append(self.state.board.last_coord)
            self.action_space.remove(opponent_action)
            assert self.state.color == self.player_color

        # ---------- REWARD SHAPING ----------
        intermediate_reward = 0.0

        board_before = prev_state.board.board_state
        board_after  = self.state.board.board_state
        player_color = gomoku_util.color_dict[self.player_color]

        # Convert flat action → (r,c)
        if isinstance(action, tuple):
            r, c = action
        else:
            n = len(board_after)
            r, c = divmod(action, n)

        DIRECTIONS = [(0,1), (1,0), (1,1), (1,-1)]

        def in_bounds(x, y):
            return 0 <= x < len(board_after) and 0 <= y < len(board_after[0])

        # ✅ BLOCK OPPONENT OPEN FOUR
        if self._check_opponent_four_in_row(board_before, action):
            intermediate_reward += 0.15

        # ✅ CREATE OPEN FOUR
        for dr, dc in DIRECTIONS:
            count = 1

            i = 1
            while in_bounds(r+i*dr, c+i*dc) and board_after[r+i*dr][c+i*dc] == player_color:
                count += 1
                i += 1
            open_1 = in_bounds(r+i*dr, c+i*dc) and board_after[r+i*dr][c+i*dc] == 0

            i = 1
            while in_bounds(r-i*dr, c-i*dc) and board_after[r-i*dr][c-i*dc] == player_color:
                count += 1
                i += 1
            open_2 = in_bounds(r-i*dr, c-i*dc) and board_after[r-i*dr][c-i*dc] == 0

            if count == 4 and open_1 and open_2:
                intermediate_reward += 0.25

        # ✅ CREATE OPEN THREE
        for dr, dc in DIRECTIONS:
            count = 1

            i = 1
            while in_bounds(r+i*dr, c+i*dc) and board_after[r+i*dr][c+i*dc] == player_color:
                count += 1
                i += 1
            open_1 = in_bounds(r+i*dr, c+i*dc) and board_after[r+i*dr][c+i*dc] == 0

            i = 1
            while in_bounds(r-i*dr, c-i*dc) and board_after[r-i*dr][c-i*dc] == player_color:
                count += 1
                i += 1
            open_2 = in_bounds(r-i*dr, c-i*dc) and board_after[r-i*dr][c-i*dc] == 0

            if count == 3 and open_1 and open_2:
                intermediate_reward += 0.05

        # ✅ STEP PENALTY
        intermediate_reward -= 0.001

        # ---------- NON-TERMINAL ----------
        if not self.state.board.is_terminal():
            self.done = False
            return self.state.board.encode(use_onehot=True), intermediate_reward, False, {'state': self.state}

        # ---------- TERMINAL ----------
        self.done = True
        exist, win_color = gomoku_util.check_five_in_row(
            self.state.board.board_state
        )

        if win_color == "empty":
            final_reward = 0.0
        else:
            final_reward = 1.0 if self.player_color == win_color else -1.0

        total_reward = final_reward + intermediate_reward
        
        return self.state.board.encode(use_onehot=True), total_reward, True, {'state': self.state}
    
    def _check_opponent_four_in_row(self, board_state_before_move, action):
        """
        Check if the given action blocks an opponent's 4-in-a-row threat.
        
        Args:
            board_state_before_move: Board state before the player's move (2D list)
            action: Action that was just played
        
        Returns:
            True if action blocks opponent's 4-in-a-row, False otherwise
        """
        opponent_color = gomoku_util.other_color(self.player_color)
        opponent_color_val = gomoku_util.color_dict[opponent_color]
        
        # Patterns for 4-in-a-row that need blocking:
        # [0, X, X, X, X] or [X, X, X, X, 0] where X is opponent's color
        pattern_four_a = [0] + [opponent_color_val] * 4  # [0, X, X, X, X]
        pattern_four_b = [opponent_color_val] * 4 + [0]  # [X, X, X, X, 0]
        
        # Get coordinate of the action
        board = Board(self.board_size)
        board.copy(board_state_before_move)
        coord = board.action_to_coord(action)
        
        # Check all lines containing this coordinate
        for line in gomoku_util.iterator(board_state_before_move):
            if coord in line:
                line_values = gomoku_util.value(board_state_before_move, line)
                # Check if this line has a 4-in-a-row pattern with empty at our action position
                if gomoku_util.is_sublist(line_values, pattern_four_a):
                    # Find where the pattern starts
                    pattern_start = gomoku_util.index(line_values, pattern_four_a)
                    if pattern_start >= 0:
                        # Check if our action is at the empty position (index 0 in pattern)
                        if pattern_start < len(line) and line[pattern_start] == coord:
                            return True
                if gomoku_util.is_sublist(line_values, pattern_four_b):
                    # Find where the pattern starts
                    pattern_start = gomoku_util.index(line_values, pattern_four_b)
                    if pattern_start >= 0:
                        # Check if our action is at the empty position (index 4 in pattern)
                        if pattern_start + 4 < len(line) and line[pattern_start + 4] == coord:
                            return True
        return False
    
    def _check_player_three_in_row(self, board_state_after_move, action):
        """
        Check if the given action creates a 3-in-a-row for the player that is open on both sides.
        Only rewards threats that are open on both sides (stronger threat).
        
        Args:
            board_state_after_move: Board state after the player's move (2D list)
            action: Action that was just played
        
        Returns:
            True if action creates player's 3-in-a-row open on both sides, False otherwise
        """
        player_color_val = gomoku_util.color_dict[self.player_color]
        
        # Pattern for 3-in-a-row open on both sides: [0, X, X, X, 0]
        # This is a stronger threat as it can become 4-in-a-row in two ways
        pattern_three_open_both = [0] + [player_color_val] * 3 + [0]  # [0, X, X, X, 0]
        
        board = Board(self.board_size)
        board.copy(board_state_after_move)
        coord = board.action_to_coord(action)
        
        # Check all lines containing this coordinate
        for line in gomoku_util.iterator(board_state_after_move):
            if coord not in line:
                continue
                
            line_values = gomoku_util.value(board_state_after_move, line)
            
            # Only check for pattern open on both sides
            if gomoku_util.is_sublist(line_values, pattern_three_open_both):
                pattern_start = gomoku_util.index(line_values, pattern_three_open_both)
                if pattern_start >= 0:
                    # Check if our coord is in the pattern range
                    pattern_end = pattern_start + len(pattern_three_open_both)
                    if pattern_end <= len(line):
                        pattern_coords = line[pattern_start:pattern_end]
                        if coord in pattern_coords:
                            return True
        return False
    
    def _exec_opponent_play(self, curr_state, prev_state, prev_action):
        '''There is no resign in gomoku'''
        assert curr_state.color != self.player_color
        opponent_action = self.opponent_policy(curr_state, prev_state, prev_action)
        return curr_state.act(opponent_action), opponent_action
    
    @property
    def _state(self):
        return self.state
    
    @property
    def _moves(self):
        return self.moves
    
    def _reset_opponent(self, board):
        # Use random opponent for simplicity
        self.opponent_policy = make_random_policy(self.np_random)

class Board(object):
    '''
    Basic Implementation of a Go Board, natural action are int [0,board_size**2)
    '''
    
    def __init__(self, board_size):
        self.size = board_size
        self.board_state = [[gomoku_util.color_dict['empty']] * board_size for i in range(board_size)] # initialize board states to empty
        self.move = 0
        self.last_coord = (-1,-1)
        self.last_action = None
    
    def coord_to_action(self, i, j):
        ''' convert coordinate i, j to action a in [0, board_size**2)
        '''
        a = i * self.size + j # action index
        return a
    
    def action_to_coord(self, a):
        coord = (a // self.size, a % self.size)
        return coord
    
    def get_legal_move(self):
        ''' Get all the next legal move, namely empty space that you can place your 'color' stone
            Return: Coordinate of all the empty space, [(x1, y1), (x2, y2), ...]
        '''
        legal_move = []
        for i in range(self.size):
            for j in range(self.size):
                if (self.board_state[i][j] == 0):
                    legal_move.append((i, j))
        return legal_move
    
    def get_legal_action(self):
        ''' Get all the next legal action, namely empty space that you can place your 'color' stone
            Return: Coordinate of all the empty space, [(x1, y1), (x2, y2), ...]
        '''
        legal_action = []
        for i in range(self.size):
            for j in range(self.size):
                if (self.board_state[i][j] == 0):
                    legal_action.append(self.coord_to_action(i, j))
        return legal_action
    
    def copy(self, board_state):
        '''update board_state of current board values from input 2D list
        '''
        input_size_x = len(board_state)
        input_size_y = len(board_state[0])
        assert input_size_x == input_size_y, 'input board_state two axises size mismatch'
        assert len(self.board_state) == input_size_x, 'input board_state size mismatch'
        for i in range(self.size):
            for j in range(self.size):
                self.board_state[i][j] = board_state[i][j]
    
    def play(self, action, color):
        '''
            Args: input action, current player color
            Return: new copy of board object
        '''
        b = Board(self.size)
        b.copy(self.board_state) # create a board copy of current board_state
        b.move = self.move
        
        coord = self.action_to_coord(action)
        # check if it's legal move
        if (b.board_state[coord[0]][coord[1]] != 0): # the action coordinate is not empty
            raise error.Error("Action is illegal, position [%d, %d] on board is not empty" % ((coord[0]+1),(coord[1]+1)))
        
        b.board_state[coord[0]][coord[1]] = gomoku_util.color_dict[color]
        b.move += 1 # move counter add 1
        b.last_coord = coord # save last coordinate
        b.last_action = action
        return b
    
    def is_terminal(self):
        exist, color = gomoku_util.check_five_in_row(self.board_state)
        is_full = gomoku_util.check_board_full(self.board_state)
        if (is_full): # if the board if full of stones and no extra empty spaces, game is finished
            return True
        else:
            return exist
    
    def __repr__(self):
        ''' representation of the board class
            print out board_state
        '''
        out = ""
        size = len(self.board_state)
        
        letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:size]
        numbers = list(range(1, 100))[:size]
        
        label_move = "Move: " + str(self.move) + "\n"
        label_letters = "     " + " ".join(letters) + "\n"
        label_boundry = "   " + "+-" + "".join(["-"] * (2 * size)) + "+" + "\n"
        
        # construct the board output
        out += (label_move + label_letters + label_boundry)
        
        for i in range(size-1,-1,-1):
            line = ""
            line += (str("%2d" % (i+1)) + " |" + " ")
            for j in range(size):
                # check if it's the last move
                line += gomoku_util.color_shape[self.board_state[i][j]]
                if (i,j) == self.last_coord:
                    line += ")"
                else:
                    line += " "
            line += ("|" + "\n")
            out += line
        out += (label_boundry + label_letters)
        return out
    
    def encode(self, use_onehot=True):
        '''Return: np array
            If use_onehot=True: np.array(3, board_size, board_size) - one-hot encoded board
            If use_onehot=False: np.array(board_size, board_size) - normalized single channel (legacy)
        '''
        if use_onehot:
            # Return one-hot encoded board: (3, board_size, board_size)
            # Channels: [empty, black, white]
            return gomoku_util.encode_board_onehot(self.board_state)
        else:
            # Legacy encoding: normalized single channel
            img = np.array(self.board_state, dtype=np.float32) # shape [board_size, board_size]
            # Normalize to [0, 1] range: 0->0.0, 1->0.5, 2->1.0
            img = img / 2.0
            return img
