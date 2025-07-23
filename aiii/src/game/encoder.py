import numpy as np

def encode_action(action, board_size):
    """Encode action (row, col) or 'pass' into index."""
    if action == 'pass':
        return board_size * board_size
    row, col = action
    return row * board_size + col

def decode_action(index, board_size):
    """Decode index into (row, col) or 'pass'."""
    if index == board_size * board_size:
        return 'pass'
    return index // board_size, index % board_size

class BoardEncoder:
    """Encodes board state for neural network input."""
    def __init__(self, board_size):
        self.board_size = board_size
        self.num_planes = 4  # Standard AlphaGo Zero encoding
        
    def encode(self, board, player):
        """Encode board state and current player into input tensor."""
        encoded = np.zeros((self.num_planes, self.board_size, self.board_size), dtype=np.float32)
        
        # Plane 0: current player's stones
        encoded[0] = (board.board == player).astype(np.float32)
        # Plane 1: opponent's stones
        encoded[1] = (board.board == -player).astype(np.float32)
        # Plane 2: previous move
        if board.last_move and board.last_move != 'pass':
            row, col = board.last_move
            encoded[2, row, col] = 1.0
        # Plane 3: current player color (1 for black, 0 for white)
        encoded[3] = np.ones((self.board_size, self.board_size), dtype=np.float32) if player == 1 else 0
        
        return encoded