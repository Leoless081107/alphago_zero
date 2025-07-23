import numpy as np

class Player:
    """Base class for players."""
    def __init__(self, name):
        self.name = name
        
    def get_move(self, board):
        """Return a move (row, col) or 'pass'."""
        raise NotImplementedError

class MCTSPlayer(Player):
    """Player that uses MCTS with a policy-value network."""
    def __init__(self, name, mcts, temperature=1.0):
        super().__init__(name)
        self.mcts = mcts
        self.temperature = temperature
        self.game_history = []
        
    def get_move(self, board):
        # Get full policy array from MCTS
        _, probs = self.mcts.search(board)  # Unpack tuple (actions, probs)
        
        # Sample action from full probability distribution
        action = np.random.choice(len(probs), p=probs)
        
        # Store complete policy array in game history
        self.game_history.append((board.copy(), probs, None))
        return action