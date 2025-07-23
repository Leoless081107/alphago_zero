import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """Experience replay buffer for storing and sampling self-play data."""
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        
    def add_game(self, game_history):
        """Add a complete game to the buffer."""
        self.buffer.extend(game_history)
        
    def sample_batch(self, batch_size):
        """Sample a batch of training data."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
    def size(self):
        """Return current buffer size."""
        return len(self.buffer)