import numpy as np
import torch  # Add this import statement
from src.game.player import MCTSPlayer
from src.mcts.mcts import MCTS
from src.network.utils import encode_state

class SelfPlayWorker:
    """Worker for generating self-play games."""
    def __init__(self, policy_value_net, config):
        self.policy_value_net = policy_value_net
        self.config = config
        self.board_size = config['board_size']
        self.temperature = config['mcts']['temperature']
        self.temperature_threshold = config['mcts']['temperature_threshold']
        self.device = next(policy_value_net.parameters()).device
        self.mcts = MCTS(
            policy_value_fn=self._policy_value_fn,
            num_simulations=config['mcts']['num_simulations'],
            c_puct=config['mcts']['c_puct']
        )
    
    def _policy_value_fn(self, board):
        """Policy-value function wrapper for MCTS."""
        # Get legal moves
        legal_moves = board.get_legal_moves()
        
        # Prepare board state
        encoded_state = encode_state(board, board.current_player).unsqueeze(0).to(self.device)
        
        # Get policy and value from network
        self.policy_value_net.eval()
        with torch.no_grad():
            policy, value = self.policy_value_net(encoded_state)
        policy = policy.squeeze().cpu().numpy()
        value = value.item()
        
        # Mask illegal moves
        mask = np.zeros_like(policy)
        mask[legal_moves] = 1
        policy = policy * mask
        policy = policy / np.sum(policy) if np.sum(policy) > 0 else policy
        
        return policy, value
    
    def generate_self_play_game(self, board_size=19):
        """Generate a single self-play game."""
        from src.game.board import GoBoard
        
        board = GoBoard(size=board_size)
        player = MCTSPlayer(
            "AlphaGo Zero", 
            self.mcts,
            temperature=self.temperature  # Use existing temperature from config
        )
        
        # Play game
        while not board.is_terminal():
            # Lower temperature after certain moves
            if len(board.history) > self.temperature_threshold:
                player.temperature = 0  # Use 0 for final temperature
            
            move = player.get_move(board)
            board = board.apply_action(move)
        
        # Determine winner using board's calculate_winner method
        winner = board.calculate_winner(komi=self.config.get('komi', 7.5))
        
        # Update game history with winner
        game_history = [(state, policy, winner) for state, policy, _ in player.game_history]
        
        return game_history
    
    def generate_self_play_data(self, num_games):
        """Generate multiple self-play games."""
        game_histories = []
        
        for i in range(num_games):
            print(f"Generating game {i+1}/{num_games}")
            game_history = self.generate_self_play_game(self.config['board_size'])
            game_histories.append(game_history)
        
        return game_histories