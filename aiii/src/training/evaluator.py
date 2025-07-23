import torch

class Evaluator:
    """Evaluate model performance against previous versions."""
    def __init__(self, policy_value_net, config, device='cpu'):
        self.policy_value_net = policy_value_net
        self.config = config
        self.device = device
        
    def evaluate(self, new_model_path, old_model_path=None, num_games=10):
        """Evaluate new model against old model (or random if no old model)."""
        from src.game.board import GoBoard
        from src.self_play.worker import SelfPlayWorker
        from src.mcts.mcts import MCTS
        
        # Load new model
        new_model = torch.load(new_model_path)
        self.policy_value_net.load_state_dict(new_model)
        new_worker = SelfPlayWorker(self.policy_value_net, self.config, self.device)
        
        # Load old model or use random player
        if old_model_path:
            old_model = torch.load(old_model_path)
            old_policy_value_net = type(self.policy_value_net)(
                board_size=self.config['board_size'],
                num_res_blocks=self.config['network']['num_res_blocks'],
                num_channels=self.config['network']['num_channels']
            )
            old_policy_value_net.load_state_dict(old_model)
            old_worker = SelfPlayWorker(old_policy_value_net, self.config, self.device)
            old_mcts = MCTS(old_worker._policy_value_fn, num_simulations=self.config['num_simulations'])
        
        # Play games
        new_model_wins = 0
        total_moves = 0
        policy_entropies = []
        move_diversity = set()
        
        for i in range(num_games):
            print(f"Evaluation game {i+1}/{num_games}")
            board = GoBoard(size=self.config['board_size'])
            game_moves = 0
            
            # Alternate who plays black
            if i % 2 == 0:
                black_worker = new_worker
                white_worker = old_worker if old_model_path else None
            else:
                black_worker = old_worker if old_model_path else None
                white_worker = new_worker
            
            # Play game
            while not board.is_terminal():
                game_moves += 1
                if board.current_player == 1:
                    policy = black_worker.mcts.search(board)
                    move = np.argmax(policy)
                    # Track policy entropy and move diversity
                    if isinstance(policy, np.ndarray) and len(policy) > 0:
                        entropy = -np.sum(policy * np.log(policy + 1e-10))
                        policy_entropies.append(entropy)
                    move_diversity.add(move)
                else:
                    if old_model_path:
                        policy = white_worker.mcts.search(board)
                        move = np.argmax(policy)
                    else:
                        # Random move for baseline
                        legal_moves = board.get_legal_moves()
                        move = legal_moves[torch.randint(0, len(legal_moves), (1,)).item()]
                    
                board = board.apply_action(move)
            
            total_moves += game_moves
            # Determine winner using board's calculate_winner method
            winner = board.calculate_winner(komi=self.config.get('komi', 7.5))
            
            # Count wins for new model
            if (i % 2 == 0 and winner == 1) or (i % 2 == 1 and winner == -1):
                new_model_wins += 1
        
        win_rate = new_model_wins / num_games
        avg_game_length = total_moves / num_games
        avg_entropy = np.mean(policy_entropies) if policy_entropies else 0
        diversity_score = len(move_diversity) / (board.size * board.size + 1) if (board.size * board.size + 1) > 0 else 0
        
        print(f"New model win rate: {win_rate:.2f}")
        print(f"Average game length: {avg_game_length:.1f}")
        print(f"Average policy entropy: {avg_entropy:.3f}")
        print(f"Move diversity score: {diversity_score:.3f}")
        
        return {
            'win_rate': win_rate,
            'avg_game_length': avg_game_length,
            'avg_policy_entropy': avg_entropy,
            'move_diversity': diversity_score
        }