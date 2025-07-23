import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.network.utils import prepare_training_data
from torch.utils.data import TensorDataset, DataLoader
from src.network.losses import policy_value_loss
class Trainer:
    def __init__(self, policy_value_net, config):
        self.policy_value_net = policy_value_net
        self.device = next(policy_value_net.parameters()).device
        self.config = config
        self.optimizer = optim.Adam(
            policy_value_net.parameters(),
            lr=float(config['learning_rate']),
            weight_decay=float(config['weight_decay'])
        )
        self.writer = SummaryWriter(log_dir='data/logs')
        self.step = 0
        self.best_model = None
        
    def evaluate(self, current_model, num_eval_games=20):
        """Evaluate current model against previous best"""
        if self.best_model is None:
            self.best_model = type(current_model)()
            self.best_model.load_state_dict(current_model.state_dict())
            self.best_model.to(self.device)
            return 0.5  # Default win rate
        
        from src.game.player import MCTSPlayer
        from src.game.board import GoBoard
        
        current_wins = 0
        current_model.eval()
        self.best_model.eval()
        
        for i in range(num_eval_games):
            board = GoBoard(size=self.config['board_size'])
            
            # Alternate who plays first
            if i % 2 == 0:
                players = {
                    1: MCTSPlayer("Current", MCTS(current_model.policy_value_fn)),
                    -1: MCTSPlayer("Best", MCTS(self.best_model.policy_value_fn))
                }
            else:
                players = {
                    1: MCTSPlayer("Best", MCTS(self.best_model.policy_value_fn)),
                    -1: MCTSPlayer("Current", MCTS(current_model.policy_value_fn))
                }
            
            # Play game
            while not board.is_terminal():
                move = players[board.current_player].get_move(board)
                board = board.apply_action(move)
            
            # Determine winner
            winner = board.calculate_winner()
            if (i % 2 == 0 and winner == 1) or (i % 2 == 1 and winner == -1):
                current_wins += 1
        
        win_rate = current_wins / num_eval_games
        self.writer.add_scalar('eval/win_rate', win_rate, self.step)
        return win_rate
    def train(self, replay_buffer, epochs=10):
        """Train using data from replay buffer"""
        # Sample batch from replay buffer
        batch_size = min(len(replay_buffer.buffer), 
                     self.config['batch_size'] * epochs)
        batch = replay_buffer.sample_batch(batch_size)
        
        # Prepare training data with augmentation config
        augmentation_config = self.config.get('data_augmentation', {})
        states, policy_targets, value_targets = prepare_training_data(
            batch, self.policy_value_net, self.device, augmentation_config=augmentation_config
        )
        
        # Create DataLoader
        dataset = TensorDataset(states, policy_targets, value_targets)
        train_loader = DataLoader(dataset, 
                                batch_size=self.config['batch_size'], 
                                shuffle=True)
        
        # Training loop
        self.policy_value_net.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_states, batch_policy, batch_value in train_loader:
                # Forward pass
                policy_out, value_out = self.policy_value_net(batch_states)
                
                # Loss calculation
                loss = policy_value_loss(batch_policy, batch_value, 
                                       policy_out, value_out)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                self.step += 1
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            self.writer.add_scalar('train/loss', avg_loss, self.step)