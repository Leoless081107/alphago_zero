import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
from src.network.model import PolicyValueNetwork
from src.self_play.worker import SelfPlayWorker
from src.training.trainer import Trainer
from src.self_play.replay import ReplayBuffer
import yaml
import os

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../src/configs/default.yaml')
        config_path = os.path.abspath(config_path)
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    network = PolicyValueNetwork(
        board_size=config['board_size'],
        num_res_blocks=config['network']['num_res_blocks'],
        num_channels=config['network']['num_channels']
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)
    replay_buffer = ReplayBuffer(config['training']['replay_buffer_size'])
    self_play_worker = SelfPlayWorker(network, config)
    trainer = Trainer(
        policy_value_net=network,
        config=config['training']  
    )
    
    best_win_rate = 0
    best_model_path = ""
    total_games = 0 

    for iteration in range(config['training']['iterations']):
        print(f"Starting iteration {iteration+1}/{config['training']['iterations']}")


        print("Generating self-play data...")
        num_games = config['training']['num_games_per_iteration']
        self_play_data = self_play_worker.generate_self_play_data(num_games)

        total_games += num_games
        print(f"Completed {total_games} total games so far")
        
        for game in self_play_data:
            replay_buffer.add_game(game)
        
        print("Training network...")
        trainer.train(replay_buffer, config['training']['num_epochs'])
        
        os.makedirs('data/trained_models', exist_ok=True)
        os.makedirs('data/checkpoints', exist_ok=True)
        os.makedirs('data/experience', exist_ok=True)
        
        checkpoint_path = f"data/checkpoints/model_{iteration+1}.pth"
        torch.save(network.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        latest_path = "data/checkpoints/latest.pth"
        torch.save(network.state_dict(), latest_path)
        
        buffer_path = "data/checkpoints/latest_buffer.pth"
        torch.save(replay_buffer, buffer_path)
        
        if (iteration + 1) % 5 == 0:
            backup_path = f"data/checkpoints/backup_{iteration+1}.pth"
            torch.save(network.state_dict(), backup_path)
            print(f"Created backup checkpoint: {backup_path}")
        
        for game_idx, game in enumerate(self_play_data):
            exp_path = f"data/experience/game_{iteration+1}_{game_idx}.pth"
            torch.save(game, exp_path)
        
        if (iteration + 1) % config['training'].get('eval_interval', 10) == 0:
            win_rate = trainer.evaluate(network)
            print(f"Evaluation Win Rate: {win_rate:.2f}")
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_model_path = f"data/trained_models/best_model.pth"
                torch.save(network.state_dict(), best_model_path)
                print(f"New best model saved with win rate {win_rate:.2f}")

if __name__ == "__main__":
    main()