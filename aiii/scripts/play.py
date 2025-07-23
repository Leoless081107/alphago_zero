import sys
import re
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import yaml
import argparse
from src.game.board import GoBoard
from src.game.player import MCTSPlayer
from src.mcts.mcts import MCTS
from src.network.model import PolicyValueNetwork
from src.network.utils import encode_state
import numpy as np
import os

def human_player_move(board):
    """Get move from human player."""
    while True:
        move = input("Enter your move (x,x) or 'pass': ").strip()
        if move.lower() == 'pass':
            return board.size * board.size
        
        try:
            match = re.match(r"\(?(\d+)\s*,\s*(\d+)\)?", move)
            if not match:
                raise ValueError("Invalid format")
                
            row = int(match.group(1))
            col = int(match.group(2))
            
            if 0 <= row < board.size and 0 <= col < board.size:
                return row * board.size + col
            print(f"Please enter coordinates between 0 and {board.size-1}")
        except ValueError:
            print("Invalid format. Use (x,x) where x is a number (e.g., (3,4)) or 'pass'")

def main():
    parser = argparse.ArgumentParser(description='Play against AlphaGo Zero')
    parser.add_argument('--model-path', type=str, default='data/trained_models/model_1.pth',
                      help='Path to trained model')
    parser.add_argument('--config', type=str, default='src/configs/default.yaml',
                      help='Path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    while True:
        first_player = input("Who goes first? (human/ai): ").strip().lower()
        if first_player in ['human', 'ai']:
            human_first = (first_player == 'human')
            break
        print("Please enter 'human' or 'ai'")
    policy_value_net = PolicyValueNetwork(
        board_size=config['board_size'],
        num_res_blocks=config['network']['num_res_blocks'],
        num_channels=config['network']['num_channels']
    )
    if args.model_path:
        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found at {args.model_path}")
            print("Please train a model first using train.py or specify a valid model path with --model-path")
            sys.exit(1)
        policy_value_net.load_state_dict(torch.load(args.model_path))
        policy_value_net.eval()
    def policy_value_fn(board):
        encoded_state = encode_state(board, board.current_player).unsqueeze(0)
        policy, value = policy_value_net(encoded_state)
        policy = policy.squeeze().detach().numpy()
        value = value.item()
        legal_moves = board.get_legal_moves()
        mask = np.zeros_like(policy)
        mask[legal_moves] = 1
        policy = policy * mask
        policy = policy / np.sum(policy) if np.sum(policy) > 0 else policy
        
        return policy, value

    mcts = MCTS(policy_value_fn, num_simulations=config['mcts']['num_simulations'])
    ai_player = MCTSPlayer("AI", mcts, temperature=0)
    board = GoBoard(size=config['board_size'])
    print("Starting game! Enter moves as 'row,col' (0-based) or 'pass'")
    print("Board size: {0}x{0}".format(config['board_size']))

    while not board.is_terminal():
    
        if (board.current_player == 1 and not human_first) or (board.current_player == -1 and human_first):
            print("AI's turn...")
            move = ai_player.get_move(board)
            move_str = f"({move//board.size},{move%board.size})" if move != board.size*board.size else 'pass'
            print(f"AI plays: {move_str}")
        else:
            move = human_player_move(board)

        board = board.apply_action(move)

    print("Game over!")
    winner = board.calculate_winner()
    if winner == 1:
        print("Black (X) wins by {} points!\n".format(board.get_black_score() - board.get_white_score()))
    elif winner == -1:
        print("White (O) wins by {} points!\n".format(board.get_white_score() - board.get_black_score()))
    else:
        print("Game ended in a tie!\n")

if __name__ == "__main__":
    main()