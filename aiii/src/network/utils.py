import numpy as np
import torch
import random


def encode_state(board, current_player):
    """Encode board state into input tensor for neural network."""
    
    size = board.size
    # Create 4-channel input: own stones, opponent stones, recent moves, color
    encoded = np.zeros((4, size, size), dtype=np.float32)
    
    # Own stones
    encoded[0] = (board.board == current_player).astype(np.float32)
    # Opponent stones
    encoded[1] = (board.board == -current_player).astype(np.float32)
    # Recent move (simplified)
    if board.last_move != 'pass' and board.last_move is not None:
        row, col = board.last_move
        encoded[2, row, col] = 1.0
    # Player color (1 for black, 0 for white)
    encoded[3] = np.ones((size, size), dtype=np.float32) if current_player == 1 else 0
    
    return torch.from_numpy(encoded)


def prepare_training_data(game_history, policy_value_net, device='cpu', augmentation_config=None):
    """Prepare self-play data for training."""
    states = []
    policy_targets = []
    value_targets = []
    
    for state, policy, winner in game_history:
        encoded_state = encode_state(state, state.current_player).numpy()
        board_size = state.size
        
        # Generate configured augmentations
        augmented = augment_state(encoded_state, policy, board_size, augmentation_config)
        for aug_state, aug_policy in augmented:
            states.append(torch.from_numpy(aug_state))
            policy_targets.append(torch.from_numpy(aug_policy))
            # Value target remains the same for symmetric positions
            value_target = 1 if winner == state.current_player else -1
            value_targets.append(torch.tensor(value_target, dtype=torch.float32))
    
    # Convert to batches
    states = torch.stack(states).to(device)
    policy_targets = torch.stack(policy_targets).to(device)
    value_targets = torch.stack(value_targets).to(device)
    
    return states, policy_targets, value_targets


def rotate_board(board, k):
    """Rotate board 90 degrees clockwise k times"""
    return np.rot90(board, k, axes=(1, 2)).copy()


def flip_board(board):
    """Flip board horizontally"""
    return np.flip(board, axis=2).copy()


def rotate_policy(policy, board_size, k):
    """Rotate policy 90 degrees clockwise k times"""
    policy_2d = policy[:-1].reshape(board_size, board_size)
    rotated_2d = np.rot90(policy_2d, k)
    rotated_flat = rotated_2d.flatten()
    return np.concatenate([rotated_flat, [policy[-1]]])  # Keep pass move unchanged


def flip_policy(policy, board_size):
    """Flip policy horizontally"""
    policy_2d = policy[:-1].reshape(board_size, board_size)
    flipped_2d = np.flip(policy_2d, axis=1)
    flipped_flat = flipped_2d.flatten()
    return np.concatenate([flipped_flat, [policy[-1]]])  # Keep pass move unchanged


def augment_state(state, policy, board_size, config=None):
    """Generate configured symmetric variations of a state-policy pair"""
    config = config or {}
    enabled = config.get('enabled', True)
    if not enabled:
        return [(state, policy)]
        
    use_rotations = config.get('rotations', True)
    use_flips = config.get('flips', True)
    random_rotation = config.get('random_rotation', False)
    rotation_prob = config.get('rotation_probability', 1.0)
    flip_prob = config.get('flip_probability', 1.0)
    
    augmentations = []
    # Original state
    augmentations.append((state, policy))
    
    # Rotations
    if use_rotations and random.random() < rotation_prob:
        if random_rotation:
            # Randomly select one rotation
            k = random.choice([1, 2, 3])
            rotated_state = rotate_board(state, k)
            rotated_policy = rotate_policy(policy, board_size, k)
            augmentations.append((rotated_state, rotated_policy))
        else:
            # Apply all rotations
            for k in [1, 2, 3]:
                rotated_state = rotate_board(state, k)
                rotated_policy = rotate_policy(policy, board_size, k)
                augmentations.append((rotated_state, rotated_policy))
    
    # Flip and rotations
    if use_flips and random.random() < flip_prob:
        flipped_state = flip_board(state)
        flipped_policy = flip_policy(policy, board_size)
        augmentations.append((flipped_state, flipped_policy))
        
        if use_rotations and random.random() < rotation_prob:
            if random_rotation:
                k = random.choice([1, 2, 3])
                rotated_flipped_state = rotate_board(flipped_state, k)
                rotated_flipped_policy = rotate_policy(flipped_policy, board_size, k)
                augmentations.append((rotated_flipped_state, rotated_flipped_policy))
            else:
                for k in [1, 2, 3]:
                    rotated_flipped_state = rotate_board(flipped_state, k)
                    rotated_flipped_policy = rotate_policy(flipped_policy, board_size, k)
                    augmentations.append((rotated_flipped_state, rotated_flipped_policy))
    
    return augmentations