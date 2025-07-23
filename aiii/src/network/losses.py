import torch
import torch.nn as nn

def policy_value_loss(policy_targets, value_targets, policy_outputs, value_outputs):
    """Combined loss function for policy and value networks."""
    # Policy loss (cross-entropy)
    policy_loss = nn.CrossEntropyLoss()(policy_outputs, policy_targets)
    # Value loss (mean squared error)
    value_loss = nn.MSELoss()(value_outputs.squeeze(), value_targets)
    # Total loss
    return policy_loss + value_loss