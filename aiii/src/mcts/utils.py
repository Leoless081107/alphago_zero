import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def ucb_score(parent_visit_count, child_visit_count, child_value_sum, child_prior, c_puct=1.0):
    """Calculate the Upper Confidence Bound score for a child node."""
    if child_visit_count == 0:
        return float('inf')
    # Exploitation term
    q_value = (child_value_sum / child_visit_count) if child_visit_count > 0 else 0
    # Exploration term
    exploration_term = c_puct * child_prior * np.sqrt(parent_visit_count) / (1 + child_visit_count)
    return q_value + exploration_term