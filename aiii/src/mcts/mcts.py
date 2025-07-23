import numpy as np
from .node import MCTSNode

class MCTS:
    def __init__(self, policy_value_fn, num_simulations=1000, c_puct=1.0):
        self.policy_value_fn = policy_value_fn  # (state) -> (policy, value)
        self.num_simulations = num_simulations  # Number of simulations per move
        self.c_puct = c_puct  # Exploration constant

    def _get_action_probs(self, root, temp=1e-3):
        # Get board size from root state to calculate total possible moves
        board_size = root.state.size
        total_moves = board_size * board_size + 1  # +1 for pass move
        probs = np.zeros(total_moves)
        
        visit_counts = [child.visit_count for child in root.children]
        actions = [child.action for child in root.children]

        # Handle edge case: no children or all visit counts are zero
        if not visit_counts or sum(visit_counts) == 0:
            # Return uniform distribution over all possible moves
            return np.ones(total_moves) / total_moves

        if temp == 0:
            # Greedy action selection
            best_action = actions[np.argmax(visit_counts)]
            probs[best_action] = 1.0
        else:
            # Convert to numpy array and add numerical stability
            visit_counts = np.array(visit_counts, dtype=np.float64)
            visit_counts = np.clip(visit_counts, 1e-10, None)  # Prevent zeros
            logits = np.log(visit_counts) / temp
            # Subtract max for numerical stability
            logits -= np.max(logits)
            probs_softmax = np.exp(logits)
            probs_softmax /= np.sum(probs_softmax) + 1e-10  # Prevent division by zero
            for action, prob in zip(actions, probs_softmax):
                probs[action] = prob
        return probs
    
    def search(self, root_state):
        root = MCTSNode(root_state)

        for _ in range(self.num_simulations):
            node = root
            # Select
            while not node.is_leaf():
                # Choose child with highest UCB score
                node = max(node.children, key=lambda x: x.ucb_score(self.c_puct))

            # Expand
            policy, value = self.policy_value_fn(node.state)
            if not node.is_terminal():
                for action, prob in enumerate(policy):
                    if prob > 0:
                        child_state = node.state.apply_action(action)
                        child = MCTSNode(child_state, parent=node, action=action)
                        child.prior = prob
                        node.children.append(child)

            # Backup
            self._backup(node, value)

        # Return (actions, probs) tuple instead of just probabilities
        probs = self._get_action_probs(root)
        actions = np.arange(len(probs))
        return actions, probs

    def _backup(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip value for opponent
            node = node.parent