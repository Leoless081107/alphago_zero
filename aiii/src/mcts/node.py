class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # Game state
        self.parent = parent  # Parent node
        self.action = action  # Action taken to reach this node
        self.children = []  # Child nodes
        self.visit_count = 0  # Number of visits
        self.value_sum = 0  # Sum of values from this node
        self.prior = 0  # Prior probability from policy network

    def is_terminal(self):
        return self.state.is_terminal()

    def is_leaf(self):
        return len(self.children) == 0

    def ucb_score(self, c_puct=1.0):
        if self.visit_count == 0:
            return float('inf')
        # Upper Confidence Bound formula
        return (self.value_sum / self.visit_count) + c_puct * self.prior * \
               (self.parent.visit_count **0.5 / (1 + self.visit_count))