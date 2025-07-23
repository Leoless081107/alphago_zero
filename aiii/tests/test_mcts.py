import unittest
import numpy as np
from src.mcts.node import MCTSNode
from src.mcts.mcts import MCTS
from src.game.board import GoBoard

class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.board = GoBoard(size=5)
        # Simple policy-value function for testing
        def dummy_policy_value_fn(board):
            policy = np.zeros(board.size * board.size + 1)
            policy[0] = 1.0  # Always choose first move
            return policy, 0.5
        self.mcts = MCTS(dummy_policy_value_fn, num_simulations=10)

    def test_node_creation(self):
        node = MCTSNode(self.board)
        self.assertEqual(node.visit_count, 0)
        self.assertEqual(node.value_sum, 0)
        self.assertTrue(node.is_leaf())

    def test_ucb_score(self):
        node = MCTSNode(self.board)
        child = MCTSNode(self.board, parent=node, action=0)
        child.visit_count = 10
        child.value_sum = 5
        child.prior = 0.5
        node.visit_count = 100

        score = child.ucb_score(c_puct=1.0)
        self.assertGreater(score, 0)

    def test_mcts_search(self):
        actions, probs = self.mcts.search(self.board)
        self.assertIsInstance(actions, np.ndarray)
        self.assertIsInstance(probs, np.ndarray)
        self.assertEqual(len(actions), len(probs))
        self.assertAlmostEqual(np.sum(probs), 1.0, delta=0.01)

if __name__ == '__main__':
    unittest.main()