import unittest
from src.game.board import GoBoard

class TestGoBoard(unittest.TestCase):
    def setUp(self):
        self.board = GoBoard(size=5)

    def test_initial_state(self):
        self.assertEqual(self.board.current_player, 1)
        self.assertEqual(np.sum(self.board.board), 0)
        self.assertIsNone(self.board.last_move)

    def test_apply_action(self):
        new_board = self.board.apply_action(0)
        self.assertEqual(new_board.board[0, 0], 1)
        self.assertEqual(new_board.current_player, -1)

    def test_legal_moves(self):
        legal_moves = self.board.get_legal_moves()
        self.assertIn(0, legal_moves)
        self.assertIn(5*5, legal_moves)  # pass

    def test_is_terminal(self):
        self.assertFalse(self.board.is_terminal())
        # Make two consecutive passes
        board = self.board.apply_action(5*5)  # pass
        board = board.apply_action(5*5)  # pass
        self.assertTrue(board.is_terminal())

if __name__ == '__main__':
    unittest.main()