import numpy as np

class GoBoard:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1  # 1 for black, -1 for white
        self.history = []
        self.last_move = None

    def copy(self):
        new_board = GoBoard(self.size)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.history = list(self.history)
        new_board.last_move = self.last_move
        return new_board

    def apply_action(self, action):
        # action: size*size for pass
        if action == self.size * self.size:
            new_board = self.copy()
            new_board.history.append(('pass', self.current_player))
            new_board.current_player *= -1
            new_board.last_move = 'pass'
            return new_board

        row, col = action // self.size, action % self.size
        if self.board[row, col] != 0:
            raise ValueError(f"Invalid move at ({row}, {col})")

        new_board = self.copy()
        new_board.board[row, col] = self.current_player
        new_board.history.append(((row, col), self.current_player))
        new_board._remove_captured(-self.current_player)
        new_board.current_player *= -1
        new_board.last_move = (row, col)
        return new_board

    def _remove_captured(self, player):
        # Implementation of capture logic
        visited = np.zeros((self.size, self.size), dtype=bool)
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == player and not visited[i, j]:
                    group, liberties = self._get_group_and_liberties(i, j, visited)
                    if liberties == 0:
                        for (x, y) in group:
                            self.board[x, y] = 0

    def _get_group_and_liberties(self, i, j, visited):
        group = []
        liberties = 0
        queue = [(i, j)]
        visited[i, j] = True
        player = self.board[i, j]

        while queue:
            x, y = queue.pop(0)
            group.append((x, y))

            # Check all four directions
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if not visited[nx, ny]:
                        if self.board[nx, ny] == 0:
                            liberties += 1
                        elif self.board[nx, ny] == player:
                            visited[nx, ny] = True
                            queue.append((nx, ny))
        return group, liberties

    def get_legal_moves(self):
        # Return list of legal moves
        legal_moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    # Check if move is legal (would implement full Go rules)
                    legal_moves.append(i * self.size + j)
        legal_moves.append(self.size * self.size)  # Pass
        return legal_moves

    def is_terminal(self):
        # Check if game is over (simplified for example)
        return len(self.history) > 1 and self.history[-1][0] == 'pass' and self.history[-2][0] == 'pass'

    def get_state(self):
        # Return board state for neural network input
        return self.board

    def calculate_winner(self, komi=7.5):
        # Area scoring: stones + territory
        black_area = np.sum(self.board == 1)
        white_area = np.sum(self.board == -1) + komi
        return 1 if black_area > white_area else -1 if white_area > black_area else 0