from copy import deepcopy


class GO:
    def __init__(self, n):
        self.board = None
        self.previous_board = None
        self.size = n
        self.piece_type = 1
        self.died_pieces = []
        self.move_num = 0
        self.max_move = n * n - 1
        self.KOMI = n / 2
        self.whetherManual = False

    def set_move(self, n_move):
        self.move_num = n_move

    def get_move(self):
        return self.move_num

    def get_max_move(self):
        return self.max_move

    def initialize_board(self, piece_type, previous_board, board):
        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))

        self.piece_type = piece_type
        self.board = board
        self.previous_board = previous_board

    def is_the_same_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def copy_board(self):
        return deepcopy(self)

    def find_all_neighbors(self, i, j):
        board = self.board
        neighbors = []

        # filter border
        if i > 0:
            neighbors.append((i - 1, j))
        if i < len(board) - 1:
            neighbors.append((i + 1, j))
        if j > 0:
            neighbors.append((i, j - 1))
        if j < len(board) - 1:
            neighbors.append((i, j + 1))
        return neighbors

    def find_all_neighbor_allies(self, i, j):
        board = self.board
        neighbors = self.find_all_neighbors(i, j)  # find neighbors
        group_allies = []

        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def find_neighbor_ally_enemy_num(self, i, j, piece_type):
        board = self.board
        neighbors = self.find_all_neighbors(i, j)  # find neighbors
        ally_num = 0
        enemy_num = 0

        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == piece_type:
                ally_num += 1
            elif board[piece[0]][piece[1]] == 3 - piece_type:
                enemy_num += 1
        return ally_num, enemy_num

    def find_neighbor_empties(self, i, j):
        board = self.board
        neighbors = self.find_all_neighbors(i, j)  # find neighbors
        group_empty = []

        # Iterate through neighbors
        for piece in neighbors:
            # Add to enemies list if having the different color
            if board[piece[0]][piece[1]] == 0:
                group_empty.append(piece)
        return group_empty

    def find_neighbor_allies_with_type(self, i, j, piece_type):
        board = self.board
        neighbors = self.find_all_neighbors(i, j)  # find neighbors
        group_allies = []

        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == piece_type:
                group_allies.append(piece)
        return group_allies

    def find_all_group_members(self, i, j):
        stack = [(i, j)]
        group_members = []
        while stack:
            piece = stack.pop()
            group_members.append(piece)
            neighbor_allies = self.find_all_neighbor_allies(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in group_members:
                    stack.append(ally)
        return group_members

    def has_liberty(self, i, j):
        board = self.board
        ally_members = self.find_all_group_members(i, j)
        for member in ally_members:
            neighbors = self.find_all_neighbors(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    def find_liberty_pieces_position(self, i, j):
        board = self.board
        ally_members = self.find_all_group_members(i, j)
        liberty = {}
        for member in ally_members:
            neighbors = self.find_all_neighbors(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    liberty[piece] = 1
        return liberty

    def is_connection_of_groups(self, i, j, piece_type):
        board = self.board
        if board[i][j] == 0:
            neighbor_allies = self.find_neighbor_allies_with_type(i, j, piece_type)
            if len(neighbor_allies) > 1:
                initial_liberty = {}
                for ally in neighbor_allies:
                    initial_liberty.update(self.find_liberty_pieces_position(ally[0], ally[1]))
                neighbor_empties = self.find_neighbor_empties(i, j)
                final_liberty = len(initial_liberty) - 1 + len(neighbor_empties)
                if final_liberty >= len(initial_liberty):
                    return True
        return False

    def find_died_pieces(self, piece_type):
        board = self.board
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.has_liberty(i, j):
                        died_pieces.append((i, j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces:
            return []
        self.remove_these_pieces(died_pieces)
        return died_pieces

    def remove_these_pieces(self, positions):
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_now_board(board)

    def place_chess(self, i, j, piece_type):
        board = self.board

        valid_place = self.valid_place_check(i, j, piece_type)
        if not valid_place:
            return False
        self.previous_board = deepcopy(board)
        board[i][j] = piece_type
        self.update_now_board(board)
        # Remove the following line for HW2 CS561 S2020
        self.move_num += 1
        return True

    def valid_place_check(self, i, j, piece_type):
        board = self.board

        # Check if the place is in the board range
        if not (i >= 0 and i < len(board)):
            return False
        if not (j >= 0 and j < len(board)):
            return False

        # Check if the place already has a piece
        if board[i][j] != 0:
            return False

        # Copy the board for testing
        test_go = self.copy_board()
        test_board = test_go.board

        # Check if the place has liberty
        test_board[i][j] = piece_type
        test_go.update_now_board(test_board)
        if test_go.has_liberty(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.has_liberty(i, j):
            return False

        # Check special case: repeat placement causing the repeat board go_state (KO rule)
        else:
            if self.died_pieces and self.is_the_same_board(self.previous_board, test_go.board):
                return False
        return True

    def update_now_board(self, new_board):
        self.board = new_board

    def game_end(self, action="MOVE"):
        # Case 1: max move reached
        if self.move_num >= self.max_move:
            return True
        # Case 2: two players all pass the move.
        if self.is_the_same_board(self.previous_board, self.board) and action == "PASS":
            return True
        return False