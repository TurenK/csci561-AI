import random
from host import GO
from copy import deepcopy
import queue
from read import readInput
from write import writeOutput


class MiniMaxPlayer():
    def __init__(self, side=None):
        self.MAX_SCORE = 999999
        self.MIN_SCORE = -999999
        self.type = 'minimax'
        self.side = side
        self.num = -1

    def set_side(self, side):
        self.side = side

    def judge_num(self, board):
        num = 0
        for i in range(5):
            for j in range(5):
                if board[i][j] != 0:
                    num += 1
        return num

    def white_start_policy_second(self, go):
        res = None
        distance = 3
        for i in range(5):
            for j in range(5):
                if go.board[i][j] == 2:
                    if go.board[i][j + 1] == 0:
                        dis = (abs(i - 2) ** 2 + abs(j + 1 - 2) ** 2) ** (1 / 2)
                        if dis < distance:
                            distance = dis
                            res = (i, j + 1)
                    if go.board[i + 1][j] == 0:
                        dis = (abs(i + 1 - 2) ** 2 + abs(j - 2) ** 2) ** (1 / 2)
                        if dis < distance:
                            distance = dis
                            res = (i + 1, j)
                    if go.board[i][j - 1] == 0:
                        dis = (abs(i - 2) ** 2 + abs(j - 1 - 2) ** 2) ** (1 / 2)
                        if dis < distance:
                            distance = dis
                            res = (i, j - 1)
                    if go.board[i - 1][j] == 0:
                        dis = (abs(i - 1 - 2) ** 2 + abs(j - 2) ** 2) ** (1 / 2)
                        if dis < distance:
                            distance = dis
                            res = (i - 1, j)
        return res

    def white_start_policy_third(self, go, moves):
        res = moves
        # 不走左不走上，因为第二步走右下
        for i in range(5):
            if (0, i) in res:
                res.remove((0, i))
            if (i, 0) in res:
                res.remove((i, 0))
        for r in res:
            if 1<=r[0]<=3 and 1<=r[1]<=3:
                if go.board[r[0] + 1][r[1]] != 2 and go.board[r[0] - 1][r[1]] != 2 and go.board[r[0]][r[1] + 1] != 2 and go.board[r[0]][r[1] - 1] != 2:
                    if go.board[r[0] + 1][r[1]] + go.board[r[0] - 1][r[1]] + go.board[r[0]][r[1] + 1] + go.board[r[0]][r[1] - 1] == 2:
                        res.remove(r)
        return res

    def black_start_policy_second(self, go):
        if go.board[1][2] == 2:
            return (1, 3), True
        if go.board[2][1] == 2:
            return (1, 1), True
        if go.board[3][2] == 2:
            return (3, 1), True
        if go.board[2][3] == 2:
            return (3, 3), True
        return None, False

    def black_start_policy_third(self, go):
        if go.board[1][2] == 2:
            if go.board[1][3] == 1 and go.board[1][1] == 0:
                return (1, 1), True
            if go.board[1][3] == 1 and go.board[1][1] == 2:
                return (2, 1), True
        if go.board[2][1] == 2:
            if go.board[1][1] == 1 and go.board[3][1] == 0:
                return (3, 1), True
            if go.board[1][1] == 1 and go.board[3][1] == 2:
                return (3, 2), True
        if go.board[3][2] == 2:
            if go.board[3][1] == 1 and go.board[3][3] == 0:
                return (3, 3), True
            if go.board[3][1] == 1 and go.board[3][3] == 2:
                return (2, 3), True
        if go.board[2][3] == 2:
            if go.board[3][3] == 1 and go.board[1][3] == 0:
                return (1, 3), True
            if go.board[3][3] == 1 and go.board[1][3] == 2:
                return (1, 2), True
        return None, False

    def black_aggressive_when_too_many_moves(self, go, moves):
        candidates = []
        for i in range(1, 4):
            for j in range(1, 4):
                if go.board[i][j] == 2:
                    if go.board[i][j + 1] != 2 and go.board[i][j - 1] != 2 and go.board[i + 1][j] != 2 and \
                            go.board[i - 1][j] != 2:
                        if go.board[i][j + 1] + go.board[i][j - 1] + go.board[i + 1][j] + go.board[i - 1][j] == 2:
                            if go.board[i][j + 1] == 0:
                                candidates.append((i, j + 1))
                            if go.board[i][j - 1] == 0:
                                candidates.append((i, j - 1))
                            if go.board[i + 1][j] == 0:
                                candidates.append((i + 1, j))
                            if go.board[i - 1][j] == 0:
                                candidates.append((i - 1, j))
        max_distance = 3
        res = None
        for c in candidates:
            if c in moves:
                distance = (abs(c[0] - 2) ** 2 + abs(c[1] - 2) ** 2) ** (1 / 2)
                if distance < max_distance:
                    res = c
                    max_distance = distance
        return res

    def judge_surround(self, position, board, piece_type):
        q = queue.PriorityQueue(25)  # 优先级,优先级用数字表示,数字越小优先级越高
        for i in range(5):
            for j in range(5):
                if board[i][j] != 0:
                    distance = ((position[0] - i) ** 2 + (position[1] - j) ** 2) ** (1 / 2)
                    q.put((distance, board[i][j]))
        # 前三个最近的棋子都是对方的
        i = 0
        while not q.empty():
            v = q.get()
            if v[1] == piece_type:
                return False
            if i == 2 and v[1] != piece_type:
                return True
            i += 1
        return False

    def filter_moves_when_too_many_moves(self, go, moves, piece_type):
        res = moves
        # 不走角落
        if (0, 0) in res:
            res.remove((0, 0))
        if (0, 4) in res:
            res.remove((0, 4))
        if (4, 0) in res:
            res.remove((4, 0))
        if (4, 4) in res:
            res.remove((4, 4))
        # 不走对方的包围圈内
        for r in res:
            if r[0] == 0 or r[0] == 4 or r[1] == 0 or r[1] == 4:
                if self.judge_surround(r, go.board, piece_type):
                    res.remove(r)
        return res

    def black_aggressive_2_when_too_many_moves(self, go, moves):
        for m in moves:
            if m[0] == 0:
                if go.board[0][m[1] + 1] == 2 or go.board[0][m[1] - 1] == 2:
                    if go.board[1][m[1]] == 1:
                        return m
            if m[0] == 4:
                if go.board[4][m[1] + 1] == 2 or go.board[4][m[1] - 1] == 2:
                    if go.board[3][m[1]] == 1:
                        return m
            if m[1] == 0:
                if go.board[m[0] + 1][0] == 2 or go.board[m[0] - 1][0] == 2:
                    if go.board[m[0]][1] == 1:
                        return m
            if m[1] == 4:
                if go.board[m[0] + 1][4] == 2 or go.board[m[0] - 1][4] == 2:
                    if go.board[m[0]][3] == 1:
                        return m
        return None

    def get_input(self, go, piece_type):
        '''
        遍历所有可行走法，根据best_result返回结果挑选最合适的走法
        '''

        moves = []

        start = [(2, 2), (3, 3), (3, 1), (1, 3), (1, 1)]
        if self.judge_num(go.board) <= 4:
            # 白棋第二步
            if piece_type == 2 and self.judge_num(go.board) == 3 and self.white_start_policy_second(go):
                return self.white_start_policy_second(go)
            # 黑棋第二步
            if piece_type == 1 and self.judge_num(go.board) == 2 and self.black_start_policy_second(go)[1]:
                return self.black_start_policy_second(go)[0]
            # 黑棋第三步
            if piece_type == 1 and self.judge_num(go.board) == 4 and self.black_start_policy_third(go)[1]:
                return self.black_start_policy_third(go)[0]
            for move in start:
                if go.valid_place_check(move[0], move[1], piece_type):
                    return move

        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type):
                    possible_placements.append((i, j))

        if not possible_placements:
            return "PASS"

        res = self.MIN_SCORE - 100
        for possible_move in possible_placements:
            next_go = deepcopy(go)
            if possible_move != "PASS":
                # If invalid input, continue the loop. Else it places a chess on the board.
                if not next_go.place_chess(possible_move[0], possible_move[1], piece_type):
                    continue
                next_go.died_pieces = next_go.remove_died_pieces(3 - piece_type)  # Remove the dead pieces of opponent
            else:
                next_go.previous_board = deepcopy(next_go.board)
            # 尝试一种可行走法后，看对手可以得到什么好处
            next_go.n_move += 1
            opponent_best_outcome = self.best_result(next_go, 2, self.capture_diff, 3 - piece_type, self.MIN_SCORE,
                                                     self.MIN_SCORE)

            our_best_outcome = -1 * opponent_best_outcome + len(next_go.died_pieces) * 0.01

            if our_best_outcome > res:
                res = our_best_outcome
                moves.clear()
                moves.append(possible_move)
            elif our_best_outcome == res:
                moves.append(possible_move)
        # 白棋第三步
        if piece_type == 2 and go.n_move == 5 and len(moves) >= 8:
            moves = self.white_start_policy_third(go, moves)
        # 黑棋第六步,打aggressive
        if piece_type == 1 and go.n_move == 10 and len(moves) >= 8:
            return self.black_aggressive_when_too_many_moves(go, moves) if self.black_aggressive_when_too_many_moves(go,
                                                                                                                     moves) else random.choice(
                moves)
        # 白棋第四五六步
        if piece_type == 2 and 7 <= go.n_move <= 11 and len(moves) >= 6:
            moves = self.filter_moves_when_too_many_moves(go, moves, piece_type)
        # 黑棋白棋七八九十步（中期）
        if 12 <= go.n_move <= 19 and len(moves) >= 4:
            moves = self.filter_moves_when_too_many_moves(go, moves, piece_type)
            # 黑棋第八步,打aggressive
            if piece_type == 1 and go.n_move == 14:
                return self.black_aggressive_2_when_too_many_moves(go,
                                                                   moves) if self.black_aggressive_2_when_too_many_moves(
                    go, moves) else random.choice(moves)
        if len(moves) < 1:
            return "PASS"
        return random.choice(moves)

    def best_result(self, go, max_depth, eval_fn, piece_type, best_black, best_white):
        if go.game_end():
            result = go.judge_winner()
            if result == piece_type:
                return self.MAX_SCORE
            elif result == 3 - piece_type:
                return self.MIN_SCORE

            # max_depth变量控制遍历深度
        if max_depth == 0:
            # 抵达允许的最大层次后评估局面好坏
            re = eval_fn(go, piece_type)
            return re

        best_so_far = self.MIN_SCORE - 100
        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type):
                    possible_placements.append((i, j))
        for possible_move in possible_placements:
            next_go = deepcopy(go)
            if possible_move != "PASS":
                # If invalid input, continue the loop. Else it places a chess on the board.
                if not next_go.place_chess(possible_move[0], possible_move[1], piece_type):
                    continue
                next_go.died_pieces = next_go.remove_died_pieces(3 - piece_type)  # Remove the dead pieces of opponent
            else:
                next_go.previous_board = deepcopy(next_go.board)
            next_go.n_move += 1
            opponent_best_result = self.best_result(next_go, max_depth - 1, eval_fn, 3 - piece_type, best_black,
                                                    best_white)

            our_result = -1 * opponent_best_result + len(next_go.died_pieces) * 0.01

            if our_result > best_so_far:
                best_so_far = our_result

            if piece_type == 2:
                if best_so_far > best_white:
                    best_white = best_so_far
                outcome_for_black = -1 * best_so_far
                if outcome_for_black < best_black:
                    # 如果白棋落子，选择能够减少黑棋当前最好得分的步骤
                    return best_so_far
            elif piece_type == 1:
                # 如果是黑棋落子，那么选择能够减少白棋当前最好得分的步骤
                if best_so_far > best_black:
                    best_black = best_so_far
                outcome_for_white = -1 * best_so_far
                if outcome_for_white < best_white:
                    return best_so_far

        return best_so_far

    def capture_diff(self, go, piece_type):
        '''
        计算双方在棋盘上的棋子数量，进而评估当前局面好坏，己方的棋子数比对方多意味着局面好
        '''
        one_stones = go.score(1)
        two_stones = go.score(2)

        diff = one_stones - two_stones
        if piece_type == 1:
            return diff
        return -1 * diff


if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = MiniMaxPlayer()
    player.set_side(piece_type)

    num = player.judge_num(go.board)
    if num == 0:
        move = 0
        with open("1.txt", 'w') as f:
            f.write(str(move))
        f.close()
    elif num == 1:
        move = 1
        with open("1.txt", 'w') as f:
            f.write(str(move))
        f.close()
    else:
        with open("1.txt", 'r') as f:
            lines = f.readlines()
            move1 = int(lines[0])
            move1 += 2
        move = move1
        with open("1.txt", 'w') as f:
            f.write(str(move))
        f.close()
    go.n_move = move

    action = player.get_input(go, piece_type)

    writeOutput(action)
