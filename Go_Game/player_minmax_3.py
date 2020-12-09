import random
import time
from read import readInput
from write import writeOutput
from my_Go import GO
import queue
from copy import deepcopy
import numpy as np
import collections


class MinMaxPlayer:
    def cal_play_num(self, board):
        play_num = 0
        for i in range(5):
            for j in range(5):
                if board[i][j] != 0:
                    play_num += 1
        return play_num

    def random_play(self, go, piece_type):
        possible_placements = []
        for i in range(5):
            for j in range(5):
                if go.valid_place_check(i, j, piece_type):
                    possible_placements.append((i, j))

        if len(possible_placements) == 0:
            return "PASS"
        else:
            return random.choice(possible_placements)

    # try to occupy four coner of the board [(1,1), (1,3), (3,1), (3,3)] and center of board
    def check_key_places(self, go, piece_type):
        count = 0
        for place in [(2, 2), (1, 1), (1, 3), (3, 1), (3, 3)]:
            if go.valid_place_check(place[0], place[1], piece_type):
                count += 1
        return count

    def try_play_key_places(self, go, piece_type):
        solution = []
        for place in [(2, 2), (1, 1), (1, 3), (3, 1), (3, 3)]:
            if go.valid_place_check(place[0], place[1], piece_type):
                ally_num, enemy_num = go.find_neighbor_ally_enemy_num(place[0], place[1], piece_type)
                if ally_num > 3 or enemy_num > 2:
                    continue
                solution.append(place)
        if len(solution) < 1:
            return []
        if (2, 2) in solution:
            return [(2, 2)]
        # return [solution[0]]
        return [random.choice(solution)]

    # try to protect myself
    def try_protect_myself(self, go, piece_type):
        next_piece = 3 - piece_type
        protect_place = []
        protect_place_weight = []
        for i in range(5):
            for j in range(5):
                if go.board[i][j] == piece_type:
                    liberty_group = go.find_liberty_pieces_position(i, j)
                    if len(liberty_group) < 2:  # if my_ally liberty == 1
                        ally_group = go.find_all_group_members(i, j)
                        for liberty_empty in liberty_group:
                            if go.valid_place_check(liberty_empty[0], liberty_empty[1], next_piece):
                                next_go_me = go.copy_board()
                                next_go_me.place_chess(liberty_empty[0], liberty_empty[1], piece_type)
                                next_go_me.died_pieces = next_go_me.remove_died_pieces(next_piece)
                                next_liberty = next_go_me.find_liberty_pieces_position(liberty_empty[0], liberty_empty[1])
                                if len(next_liberty) < 2:
                                    continue
                                my_allies = len(ally_group)
                                protect_place.append((liberty_empty[0], liberty_empty[1]))
                                protect_place_weight.append(my_allies)
        if len(protect_place_weight) > 2:
            index = sorted(range(len(protect_place_weight)), key=lambda i: protect_place_weight[i])[-2:]
            protect_place = np.array(protect_place)[index].tolist()
            protect_place_weight = np.array(protect_place_weight)[index].tolist()
        protect_place_t = []
        for place in protect_place:
            protect_place_t.append((place[0], place[1]))
        return protect_place_t, protect_place_weight

    # try to eat pieces
    def try_attack(self, go, piece_type):  # greedy
        attack_place = []
        attack_place_weight = []
        for i in range(5):
            for j in range(5):
                if go.valid_place_check(i, j, piece_type):
                    next_board = go.copy_board()
                    next_board.place_chess(i, j, piece_type)
                    next_board.died_pieces = next_board.find_died_pieces(3 - piece_type)
                    remove_count = len(next_board.died_pieces)
                    if remove_count > 0:
                        attack_place.append((i, j))
                        attack_place_weight.append(remove_count)
        if len(attack_place_weight) > 2:
            index = sorted(range(len(attack_place_weight)), key=lambda i: attack_place_weight[i])[-2:]
            attack_place = np.array(attack_place)[index].tolist()
            attack_place_weight = np.array(attack_place_weight)[index].tolist()
        attack_place_t = []
        for place in attack_place:
            attack_place_t.append((place[0], place[1]))
        return attack_place_t, attack_place_weight

    # try to make a group of enemies only one liberty left
    def try_capture_enemy_1_liberty_left(self, go, piece_type):
        capture_enemy = []
        threaten_weight = []
        for x in range(5):
            for y in range(5):
                if go.board[x][y] == 3 - piece_type:
                    liberty = go.find_liberty_pieces_position(x, y)
                    if len(liberty) == 2:
                        empties = liberty.keys()
                        empty_now = []
                        flag = True
                        for empty in empties:
                            if go.valid_place_check(empty[0], empty[1], piece_type):
                                next_board = go.copy_board()
                                next_board.place_chess(empty[0], empty[1], piece_type)
                                next_board.died_pieces = next_board.remove_died_pieces(3 - piece_type)
                                liberty_new = next_board.find_liberty_pieces_position(empty[0], empty[1])
                                empty_neighbor_num = len(next_board.find_all_neighbors(empty[0], empty[1]))
                                if (empty_neighbor_num > 2 and len(liberty_new) < 2) or (
                                        empty_neighbor_num == 2 and len(liberty_new) == 0):
                                    flag = False
                                    break
                                elif (empty_neighbor_num == 2 and len(liberty_new) == 1):
                                    continue
                                empty_now.append((empty[0], empty[1]))
                        if flag:
                            allies_of_enemy_num = len(go.find_all_group_members(x, y))
                            capture_enemy.append(empty_now)
                            threaten_weight.append(allies_of_enemy_num)
        capture_enemy_temp = []
        if len(threaten_weight) > 2:
            index = sorted(range(len(threaten_weight)), key=lambda i: threaten_weight[i])[-2:]
            capture_enemy = np.array(capture_enemy, dtype=object)[index].tolist()
            for eachpair in capture_enemy:
                capture_enemy_temp += eachpair
            threaten_weight = np.array(threaten_weight)[index].tolist()
        else:
            for eachpair in capture_enemy:
                capture_enemy_temp += eachpair
        capture_enemy_t = []
        for pairs in capture_enemy_temp:
            capture_enemy_t.append((pairs[0], pairs[1]))
        return capture_enemy_t, threaten_weight

    # make a trap to make one place become invalid (fake eye)
    def try_make_trap(self, go, piece_type):
        trap_action = []
        for i in range(5):
            for j in range(5):
                if go.valid_place_check(i, j, piece_type):
                    next_go = go.copy_board()
                    next_go.place_chess(i, j, piece_type)
                    next_go.died_pieces = next_go.remove_died_pieces(3 - piece_type)
                    for x, y in next_go.find_all_neighbors(i, j):
                        if next_go.board[x][y] == 0 and not next_go.valid_place_check(x, y, 3 - piece_type):
                            trap_action.append((i, j))
        return trap_action

    # try to connect ally and make them have more liberty
    def try_connect_ally(self, go, piece_type):
        connect_ally = []
        for x in range(5):
            for y in range(5):
                if go.board[x][y] == piece_type:
                    liberty = go.find_liberty_pieces_position(x, y)
                    if len(liberty) <= 2:
                        for empty in liberty.keys():
                            if go.valid_place_check(empty[0], empty[1], piece_type):
                                next_go = go.copy_board()
                                next_go.place_chess(empty[0], empty[1], piece_type)
                                next_go.died_pieces = next_go.remove_died_pieces(piece_type)
                                new_liberty = next_go.find_liberty_pieces_position(empty[0], empty[1])
                                if len(new_liberty) >= len(liberty):
                                    connect_ally.append((empty[0], empty[1]))
        return connect_ally

    # try to truncate enemy or just make enemy liberty less
    # truncate is prior to liberty less
    def try_truncate_offense(self, go, piece_type):
        opponent_type = 3 - piece_type
        truncate_offense_place = []
        ally_dict = collections.defaultdict(int)
        open_dict = collections.defaultdict(list)
        for x in range(5):
            for y in range(5):
                if go.is_connection_of_groups(x, y, opponent_type):
                    next_go_me = go.copy_board()
                    next_go_me.place_chess(x, y, piece_type)
                    next_go_me.died_pieces = next_go_me.remove_died_pieces(opponent_type)
                    me_liberty_after_place = next_go_me.find_liberty_pieces_position(x, y)
                    if len(me_liberty_after_place) > 2:
                        truncate_offense_place.append((x, y))
                elif go.board[x][y] == 3 - piece_type:
                    cur_neighbor = go.find_all_neighbors(x, y)
                    for place in cur_neighbor:
                        if go.board[place[0]][place[1]] == piece_type:
                            ally_dict[(x, y)] += 1
                        if go.board[place[0]][place[1]] == 0 and go.valid_place_check(place[0], place[1], piece_type):
                            open_dict[(x, y)].append((place[0], place[1]))

        res = list()
        for key, value in ally_dict.items():
            if value == max(ally_dict.values()) and key in open_dict.keys():
                res += open_dict[key]
        for temp in res:
            truncate_offense_place.append((temp[0], temp[1]))

        return truncate_offense_place

    # get all possible choices
    def get_all_possible(self, go, piece_type):
        self_fill = []
        all_except_self_fill = []
        for x in range(5):
            for y in range(5):
                if go.valid_place_check(x, y, piece_type):
                    ally_num, enemy_num = go.find_neighbor_ally_enemy_num(x, y, piece_type)
                    if ally_num > 3 or (len(go.find_all_neighbors(x, y)) < 4 and ally_num > 2) or (
                            len(go.find_all_neighbors(x, y)) < 3 and ally_num > 1):
                        self_fill.append((x, y))
                    else:
                        all_except_self_fill.append((x, y))
        return self_fill, all_except_self_fill

    # filter corner and danger surrounding if a lot pieces
    def filter_corner_surrounding(self, go, possible_places, piece_type):
        temp = deepcopy(possible_places)
        # filter corner
        if (0, 0) in temp:
            temp.remove((0, 0))
        if (0, 4) in temp:
            temp.remove((0, 4))
        if (4, 0) in temp:
            temp.remove((4, 0))
        if (4, 4) in temp:
            temp.remove((4, 4))

        # filter danger surrounding
        for r in temp:
            if r[0] == 0 or r[0] == 4 or r[1] == 0 or r[1] == 4:
                if self.detect_surround(r[0], r[1], go.board, piece_type):
                    temp.remove(r)

        return temp

    # filter pieces will be eaten
    def filter_piece_will_be_eaten(self, go, possible_places, piece_type):
        temp = deepcopy(possible_places)
        for piece in temp:
            if go.valid_place_check(piece[0], piece[1], piece_type):
                next_go = go.copy_board()
                next_go.place_chess(piece[0], piece[1], piece_type)
                next_go.died_pieces = next_go.remove_died_pieces(3 - piece_type)
                ally_liberty = next_go.find_liberty_pieces_position(piece[0], piece[1])
                if len(ally_liberty) == 1:
                    liberty_piece = list(ally_liberty.keys())[0]
                    if next_go.valid_place_check(liberty_piece[0],liberty_piece[1],3-piece_type):
                        temp.remove(piece)
        return temp

    def get_possible_good_place(self, go, piece_type):
        protect_place, protect_place_weight = self.try_protect_myself(go, piece_type)
        attack, attack_weight = self.try_attack(go, piece_type)
        all_possible_places = list(set(attack + protect_place))
        if len(all_possible_places) < 1:
            hold_corner = self.try_play_key_places(go, piece_type)
            all_possible_places = hold_corner
            if len(all_possible_places) < 1:
                capture_enemy, threaten_weight = self.try_capture_enemy_1_liberty_left(go, piece_type)
                make_trap = self.try_make_trap(go, piece_type)
                if piece_type == 2:
                    connect_ally = []
                else:
                    connect_ally = self.try_connect_ally(go, piece_type)
                truncate_offense = self.try_truncate_offense(go, piece_type)
                aggressive_black = []
                if go.get_move() == 11 and piece_type == 1:
                    self_fill, all_except_self_fill = self.get_all_possible(go, piece_type)
                    all_p = self_fill + all_except_self_fill
                    all_p = self.filter_corner_surrounding(go, all_p, piece_type)
                    temp_piece = self.aggressive_black_mid(go, all_p)
                    if temp_piece:
                        aggressive_black.append(temp_piece)
                elif go.get_move() == 15 and piece_type == 1:
                    self_fill, all_except_self_fill = self.get_all_possible(go, piece_type)
                    all_p = self_fill + all_except_self_fill
                    all_p = self.filter_corner_surrounding(go, all_p, piece_type)
                    temp_piece = self.aggressive_black_a_while(go, all_p)
                    if temp_piece:
                        aggressive_black.append(temp_piece)
                all_possible_places = list(set(aggressive_black + connect_ally + capture_enemy + make_trap + truncate_offense))
                if go.get_move() <= 23:
                    all_possible_places = self.filter_piece_will_be_eaten(go, all_possible_places, piece_type)
                if len(all_possible_places) < 1:
                    self_fill, all_except_self_fill = self.get_all_possible(go, piece_type)
                    all_possible_places = list(set(all_possible_places + all_except_self_fill))
                    if 6 < go.get_move() < 13:
                        all_possible_places = self.filter_corner_surrounding(go, all_possible_places, piece_type)
                    if go.get_move() <= 23:
                        all_possible_places = self.filter_piece_will_be_eaten(go, all_possible_places, piece_type)
                    if len(all_possible_places) <= 7:
                        all_possible_places = list(set(all_possible_places + self_fill))
                        if 6 < go.get_move() < 13:
                            all_possible_places = self.filter_corner_surrounding(go, all_possible_places, piece_type)
                        if go.get_move() <= 23:
                            all_possible_places = self.filter_piece_will_be_eaten(go, all_possible_places, piece_type)
        return all_possible_places

    def get_input(self, go, piece_type):
        # beginning try to reach key places
        if self.check_key_places(go, piece_type) > 0 and go.get_move() < 7:
            occupy_solution = self.try_play_key_places(go, piece_type)
            if len(occupy_solution) > 0:
                print('key places')
                return occupy_solution[0]

        # if cannot reach them, run min-max
        else:
            possible_place = self.get_possible_good_place(go, piece_type)
            if len(possible_place) < 1:
                return "PASS"
            if len(possible_place) == 1:
                return possible_place[0]
            elif len(possible_place) <= 5:
                max_deep = 5
            elif len(possible_place) <= 10:
                max_deep = 4
            else:
                max_deep = 3
            solution, place = self.Min_Max(go, piece_type, possible_place, max_deep, piece_type)

            return place

    # when play a while, tricks for black
    def aggressive_black_mid(self, go, possible_pieces):
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
            if c in possible_pieces:
                distance = (abs(c[0] - 2) ** 2 + abs(c[1] - 2) ** 2) ** (1 / 2)
                if distance < max_distance:
                    res = c
                    max_distance = distance
        return res

    # trick for black when play a lot
    def aggressive_black_a_while(self, go, possible_pieces):
        for m in possible_pieces:
            if m[0] == 0 and 1 <= m[1] <= 3:
                if go.board[0][m[1] + 1] == 2 or go.board[0][m[1] - 1] == 2:
                    if go.board[1][m[1]] == 1:
                        return m
            if m[0] == 4 and 1 <= m[1] <= 3:
                if go.board[4][m[1] + 1] == 2 or go.board[4][m[1] - 1] == 2:
                    if go.board[3][m[1]] == 1:
                        return m
            if m[1] == 0 and 1 <= m[0] <= 3:
                if go.board[m[0] + 1][0] == 2 or go.board[m[0] - 1][0] == 2:
                    if go.board[m[0]][1] == 1:
                        return m
            if m[1] == 4 and 1 <= m[0] <= 3:
                if go.board[m[0] + 1][4] == 2 or go.board[m[0] - 1][4] == 2:
                    if go.board[m[0]][3] == 1:
                        return m
        return None

    def detect_surround(self, i, j, board, piece_type):
        q = queue.PriorityQueue(25)  # priority - the less the bigger
        for x in range(5):
            for y in range(5):
                if board[i][j] != 0:
                    distance = ((i - x) ** 2 + (j - y) ** 2) ** (1 / 2)
                    q.put((distance, board[i][j]))
        # nearest three are opps
        i = 0
        while not q.empty():
            v = q.get()
            if v[1] == piece_type:
                return False
            if i == 2 and v[1] != piece_type:
                return True  # danger
            i += 1
        return False

    # min-max
    def Min_Max(self, go, piece_type, possible_place, max_deep, real_piece_type):
        a = (float("-inf"), float("-inf"), float("-inf"))
        b = (float("inf"), float("inf"), float("inf"))
        value, place = self.find_max_place(go, piece_type, possible_place, a, b, max_deep, real_piece_type)
        return value, place

    def find_max_place(self, go, piece_type, possible_place, a, b, deep, real_piece_type):
        if go.get_move() > go.get_max_move() or len(possible_place) == 0 or deep == 0:
            return self.get_score(go, real_piece_type), None

        cur_max = (float("-inf"), float("-inf"), float("-inf"))
        cur_place = None
        for place in possible_place:
            next_board = go.copy_board()
            next_board.place_chess(place[0], place[1], piece_type)
            next_board.died_pieces = next_board.remove_died_pieces(3 - piece_type)
            children_possible_place = self.get_possible_good_place(next_board, 3 - piece_type)
            next_value, next_position = self.find_min_place(next_board, 3 - piece_type, children_possible_place, a,
                                                            b,
                                                            deep - 1, real_piece_type)
            if cur_max < next_value:
                cur_place = place
                cur_max = next_value
            if cur_max >= b:
                return cur_max, cur_place
            a = max(a, cur_max)

        return cur_max, cur_place

    def find_min_place(self, go, piece_type, possible_place, a, b, deep, real_piece_type):
        if go.get_move() > go.get_max_move() or len(possible_place) == 0 or deep == 0:
            return self.get_score(go, real_piece_type), None

        cur_min = (float("inf"), float("inf"), float("inf"))
        cur_place = None
        for place in possible_place:
            next_board = go.copy_board()
            next_board.place_chess(place[0], place[1], piece_type)
            next_board.died_pieces = next_board.remove_died_pieces(3 - piece_type)
            children_possible_place = self.get_possible_good_place(next_board, 3 - piece_type)
            next_value, next_position = self.find_max_place(next_board, 3 - piece_type, children_possible_place, a,
                                                            b,
                                                            deep - 1, real_piece_type)
            if cur_min > next_value:
                cur_place = place
                cur_min = next_value
            if cur_min <= a:
                return cur_min, cur_place
            b = min(b, cur_min)

        return cur_min, cur_place

    # calculate score
    def get_score(self, go, piece_type):
        if piece_type == 1:
            score, possible_eat_place, liberty_black, liberty_white = 0, 0, 0, 0
            board = go.board
            board_black = go.board
            board_white = go.board
            for x in range(5):
                for y in range(5):
                    if board[x][y] == piece_type:
                        score += 1
                        cur_neighbor = go.find_all_neighbors(x, y)
                        for place in cur_neighbor:
                            if board_black[place[0]][place[1]] == 0:
                                liberty_black += 1
                                board_black[place[0]][place[1]] = -1
                    if board[x][y] == 3 - piece_type:
                        score -= 1
                        cur_neighbor = go.find_all_neighbors(x, y)
                        for place in cur_neighbor:
                            if board_white[place[0]][place[1]] == 0:
                                board_white[place[0]][place[1]] = -1
                                liberty_white += 1
                    if board[x][y] == 0 and go.valid_place_check(x, y, 3 - piece_type):
                        next_board = go.copy_board()
                        next_board.place_chess(x, y, 3 - piece_type)
                        if len(next_board.find_died_pieces(piece_type)) > 0:
                            possible_eat_place += 1

            return score, possible_eat_place, liberty_black, -liberty_white
        else:
            score, liberty_diff, liberty = 0, 0, 0
            board = go.board
            my_liberty_pieces = {}
            enemy_liberty_pieces = {}
            for x in range(5):
                for y in range(5):
                    if board[x][y] == piece_type:
                        score += 1
                        cur_neighbor = go.find_all_neighbors(x, y)
                        for place in cur_neighbor:
                            if board[place[0]][place[1]] == 0 and place not in my_liberty_pieces.keys():
                                liberty_diff += 1
                                liberty += 1
                                my_liberty_pieces[place] = 1
                    if board[x][y] == 3 - piece_type:
                        score -= 1
                        cur_neighbor = go.find_all_neighbors(x, y)
                        for place in cur_neighbor:
                            if board[place[0]][place[1]] == 0 and place not in enemy_liberty_pieces.keys():
                                liberty_diff -= 1
                                enemy_liberty_pieces[place] = 1

            return score, liberty_diff, liberty


if __name__ == "__main__":
    s = time.time()
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.initialize_board(piece_type, previous_board, board)
    player = MinMaxPlayer()

    play_num = player.cal_play_num(go.board)
    if play_num == 0:
        move = 1
        with open("play_num2.txt", 'w') as f:
            f.write(str(1))
        f.close()
    elif play_num == 1:
        move = 2
        with open("play_num2.txt", 'w') as f:
            f.write(str(2))
        f.close()
    else:
        with open("play_num2.txt", 'r') as f:
            move = int(f.readline().strip()) + 2
        f.close()
        with open("play_num2.txt", 'w') as f:
            f.write(str(move))
        f.close()
    go.set_move(move)

    action = player.get_input(go, piece_type)

    writeOutput(action)
    print('Duration:', round(time.time() - s, 2))
