import time
from queue import Queue
from queue import PriorityQueue


def writeFAIL():
    with open('output.txt', 'w') as f:
        f.write('FAIL\n')
    f.close()


def writeSucc(path, steps, cost_all):
    with open('output.txt', 'w') as f:
        f.write(str(cost_all) + '\n')
        f.write(str(steps) + '\n')
        for i in range(len(path)):
            if i == len(path) - 1:
                f.write(path[i])
            else:
                f.write(path[i] + '\n')
    f.close()


def findIndex(point, points):
    points_type = type(points)
    if points_type == type({}):
        return points.get(point, -1)
    else:
        try:
            index = points.index(point)
        except:
            index = -1
        return index


def traceback(exit, entrance, parents, weight, weight_type):
    temp = exit
    path = []
    cost_all = 0
    steps = 0
    if weight_type:
        while True:
            now = list(temp)
            next = parents[temp]
            if getEucli_dis(now, next) > 1:
                now.append(int(weight * 1.4))
                path.append(' '.join(list(map(str, now))))
                cost_all += int(weight * 1.4)
            else:
                now.append(weight)
                path.append(' '.join(list(map(str, now))))
                cost_all += weight
            steps += 1
            temp = next
            if temp == entrance:
                temp = list(temp)
                temp.append(0)
                steps += 1
                path.append(' '.join(list(map(str, temp))))
                break
    else:
        while True:
            now = list(temp)
            now.append(weight)
            cost_all += weight
            steps += 1
            path.append(' '.join(list(map(str, now))))
            temp = parents[temp]
            if temp == entrance:
                temp = list(temp)
                temp.append(0)
                steps += 1
                path.append(' '.join(list(map(str, temp))))
                break
    path.reverse()
    return path, steps, cost_all


def getEucli_dis(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2) ** 0.5


def initializeHNodes(exit, points):
    h_nodes = []
    for point in points:
        h_nodes.append(getEucli_dis(point, exit))
    return h_nodes


def checkpointinboundary(point, XYZ_Boundary):
    if 0 <= point[0] < XYZ_Boundary[0] and 0 <= point[1] < XYZ_Boundary[1] and 0 <= point[2] < XYZ_Boundary[2]:
        return True
    else:
        return False


# def sortOpenByF(open, points, f_nodes):
#     new_f_nodes = []
#     for openitem in open:
#         item_index = findIndex(openitem, points)
#         new_f_nodes.append(f_nodes[item_index])
#     sorted_q = sorted(enumerate(new_f_nodes), key=lambda x: x[1])
#     sorted_index = [i[0] for i in sorted_q]
#     # print(sorted_index)
#     new_open = [open[i] for i in sorted_index]
#     return new_open

def binarySearchInsert(open_queue, prev_flag, now_flag, k_nodes, child_k):
    temp_flag = int((prev_flag + now_flag) / 2)
    if child_k <= k_nodes[temp_flag] and prev_flag > now_flag:
        prev_flag = now_flag
        now_flag = temp_flag
        if now_flag - prev_flag < 2:
            return now_flag
        else:
            return binarySearchInsert(open_queue, prev_flag, now_flag, k_nodes, child_k)
    elif child_k <= k_nodes[temp_flag] and prev_flag < now_flag:
        now_flag = temp_flag
        if now_flag - prev_flag < 2:
            return now_flag
        else:
            return binarySearchInsert(open_queue, prev_flag, now_flag, k_nodes, child_k)
    elif child_k > k_nodes[temp_flag] and prev_flag > now_flag:
        now_flag = temp_flag
        if prev_flag - now_flag < 2:
            return prev_flag
        else:
            return binarySearchInsert(open_queue, prev_flag, now_flag, k_nodes, child_k)
    else:
        prev_flag = now_flag
        now_flag = temp_flag
        if prev_flag - now_flag < 2:
            return prev_flag
        else:
            return binarySearchInsert(open_queue, prev_flag, now_flag, k_nodes, child_k)


def AStar(points, XYZ_Boundary, entrance, exit):
    open_queue = PriorityQueue()
    closed_queue = set()
    parents = {}
    g_points = {}
    f_points = {}
    g_points[entrance] = 0
    f_points[entrance] = getEucli_dis(entrance,exit) + g_points[entrance]
    open_queue.put((f_points[entrance], entrance))
    found = False
    while not open_queue.empty():
        parent = open_queue.get()[1]
        if parent == exit:
            found = True
            break
        closed_queue.add(parent)
        parent_actions = points[parent]
        for action in parent_actions:
            weight = 10
            if action == 1:
                child = [parent[0] + 1, parent[1], parent[2]]
            elif action == 2:
                child = [parent[0] - 1, parent[1], parent[2]]
            elif action == 3:
                child = [parent[0], parent[1] + 1, parent[2]]
            elif action == 4:
                child = [parent[0], parent[1] - 1, parent[2]]
            elif action == 5:
                child = [parent[0], parent[1], parent[2] + 1]
            elif action == 6:
                child = [parent[0], parent[1], parent[2] - 1]
            elif action == 7:
                child = [parent[0] + 1, parent[1] + 1, parent[2]]
                weight = int(weight * 1.4)
            elif action == 8:
                child = [parent[0] + 1, parent[1] - 1, parent[2]]
                weight = int(weight * 1.4)
            elif action == 9:
                child = [parent[0] - 1, parent[1] + 1, parent[2]]
                weight = int(weight * 1.4)
            elif action == 10:
                child = [parent[0] - 1, parent[1] - 1, parent[2]]
                weight = int(weight * 1.4)
            elif action == 11:
                child = [parent[0] + 1, parent[1], parent[2] + 1]
                weight = int(weight * 1.4)
            elif action == 12:
                child = [parent[0] + 1, parent[1], parent[2] - 1]
                weight = int(weight * 1.4)
            elif action == 13:
                child = [parent[0] - 1, parent[1], parent[2] + 1]
                weight = int(weight * 1.4)
            elif action == 14:
                child = [parent[0] - 1, parent[1], parent[2] - 1]
                weight = int(weight * 1.4)
            elif action == 15:
                child = [parent[0], parent[1] + 1, parent[2] + 1]
                weight = int(weight * 1.4)
            elif action == 16:
                child = [parent[0], parent[1] + 1, parent[2] - 1]
                weight = int(weight * 1.4)
            elif action == 17:
                child = [parent[0], parent[1] - 1, parent[2] + 1]
                weight = int(weight * 1.4)
            elif action == 18:
                child = [parent[0], parent[1] - 1, parent[2] - 1]
                weight = int(weight * 1.4)
            else:
                print('[' + str(parent[0]) + ' ' + str(parent[1]) + ' ' + str(parent[2]) + '] ' + ' action ' + str(
                    action) + ' not valid\n')
                continue
            child = tuple(child)
            if points.get(child, -1) == -1:
                print('[' + str(parent[0]) + ' ' + str(parent[1]) + ' ' + str(parent[2]) + '] ' + ' action ' + str(
                    action) + ' not lead to an exist point ' + '[' + str(child[0]) + ' ' + str(
                    child[1]) + ' ' + str(
                    child[2]) + '] (new point not found)' + '\n')
                continue
            if not checkpointinboundary(child, XYZ_Boundary):
                print('[' + str(parent[0]) + ' ' + str(parent[1]) + ' ' + str(parent[2]) + '] ' + ' action ' + str(
                    action) + ' not lead to an exist point ' + '[' + str(child[0]) + ' ' + str(
                    child[1]) + ' ' + str(
                    child[2]) + '] (out of boundary)' + '\n')
                continue
            else:
                child_g = g_points[parent] + weight
                child_f = g_points[parent] + weight + getEucli_dis(child, exit)
                if child not in closed_queue:
                    open_queue.put((child_f, child))
                    closed_queue.add(child)
                    g_points[child] = child_g
                    f_points[child] = child_f
                    parents[child] = parent
                else:
                    if child_g < g_points[child]:
                        f_points[child] = child_f
                        g_points[child] = child_g
                        parents[child] = parent
                        closed_queue.remove(child)
                        open_queue.put((child_f, child))
    if found:
        path, steps, cost_all = traceback(exit, entrance, parents, 10, True)
        writeSucc(path, steps, cost_all)
    else:
        writeFAIL()


def BFS(points, XYZ_Boundary, entrance, exit):
    open_queue = Queue()
    closed_queue = set()
    parents = {}
    found = False
    open_queue.put(entrance)
    while not open_queue.empty():
        parent = open_queue.get()
        if parent == exit:
            found = True
            break
        closed_queue.add(parent)
        parent_actions = points[parent]
        for action in parent_actions:
            if action == 1:
                child = (parent[0] + 1, parent[1], parent[2])
            elif action == 2:
                child = (parent[0] - 1, parent[1], parent[2])
            elif action == 3:
                child = (parent[0], parent[1] + 1, parent[2])
            elif action == 4:
                child = (parent[0], parent[1] - 1, parent[2])
            elif action == 5:
                child = (parent[0], parent[1], parent[2] + 1)
            elif action == 6:
                child = (parent[0], parent[1], parent[2] - 1)
            elif action == 7:
                child = (parent[0] + 1, parent[1] + 1, parent[2])
            elif action == 8:
                child = (parent[0] + 1, parent[1] - 1, parent[2])
            elif action == 9:
                child = (parent[0] - 1, parent[1] + 1, parent[2])
            elif action == 10:
                child = (parent[0] - 1, parent[1] - 1, parent[2])
            elif action == 11:
                child = (parent[0] + 1, parent[1], parent[2] + 1)
            elif action == 12:
                child = (parent[0] + 1, parent[1], parent[2] - 1)
            elif action == 13:
                child = (parent[0] - 1, parent[1], parent[2] + 1)
            elif action == 14:
                child = (parent[0] - 1, parent[1], parent[2] - 1)
            elif action == 15:
                child = (parent[0], parent[1] + 1, parent[2] + 1)
            elif action == 16:
                child = (parent[0], parent[1] + 1, parent[2] - 1)
            elif action == 17:
                child = (parent[0], parent[1] - 1, parent[2] + 1)
            elif action == 18:
                child = (parent[0], parent[1] - 1, parent[2] - 1)
            else:
                print('[' + str(parent[0]) + ' ' + str(parent[1]) + ' ' + str(parent[2]) + '] ' + ' action ' + str(
                    action) + ' not valid\n')
                continue
            if points.get(child, -1) == -1:
                print('[' + str(parent[0]) + ' ' + str(parent[1]) + ' ' + str(parent[2]) + '] ' + ' action ' + str(
                    action) + ' not lead to an exist point ' + '[' + str(child[0]) + ' ' + str(
                    child[1]) + ' ' + str(
                    child[2]) + '] (new point not found)' + '\n')
                continue
            if not checkpointinboundary(child, XYZ_Boundary):
                print('[' + str(parent[0]) + ' ' + str(parent[1]) + ' ' + str(parent[2]) + '] ' + ' action ' + str(
                    action) + ' not lead to an exist point ' + '[' + str(child[0]) + ' ' + str(
                    child[1]) + ' ' + str(
                    child[2]) + '] (out of boundary)' + '\n')
                continue
            else:
                if child not in closed_queue:
                    parents[child] = parent
                    closed_queue.add(child)
                    open_queue.put(child)
    if found:
        path, steps, cost_all = traceback(exit, entrance, parents, 1, False)
        writeSucc(path, steps, cost_all)
    else:
        writeFAIL()
        return


def UCS(points, XYZ_Boundary, entrance, exit):
    open_queue = PriorityQueue()
    closed_queue = set()
    parents = {}
    g_points = {}
    g_points[entrance] = 0
    open_queue.put((0, entrance))
    found = False
    while not open_queue.empty():
        parent = open_queue.get()[1]
        if parent == exit:
            found = True
            break
        closed_queue.add(parent)
        parent_actions = points[parent]
        for action in parent_actions:
            weight = 10
            if action == 1:
                child = [parent[0] + 1, parent[1], parent[2]]
            elif action == 2:
                child = [parent[0] - 1, parent[1], parent[2]]
            elif action == 3:
                child = [parent[0], parent[1] + 1, parent[2]]
            elif action == 4:
                child = [parent[0], parent[1] - 1, parent[2]]
            elif action == 5:
                child = [parent[0], parent[1], parent[2] + 1]
            elif action == 6:
                child = [parent[0], parent[1], parent[2] - 1]
            elif action == 7:
                child = [parent[0] + 1, parent[1] + 1, parent[2]]
                weight = int(weight * 1.4)
            elif action == 8:
                child = [parent[0] + 1, parent[1] - 1, parent[2]]
                weight = int(weight * 1.4)
            elif action == 9:
                child = [parent[0] - 1, parent[1] + 1, parent[2]]
                weight = int(weight * 1.4)
            elif action == 10:
                child = [parent[0] - 1, parent[1] - 1, parent[2]]
                weight = int(weight * 1.4)
            elif action == 11:
                child = [parent[0] + 1, parent[1], parent[2] + 1]
                weight = int(weight * 1.4)
            elif action == 12:
                child = [parent[0] + 1, parent[1], parent[2] - 1]
                weight = int(weight * 1.4)
            elif action == 13:
                child = [parent[0] - 1, parent[1], parent[2] + 1]
                weight = int(weight * 1.4)
            elif action == 14:
                child = [parent[0] - 1, parent[1], parent[2] - 1]
                weight = int(weight * 1.4)
            elif action == 15:
                child = [parent[0], parent[1] + 1, parent[2] + 1]
                weight = int(weight * 1.4)
            elif action == 16:
                child = [parent[0], parent[1] + 1, parent[2] - 1]
                weight = int(weight * 1.4)
            elif action == 17:
                child = [parent[0], parent[1] - 1, parent[2] + 1]
                weight = int(weight * 1.4)
            elif action == 18:
                child = [parent[0], parent[1] - 1, parent[2] - 1]
                weight = int(weight * 1.4)
            else:
                print('[' + str(parent[0]) + ' ' + str(parent[1]) + ' ' + str(parent[2]) + '] ' + ' action ' + str(
                    action) + ' not valid\n')
                continue
            child = tuple(child)
            if points.get(child, -1) == -1:
                print('[' + str(parent[0]) + ' ' + str(parent[1]) + ' ' + str(parent[2]) + '] ' + ' action ' + str(
                    action) + ' not lead to an exist point ' + '[' + str(child[0]) + ' ' + str(
                    child[1]) + ' ' + str(
                    child[2]) + '] (new point not found)' + '\n')
                continue
            if not checkpointinboundary(child, XYZ_Boundary):
                print('[' + str(parent[0]) + ' ' + str(parent[1]) + ' ' + str(parent[2]) + '] ' + ' action ' + str(
                    action) + ' not lead to an exist point ' + '[' + str(child[0]) + ' ' + str(
                    child[1]) + ' ' + str(
                    child[2]) + '] (out of boundary)' + '\n')
                continue
            else:
                child_g = g_points[parent] + weight

                if child not in closed_queue:
                    open_queue.put((child_g, child))
                    closed_queue.add(child)
                    g_points[child] = child_g
                    parents[child] = parent
                else:
                    if child_g < g_points[child]:
                        g_points[child] = child_g
                        parents[child] = parent
                        closed_queue.remove(child)
                        open_queue.put((child_g, child))
    if found:
        path, steps, cost_all = traceback(exit, entrance, parents, 10, True)
        writeSucc(path, steps, cost_all)
    else:
        writeFAIL()


def handleMethods(method: str, entrance: tuple, exit: tuple, points: dict, XYZ_Boundary: tuple):
    if len(points) <= 0:
        writeFAIL()
        return
    if entrance not in points or exit not in points:
        writeFAIL()
        return
    if method == 'BFS':
        BFS(points, XYZ_Boundary, entrance, exit)
    elif method == 'UCS':
        UCS(points, XYZ_Boundary, entrance, exit)
    elif method == 'A*':
        AStar(points, XYZ_Boundary, entrance, exit)
    else:
        print('method read failed')
        writeFAIL()


def main_function():
    start_all = time.time()
    with open("input.txt") as f:
        line = f.readline().rstrip()
        method = line
        XYZ_Boundary = tuple(map(int, f.readline().rstrip().split(' ')))
        entrance = tuple(map(int, f.readline().rstrip().split(' ')))
        exit = tuple(map(int, f.readline().rstrip().split(' ')))
        gridNum = int(f.readline().rstrip())
        points = {}
        for i in range(gridNum):
            grid_Temp = list(map(int, f.readline().rstrip().split(' ')))
            points[(grid_Temp[0], grid_Temp[1], grid_Temp[2])] = grid_Temp[3:]
        print('read success')
        handleMethods(method, entrance, exit, points, XYZ_Boundary)
    f.close()
    end_all = time.time()
    print('All take time: ' + str(end_all - start_all))


def testFunc():
    list_a = []
    dict_b = {}
    findIndex(1, list_a)


if __name__ == '__main__':
    main_function()
    # testFunc()
