from itertools import permutations
import time
import random
import sys


from heapq import heappush, heappop
from collections import deque

size = 4
goal = "0ABCDEFGHIJKLMNO"
index_dict = {0: "R", 1: "D", 2: "L", 3: "U", "R":0, "D":1, "L":2, "U": 3, (1,0):"R", (-1,0):"L", (0,1):"D", (0,-1): "U"}
position_dict = {}

for x in range(0, size * size):
    position_dict[goal[x]] = (x % size, int(x / size))
    position_dict[(x % size, int(x / size))] = goal[x]
def swap (a: int,b: int ,string: str):
    # returns a new string with the values swapped
    ans = string[:a] + string[b] + string[a+1 : b] + string[a] + string[b+1 :]
    return ans
def reverse(path):
    string = ""
    inverses = {"U":"D", "D":"U", "R":"L", "L":"R"}
    for char in path[::-1]:
        string+= inverses[char]
    return string
def print_puzzle(state):
    if (state == None):
        return
    for y in range(0,size):
        str = ""
        for x in range (0,size):
            str += state[x + size*y] + " "
        print(str)
    print()
def get_children(state):
    # returns children in order Right Down Left Up
    # Index is Null if there is no valid move that way
    zero_index = state.index("0")
    x = zero_index%size
    y = int(zero_index/size)

    children = [None, None, None, None]

    if(x +1 < size):
        children[0] = swap(zero_index,zero_index+1,state)
    if (y+1 <size):
        children[1] = swap(zero_index,zero_index+size,state)
    if (x > 0):
        children[2] = swap(zero_index-1,zero_index,state)
    if (y>0):
        children[3] = swap(zero_index-size,zero_index,state)
    return children
def pairs(state):
    #returns the number of unordered pairs for the given state
    unordered = 0
    for x in range (0,size*size-1):
        a = state[x]
        if state[x] == "0":
            continue
        for y in range (x+1,size*size):
            b = state[y]
            if state[y] == "0":
                pass
            elif state[y]<state[x]:
                unordered += 1


    return unordered



def solveable(state:str):
    #p = pairs(state)
    #if len(goal)%2 == 1:
    #    return (p%2 == 0)
    #if len(goal)%2 == 0:
    #    return int(state.index(" ")/size) %2 == p%2
    return True

def taxicab(state):
    d = position_dict
    distance = 0
    for x in range (0,size*size):
        if not state[x] == "0":
            distance += abs(x%size - d[state[x]][0]) + abs(int(x/size) - d[state[x]][1])
    return distance
def taxicab_inc (state, dir: int):
    d = {"A": (1,0),"B": (2,0), "C": (3,0), "D":(0,1), "E": (1,1), "F": (2,1), "G": (3,1), "H": (0,2), "I": (1,2), "J": (2,2), "K": (3,2), "L": (0,3), "M": (1,3), "N": (2,3), "O":(3,3) }
    zero_index = state.index("0")
    if dir == 0:
        char = state[zero_index + 1]
        if  zero_index%size < d[char][0]:
            return 1
        return -1
    if dir == 1:
        char = state[zero_index +size]
        if int(zero_index/size) < d[char][1]:
            return 1
        return -1
    if dir == 2:
        char = state[zero_index - 1]
        if zero_index%size > d[char][0]:
            return 1
        return -1
    if dir == 3:
        char = state[zero_index-size]
        if int(zero_index/size) > d[char][1]:
            return 1
        return -1
    pass



def bfs(state):
    #performs breadth first search
    nodes = 0
    try:

        visited = set([state])
        fringe = deque([(state,"")])

        start = time.process_time()
        if not solveable(state):
            return -1

        while (len(fringe)>0):
            s,p = fringe.popleft()
            nodes += 1
            if s == goal:
                end = time.process_time()
                return p, end-start, nodes

            children = get_children(s)
            for x in range(0,4):
                if not children[x] is None and not children[x] in visited:
                    visited.add(children[x])
                    fringe.append((children[x],p + index_dict[x]))
    except (MemoryError):
        print("BFS Memory error")
        pass
def kdfs(state, k):
    nodes = 0
    start = time.process_time()
    if not solveable(state):
        return -1
    fringe = deque()
    start_ancestors = set()
    path = ""
    fringe.append((state,path,start_ancestors))

    while len(fringe)>0:
        nodes +=1
        s,p,ancestors = fringe.pop()
        if s == goal:
            return p,time.process_time()-start, nodes
        children = get_children(s)
        for x in range (0,4):
            if not children[x] is None and not children[x] in ancestors and len(p)<=k:
                a = ancestors.copy()
                a.add(s)
                fringe.append((children[x],p + index_dict[x],a))
    return None,time.process_time()-start,nodes
def iddfs(state):
    total_time = 0
    nodes = 0
    try:
        k = 1
        while (True):
            ans = kdfs(state,k)
            total_time += ans[1]
            nodes += ans[2]
            if not ans[0] is None:
                return ans[0],total_time,nodes
            k += 1
    except (MemoryError):
        return
def a_star(state, m = 1, heuristic = taxicab):

    #try:
        nodes = 0
        #####<Initialization>#####
        start = time.process_time()
        fringe = []
        heappush(fringe,(heuristic(state),random.random(),"",state))
        visited = set()
        #####</Initialization>#####

        while(len(fringe)>0):
            taxi,r,p,v = heappop(fringe)
            nodes += 1
            #<goal check>
            if v == goal:
                return p,time.process_time()-start, nodes
            #</goal check>

            if v in visited:
                continue
            visited.add(v)

            #print(p, b)
            #time.sleep(0.01)
            children = get_children(v)
            for x in range(0,4):

                if not children[x] is None and not children[x] in visited:
                    g = len(p) + 1
                    h = heuristic(children[x])
                    f = m*g+h
                    heappush(fringe,(f ,random.random(),p + index_dict[x],children[x]))
    #except:
        return
def bucket_a_star(state, m = 1):

    #try:

        lowest = taxicab(state)
        nodes = 0
        visited = set()
        #####<Initialization>#####
        start = time.process_time()
        fringe = []
        for x in range(lowest+1):
            fringe.append([])
        fringe[lowest].append(("", state))
        # heappush(fringe,(random.random(),taxicab(state),"",state))

        #####</Initialization>#####

        while(len(fringe)>0):
            p,v = fringe[lowest].pop()
            if v == goal:
                return p,time.process_time()-start, nodes
            if len(fringe[lowest]) == 0 and lowest<len(fringe)-1:
                x = 1
                while True:
                    if len(fringe[lowest+x])>0:
                        lowest += x
                        break
                    x+= 1

            nodes += 1

            if v in visited:
                continue
            visited.add(v)

            #print(p, cycles(permutation(v,inv(state))))
            #time.sleep(0.01)
            children = get_children(v)
            for x in range(0,4):

                if not children[x] is None and not children[x] in visited:
                    g = len(p) + 1
                    h = taxicab(children[x]) #taxi + taxicab_inc(v, x)
                    f = m*g+h

                    while f>=len(fringe):
                        fringe.append([])

                    fringe[f].append((p + index_dict[x],children[x]))
                    if f<lowest:
                        lowest = f
    #except:
        return
def bi_directional_bfs(state):
    nodes = 0
    start = time.process_time()
    visited_front = {state: ""}
    visited_back = {goal: ""}

    fringe_front = deque([(state,"")])
    fringe_back = deque([(goal,"")])

    while (len(fringe_back)>0 and len(fringe_front)):
        #v front, p front
        vf, pf = fringe_front.popleft()
        vb, pb = fringe_back.popleft()
        nodes += 2
        if vf in visited_back:
            return (pf+pb, time.process_time()-start, nodes)
        if vb in visited_front:
            return (pf+pb, time.process_time()-start, nodes)


        f_children = get_children(vf)
        for x in range(0,4):
            path = pf + index_dict[x]
            if not f_children[x] is None and not f_children[x] in visited_front:

                visited_front[f_children[x]] = path
                fringe_front.append((f_children[x], path))
                pass
            if f_children[x] in visited_back:
                return path + visited_back[f_children[x]], time.process_time()-start

        b_children = get_children(vb)
        for x in range (0,4):

            path = index_dict[(x+2)%size] + pb

            if not b_children[x] is None and not b_children[x] in visited_back:
                visited_back[b_children[x]] = path
                fringe_back.append((b_children[x],path))

            if b_children[x] in visited_front:
                return visited_front[b_children[x]] + path, time.process_time()



    pass
def path_valid(state, path):
    current = state
    for char in path:
        current = get_children(current)[index_dict[char]]
    return current == goal
def compare_searches(line):
    ans1 = bfs(line)
    ans2 = iddfs(line)
    ans3 = a_star(line)

    if not ans1 is None:
        print("BFS:   ", len(ans1[0]), ans1[1], ans1[2]/ans1[1], " nodes/sec")
    else:
        print("BFS MEMORY ERROR")
    if not ans2 is None:
        print("IDDFS: ", len(ans2[0]), ans2[1], ans2[2]/ans2[1], " nodes/sec")
    else:
        print("IDDFS MEMORY ERROR")
    if not ans3 is None:
        print("A* :   ", len(ans3[0]), ans3[1], ans3[2]/ans3[1], " nodes/sec")
    else:
        print("A* MEMORY ERROR")
def compare_m(line, increasing: bool):
    ans = a_star(line,1)
    length = len(ans[0])
    print(length, ans[1])
    if increasing:

        for m in range (11,31):
            ans = a_star(line,m/10)
            print("\t m = ", m/10, ": ", len(ans[0]), ans[1] )
        pass
    else:
        for m in range(1,10):
            ans = a_star(line, 1- m/10)
            print("\t m = ", 1- m/10, ": ", len(ans[0]), ans[1])

    pass



#cycles to value
def ctv(cycles):
    result = ""
    d = {'0': '0', 'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J',
     'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O'}
    for cycle in cycles:
        for x in range(len(cycle)):
            d[cycle[x]] = cycle[(x+1)%len(cycle)]
    for x in d:
        result+= d[x]
    return result
def goto(state, path): # starts at state, and travels along path and returns the puzzle that results
    current = state
    for char in path:
        children = get_children(current)
        current = children[index_dict[char]]
        if current == None:
            print("fuck you spiderman I'm the boat you rode on")

    return current
def cycles (state): #to get something useful out of this algorithm, call cycles(permutation(goal,inv(state)))
    d = {}
    cycles = []
    cycles_index = -1
    visited = set()

    for x in range (0,size*size):
        d[state[x]] = goal[x]

    for x in goal:
        if x in visited:
            continue
        if not d[x] == x:
            cycles.append("")
            cycles_index += 1
            while not x in visited:
                visited.add(x)
                cycles[cycles_index] += x
                x = d[x]


    return cycles
def permutation(first, second):
    first_dict = {}
    second_dict = {}
    result = ""
    for x in range(len(goal)):
        first_dict[goal[x]] = first[x]
    for x in range(len(goal)):

        second_dict[goal[x]] = second[x]
    for key in first_dict:
        result += second_dict[first_dict[key]]

    return result
def inv(element):
    d = {}
    inverse = ""
    for x in range(len(goal)):
        d[element[x]] = goal[x]

    for char in goal:
        inverse += d[char]

    return inverse

def unmoved(visited_set, all_cycles): # returns a set of all the places where implied path says
    moved = set()
    for c in all_cycles:
        for char in c:
            moved.add(char)
    not_moved = set(['0','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']).difference(moved)
    interesting_unmoved = not_moved.intersection(visited_set)
    return interesting_unmoved
def simpler_implied_paths(state):

    all_cycles = cycles(permutation(goal,inv(state)))
    prev = dict()
    paths = []
    for c in all_cycles:
        path,first = parse_cycle(c)
        # prev = combine_complex_paths(prev,path)
        paths.append(path)
    prev = merge_paths(paths,all_cycles)
    if state[0] == '0':
        prev = connect_zero(prev)
    # branch_from_intersection(prev, all_cycles)
    return prev
def merge_paths(input_paths,all_cycles): #
    n = len(input_paths)
    output_paths = dict()
    if all_cycles[0][0] == '0':
        l = list(permutations(range(1,n)))
        for permutation in l:
            path = input_paths[0]
            for index in permutation:
                path = combine_complex_paths(path,input_paths[index])
            output_paths = {**output_paths, **path}


    else:
        l = list(permutations(range(n)))
        for permutation in l:
            path = dict()
            for index in permutation:
                path = combine_complex_paths(path,input_paths[index])
            output_paths = {**output_paths, **path}
    return output_paths
def gen_simple_paths(start,end): #Method that returns all the ways to get from start to finish
    tile_pos = {'0': (0,0), 'A':(1,0), 'B':(2,0), "C":(3,0), 'D':(0,1), "E": (1,1), "F":(2,1), "G":(3,1), "H":(0,2), "I":(1,2), "J":(2,2), "K":(3,2), "L":(0,3), "M":(1,3), "N":(2,3), "O": (3,3)}
    tile_to_index = {'0':0, "A":1, 'B':2, 'C':3, 'D':4,'E':5,'F':6,'G':7,"H":8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14,'O':15}

    start_index = tile_to_index[start]

    startx = tile_pos[start][0]
    starty = tile_pos[start][1]

    endx = tile_pos[end][0]
    endy = tile_pos[end][1]

    travelx = endx-startx
    travely = endy-starty

    stepx = 0
    stepy = 0
    if not travelx == 0:
        stepx = travelx/abs(travelx)
    if not travely == 0:
        stepy = travely/abs(travely)

    x_remaining = abs(travelx)
    y_remaining = abs(travely)

    return simple_paths_helper(start,stepx,stepy,x_remaining,y_remaining,"")
def has_duplicates(paths):
    for path in paths:
        for x in range(1,len(paths[path][1])):
            if paths[path][1][x] == paths[path][1][x-1]:
                return True
    return False


def simple_paths_helper(start,stepx,stepy,x_remaining,y_remaining, path): #recursive method that returns all the ways to get from
                                                                          #start to finish

    step_dict_x = {1:"R", -1:"L"}
    step_dict_y = {1:"D", -1:"U"}
    if x_remaining == 0:
        path = path + step_dict_y[stepy]*y_remaining
        return get_visited_dict(path,start)
    if y_remaining == 0:
        path = path + step_dict_x[stepx] * x_remaining
        return get_visited_dict(path,start)

    a = simple_paths_helper(start,stepx,stepy,x_remaining-1,y_remaining,path+step_dict_x[stepx])
    b = simple_paths_helper(start,stepx,stepy,x_remaining,y_remaining-1,path+step_dict_y[stepy])
    combined =  {**a, **b}

    return combined

def get_visited_dict(path,start): #helper method for simple_paths_helper to take in a completed path and then return the set of
                                  #places it had been and a list describing where it had gone in order of when it had gotten there
    tile_to_index = {'0':0, "A":1, 'B':2, 'C':3, 'D':4,'E':5,'F':6,'G':7,"H":8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14,'O':15}
    index_to_tile = '0ABCDEFGHIJKLMNO'
    dir_dict = {'R': 1, 'L': -1, 'U': -4, 'D': 4}

    visited_set = set()
    visited_list = []

    index = tile_to_index[start]
    visited_list.append(index_to_tile[index])
    visited_set.add(index_to_tile[index])
    for char in path:
        index += dir_dict[char]
        visited_list.append(index_to_tile[index])
        visited_set.add(index_to_tile[index])
    return {path:(visited_set, visited_list)}
def combine_paths(paths1,paths2, exclude_bad_combos = True): #method that takes two paths and finds the simplest way of combining them.
                                  #Returns set of possible paths as well as a number indicating how much it must be wrong by
                                  #based on whether or not there was a "bad combo"
    bad_combos = set(["LR", "RL","UD","DU"])
    combined = dict()
    backup = dict()
    bad_combo_num = 0

    for path1 in paths1:
        for path2 in paths2:
            if (path1[-1] + path2[0])  in bad_combos and exclude_bad_combos:
                backup[path1+path2] = paths1[path1][0].union(paths2[path2][0]), paths1[path1][1] + paths2[path2][1][1:]
                continue
            combined[path1+path2] = paths1[path1][0].union(paths2[path2][0]), paths1[path1][1] + paths2[path2][1][1:]
    if len(combined) == 0:
        combined  = backup
        backup = dict()
        bad_combo_num+=1
    return combined,bad_combo_num
def combine_complex_paths(paths1,paths2):
    #paths are of the form path: (set, list)
    if len(paths1) == 0:
        return paths2
    best_candidates = []
    bad_combos = set(["LR", "RL","UD","DU"])

    for path1 in paths1:
        for path2 in paths2:
            if not len(paths1[path1][0].intersection(paths2[path2][0])) == 0:
                best_candidates.append(((path1,paths1[path1][0], paths1[path1][1]),(path2,paths2[path2][0], paths2[path2][1])))
                # appends tuple of a ((path1, visited set1, visited list1), (path2, visited set2, visited list2))
    new_bois = dict()
    badbois = dict()
    if not len(best_candidates) == 0:
        for pairs in best_candidates:
            intersect = pairs[0][1].intersection(pairs[1][1])



            for intersection_point in intersect:
                badcombo = False
                index_of_intersect1 = pairs[0][2].index(intersection_point)
                index_of_intersect2 = pairs[1][2].index(intersection_point)

                new_path = pairs[0][0][:index_of_intersect1] + pairs[1][0][index_of_intersect2:] + pairs[1][0][:index_of_intersect2] + pairs[0][0][index_of_intersect1:]
                #should merge this thing correctly
                if index_of_intersect1>0 and index_of_intersect2<len(pairs[1][0]) and pairs[0][0][index_of_intersect1-1] + pairs[1][0][index_of_intersect2] in bad_combos:
                    badcombo == True
                if index_of_intersect2>0 and index_of_intersect2<len(pairs[1][0])  and pairs[1][0][index_of_intersect2] + pairs[1][0][index_of_intersect2-1] in bad_combos:
                    badcombo = True
                if index_of_intersect2>0 and index_of_intersect1<len(pairs[0][0]) and pairs[1][0][index_of_intersect2-1] + pairs[0][0][index_of_intersect1] in bad_combos:
                    badcombo = True

                list1 = pairs[0][2][:index_of_intersect1]
                list2 = pairs[1][2][index_of_intersect2:]
                list3 =  pairs[1][2][:index_of_intersect2]
                list4 = pairs[0][2][index_of_intersect1:]

                clean_so_no_duplicates(list1,list2,list3,list4)


                new_visited_set = pairs[0][1].union(pairs[1][1])
                new_visited_list =  list1 + list2 + list3 + list4
                # if badcombo:
                #     badbois[new_path] = (new_visited_set,new_visited_list)
                # else:
                new_bois[new_path] = (new_visited_set,new_visited_list)
    else:
        #if the paths look like they are disconnected
        for path1 in paths1:
            for path2 in paths2:
                pair_set = get_closest_pairs(paths1[path1][0], paths2[path2][0])
                for pair in pair_set:

                    connecting_path = combine_paths(gen_simple_paths(pair[0], pair[1]), gen_simple_paths(pair[1],pair[0]),exclude_bad_combos= False)[0]
                    new = combine_complex_paths(combine_complex_paths(paths1,connecting_path),paths2)
                    new_bois.update(new)

    if not len(new_bois) == 0:
        return new_bois
    return badbois
def get_closest_pairs(set1, set2):
    minimum = 10000
    pair = None
    tile_to_index = {'0':0, "A":1, 'B':2, 'C':3, 'D':4,'E':5,'F':6,'G':7,"H":8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14,'O':15}

    pairs = set()
    for x1 in set1:
        for x2 in set2:
            distance = abs(tile_to_index[x1]%4 - tile_to_index[x2]%4) + abs(int(tile_to_index[x1]/4) - int(tile_to_index[x2]/4))
            if distance == 0:
                continue
            if distance<minimum:
                minimum = distance
                pairs = set()
            if distance == minimum:
                pairs.add((x1,x2))
    return pairs
def get_distance(set1, set2):
    minimum = 10000
    pair = None
    tile_to_index = {'0':0, "A":1, 'B':2, 'C':3, 'D':4,'E':5,'F':6,'G':7,"H":8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14,'O':15}

    pairs = set()
    for x1 in set1:
        for x2 in set2:
            distance = abs(tile_to_index[x1]%4 - tile_to_index[x2]%4) + abs(int(tile_to_index[x1]/4) - int(tile_to_index[x2]/4))
            if distance == 0:
                continue
            if distance<minimum:
                minimum = distance
                pairs = set()
            if distance == minimum:
                pairs.add((x1,x2))
    return minimum



def clean_so_no_duplicates(list1,list2,list3,list4):
    not_empty_1 = len(list1) > 0
    not_empty_2 = len(list2) > 0
    not_empty_3 = len(list3) > 0
    not_empty_4 = len(list4) > 0

    if not_empty_1:
        if not_empty_2:
            if list1[-1] == list2[0]:
                del list1[-1]
        elif not_empty_3:
            if list1[-1] == list3[0]:
                del list1[-1]
        elif not_empty_4:
            if list1[-1] == list4[0]:
                del list1[-1]
    if not_empty_2:
        if not_empty_3:
            if list2[-1] == list3[0]:
                del list2[-1]
        elif not_empty_4:
            if list2[-1] == list4[0]:
                del list2[-1]
    if not_empty_3:
        if not_empty_4:
            if list3[-1] == list4[0]:
                del list3[-1]

def parse_cycle(c):
    paths = dict()
    first_element = None
    is_connected = c[0] == "0"
    for x in range(len(c) - 1):
        start = c[x]
        end = c[x + 1]
        if len(paths) == 0:
            paths = gen_simple_paths(start, end)
        else:
            results = combine_paths(paths, gen_simple_paths(start, end),exclude_bad_combos= False)
            paths = results[0]
    if not c[0] == '0':
        first_element = c[0]
        start = c[-1]
        end = c[0]
        results = combine_paths(paths, gen_simple_paths(start, end),exclude_bad_combos=False)
        paths = results[0]
    return paths, first_element

def addition(state,paths,path):
    tile_to_index = {'0':0, "A":1, 'B':2, 'C':3, 'D':4,'E':5,'F':6,'G':7,"H":8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14,'O':15}

    moved_bois = 0
    all_cycles = cycles(permutation(goal, inv(state)))
    interesting = unmoved(paths[path][0],all_cycles)
    final = goto(state,reverse(path))
    weird = []
    for char in interesting:
        if not final[tile_to_index[char]] == char:
            weird.append(char)
            moved_bois += 1
    # print(weird)
    return moved_bois*2

def tentative_heuristic(state):
    if state == goal:
        return 0
    paths = simpler_implied_paths(state)
    real_paths = get_real_paths(createMapDict(paths))
    if len(paths) == 0:
        pass
    est = 1000000000
    for path in real_paths:
        # add = addition(state,paths,path)
        add = taxicab(goto(state,reverse(path))) /2

        temp_est =  len(path) + add
        if temp_est<est:
            est = temp_est
    if est == 1000000000:
        return 0
    return est
def get_badcombos(path):
    bad_combos = set(["LR", "RL","UD","DU"])
    num = 0
    for x in range(len(path)-1):
        if path[x] + path[x+1] in bad_combos:
            num += 1
    return num
def gen_puzzles_up_to(num):
    nodes = 0
    state = goal
    fout = open("allpuzzles.txt", "w+")

    try:

        visited = set([state])
        fringe = deque([(state,"")])

        start = time.process_time()

        while (len(fringe)>0):
            s,p = fringe.popleft()
            nodes += 1
            fout.write(s + " " + str(len(p)) + "\n")
            if len(p) == num:
                continue
            children = get_children(s)
            for x in range(0,4):
                if not children[x] is None and not children[x] in visited:
                    visited.add(children[x])
                    fringe.append((children[x],p + index_dict[x]))
    except (MemoryError):
        print("Puzzle Memory error")
        pass
    fout.close()



def test_allpuzles():
    with open("allpuzzles.txt") as file:
        lines = file.readlines()
        errored = set()
    # error_file = open("errored.txt", "w+")
    over_est = open("overestimates.txt", "w+")

    for line in lines:
        state = line.split(" ")[0]
        a = line.split(" ")[1]
        h = tentative_heuristic(state)
        if h>int(a):
            over_est.write(state + "\n")
    over_est.close()
        # print(state, a,h, taxicab(state))


def connect_zero(paths):
    #connects the zero to a closed path
    tile_to_index = {'0':0, "A":1, 'B':2, 'C':3, 'D':4,'E':5,'F':6,'G':7,"H":8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14,'O':15}


    promising_options = []
    final_paths = dict()
    for path in paths:
        if '0' in paths[path][0]:
            promising_options.append(path)
    if len(promising_options) > 0:
        for path in promising_options:
            zero_index = paths[path][1].index('0')
            new_path = path[zero_index:]+ path[:zero_index]
            new_set = paths[path][0]
            old_list = paths[path][1]
            new_list = old_list[zero_index:]+ old_list[1:zero_index] + ["0"]
            final_paths[new_path] = (new_set,new_list)
    else:
        prev = combine_complex_paths({"":(set(['0']),['0'])}, paths)
        return connect_zero(prev)
    return final_paths

def branch_from_intersection(paths, all_cycles):
    tile_to_index = {'0':0, "A":1, 'B':2, 'C':3, 'D':4,'E':5,'F':6,'G':7,"H":8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14,'O':15}


    new = []
    for path in paths:
        branch_points = intersection_points(paths,path)
        if len(branch_points)>1 and len(all_cycles) == 1:
            earliest_index = 10000
            earliest = None
            latest_index = 0
            latest = None
            for char,indexes in branch_points:
                min_of_char = min(indexes)
                if min_of_char < earliest_index:
                    earliest_index = min_of_char
                    earliest = char,indexes
                max_of_char = max(indexes)
                if max_of_char > latest_index:
                    latest_index = max_of_char
                    latest = char,indexes
            list1 = paths[path][1][:earliest_index]
            list2 = paths[path][1][earliest[1][-1]: latest[1][-1]]
            list3 = paths[path][1][latest[1][0]: earliest[1][-1]]
            list4 = paths[path][1][earliest[1][0]:latest[1][0]]
            list5 = paths[path][1][latest[1][-1]:]

            new_list = list1 + list2 + list3 + list4 + list5
            new_path = ""
            for x in range(len(path)):
                change = tile_to_index[new_list[x+1]] - tile_to_index[new_list[x]]
                dir = {1:'R', -1:'L', 4:'D', -4:'U'}[change]
                new_path += dir
            new.append((new_path,paths[path][0], new_list))
    for thing in new:
        paths[thing[0]] = (thing[1],thing[2])
def intersection_points(paths,path):
    dictionary = {}
    intersect_points = set()

    threes = 0
    fours = 0
    for x in range(len(paths[path][1])):
        if paths[path][1][x] not in dictionary:
            dictionary[paths[path][1][x]] = []
        dictionary[paths[path][1][x]].append(x)
    for char in dictionary:
        if len(dictionary[char])>1:
            adjacent = set()
            for index in dictionary[char]:
                if index< len(paths[path][1])-1:
                    adjacent.add(paths[path][1][index + 1])
                if index>0:
                    adjacent.add(paths[path][1][index - 1])
            if len(adjacent) == 4:
                intersect_points.add((char,tuple(dictionary[char])))
                fours += 1
            if len(adjacent) == 3:
                threes += 1
    if threes>1 and fours>1:
        return intersect_points
    return set()


def createMapDict(paths):
    bad_combos = set(["LR", "RL","UD","DU"])
    dictionaries = []
    index = -1
    for path in paths:
        mapDictHelper = {}
        is_bad_combo = False
        for x in range(len(path)):
            char1 = paths[path][1][x]
            char2 = paths[path][1][x + 1]

            if x< len(path)-1 and path[x]+path[x+1]in bad_combos:
                is_bad_combo = True
                break

            if not char1+char2 in mapDictHelper:
                mapDictHelper[char1+char2] = 0
            mapDictHelper[char1+char2] += 1
        if is_bad_combo:
            continue
        map_dict = {}
        for thing in mapDictHelper:
            if not thing[0] in map_dict:
                map_dict[thing[0]] = dict()
            map_dict[thing[0]][thing[1]]= mapDictHelper[thing]
        if index<0 or not map_dict == dictionaries[index]:
            dictionaries.append(map_dict)
            index += 1
    return dictionaries
def get_real_paths(dictionaries):
    results = set()
    for dictt in dictionaries:
        results = results.union(traverse("",{},'0',dictt))
    return results
def continue_options(prev,pos,board_map):
    options = []
    if not pos in board_map:
        return []
    for thing in board_map[pos]:
        if not pos in prev or not thing in prev[pos] or prev[pos][thing]<board_map[pos][thing]:
            options.append(thing)
    return options



def copy_prev(prev):
    new_thing = dict()
    for key in prev:
        new_dict = dict()
        for thing in prev[key]:
            new_dict[thing] = prev[key][thing]
        new_thing[key] = new_dict
    return new_thing

def is_complete(traversal, board_map):
    for index in board_map:
        if not index in traversal:
            return False
        for index2 in board_map[index]:
            if not index2 in traversal[index]:
                return False
            if traversal[index][index2]<board_map[index][index2]:
                return False
    return True
def traverse(path,prev,pos,board_map):
    tile_to_index = {'0':0, "A":1, 'B':2, 'C':3, 'D':4,'E':5,'F':6,'G':7,"H":8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14,'O':15}
    dir_dict = { 1:'R', -1:'L',  -4:'U',  4:'D'}

    while True:
        options = continue_options(prev,pos,board_map)
        if not pos in prev:
            prev[pos] = dict()
        if len(options) == 0:
            if is_complete(prev,board_map):
                return set([path])
            return set()
        if len(options) == 1:
            if not options[0] in prev[pos]:
                prev[pos][options[0]] = 0
            prev[pos][options[0]] += 1
            increment = dir_dict[tile_to_index[options[0]]-tile_to_index[pos]]
            pos = options[0]
            path += increment
        if len(options)>1:
            the_sets = set()
            for opt in options:
                if not opt in prev[pos]:
                    prev[pos][opt] = 0

                new_prev = copy_prev(prev)
                new_prev[pos][opt] += 1
                increment = dir_dict[tile_to_index[opt] - tile_to_index[pos]]
                next = traverse(path+increment,new_prev,opt,board_map)
                the_sets = the_sets.union(next)
            return the_sets

# state2 = "BDFCAE0GHIJKLMNO"
# state2 = "EDBCN0MIJLGFAHOK"
state2 = "IABCDEFGH0JKLMNO"
# state2 = "LABCDEFGHIJK0MNO"
# print_puzzle(state2)
# print_puzzle(state2)
# print(tentative_heuristic(state2))
# print(taxicab(state2))
all_cycles = (cycles(permutation(goal,inv(state2))))
print(all_cycles)

paths = (simpler_implied_paths(state2))
print(paths)


maps = createMapDict(paths)
# print(get_real_paths(maps))


#
print(reverse(a_star(state2)[0]))
# real_paths = get_real_paths(createMapDict(paths))
# print(tentative_heuristic(state2))
print(len(a_star(state2)[0]), tentative_heuristic(state2), taxicab(state2))

# for path in paths:
#     print()
#     print(path)
print()
print("NOW FOR TRAVERSALS")
for t in get_real_paths(createMapDict(paths)):
    print(t)

#     print_puzzle(final)
#     print(cycles(permutation(goal,inv(final))))
#     print(intersection_points(paths,path))
#     print(taxicab(final))
#       print(unmoved(paths[path][1],all_cycles))

# start = time.process_time()
# test_allpuzles()
# print(time.process_time()-start)

# state2 = "0DBCEAFGHIJKLMNO"
# print_puzzle(state2)
# print(cycles(permutation(goal,inv(state2))))
# print(taxicab(state2))
# print(tentative_heuristic(state2))
# print(a_star(state2))
# imp = (implied_paths(state2))[0]
# for x in imp:
#     print(reverse(x),goto(state2,reverse(x)))
#     goto(state2,reverse(x))

# test_allpuzles()

# for path in simpler_implied_paths(state2):
#     index = 0
#     print()
#     print(cycles(permutation(goal,inv(goto(state2,reverse(path))))))
#     print(path, taxicab(goto(state2,reverse(path))))
#     print(paths[path][1])
#     print(list(path))
#
#     print_puzzle(goto(goal,(path)))







