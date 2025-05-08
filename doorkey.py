from math import inf
from venv import create
from utils import *
from example import example_use_of_gym_env
from gymnasium.envs.registration import register
from minigrid.envs.doorkey import DoorKeyEnv
import numpy as np

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

class DoorKey10x10Env(DoorKeyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=10, **kwargs)

register(
    id='MiniGrid-DoorKey-10x10-v0',
    entry_point='__main__:DoorKey10x10Env'
)
def doorkey_problem_B(info):
    # [0,-1] up, [-1,0] left, [0,1] down, [1,0] right 
    Dir = [[0,-1], [-1,0], [0,1], [1,0]]
    Dir_index = Dir.index(info['init_agent_dir'].tolist())
    
    # all elements with 0 are walls
    obstacles = [[5,0],[5,1],[5,2],[5,4],[5,5],[5,6],[5,8],[5,9]]
    print(f"obsatcles : {obstacles}")  
    
    # Door State
    door_postion = [[5,3],[5,7]]
    door_state = [int(info['door_open'][0]), int(info['door_open'][1])]
    print(door_state)
    
    # state space
    # (x, y, direction, key state, door1 state, door2 state,key location index, goal location index)
    X = np.zeros((info['width'],info['height'],4,2,2,2,3,3))
    key_location = [[2, 2],[2, 3],[1, 6]]
    goal_location = [[6, 1],[7, 3],[6, 6]]
    key_index = key_location.index(info['key_pos'].tolist())
    goal_index = goal_location.index(info['goal_pos'].tolist())
    # control space
    U = ["MF","TL","TR","PK","UD"]
    
    
    # initial state x0
    X0 =  np.array((info['init_agent_pos'][0], info['init_agent_pos'][1], Dir_index, 0, door_state[0],door_state[1],key_index,goal_index))
    # planning horizon T
    T = 200
    
    #Value
    V = np.ones((info['width'],info['height'],4,2,2,2,3,3,T)) * np.inf  
    V[:,:,:,:,:,:,:,:,T-1] = Termial_Cost(X, info)
   
    for t in range(T-1, -1, -1):
        for x in np.ndindex(info['width'], info['height'], 4, 2, 2,2,3,3):
            Q = np.zeros(5)
            for i, u in enumerate(U):
                x_next = motion_model_B(x, u, obstacles, info)
                Q[i] = Stage_Cost_B(x, u, obstacles, info) + V[x_next[0], x_next[1], x_next[2], x_next[3], x_next[4],x_next[5], x_next[6],x_next[7],t]
            V[x[0], x[1], x[2], x[3], x[4],x[5],x[6],x[7], t-1] = min(Q)
        if np.array_equal(V[:,:,:,:,:,:,:,:,t-1],V[:,:,:,:,:,:,:,:,t]) and t != T-1:
            print('Converged at t:', t)
            break

    return V  
    # for t in range(T , 1, -1):
    #     Currrent_State = X0
    #     optim_act_seq = []
    #     Value = []
    #     while t < T:
    #         if np.array_equal([Currrent_State[0], Currrent_State[1]], info['goal_pos']):
    #             print("Reached goal!")
    #             break
    #         Q = np.zeros(5)
    #         for i, u in enumerate(U):
    #             x_next = motion_model_B(Currrent_State, u, obstacles, info)
    #             Q[i] = Stage_Cost_B(Currrent_State, u, obstacles, info) + V[x_next[0], x_next[1], x_next[2], x_next[3], x_next[4],x_next[5], t]
    #         optim_act = np.argmin(Q)
    #         optim_act_seq.append(optim_act)
    #         Currrent_State = motion_model_B(Currrent_State, U[optim_act], obstacles, info)
    #         Value.append(V[Currrent_State[0], Currrent_State[1], Currrent_State[2], Currrent_State[3], Currrent_State[4],Currrent_State[5], t])
    #         t += 1
    #     if np.array_equal([Currrent_State[0], Currrent_State[1]], info['goal_pos']):
    #         print("Reached goal!")
    #         print("Reached goal!")
    #         break
    # for i in optim_act_seq :
    #     print(U[i])
    # print(len(optim_act_seq))
    # return optim_act_seq, V


def Create_policy(info,V) :
    # [0,-1] up, [-1,0] left, [0,1] down, [1,0] right 
    Dir = [[0,-1], [-1,0], [0,1], [1,0]]
    Dir_index = Dir.index(info['init_agent_dir'].tolist())
    
    # all elements with 0 are walls
    obstacles = [[5,0],[5,1],[5,2],[5,4],[5,5],[5,6],[5,8],[5,9]]
    print(f"obsatcles : {obstacles}")  
    
    # Door State
    door_postion = [[5,3],[5,7]]
    door_state = [int(info['door_open'][0]), int(info['door_open'][1])]
    print(door_state)
    
    # state space
    # (x, y, direction, key state, door1 state, door2 state)
    X = np.zeros((info['width'],info['height'],4,2,2,2))

    # control space
    U = ["MF","TL","TR","PK","UD"]
    
    key_location = [[2, 2],[2, 3],[1, 6]]
    goal_location = [[6, 1],[7, 3],[6, 6]]
    key_index = key_location.index(info['key_pos'].tolist())
    goal_index = goal_location.index(info['goal_pos'].tolist())
    # initial state x0
    X0 =  np.array((info['init_agent_pos'][0], info['init_agent_pos'][1], Dir_index, 0, door_state[0],door_state[1],key_index,goal_index))
    # planning horizon T
    T = 200
    
    for t in range(T , 1, -1):
        Currrent_State = X0
        optim_act_seq = []
        Value = []
        while t < T:
            if np.array_equal([Currrent_State[0], Currrent_State[1]], info['goal_pos']):
                print("Reached goal!")
                break
            Q = np.zeros(5)
            for i, u in enumerate(U):
                x_next = motion_model_B(Currrent_State, u, obstacles, info)
                Q[i] = Stage_Cost_B(Currrent_State, u, obstacles, info) + V[x_next[0], x_next[1], x_next[2], x_next[3], x_next[4],x_next[5],x_next[6],x_next[7], t]
            optim_act = np.argmin(Q)
            optim_act_seq.append(optim_act)
            Currrent_State = motion_model_B(Currrent_State, U[optim_act], obstacles, info)
            Value.append(V[Currrent_State[0], Currrent_State[1], Currrent_State[2], Currrent_State[3], Currrent_State[4],Currrent_State[5],Currrent_State[6],Currrent_State[7], t])
            t += 1
        if np.array_equal([Currrent_State[0], Currrent_State[1]], info['goal_pos']):
            print("Reached goal!")
            print("Reached goal!")
            break
    for i in optim_act_seq :
        print(U[i])
    print(len(optim_act_seq))
    return optim_act_seq
def doorkey_problem_A(env_name, info):
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env

    Feel Free to modify this fuction
    """
    print(info)
    # [0,-1] up, [-1,0] left, [0,1] down, [1,0] right 
    Dir = [[0,-1], [-1,0], [0,1], [1,0]]
    Dir_index = Dir.index(info['init_agent_dir'].tolist())
    # all elements with 0 are walls
    obstacles_dict = { 
                1:[[2,1],[2,3]], # 5*5
                2:[[2,3],[3,2],[3,3]], # 6*6
                3:[[2,1],[3,3],[3,4]], # 6*6
                4:[[2,1],[3,2],[3,4]], # 6*6
                5:[[1,3],[2,3],[3,1],[4,3],[4,4],[4,5]], # 8*8
                6:[[1,3],[2,3],[3,1],[4,3],[4,4],[4,5],[4,6]], # 8*8
                7:[[1,3],[2,3],[2,5],[3,5],[4,2],[4,3],[4,4],[4,5]] # 8*8
                }
    if "5x5" in env_name:
        obstacles = obstacles_dict[1]
    elif '6x6-direct' in env_name:
        obstacles = obstacles_dict[2]
    elif '6x6-normal' in env_name:
        obstacles = obstacles_dict[3]
    elif '6x6-shortcut' in env_name:
        obstacles = obstacles_dict[4]
    elif '8x8-direct' in env_name:
        obstacles = obstacles_dict[5]
    elif '8x8-normal' in env_name:
        obstacles = obstacles_dict[6]
    elif '8x8-shortcut' in env_name:
        obstacles = obstacles_dict[7]
    else:
        obstacles = []
    print(f"obsatcles : {obstacles}")  


    # state space
    # (x, y, direction, key state, door state)
    X = np.zeros((info['width'],info['height'],4,2,2))
    
    # control space
    U = ["MF","TL","TR","PK","UD"]
    
    
    # initial state x0
    
    X0 =  np.array((info['init_agent_pos'][0], info['init_agent_pos'][1], Dir_index, 0, 0))
    # planning horizon T
    T = 200
    
    #Value
    V = np.ones((info['width'],info['height'],4,2,2,T)) * np.inf  
    V[:,:,:,:,:,T-1] = Termial_Cost(X, info)
   
    for t in range(T-1, -1, -1):
        for x in np.ndindex(info['width'], info['height'], 4, 2, 2):
            Q = np.zeros(5)
            for i, u in enumerate(U):
                x_next = motion_model_A(x, u, obstacles, info)
                Q[i] = Stage_Cost_A(x, u, obstacles, info) + V[x_next[0], x_next[1], x_next[2], x_next[3], x_next[4], t]
            V[x[0], x[1], x[2], x[3], x[4], t-1] = min(Q)
        if np.array_equal(V[:,:,:,:,:,t-1],V[:,:,:,:,:,t]) and t != T-1:
            print('Converged at t:', t)
            break

    
    for t in range(T , 1, -1):
        Currrent_State = X0
        optim_act_seq = []
        Value = []
        while t < T:
            if np.array_equal([Currrent_State[0], Currrent_State[1]], info['goal_pos']):
                print("Reached goal!")
                break
            Q = np.zeros(5)
            for i, u in enumerate(U):
                x_next = motion_model_A(Currrent_State, u, obstacles, info)
                Q[i] = Stage_Cost_A(Currrent_State, u, obstacles, info) + V[x_next[0], x_next[1], x_next[2], x_next[3], x_next[4], t]
            optim_act = np.argmin(Q)
            optim_act_seq.append(optim_act)
            # print("Action:", U[optim_act])
            Currrent_State = motion_model_A(Currrent_State, U[optim_act], obstacles, info)
            Value.append(V[Currrent_State[0], Currrent_State[1], Currrent_State[2], Currrent_State[3], Currrent_State[4], t])
            t += 1
        if np.array_equal([Currrent_State[0], Currrent_State[1]], info['goal_pos']):
            print("Reached goal!")
            break
    for i in optim_act_seq :
        print(U[i])
    print(len(optim_act_seq))
    return optim_act_seq

def Stage_Cost_A(x, u, obstacles, info):
    '''
    analyze the state x and return the satge cost
    ----------------
    return cost = legal or illegal ? 1 : inf
    '''
    if(Legal_A(x, u, obstacles, info)) : 
        return 1
    else:
        return inf # Stop
def Stage_Cost_B(x, u, obstacles, info):
    '''
    analyze the state x and return the satge cost
    ----------------
    return cost = legal or illegal ? 1 : inf
    '''
    if(Legal_B(x, u, obstacles, info)) : 
        return 1
    else:
        return inf # Stop 
def Termial_Cost(x,info):     
    '''
    analyze the state x and return the satge cost
    ----------------
    return cost = legal or illegal ? 1 : inf
    '''
    q = np.ones(x.shape)*np.inf
    # Part A
    if len(info['goal_pos']) == 2:
        q[info['goal_pos'][0],info['goal_pos'][1]] = 0        
    # Part B
    else: 
        for n in range(0,len(info['goal_pos'])):
            q[info['goal_pos'][n][0],info['goal_pos'][n][1],:,:,:,n,:,:] = 0
   
    return q

def Legal_A(x, u, obstacles, info):
    '''
    check if the agent is legal
    ----------------
    return true or false
    '''
    X = [x[0],x[1],x[2],x[3],x[4]]
    Dir_index = x[2]
    P = MoveForward(X[0], X[1], Dir_index) # next step
    if(X[0] <= 0 or X[0] >= info['width'] -1 or X[1] <= 0 or X[1] >= info['height'] -1):
        # print("at walls")
        return False
    if(obstacles.count([X[0],X[1]]) != 0):# obstacles
        # print("at obstacles")
        return False
    if(X[3] == 0 and ([X[0],X[1]] == [5,3] or [X[0],X[1]] == [5,7])):
        return False
    if(u == "MF"):
        if (P[0] <= 0 or P[0] >= info['width'] -1 or P[1] <= 0 or P[1] >= info['height'] -1): # walls
            # print("hit walls")
            return False
        if(obstacles.count(P) != 0):# obstacles
            # print("hit obstacles")
            return False
        if(X[4] == 0 and np.array_equal(P,info["door_pos"])):# door without key
            # print("hit door")
            return False
    elif(u == "PK"):
        if(X[3] == 1): # already keep key
            # print("holded key")
            return False
        if(not np.array_equal(P,info["key_pos"])):
            # print("no key")
            return False
    elif(u == "UD") :
        if(X[3] == 0 or X[4] == 1): # no key or door opened
            # print("can't UD")
            return False
        if(not np.array_equal(P,info["door_pos"])):
            # print("no door")
            return False
    return True
def Legal_B(x, u, obstacles, info):
    '''
    check if the agent is legal
    ----------------
    return true or false
    '''
    X = [x[0],x[1],x[2],x[3],x[4],x[5]]
    Dir_index = x[2]
    P = MoveForward(X[0], X[1], Dir_index) # next step
    if(X[0] <= 0 or X[0] >= info['width'] -1 or X[1] <= 0 or X[1] >= info['height'] -1):
        # print("at walls")
        return False
    if(obstacles.count([X[0],X[1]]) != 0):# obstacles
        # print("at obstacles")
        return False
    if(X[4] == 0 and [X[0],X[1]] == [5,3] ):
        return False
    if(X[5] == 0 and [X[0],X[1]] == [5,7] ):
        return False
    if(u == "MF"):
        if (P[0] <= 0 or P[0] >= info['width'] -1 or P[1] <= 0 or P[1] >= info['height'] -1): # walls
            # print("hit walls")
            return False
        if(obstacles.count(P) != 0):# obstacles
            # print("hit obstacles")
            return False
        if(X[4] == 0 and np.array_equal(P,np.array([5,3]))):# door without key
            # print("hit door")
            return False
        if(X[5] == 0 and np.array_equal(P,np.array([5,7]))):# door without key
            # print("hit door")
            return False
    elif(u == "PK"):
        if(X[3] == 1): # already keep key
            # print("holded key")
            return False
        if(not np.array_equal(P,info["key_pos"])):
            # print("no key")
            return False
    elif(u == "UD") :
        if([P[0],P[1] == [5,3]]):
            if(X[3] == 0 or X[4] == 1): # no key or door opened
                # print("can't UD")
                return False
        
        if([P[0],P[1] == [5,7]]):
            if(X[3] == 0 or X[5] == 1): # no key or door opened
                # print("can't UD")
                return False
    return True
def motion_model_A(x, u, obstacles, info):
    '''
    the next step with control command u
    ----------------
    return x_next = (x, y, direction, key state, door state)
    '''

    x_next = [x[0],x[1],x[2],x[3],x[4]]
    Dir_index = x[2]
    P = MoveForward(x[0], x[1], Dir_index)
    

    if(not Legal_A(x, u, obstacles, info)):
        return x_next
    if(np.array_equal(x_next,info['goal_pos'])):
        return x_next

    if(u == "TL"):
        x_next[2] = (Dir_index + 1) % 4
    elif(u == "TR"):
        x_next[2] = (Dir_index - 1) % 4
    elif(u == "MF"):
        x_next[0] = P[0]
        x_next[1] = P[1]
    elif(u == "PK"):
        # print("PK")
        x_next[3] = 1
    elif(u == "UD") :
        # print("UD")
        x_next[4] = 1
    return x_next
def motion_model_B(x, u, obstacles, info):
    '''
    the next step with control command u
    ----------------
    return x_next = (x, y, direction, key state, door state)
    '''

    x_next = [x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]]
    Dir_index = x[2]
    P = MoveForward(x[0], x[1], Dir_index)
    

    if(not Legal_B(x, u, obstacles, info)):
        return x_next
    if(np.array_equal(x_next,info['goal_pos'])):
        return x_next

    if(u == "TL"):
        x_next[2] = (Dir_index + 1) % 4
    elif(u == "TR"):
        x_next[2] = (Dir_index - 1) % 4
    elif(u == "MF"):
        x_next[0] = P[0]
        x_next[1] = P[1]
    elif(u == "PK"):
        # print("PK")
        x_next[3] = 1
    elif(u == "UD") :
        if([P[0],P[1]] == [5,3]):
            x_next[4] = 1
        if([P[0],P[1]] == [5,7]):
            x_next[5] = 1
        # print("UD")
        
    return x_next
def MoveForward(x,y,Dir_index):
    '''
    the next step with "Move Forward" with the direction
    ----------------
    return x_next = (x, y)
    '''
    # [0,-1] up, [-1,0] left, [0,1] down, [1,0] right 
    if(Dir_index == 0):
        y -= 1
    elif(Dir_index == 1):
        x -= 1
    elif(Dir_index == 2):
        y += 1
    elif(Dir_index == 3):
        x += 1
    return [x,y]

def partA():
    # env_path = "./envs/known_envs/doorkey-5x5-normal.env"
    # env_path = "./envs/known_envs/doorkey-6x6-direct.env"
    # env_path = "./envs/known_envs/doorkey-6x6-normal.env"
    # env_path = "./envs/known_envs/doorkey-6x6-shortcut.env"
    # env_path = "./envs/known_envs/doorkey-8x8-direct.env"
    # env_path = "./envs/known_envs/doorkey-8x8-normal.env"
    env_path = "./envs/known_envs/doorkey-8x8-shortcut.env"
    env, info = load_env(env_path)  # load an environment
    env_name = env_path.split('/')[-1]
    seq = doorkey_problem_A(env_name, info)  # find the optimal action sequence
    draw_gif_from_seq(seq, load_env(env_path)[0])  # draw a GIF & save

def partB():
    env_folder = "./envs/random_envs"
    env, info, env_path = load_random_env(env_folder)
    print(info)
    V = doorkey_problem_B(info)  # find the optimal action sequence
    # draw_gif_from_seq(seq, load_env(env_path)[0])  # draw a GIF & save
    
    # use the DP to create policy 
    for i in range(5) :
        print("A")
        env, info, env_path = load_random_env(env_folder)
        seq = Create_policy(info,V)  # find the optimal action sequence
        path = f"./gif/doorkey_{i}.gif"
        draw_gif_from_seq(seq, load_env(env_path)[0],path)  # draw a GIF & save

if __name__ == "__main__":
    # example_use_of_gym_env()
    # partA()
    partB()

