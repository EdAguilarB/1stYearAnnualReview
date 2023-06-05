
import pandas as pd
import numpy as np
from math import dist
from colorama import Fore, Style

def get_composition(df: pd.DataFrame):
    composition = df.copy()
    composition = pd.DataFrame(df.loc[0].str.split('   ', expand = True))
    composition = composition.replace("", float('NaN'))
    composition = composition.dropna(axis=1)
    composition = composition.rename(columns = {composition.columns[0]: 'Mo', composition.columns[1]: 'C' })
    composition = composition.astype(int)
    return composition

def float_converter(n):
    try:
        float(n)
        return False
    except:
        return True

def get_coords(df: pd.DataFrame):
    coords = df.copy()
    coords = coords.loc[2:]
    coords = pd.DataFrame(coords['Mo C'].str.split('  ', expand = True))
    coords = coords.reset_index(drop = True)
    Align = coords[3].apply(lambda m : float_converter(m))
    coords.loc[Align] = coords.loc[Align].shift(periods=1, axis='columns')
    coords = coords.drop(0, axis = 1)
    coords = coords.rename(columns = {1:'x', 2:'y', 3:'z'})
    coords = coords.astype(float)
    return coords.to_numpy()


def get_distances(coords:np.array):
    distances = np.zeros((len(coords), len(coords)), float)
    for i in range(len(coords)-1):
        a1 = [coords[i, 0], coords[i, 1], coords[i, 2]]
        for j in range(i+1, len(coords)):
            a2 = [coords[j, 0], coords[j, 1], coords[j, 2]]
            distance = dist(a1,a2)
            distances[i, j] = distance
            distances[j, i] = distance
    distances[distances == 0] = 'nan'

    return distances 

def get_adj_matrix(distances: np.array, max_neighbours:int, max_distance):

    dist = distances[distances <= max_distance]
    dist = np.sort(dist) 
    adj = np.zeros((len(distances), len(distances)), dtype=int)

    for i in range(0,len(dist),2):
        rc = np.nonzero(distances == dist[i])[0]

        if len(rc) == 2:
            r = rc[0]
            c = rc[1]
            if len(adj[adj[:,c] != 0]) < max_neighbours and  len(adj[adj[r,:] != 0]) < max_neighbours:
                adj[r, c] = 1
                adj[c,r] = 1

        else:
            r = np.nonzero(distances == dist[i])[0]
            c = np.nonzero(distances == dist[i])[1]

            for r,c in zip(r,c):

                if len(adj[adj[:,c] != 0]) < max_neighbours and len(adj[adj[r,:] != 0]) < max_neighbours:
                    adj[r, c] = 1
                    adj[c,r] = 1

    neigh =  adj.sum(axis = 1)

    if len(neigh[neigh == 0]) != 0:
        zeros = np.argwhere(neigh == 0) + 1
        zeros = zeros.tolist()
        print(Fore.RED + f'Warning! Atoms {zeros} are completly disconnected from the rest of the graph.')
        print(Style.RESET_ALL)
    
    if len(neigh[neigh > max_neighbours]) != 0:

        nei = np.argwhere(neigh > max_neighbours) + 1
        nei = nei.tolist()
        print(Fore.RED + f'Warning! Atoms {nei} have more neighbours than the given limit.')
        print(Style.RESET_ALL)
    
    return adj


def direction(c1, c2):
    d = c2 - c1
    if -1 < d[0] < 1:
        xdir = None
    elif d[0] < 1:
        xdir = 'r' #right
    else:
        xdir = 'l' #left

    if -1 < d[1] < 1:
        ydir = None 
    elif d[1] > 1:
        ydir = 'u' #up
    else:
        ydir = 'd' #down

    if -1 < d[2] < 1:
        zdir = None
    elif d[2] > 1:
        zdir = 'a' #above
    else:
        zdir = 'b' #below
    
    return xdir, ydir, zdir

def onek_encoding_unk(x, allowable_set):
    return list(map(lambda s: x == s, allowable_set))

def edge_features(c1, c2):
    return np.array(onek_encoding_unk(direction(c1, c2)[0], ['r','l']) 
            + onek_encoding_unk(direction(c1, c2)[1], ['u','d']) 
            + onek_encoding_unk(direction(c1, c2)[2], ['a','b']), 
            dtype=np.float32)
