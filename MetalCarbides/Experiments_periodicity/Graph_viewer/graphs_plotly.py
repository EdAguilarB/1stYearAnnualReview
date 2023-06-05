import pandas as pd
import numpy as np
import torch
import os.path as osp
import plotly.express as px
import plotly.graph_objects as go

print('Enter the type of the file you would like to see: \n \
      1. Coords and distance  \
      2. Direction \
      3. Coords and direction  \
      4. Coords, distance, and direction')

type = int(input())

if type == 1:
    path = '/Users/ed/Documents/PhD_Documents/Experiments/Metal_carbides/Experiments_periodicity/Data/coords_distance/processed'

if type == 2:
    path = '/Users/ed/Documents/PhD_Documents/Experiments/Metal_carbides/Experiments_periodicity/Data/direction/processed'

if type == 3:
    path = '/Users/ed/Documents/PhD_Documents/Experiments/Metal_carbides/Experiments_periodicity/Data/coords_direction/processed'

if type == 4:
    path = '/Users/ed/Documents/PhD_Documents/Experiments/Metal_carbides/Experiments_periodicity/Data/coords_distance_direction/processed'

print('Enter the number of the file you would like to see:')

file = input()

print('Enter the max neighbours considered to produce the graph:')

max_n = input()

print('Enter the max distance considered to produce the graph:')

max_d = input()

graph = torch.load(osp.join(path, f'processed_{file}_mn_{max_n}_md_{max_d}.pt'))

data = pd.DataFrame(graph.x, columns=['Mo', 'C', 'x', 'y', 'z'])
data['Atom'] = np.where(data['Mo'] == 1, 'Mo', 'C')
data['Num.Atom'] = data.index

Mo = data.loc[data['Mo'] == 1]
C = data.loc[data['Mo'] != 1]

row = graph.edge_index[0]
col = graph.edge_index[1]


fig = go.Figure()

fig.add_trace(go.Scatter3d(x=Mo['x'], y=Mo['y'], z=Mo['z'], name = 'Mo', text=Mo['Num.Atom'],
                    mode='markers', 
                    marker=dict(
                        size = 15,
                        color = 'rgb(128, 128, 128)'
                    )))

fig.add_trace(go.Scatter3d(x=C['x'], y=C['y'], z=C['z'], name = 'C', text=C['Num.Atom'],
                    mode='markers', 
                    marker=dict(
                        size = 5,
                        color = 'blue'
                    )))



for i in range(int(len(row))):
    x = np.array((graph.x[row[i]][2], graph.x[col[i]][2]))
    y = np.array((graph.x[row[i]][3], graph.x[col[i]][3]))
    z = np.array((graph.x[row[i]][4], graph.x[col[i]][4]))

    name = str(f'atom{row[i]}-atom{col[i]}')
    text = str(graph.edge_attr[i,0])

    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, name=name, text = text,
                               line=dict(color='black', width=1, dash='dot')))


fig.show()


