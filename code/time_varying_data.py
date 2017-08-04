from __future__ import division

import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from wavelets import app_gsp_filter


folder='datasets\\'
topological_spatial_change=False

nNodes=72
nTimeSlices=20

center_node=42
pos=np.load(folder+'sensor_network_position_delaunay.npy').item()
with open(folder+'sensor_network_edges_delaunay.txt', 'rb') as inFile:
    spatial_edges=pickle.load(inFile)

if (topological_spatial_change):
    spatial_edges.remove((42,31))
    spatial_edges.remove((31,42))
    
    spatial_edges.remove((42,43))
    spatial_edges.remove((42,43))
    
    spatial_edges.remove((42,50))
    spatial_edges.remove((50,42))
    
    spatial_edges.remove((59,42))
    spatial_edges.remove((59,42))
    
    spatial_edges.remove((47,42))
    spatial_edges.remove((47,42))


G=nx.Graph()
G.add_nodes_from(range(nNodes))
G.add_edges_from(spatial_edges)

center_time=9

###
f1=np.zeros((nTimeSlices,nNodes))
f1[center_time,center_node]=1


###
f2=np.zeros((nTimeSlices,nNodes))
f2[center_time,center_node]=1

neighbors_order=nx.single_source_shortest_path_length(G,center_node,cutoff=6)
for i in range(nNodes):
    f2[center_time,i]=f2[center_time,center_node]-0.1*neighbors_order[i]


###
k=0.0
f3=np.zeros((nTimeSlices,nNodes))
for j in range(10):
    k+=0.1
    f3[j,center_node]+=k

for j in range(10,20):
    k-=0.1
    f3[j,center_node]+=k


###
k=0.0
f4=np.zeros((nTimeSlices,nNodes))
for j in range(10):
    k+=0.1
    f4[j,center_node]+=k

for j in range(10,20):
    k-=0.1
    f4[j,center_node]+=k

other_nodes=range(nNodes)
other_nodes.remove(center_node)  

neighbors_order=nx.single_source_shortest_path_length(G,center_node,cutoff=6)
for j in range(nTimeSlices):
    for i in other_nodes:
        f4[j,i]=f4[j,center_node]-0.1*neighbors_order[i]

for j in range(nTimeSlices):
    for i in range(nNodes):
        if (f4[j,i] < 0):
            f4[j,i]=0


for f in [f1,f2,f3,f4]:
    node_color=f[9,:]
    
    plt.figure()
    nx.draw(G,pos,node_color=node_color)
    plt.show()
    
    node_color=f[:,42]
    H=nx.path_graph(nTimeSlices)
    path_pos=dict()
    for i in range(nTimeSlices):
        path_pos[i]=np.array([i,0])
    
    plt.figure()
    nx.draw(H,path_pos,node_color=node_color)
    plt.show()
    
GH=nx.cartesian_product(G,H)

edges=[[] for i in range(nTimeSlices)]

index=0
indexs=[]
for j in range(nTimeSlices):
    for i in spatial_edges:
        edges[j].append(i)
        edges[j].append(i)

GH_edges=[]
for i in range(nTimeSlices):
    for j in range(len(edges[i])):
        if (edges[i][j] != [None, None]):
            GH_edges.append(((edges[i][j][0],i),(edges[i][j][1],i)))

for i in GH_edges:
    GH.add_edge(i[0],i[1],weight=1.0)

index=0    
nodelist=[]
for t in range(nTimeSlices):
    for n in range(nNodes):
        nodelist.append((n,t))
        index+=1

L=nx.laplacian_matrix(GH,nodelist=nodelist)

n_scales=8
filter_type='sagw'

for f in [f1,f2,f3,f4]:
    total_nodes=nNodes*nTimeSlices
    
    w=app_gsp_filter(f.ravel(),L,total_nodes,n_scales,M=40,filter_type=filter_type)
    wav_coeff=np.abs(w.T)
    
    node=42
    time=9
    
    x=range(n_scales)
    y=wav_coeff[node+time*nNodes,:]
    width=1/1.5
    
    plt.figure()
    plt.bar(x,y,width,color="blue")
    plt.ylim(ymax=0.9)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()