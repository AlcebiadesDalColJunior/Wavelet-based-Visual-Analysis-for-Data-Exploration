from __future__ import division

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from math import sin, cos, pi

from util import graph_Fourier_transform,comet_plot
from wavelets import app_gsp_filter


#%% 

N=15
center_degree=10

G=nx.Graph()
G.add_nodes_from(range(N))

for i in range(1,center_degree+1):
    G.add_edge(0,i)
    
for i in range(center_degree,N-1):
    G.add_edge(i,i+1)

r=5
theta=(2*pi)/center_degree
circle=[]
for i in range(center_degree+1):
    circle.append((r*cos(i*theta),r*sin(i*theta)))

pos=[]
ind=5.0
for i in range(N):
    if (i == 0):
        pos.append((0,0))
    if ((1 <= i) and (i <= center_degree)):
        pos.append(circle[i])
    if ((center_degree+1) <= i):
        ind+=5.0
        pos.append((ind,0))
        
# Comet with 15 nodes
comet_plot(G,pos)



#%%

N=64
center_degree=30

G=nx.Graph()
G.add_nodes_from(range(N))

for i in range(1,center_degree+1):
    G.add_edge(0,i)
    
for i in range(center_degree,N-1):
    G.add_edge(i,i+1)

r=5
theta=(2*pi)/center_degree
circle=[]
for i in range(center_degree+1):
    circle.append((r*cos(i*theta),r*sin(i*theta)))

pos=[]
ind=5.0
for i in range(N):
    if (i == 0):
        pos.append((0,0))
    if ((1 <= i) and (i <= center_degree)):
        pos.append(circle[i])
    if ((center_degree + 1) <= i):
        ind+=1.5
        pos.append((ind,0))
        
L=nx.laplacian_matrix(G).asfptype()

spectrum,U=np.linalg.eigh(L.toarray())
  
f=U[:,62]

# Comet with 64 nodes
comet_plot(G,pos,f) # Plot with colorbar: comet_plot(G,pos,f,True)
  
fhat=graph_Fourier_transform(L,f)

# Graph Fourier transform
y=[0 for i in range(N)]
plt.figure()
plt.plot(spectrum,y,'k')
plt.plot(spectrum,fhat[0],'bo')
for i in range(N):
    plt.plot((spectrum[i],spectrum[i]),(0,fhat[0][i]),'b')

xmax=spectrum[-1]
plt.xlim(xmin=-0.5,xmax=xmax+0.5)
plt.ylim([-0.05,1.05])
plt.grid('off')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

for filter_type in ['sgw','sagw']:
    n_scales=8
    
    w=app_gsp_filter(f,L,N,n_scales,M=40,filter_type=filter_type,view_filters=True)
    wav_coeff=np.abs(w.T)
    
    x=range(n_scales)
    y=wav_coeff[46,:]
    ymax=np.max(wav_coeff)
    width=1/1.5
    
    plt.figure()
    plt.bar(x,y,width,color="blue")
    plt.ylim(ymax=ymax)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()