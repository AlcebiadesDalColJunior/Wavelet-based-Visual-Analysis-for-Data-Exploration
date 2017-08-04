from __future__ import division

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from util import graph_Fourier_transform
from wavelets import app_gsp_filter


N=64

G=nx.path_graph(N)
L=nx.laplacian_matrix(G).asfptype()

functions=dict()
functions[0]='low'
functions[1]='high'
functions[2]='low and high'
function=functions[2]

spectrum,U=np.linalg.eigh(L.toarray())

if (function == 'low'):
    f=U[:,10]

if (function == 'high'):
    f=U[:,60]
    
if (function == 'low and high'):
    f=np.zeros((N,))
    for i in range(30):
        f[i]=U[i,10]
    for i in range(30,N):
        f[i]=U[i,60]

x=range(N)
y=[0 for i in range(N)]

plt.figure()
plt.plot(x,f,'bo')
for i in range(N):
    plt.plot((x[i],x[i]),(0,f[i]),'b')

# Signal
plt.plot(x,y,'k')
plt.xlim([-0.8,x[-1]+0.8])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(x,y,'ro')
plt.grid('off')
plt.show()

fhat=graph_Fourier_transform(L,f)

# Graph Fourier transform
y=[0 for i in range(N)]
plt.figure()
plt.plot(spectrum,y,'k')
plt.plot(spectrum,fhat[0],'bo')
for i in range(N):
    plt.plot((spectrum[i],spectrum[i]),(0,fhat[0][i]),'b')

xmax=spectrum[-1]
plt.xlim(xmin=-0.1,xmax=xmax+0.1)
plt.ylim([-0.05,1.05])
plt.grid('off')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

if (function == 'low and high'):
    n_scales=8
    filter_type='sgw'

    w=app_gsp_filter(f,L,N,n_scales,M=40,filter_type=filter_type)
    wav_coeff=np.abs(w.T)
    
    ymax=np.max(wav_coeff)
    
    # Wavelet coefficients of the nodes 15 and 45
    for i in [14,44]:
        y=wav_coeff[i,:]
        x=range(n_scales)
        width=1/1.5
        
        plt.figure()
        plt.bar(x,y,width,color="blue")
        plt.ylim(ymax=ymax)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()