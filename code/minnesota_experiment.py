from __future__ import division

import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering

from wavelets import app_gsp_filter


folder='datasets\\'
Q=scipy.io.loadmat(folder+'minnesota.mat')

G=dict()
A=Q['A'].toarray()
N=A.shape[0]

A=A-np.diag(np.diag(A))

A[348,354]=1
A[354,348]=1

A[85,87]=1
A[87,85]=1

A[344,345]=1
A[345,344]=1

A[1706,1708]=1
A[1708,1706]=1

A[2288,2289]=1
A[2289,2288]=1

d=np.zeros((N,))
for i in range(N):
    d[i]=np.sum(A[i,:])

D=np.diag(d)

L=scipy.sparse.csc_matrix(D-A)

n_clusters=3

spectral=SpectralClustering(n_clusters=n_clusters,affinity='precomputed')
classification=spectral.fit_predict(A) 

visited=[]
for i in range(N):
    if (classification[i] not in visited):
        visited.append(classification[i])

converter=dict()            
for i in range(n_clusters):
    converter[visited[i]]=i 
    
for i in range(N):
    classification[i]=converter[classification[i]]

spectrum,U=np.linalg.eigh(L.toarray())

cluster=[[] for i in range(n_clusters)]
for i in range(N):
    cluster[classification[i]].append(i)

f_array=[]
for i in range(n_clusters):
    f_array.append(np.zeros((N,1)))

for j in range(n_clusters):
    for i in range(N):
        if (i in cluster[j]):
            for l in range(N):
                if ((j==1) and (0.0 <= spectrum[l] <= 0.08)):
                    # Blue class
                    f_array[j][i]+=U[i,l]
                    
                if ((j==2) and (2.0 <= spectrum[l] <= 2.5)):
                    # Red class
                    f_array[j][i]+=U[i,l]
                    
                if ((j==0) and (5.0 <= spectrum[l] <= 8.0)):
                    # Green class
                    f_array[j][i]+=U[i,l]

f=np.zeros((N,1))
for i in range(n_clusters):
    norm=np.linalg.norm(f_array[i], np.inf)
    if (norm != 0):
        f+=f_array[i]/norm
        
G=nx.from_numpy_matrix(A)
pos=Q['xy']


node_color=[]
for i in range(N):            
    if (i in cluster[0]):
        node_color.append('g')
    if (i in cluster[1]):
        node_color.append('b')
    if (i in cluster[2]):
        node_color.append('r')

xmin=np.min(pos[:,0])-0.1
xmax=np.max(pos[:,0])+0.1
ymin=np.min(pos[:,1])-0.1
ymax=np.max(pos[:,1])+0.1
        
plt.figure()
plt.xlim(xmin=xmin,xmax=xmax)
plt.ylim(ymin=ymin,ymax=ymax)
nx.draw_networkx_nodes(G,pos,node_color=node_color,node_size=15,linewidths=0.0)
nx.draw_networkx_edges(G,pos,edge_color='gray')
plt.axis('off')
plt.show()

f=f.reshape((N,))

vmin=np.min(f)
vmax=np.max(f)

cmap=plt.get_cmap('seismic')

nodelist=G.nodes()
zorder=list(np.argsort(np.abs(f)))

plt.figure()
nx.draw_networkx_edges(G,pos,width=0.5,edge_color='gray')  
for j in zorder:
    nx.draw_networkx_nodes(G,pos,nodelist=[nodelist[j]],node_color=f[j],
            node_size=40,cmap=cmap,with_labels=False,vmin=vmin,vmax=vmax,
            linewidths=0.1,width=0.5,edge_color='gray')

colorbar=False
if (colorbar):
    sm=plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin,vmax=vmax))
    sm._A=[]
    cbar=plt.colorbar(sm,orientation='horizontal')
    cbar.solids.set_edgecolor("face")
    
plt.axis('off')
plt.show()

n_scales=8
filter_type='sagw'

w=app_gsp_filter(f,L,N,n_scales,M=40,filter_type=filter_type)
wav_coeff=np.abs(w.T)
    
node_color=['gray' for i in range(N)]
node_color[132]='g'          # Green
node_color[1374]='b'         # Blue
node_color[1996]='r'         # Red
node_color[1970]='deeppink'  # Pink

ymax=0.82
for i in [132,1374,1996,1970]:
    x=range(n_scales)
    y=wav_coeff[i,:]
    width=1/1.5
    
    plt.figure()
    plt.bar(x,y,width,color="blue")
    plt.scatter(7.2,0.72,c=node_color[i],s=450,linewidth='0')
    plt.ylim(ymin=0,ymax=ymax)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    
node_size=[15 for i in range(N)]
node_size[132]=70
node_size[1374]=70
node_size[1996]=70
node_size[1970]=70

xmin=np.min(pos[:,0])-0.1
xmax=np.max(pos[:,0])+0.1
ymin=np.min(pos[:,1])-0.1
ymax=np.max(pos[:,1])+0.1
        
plt.figure()
plt.xlim(xmin=xmin,xmax=xmax)
plt.ylim(ymin=ymin,ymax=ymax)
nx.draw_networkx_nodes(G,pos,node_color=node_color,node_size=node_size,linewidths=0.0)
nx.draw_networkx_edges(G,pos,edge_color='gray')
plt.axis('off')
plt.show()

xmax=spectrum[-1]    

plt.figure()
plt.xlim(xmin=-0.1,xmax=xmax+0.1)
plt.ylim(ymin=-0.1,ymax=0.1)
plt.plot(spectrum,np.zeros(len(spectrum)),'kx',mew=1.5)
plt.grid('off')
plt.show()