from __future__ import division

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def zero_crossings_count(G,f):
    count=0
    for i in G.nodes():
        neighbors=G.neighbors(i)
        
        for j in neighbors:
            if ((np.sign(f[i]) * np.sign(f[j])) < 0):
                count+=1
                
    count=count/2

    return(count)

def graph_Fourier_transform(L,f):
    spectrum,U=np.linalg.eigh(L.toarray())
    N=len(spectrum)
    
    fhat=np.zeros((1,N))
    for l in range(N):
        for i in range(N):    
            fhat[0][l]+=f[i]*U[i][l]
            
    fhat=np.abs(fhat)
    
    return(fhat)
    
def comet_plot(G,pos,f=None,colorbar=False):    
    if (f is None):
        plt.figure()
        nx.draw(G,pos,node_size=100,linewidths=0.15,width=1.5)
        plt.axis('equal')
        plt.show()     
    
    if (f is not None):
        cmap=plt.cm.Blues
        
        vmin=np.min(f)
        vmax=np.max(f)
        
        plt.figure()
        nx.draw(G,pos,node_color=f,node_size=60,cmap=cmap,
                with_labels=False,vmin=vmin,vmax=vmax,linewidths=0.2,
                width=0.5,edge_color='r')
        
        if (colorbar):
            norm=plt.Normalize(vmin=vmin,vmax=vmax)
            sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm)
            sm._A=[]
            cbar=plt.colorbar(sm,orientation='horizontal')
            cbar.solids.set_edgecolor("face")
        plt.axis('equal')
        plt.show()