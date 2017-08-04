from __future__ import division

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import csv
import operator

from util.custom_colormaps import THot

from wavelets import app_gsp_filter


folder='datasets\\'
values=False
colorbar=False
spatio_temporal_wavelet_coefficients=False

#%% Spatial wavelet coefficients

if (not spatio_temporal_wavelet_coefficients):
    cmap=THot
    
    G=nx.read_edgelist(folder+'nyc_base.el',nodetype=int)
    inFile=open(folder+'nyc_base.xy','r')
    cscFile=csv.reader(inFile)
    
    for row in cscFile:
        node_id=int(row[0])
        G.node[node_id]['x']=float(row[1])
        G.node[node_id]['y']=float(row[2])
    
    N=len(G.nodes())
    
    f=np.genfromtxt(folder+'nycCab_aug_12_0730_0800_signal',delimiter=';',usecols=range(N))
    
    a=-73.9762
    b=40.7688
    
    for i in range(N):
        if (G.node[i]['x']*0.5+G.node[i]['y'] > a*0.5+b):
            G.remove_node(i)
    
    N=len(G.nodes())
    
    signal=np.array(())
    for i in G.nodes():
        signal=np.append(signal,f[i])
    
    f=signal[:]
    
    L=nx.laplacian_matrix(G).asfptype()
    
    n_scales=8
    filter_type='sagw'
    
    w=app_gsp_filter(f,L,N,n_scales,M=40,filter_type=filter_type)
    wav_coeff=np.abs(w.T)
    
    
    # 3889: 8th av with 41st st
    # 7055: church st with duane st
    # 2194: 6th av with 53rd st
    
    for selected_node in [3889,7055,2194]:  
        node_size=[]
        for i in range(N):
            if (f[i] != 0):
                node_size.append(40)
            else:
                node_size.append(0)
        
        pos=dict()
        for i in G.nodes():
            pos[i]=(G.node[i]['x'],G.node[i]['y'])
        
        vmin=1
        vmax=np.max(f)
        
        xpos=[]
        ypos=[]
        for i in G.nodes():
            xpos.append(pos[i][0])
            ypos.append(pos[i][1])
            
        xmax=max(xpos)+0.001
        xmin=min(xpos)-0.001
        ymax=max(ypos)+0.001
        ymin=min(ypos)-0.001
        
        nodelist=G.nodes()
        zorder=list(np.argsort(f))
        
        plt.figure()
        nx.draw_networkx_edges(G,pos,width=0.5,edge_color='gray')
        for i in zorder:
            nx.draw_networkx_nodes(G,pos,nodelist=[nodelist[i]],node_color=f[i],
                    node_size=node_size[i],cmap=cmap,with_labels=False,vmin=vmin,
                    vmax=vmax,linewidths=0.1)
        
        if (values):
            values_pos=dict()
            for i in G.nodes():
                values_pos[i]=tuple(map(operator.add,pos[i],(0.0001,0.0001)))        
            
            neighbors=nx.single_source_shortest_path_length(G,selected_node,cutoff=2)
            
            signal_labels=dict()
            for i in neighbors.keys():
                signal_labels[i]=str(int(f[nodelist.index(i)]))
                
            nx.draw_networkx_labels(G,values_pos,signal_labels)
        
        if (colorbar):
            norm=plt.Normalize(vmin=vmin,vmax=vmax)
            sm=plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A=[]
            cbar=plt.colorbar(sm,orientation='horizontal')
            cbar.solids.set_edgecolor("face")
            
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        plt.axis('off')
        plt.show()
        
        x=range(n_scales)
        y=wav_coeff[nodelist.index(selected_node),:]
        width=1/1.5
        
        if (selected_node == 3889):
            ymax=50.0
        if (selected_node == 7055):
            ymax=3.0
        if (selected_node == 2194):
            ymax=4.5
        
        plt.figure()
        plt.bar(x,y,width,color="blue")
        plt.ylim(ymax=ymax)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()
    
    
#%% Spatio-temporal wavelet coefficients

if (spatio_temporal_wavelet_coefficients):
    cmap=THot
    
    G=nx.read_edgelist(folder+'nyc_base.el',nodetype=int)
    inFile=open(folder+'nyc_base.xy','r')
    cscFile=csv.reader(inFile)
    
    for row in cscFile:
        node_id=int(row[0])
        G.node[node_id]['x']=float(row[1])
        G.node[node_id]['y']=float(row[2])
    
    N=len(G.nodes())
    
    f=np.genfromtxt(folder+'nycCab_aug_12_dur_30min_signal',delimiter=';',usecols=range(N))
    
    a=-73.9762
    b=40.7688
    
    for i in range(N):
        if (G.node[i]['x']*0.5+G.node[i]['y'] > a*0.5+b):
            G.remove_node(i)
    
    N=len(G.nodes())
    nTimeSlices=24
    #nTimeSlices=f.shape[0]  # Require more memory (all day)
    
    signal=np.zeros((nTimeSlices,N))
    for j in range(nTimeSlices):
        ind=0
        for i in G.nodes():
            signal[j,ind]=f[j,i]
            ind+=1
    
    f=signal[:]
    f=f.ravel()
    
    H=nx.path_graph(nTimeSlices)
    GH=nx.cartesian_product(G,H)
    
    edges=[]
    for j in range(nTimeSlices):
        edges.append(G.edges())
    
    # Setting network graph edges
    GH_edges=[]
    for i in range(nTimeSlices):
        for j in range(len(edges[i])):
            GH_edges.append(((edges[i][j][0],i),(edges[i][j][1],i)))
    
    # Adding spatial edges
    for i in GH_edges:
        GH.add_edge(i[0],i[1],weight=1.0)
    
    # Laplacian matrix   
    nodelist=[]
    for t in range(nTimeSlices):
        for n in G.nodes():
            nodelist.append((n,t))
    
    L=nx.laplacian_matrix(GH,nodelist=nodelist)
    
    n_scales=8
    filter_type='sagw'
    
    w=app_gsp_filter(f,L,N*nTimeSlices,n_scales,M=40,filter_type=filter_type)
    wav_coeff=np.abs(w.T)
    
    
    time=15  # 07:30 to 08:00
    f=signal[time,:]
    
    # 3889: 8th av with 41st st
    # 7055: church st with duane st
    # 2194: 6th av with 53rd st
    
    for selected_node in [3889,7055,2194]:  
        node_size=[]
        for i in range(N):
            if (f[i] != 0):
                node_size.append(40)
            else:
                node_size.append(0)
        
        pos=dict()
        for i in G.nodes():
            pos[i]=(G.node[i]['x'],G.node[i]['y'])
        
        vmin=1
        vmax=np.max(f)
        
        xpos=[]
        ypos=[]
        for i in G.nodes():
            xpos.append(pos[i][0])
            ypos.append(pos[i][1])
            
        xmax=max(xpos)+0.001
        xmin=min(xpos)-0.001
        ymax=max(ypos)+0.001
        ymin=min(ypos)-0.001
        
        nodelist=G.nodes()
        zorder=list(np.argsort(f))
        
        plt.figure()
        nx.draw_networkx_edges(G,pos,width=0.5,edge_color='gray')
        for i in zorder:
            nx.draw_networkx_nodes(G,pos,nodelist=[nodelist[i]],node_color=f[i],
                    node_size=node_size[i],cmap=cmap,with_labels=False,vmin=vmin,
                    vmax=vmax,linewidths=0.1)
        
        if (values):
            values_pos=dict()
            for i in G.nodes():
                values_pos[i]=tuple(map(operator.add,pos[i],(0.0001,0.0001)))        
            
            neighbors=nx.single_source_shortest_path_length(G,selected_node,cutoff=2)
            
            signal_labels=dict()
            for i in neighbors.keys():
                signal_labels[i]=str(int(f[nodelist.index(i)]))
                
            nx.draw_networkx_labels(G,values_pos,signal_labels)
        
        colorbar=False
        if (colorbar):
            norm=plt.Normalize(vmin=vmin,vmax=vmax)
            sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm)
            sm._A=[]
            cbar=plt.colorbar(sm,orientation='horizontal')
            cbar.solids.set_edgecolor("face")
            
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        plt.axis('off')
        plt.show()
        
        x=range(n_scales)
        y=wav_coeff[nodelist.index(selected_node)+time*N,:]
        width=1/1.5
        
        if (selected_node == 3889):
            ymax=50.0
        if (selected_node == 7055):
            ymax=3.0
        if (selected_node == 2194):
            ymax=4.5
        
        plt.figure()
        plt.bar(x,y,width,color="blue")
        plt.ylim(ymax=ymax)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()