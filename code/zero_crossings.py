import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from util import zero_crossings_count


N=64

G=nx.path_graph(N)
L=nx.laplacian_matrix(G).asfptype()

spectrum,U=np.linalg.eigh(L.toarray())

counts=[]
for j in range(N):
    counts.append(zero_crossings_count(G,U[:,j]))
    
plt.figure()
plt.plot(spectrum,counts,'bo')
plt.xlabel('$\lambda_{\ell}$',fontsize=30)
plt.ylabel('Number of zero crossings',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()