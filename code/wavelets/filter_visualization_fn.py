import numpy as np
import matplotlib.pyplot as plt

def filter_visualization_fn(G,filters,n_filters):
    L=G['L']
    lmax=G['lmax']
    spectrum=np.linalg.eigvalsh(L.toarray())
    
    x=np.arange(0, lmax-0.0001, 0.01)
    N=len(x)

    ymax=0
    y=np.zeros((N,))
    plt.figure()
    for j in range(n_filters):
        for l in range(N):
            y[l]=filters[j](x[l])
            
        if (ymax < np.max(y)):
            ymax=np.max(y)
        
        plt.plot(x,y,linewidth=2)
        
    xmax=spectrum[-1]
    
    plt.xlim(xmin=-0.1,xmax=xmax+0.1)
    plt.ylim(ymin=-0.1,ymax=ymax+0.1)
    plt.plot(spectrum,np.zeros(len(spectrum)),'kx',mew=1.5)    
    plt.grid('off')
    plt.show()