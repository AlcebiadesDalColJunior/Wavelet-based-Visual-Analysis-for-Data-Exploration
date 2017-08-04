from __future__ import division

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import identity

from cvxopt import cholmod, matrix, spmatrix
from cvxopt.cholmod import options

# Compute an approximation of the cumulative density function of the graph Laplacian eigenvalues
def gsp_spectrum_cdf_approx(G):
    num_pts=8
    
    n=G['N']
    lmax=G['lmax']
    
    counts=np.zeros((num_pts,))
    counts[-1]=n-1
    
    interp_x=np.arange(num_pts)*lmax/(num_pts-1)
    
    I=identity(n)
    A=csc_matrix(G['L'])
    
    options['supernodal']=0
    
    for i in range(1,num_pts-1):
        shift_matrix=csc_matrix(interp_x[i]*I)
        
        mat=A-shift_matrix
        
        Acoo=mat.tocoo()
        mats=spmatrix(Acoo.data,Acoo.row.tolist(),Acoo.col.tolist())
        F=cholmod.symbolic(mats)
        cholmod.numeric(mats,F)
        
        Di=matrix(1.0,(n,1))
        cholmod.solve(F,Di,sys=6)
        
        D=np.zeros((n,))
        for ii in range(n):
            D[ii]=Di[ii]
        
        counts[i]=np.sum(D<0)
    
    interp_y=counts/(n-1)
    
    approx_spectrum=dict()
    approx_spectrum['x']=interp_x
    approx_spectrum['y']=interp_y
    
    return(approx_spectrum)