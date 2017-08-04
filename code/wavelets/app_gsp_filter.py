from __future__ import division

from math import pi
import numpy as np
from scipy.integrate import trapz
from scipy.sparse.linalg import eigsh

from gsp_filter_design import gsp_filter_design
from gsp_spectrum_cdf_approx import gsp_spectrum_cdf_approx

from filter_visualization_fn import filter_visualization_fn


def coeff_c(L,N,M,n_filters,lambdaMax,alpha,filter_type,view_filters):
    
    theta=np.arange(0,pi+0.001,pi/100)
    w=len(theta)

    G=dict()    
    G['N']=N
    G['L']=L
    G['lmax']=lambdaMax
    
    if (filter_type == 'sagw'):
        approx_spectrum=gsp_spectrum_cdf_approx(G)
        filters=gsp_filter_design(filter_type,n_filters,lambdaMax,approx_spectrum)
    else:
        filters=gsp_filter_design(filter_type,n_filters,lambdaMax)
        
    if (view_filters):
        filter_visualization_fn(G,filters,n_filters)
    
    c=np.zeros((n_filters,M))
    a=alpha*(np.cos(theta)+1)
    for j in range(n_filters):
        b=np.zeros((w,))
        for i in range(w):
            b[i]=filters[j](a[i])
        for k in range(M):
            d=np.cos(k * theta)
            y=d * b
            c[j][k]=(2/pi)*trapz(y,x=theta)
    
    return(c)

def pol_chebyshev(f,L,N,M,alpha):

    pol=np.zeros((M, N))

    pol[0,:]=f
    
    sm=L.dot(f)
    pol[1,:]=((1/alpha)*sm)-f

    for k in range(2, M):
        sm=L.dot(pol[k-1,:])
        pol[k,:]=(2/alpha)*sm-2*pol[k-1,:]-pol[k-2,:]

    return(pol)

def pol_chebyshev_approximation(f,L,N,n_filters,M,filter_type,view_filters):
    
    lambdaMax=eigsh(L,k=1,which='LA',maxiter=1e9)[0]
    alpha=lambdaMax/2

    pol=pol_chebyshev(f,L,N,M,alpha)

    c=coeff_c(L,N,M,n_filters,lambdaMax,alpha,filter_type,view_filters)
    
    w=np.zeros((n_filters, N))
    for i in range(N):
        for j in range(n_filters):
            sm=0
            sm+=0.5*c[j][0]*f[i]
            sm+=np.dot(c[j,1:M],pol[1:M,i])
            w[j][i]=sm

    return(w)

def app_gsp_filter(f,L,N,n_filters=8,M=40,filter_type='sagw',view_filters=False):
    w=pol_chebyshev_approximation(f,L,N,n_filters,M,filter_type,view_filters)
    
    return(w)
    
