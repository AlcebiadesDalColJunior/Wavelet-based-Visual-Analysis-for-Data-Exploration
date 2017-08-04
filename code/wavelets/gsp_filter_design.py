from __future__ import division

import cmath
import numpy as np
import scipy.optimize as opt

from math import pi, cos, log, sqrt

def gsp_filter_design(filter_type, n_filters, lmax, approx_spectrum=None):
    if (filter_type == 'ut'): 
        filters=gsp_uniform_translates_filter_design(n_filters,lmax)
        
    if (filter_type == 'log'):
        filters=gsp_warped_translates_log(n_filters,lmax)
        
    if (filter_type == 'sagw_lin'):
        filters=gsp_warped_translates_pwl(n_filters,lmax,approx_spectrum)
        
    if (filter_type == 'sagw'):
        filters=gsp_warped_translates_mono_cubic(n_filters,lmax,approx_spectrum)
        
    if (filter_type == 'sgw'):
        filters=gsp_spectral_graph_wavelet(n_filters,lmax,lpfactor=20,a=2,b=2,t1=1,t2=2)
    
    return filters

# 
def gsp_uniform_translates_filter_design(n_filters,upper_bound_translates):
    dilation_factor=upper_bound_translates*(3/(n_filters-2))
    main_window=lambda x: .5+.5*cos(2*pi*(x/dilation_factor-1/2)) if (x>=0) and (x<=dilation_factor) else 0
    
    filters=[None]*(n_filters)
    for j in range(n_filters):
        filters[j]=lambda x, t=j: main_window(x-dilation_factor/3*(t+1-3))
    
    return(filters)

#
def gsp_log_wavelet_scaling_fn(wavelet_filters,x):
    output=9/8
    for i in range(len(wavelet_filters)):
        output=output-(wavelet_filters[i](x))**2
    
    output=cmath.sqrt(output)
    output=output.real
    
    return(output)

def gsp_warped_translates_log(n_filters,lmax):
    eps=np.finfo(np.float).eps
    warp_function=lambda x: log(x+eps)
    
    uniform_filters=gsp_uniform_translates_filter_design(n_filters-1,log(lmax))
    
    wavelet_filters=[None]*(n_filters-1)
    for j in range(n_filters-1):
        wavelet_filters[j]=lambda x, j=j: uniform_filters[j-1](warp_function(x))    
    
    filters=[None]*(n_filters)
    for j in range(1,n_filters):
        filters[j]=wavelet_filters[j-1]
        
    filters[0]=lambda x: gsp_log_wavelet_scaling_fn(wavelet_filters,x)
    
    return(filters)
    
#
def gsp_warped_translates_pwl(n_filters,lmax,approx_spectrum):    
    warp_function=lambda s: gsp_pwl_warp_fn(approx_spectrum['x'], approx_spectrum['y'], [s])
    upper_bound_translates=max(approx_spectrum['y'])
    uniform_filters=gsp_uniform_translates_filter_design(n_filters,upper_bound_translates)
    
    filters=[None]*(n_filters)
    for j in range(n_filters):
        filters[j]=lambda x, j=j: uniform_filters[j](warp_function(x))
    
    return(filters)
    
def gsp_pwl_warp_fn(x,y,x0):
    cut=1e-4
    num_pts=len(x)
    num_pts_to_interpolate=len(x0)
    interpolated_values=np.zeros((num_pts_to_interpolate,1))
    for i in range(num_pts_to_interpolate):
        closest_ind=np.argmin(np.abs(x-x0[i]))
        
        if (x[closest_ind]-x0[i])<(-cut) or (abs(x[closest_ind]-x0[i])<cut and closest_ind < num_pts-1):
            lower_ind=closest_ind
        else:
            lower_ind=closest_ind-1
            
        interpolated_values[i]=y[lower_ind]*(x[lower_ind+1]-x0[i])/float(x[lower_ind+1]-x[lower_ind])+y[lower_ind+1]*(x0[i]-x[lower_ind])/float(x[lower_ind+1]-x[lower_ind])
    
    return(interpolated_values)
    
#
def gsp_warped_translates_mono_cubic(n_filters,lmax,approx_spectrum):    
    warp_function=lambda s: gsp_mono_cubic_warp_fn(approx_spectrum['x'],approx_spectrum['y'], [s])
    upper_bound_translates=max(approx_spectrum['y'])
    uniform_filters=gsp_uniform_translates_filter_design(n_filters,upper_bound_translates)
    
    filters=[None]*(n_filters)
    for j in range(n_filters):
        filters[j]=lambda x, j=j: uniform_filters[j](warp_function(x))
    
    return(filters)
    
def gsp_mono_cubic_warp_fn(x,y,x0):
    cut=1e-4
    num_pts=len(x)
    
    # 1. Compute slopes of secant lines
    Delta=np.true_divide(y[1:]-y[0:num_pts-1],x[1:]-x[0:num_pts-1])
    
    # 2. Initialize tangents m at every data point
    m=(Delta[0:num_pts-2]+Delta[1:num_pts-1])/2
    m=np.concatenate((np.array([Delta[0]]),m,np.array([Delta[-1]])))    
    
    # 3. Check for equal y's to set slopes equal to zero
    for k in range(num_pts-1):
        if (Delta[k] == 0):
            m[k]=0
            m[k+1]=0

    # 4. Initialize alpha and beta
    alpha=m[0:num_pts-1]/Delta
    beta=m[1:num_pts]/Delta 
    
    # 5. Make monotonic
    for k in range(num_pts-1):
        if (alpha[k]**2+beta[k]**2 > 9):
            tau=3/float(sqrt(alpha[k]**2+beta[k]**2))
            m[k]=tau*alpha[k]*Delta[k]
            m[k+1]=tau*beta[k]*Delta[k]
    
    # 6. Cubic interpolation
    num_pts_to_interpolate=len(x0)
    interpolated_values=np.zeros((num_pts_to_interpolate,1))
    
    for i in range(num_pts_to_interpolate):
        closest_ind=np.argmin(np.abs(x-x0[i]))
        
        if ((x[closest_ind]-x0[i])<(-cut) or (abs(x[closest_ind]-x0[i])<cut and closest_ind < num_pts-1)):
            lower_ind=closest_ind
        else:
            lower_ind=closest_ind-1

        h=x[lower_ind+1]-x[lower_ind]
        t=(x0[i]-x[lower_ind])/float(h)
          
        interpolated_values[i]=y[lower_ind]*(2*t**3-3*t**2+1)+h*m[lower_ind]*(t**3-2*t**2+t)+y[lower_ind+1]*(-2*t**3+3*t**2)+h*m[lower_ind+1]*(t**3-t**2)
    
    return(interpolated_values)

#
def gsp_spectral_graph_wavelet(n_filters,lmax,lpfactor,a,b,t1,t2):
    lmin=lmax/lpfactor
    t=log_scales(lmin,lmax,n_filters-1)
    
    gb=lambda x: kernel_abspline3(x,a,b,t1,t2)
    g=[None]*(n_filters)
    for j in range(n_filters-1):
        g[j+1]=lambda x, t=t[j]: gb(t*x)
        
    f=lambda x: -gb(x)
    xstar=opt.fminbound(f,1,2)

    gamma_l=-f(xstar)
    lminfac=0.6*lmin
    
    gl=lambda x: np.exp(-x**4)
    g[0]=lambda x: gamma_l*gl(x/lminfac)
    
    return(g)

def log_scales(lmin,lmax,nScales,t1=1,t2=2):
  smin=t1/lmax
  smax=t2/lmin
  
  return(np.exp(np.linspace(np.log(smax),np.log(smin),nScales)))
  
def kernel_abspline3(x,alpha,beta,t1,t2):
    x=np.array(x)
    r=np.zeros(x.shape) 
    a=np.array([-5,11,-6,1])
    
    r1=(x>=0)&(x<t1)
    r2=(x>=t1)&(x<t2)
    r3=x>=t2
    
    r[r1]=x[r1]**alpha*t1**(-alpha)
    r[r2]=a[0]+a[1]*x[r2]+a[2]*x[r2]**2+a[3]*x[r2]**3
    r[r3]=x[r3]**(-beta)*t2**(beta)
    
    return(r)