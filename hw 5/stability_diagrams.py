"""
Some routines for plotting ODE stability diagrams for single-step multi-stage schemes and linear mulit-step schemes
"""

import numpy as numpy
import matplotlib.pyplot as plt

def stability_plot(x, y, C, axes, title=None, continuous=True):
    """
    Utility function to make stability diagram given complex stability scalar C
    
    parameters:
    -----------
    
    x: numpy array
        array of values for the real axis
    y: numpy array
        values to plot for the imaginary axis    C: numpy array
        Field to plot,  either |R(z)| for a single step scheme, or max(|xi_i(z)|) for a LMM scheme
    axes: matplotlib axes object
        subplot or plot to draw in. 
    title: string
        subplot title if not None 
    continuous: boolean
        if True, plot a continous coloring of C
        if False, plot Heaviside(C)
    """
    if  continuous:
        Ch = C
    else:
        Ch = numpy.heaviside(C-1,0.)
    X, Y = numpy.meshgrid(x,y)
    pcolor_plot = axes.pcolor(X, Y, Ch, vmin=0, vmax=1, cmap=plt.get_cmap('Greens_r'), shading='auto')
    axes.contour(X, Y, C, 'k', levels=[1.0])
    fig = plt.gcf()
    fig.colorbar(pcolor_plot)
    axes.plot(x, numpy.zeros(x.shape),'k--')
    axes.plot(numpy.zeros(y.shape), y,'k--')
    
    axes.set_xlabel('Re', fontsize=16)
    axes.set_ylabel('Im', fontsize=16)
    if title is not None:
        axes.set_title(title, fontsize=16)
    
    axes.set_aspect('equal')    
    
def plot_stability_ssms(R, x, y, axes=None, title=None, continuous=True):
    """ 
    plot stability regions for single-step multi-stage ODE schemes given the function R(z)
    such that U_{n+1} = R(z)U_n   and z is complex
    
    parameters:
    -----------
    
    R: calleable
        function of a complex variable z such that if |R|<=1, the scheme is absolutely stable
    x: numpy array
        array of values for the real axis
    y: numpy array
        values to plot for the imaginary axis
    axes: matplotlib axes object
        subplot or plot to draw in.  If axes=None create a new figure
    title: string
        subplot title if 
    continuous: boolean
        if True, plot a continous coloring of C
        if False, plot Heaviside(C)
    """
    
    X,Y = numpy.meshgrid(x,y)
    Z = X + 1j * Y
    if axes is None:
        fig = plt.figure(figsize=(8,6))
        axes = fig.add_subplot(1,1,1)
    
    abs_R = numpy.abs(R(Z))
    stability_plot(x, y, numpy.abs(R(Z)), axes, title, continuous)
    
def plot_stability_lmm(pi_coeff, x, y, axes=None, title=None, continuous=True):
    """ 
    plot stability regions for linear multi-step  ODE schemes given the coefficients of the stability polynomial
    pi(xi, z)
    
    parameters:
    -----------
    
    pi_coeff: calleable (function of z)
        function that returns array of stability polynomial pi(z)
    x: numpy array
        array of values for the real axis
    y: numpy array
        values to plot for the imaginary axis
    axes: matplotlib axes object
        subplot or plot to draw in.  If axes=None create a new figure
    title: string
        subplot title if not None   
    continuous: boolean
        if True, plot a continous coloring of C
        if False, plot Heaviside(C)
    """
       
    X,Y = numpy.meshgrid(x,y)
    Z = X + 1j * Y
    if axes is None:
        fig = plt.figure(figsize=(8,6))
        axes = fig.add_subplot(1,1,1)
    
    norm_max = numpy.empty(Z.shape)
    for i,row in enumerate(Z):
        for j, z in enumerate(row):
            norm_max[i,j] = max(numpy.abs(numpy.roots(pi_coeff(z))))
    
    stability_plot(x, y, norm_max, axes, title, continuous)
    