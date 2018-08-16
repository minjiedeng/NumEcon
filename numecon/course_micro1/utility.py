from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import ipywidgets as widgets

from . import consumer

##########
# figure #
##########

def _figure(par,alpha,beta):

    par.alpha = alpha
    par.beta = beta 
    
    # a. figure
    fig = plt.figure(frameon=False, figsize=(6,6),dpi=100)
    ax = fig.add_subplot(1,1,1)

    # b. 45 degrees 
    ax.plot([0,10],[0,10],'--',color="black",zorder=1,alpha=0.1)

    # c. indifference curves
    x1s = par.x1s
    x2s = par.x2s
    us = [par.u(x1,x2,par.alpha,par.beta) for x1,x2 in zip(x1s,x2s)]
    [ax.plot(x1,x2,'ro',color='black') for x1,x2 in zip(x1s,x2s)]
    [ax.text(x1*1.03,x2*1.03,f'u = {u:5.2f}') for  x1,x2,u in zip(x1s,x2s,us)]
    [consumer.indifference_curve(ax,u,par) for  u in us]
    
    # d. extra stuff
    consumer.monotonicity(ax,par,x1s[1],x2s[1])
    consumer.convex_combination(ax,par,x1s[1],x2s[1],us[1])

    # e. basic layout
    ax.grid(ls='--',lw=1)
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.show()

def figure(par):

    widgets.interact(_figure,
        par=widgets.fixed(par), 
        alpha=widgets.FloatSlider(description='$\\alpha$',min=par.alpha_min,max=par.alpha_max,step=par.alpha_step,value=par.alpha),
        beta=widgets.FloatSlider(description='$\\beta$',min=par.beta_min,max=par.beta_max,step=par.beta_step,value=par.beta))

############
# settings #
############

def settings():
    
    # a. setup
    par = SimpleNamespace()

    # b. layout
    par.x1_max = 10
    par.x2_max = 10

    # c. indifference curves
    par.x1s = [2,3,4] # x1 starting points
    par.x2s = [2,3,4] # x2 starting points
    par.N = 100 # number of points when calculating

    # d. utility
    par.u = None
    par.g = None
    par.g_inv = None
    par.alpha = 1.00
    par.beta = 1.00

    # e. slider
    par.alpha_min = 0.05
    par.alpha_max = 4.00
    par.alpha_step = 0.05

    par.beta_min = 0.05
    par.beta_max = 4.00
    par.beta_step = 0.05

    # f. technical
    par.eps = 1e-8

    return par

#########
# cases #
#########

def cobb_douglas():

    par = settings()
    consumer.utility_functions(par,'cobb_douglas')
    figure(par)

def ces():

    par = settings()
    consumer.utility_functions(par,'ces')

    par.alpha = 0.65
    par.beta = 0.85

    par.alpha_min = 0.05
    par.alpha_max = 0.99

    par.beta_min = -0.95
    par.beta_max = 10.01

    figure(par)

def perfect_substitutes():

    par = settings()
    consumer.utility_functions(par,'perfect_substitutes')
    figure(par)

def perfect_complements():

    par = settings()
    consumer.utility_functions(par,'leontief')
    figure(par)

def quasi_linear_log():

    par = settings()
    consumer.utility_functions(par,'quasi_linear',v=np.log)
    figure(par)

def quasi_linear_sqrt():

    par = settings()
    consumer.utility_functions(par,'quasi_linear',np.sqrt)
    figure(par)    

def concave():

    par = settings()
    consumer.utility_functions(par,'concave')
    figure(par) 

def quasi_quasi_linear():

    par = settings()
    consumer.utility_functions(par,'quasi_quasi_linear')
    figure(par) 

def saturated():

    par = settings()

    consumer.utility_functions(par,'saturated')

    par.alpha = 5.00
    par.beta = 5.00

    par.alpha_min = 0.0
    par.alpha_max = 8

    par.beta_min = 0.0
    par.beta_max = 8

    figure(par) 

def arbitrary(u,alpha,beta,alpha_bounds,beta_bounds,plot_type='scatter'):

    par = settings()

    par.u = u
    par.uname = ''

    par.alpha = alpha
    par.beta = beta

    par.alpha_min = alpha_bounds[0]
    par.alpha_max = alpha_bounds[1]

    par.beta_min = beta_bounds[0]
    par.beta_max = beta_bounds[1]

    figure(par)