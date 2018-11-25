from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import ipywidgets as widgets

from . import consumer_old as consumer

###########
# figure  #
###########

def _figure(par,p1,p2,I,alpha,beta,gamma):

    par.p1 = p1
    par.p2 = p2
    par.I = I
    par.alpha = alpha
    par.beta = beta

    def xs_from_gamma(gamma):
        x1 = I/p1*gamma
        x2 = (I-p1*x1)/p2
        return x1,x2
    x1,x2 = xs_from_gamma(gamma)

    # a. calculations
    x1_max,x2_max,u_max = consumer.maximization(par)

    u_alt = [par.u(x1,x2,alpha,beta),par.u(x1_max*1.2,x2_max*1.2,alpha,beta)]

    # b. figure
    fig = plt.figure(figsize=(6,6),dpi=100)
    ax = fig.add_subplot(1,1,1)

    # c. plots
    consumer.budgetline(ax,p1,p2,I)
    
    ax.plot(x1_max,x2_max,'ro',color='black')
    ax.text(x1_max*1.03,x2_max*1.03,f'$u^{{max}} = {u_max:5.2f}$')

    ax.plot(x1,x2,'o',color='firebrick')
    ax.text(x1*1.03,x2*1.03,f'$u^{{\gamma}} = {par.u(x1,x2,alpha,beta):5.2f}$')

    consumer.indifference_curve(ax,u_max,par)
    [consumer.indifference_curve(ax,u,par,ls='--') for  u in u_alt]

    # d. basic layout
    ax.grid(ls='--',lw=1)
    ax.set_xlim([0,par.x1_max])
    ax.set_ylim([0,par.x2_max])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.show()

def figure(par):

    widgets.interact(_figure,
        par=widgets.fixed(par), 
        p1=widgets.FloatSlider(description='$p_1$',min=par.p1_min, max=par.p1_max, step=par.p1_step, value=par.p1),
        p2=widgets.FloatSlider(description='$p_2$',min=par.p2_min, max=par.p2_max, step=par.p2_step, value=par.p2),
        I=widgets.FloatSlider(description='$I$',min=par.I_min, max=par.I_max, step=par.I_step, value=par.I),
        alpha=widgets.FloatSlider(description='$\\alpha$',min=par.alpha_min, max=par.alpha_max, step=par.alpha_step, value=par.alpha),
        beta=widgets.FloatSlider(description='$\\beta$',min=par.beta_min, max=par.beta_max, step=par.beta_step, value=par.beta),
        gamma=widgets.FloatSlider(description='$\\gamma$',min=0.01, max=0.99, step=0.01, value=0.25))

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
    par.N = 100 # number of points when calculating

    # c. utility
    par.u = None
    par.g = None
    par.g_inv = None
    par.monotone = True
    par.alpha = 1.00
    par.beta = 1.00

    # d. budgetset
    par.p1 = 1
    par.p2 = 1
    par.I = 8

    # e. slider
    par.alpha_min = 0.05
    par.alpha_max = 4.00
    par.alpha_step = 0.05

    par.beta_min = 0.05
    par.beta_max = 4.00
    par.beta_step = 0.05

    par.p1_min = 0.05   
    par.p1_max = 4.00    
    par.p1_step = 0.05

    par.p2_min = 0.05
    par.p2_max = 4.00    
    par.p2_step = 0.05

    par.I_min = 0.5
    par.I_max = 20
    par.I_step = 0.05

    # e. technical
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

    par.alpha = 0.50
    par.beta = 0.85

    par.alpha_min = 0.05
    par.alpha_max = 0.99

    par.beta_min = -0.95
    par.beta_max = 10.01

    figure(par)

def perfect_substitutes():

    par = settings()
    consumer.utility_functions(par,'perfect_substitutes')

    par.p1 = 1.5
    par.I = 5
    
    figure(par)

def perfect_complements():

    par = settings()
    consumer.utility_functions(par,'leontief')
    figure(par)

def quasi_linear_log():

    par = settings()
    consumer.utility_functions(par,'quasi_linear',v=np.log)
    
    par.alpha = 3.00
    par.beta = 1.00

    figure(par)

def quasi_linear_sqrt():

    par = settings()
    consumer.utility_functions(par,'quasi_linear',np.sqrt)
    
    par.alpha = 3.00
    par.beta = 1.00

    figure(par)    

def concave():

    par = settings()
    consumer.utility_functions(par,'concave')
    
    par.p2 = 2

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

    par.monotone = False

    figure(par) 

def arbitrary(u,alpha,beta,alpha_bounds,beta_bounds,
        p1,p2,I,p1_bounds,p2_bounds,I_bounds,
        monotone=True):

    par = settings()

    # a. budget set
    par.p1 = p1
    par.p2 = p2
    par.I = I

    par.p1_min = p1_bounds[0]
    par.p1_max = p1_bounds[1]

    par.p2_min = p2_bounds[0]
    par.p2_max = p2_bounds[1]

    par.I_min = I_bounds[0]
    par.I_max = I_bounds[1]

    # b. utility
    par.u = u
    par.uname = ''
    par.monotone = monotone

    par.alpha = alpha
    par.beta = beta

    par.alpha_min = alpha_bounds[0]
    par.alpha_max = alpha_bounds[1]

    par.beta_min = beta_bounds[0]
    par.beta_max = beta_bounds[1]

    figure(par)    