from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import ipywidgets as widgets

from . import consumer

###########
# figure  #
###########

def _figure(par,p1,omega_1,omega_2,alpha_A,beta_A,alpha_B,beta_B):

    par.p1 = p1

    # a. figure
    fig = plt.figure(figsize=(6,6),dpi=100)
    ax = fig.add_subplot(1,1,1)

    # b. plots

    # A
    e1_A = omega_1
    e2_A = omega_2
    par.I = p1*e1_A + e2_A
    par.alpha = alpha_A
    par.beta = beta_A
    x1_A,x2_A,u_max_A = consumer.maximization(par)    
    ax.plot(x1_A,x2_A,'ro',color='black')
    ax.text(x1_A*1.03,x2_A*1.03,'A')
    consumer.indifference_curve(ax,u_max_A,par)

    # E
    ax.plot(e1_A,e2_A,'ro',color='black')
    ax.text(e1_A*1.03,e2_A*1.03,'E')
    I = par.p1*e1_A + par.p2*e2_A
    consumer.budgetline(ax,par.p1,par.p2,I)

    # B
    e1_B = 1-omega_1
    e2_B = 1-omega_2
    par.I = p1*e1_B + e2_B
    par.alpha = alpha_B
    par.beta = beta_B
    x1_B,x2_B,u_max_B = consumer.maximization(par)
    ax.plot(1-x1_B,1-x2_B,'ro',color='black')
    ax.text((1-x1_B)*1.03,(1-x2_B)*1.03,'B')
    consumer.indifference_curve(ax,u_max_B,par,inv=True)

    line1 = f'excess demand of good 1: {x1_A+x1_B-1:6.2f}\n'
    line2 = f'excess demand of good 2: {x2_A+x2_B-1:6.2f}\n'
    ax.text(0.05,0.01,line1+line2)

    # dc. basic layout
    ax.grid(ls='--',lw=1)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel('$x_1^A$')
    ax.set_ylabel('$x_2^A$')

    ax_top = ax.twiny()
    ax_top.set_xlabel('$x_1^B$')
    ax_top.invert_xaxis()

    ax_right = ax.twinx()
    ax_right.set_ylabel('$x_2^B$')
    ax_right.invert_yaxis()

    plt.show()

def figure(par):

    widgets.interact(_figure,
        par=widgets.fixed(par), 
        p1=widgets.BoundedFloatText(description='$p_1$',min=par.p1_min, max=par.p1_max, step=par.p1_step, value=par.p1),
        omega_1=widgets.FloatSlider(description='$\\omega_1$',min=0.05, max=0.99, step=0.05, value=par.omega_1),
        omega_2=widgets.FloatSlider(description='$\\omega_2$',min=0.05, max=0.99, step=0.05, value=par.omega_2),
        alpha_A=widgets.FloatSlider(description='$\\alpha^A$',min=par.alpha_min, max=par.alpha_max, step=par.alpha_step, value=par.alpha_A),
        beta_A=widgets.FloatSlider(description='$\\beta^A$',min=par.beta_min, max=par.beta_max, step=par.beta_step, value=par.beta_A),
        alpha_B=widgets.FloatSlider(description='$\\alpha^B$',min=par.alpha_min, max=par.alpha_max, step=par.alpha_step, value=par.alpha_B),
        beta_B=widgets.FloatSlider(description='$\\beta^B$',min=par.beta_min, max=par.beta_max, step=par.beta_step, value=par.beta_B)
    )

############
# settings #
############

def settings():
    
    # a. setup
    par = SimpleNamespace()

    # b. indifference curves
    par.x1_max = 1
    par.x2_max = 1
    par.N = 100 # number of points when calculating

    # c. utility
    par.u = None
    par.g = None
    par.g_inv = None
    par.monotone = True
    par.alpha_A = 1.00
    par.beta_A = 1.50
    par.alpha_B = 1.00
    par.beta_B = 1.50

    # d. budgetset
    par.p1 = 1
    par.p1_new = 3
    par.p2 = 1

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

    # f. endowment
    par.omega_1 = 0.80
    par.omega_2 = 0.20

    # g. technical
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

    par.p1 = 2

    par.alpha_A = 0.50
    par.beta_A = 0.85

    par.alpha_B = 0.50
    par.beta_B = 0.85

    par.alpha_min = 0.05
    par.alpha_max = 0.99

    par.beta_min = -0.95
    par.beta_max = 10.01

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