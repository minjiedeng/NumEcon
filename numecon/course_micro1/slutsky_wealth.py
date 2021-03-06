from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import ipywidgets as widgets

from . import consumer_old as consumer

###########
# figure  #
###########

def _figure(par,steps,p1_old,p1_new,e1,e2,alpha,beta):

    I_old = p1_old*e1 + e2
    I_new = p1_new*e1 + e2
    par.alpha = alpha
    par.beta = beta

    # a. calculations
    par.I = I_old
    par.p1 = p1_old
    x1,x2,u_max = consumer.maximization(par)

    par.p1 = p1_new
    x1_fixI,x2_fixI,u_max_fixI = consumer.maximization(par)
    h1, h2 = consumer.costminimization(par,u_max)

    par.I = I_new
    x1_new,x2_new,u_max_new = consumer.maximization(par)

    # b. figure
    fig = plt.figure(figsize=(6,6),dpi=100)
    ax = fig.add_subplot(1,1,1)

    # c. plots
    consumer.budgetline(ax,p1_old,par.p2,I_old,ls='--',alpha=0.50,label='original')
    if steps > 1:
        consumer.budgetline(ax,p1_new,par.p2,I_new,label='final')
    if steps > 2:
        consumer.budgetline(ax,p1_new,par.p2,p1_new*h1+par.p2*h2,ls=':',alpha=0.50,label='compensated')
    if steps > 3:
        consumer.budgetline(ax,p1_new,par.p2,I_old,ls='--',label='constant income')
    ax.plot(e1,e2,'ro',color='black')
    ax.text(e1*1.03,e2*1.03,'$E$')

    # A
    ax.plot(x1,x2,'ro',color='black')
    ax.text(x1*1.03,x2*1.03,'$A$')
    consumer.indifference_curve(ax,u_max,par,ls='--',label='original')

    # B
    if steps > 2:
        ax.plot(h1,h2,'ro',color='black')
        ax.text(h1*1.03,h2*1.03,'$B$')
    
    # C2
    if steps > 1:
        ax.plot(x1_new,x2_new,'ro',color='black')
        ax.text(x1_new*1.03,x2_new*1.03,'$C_2$')
        consumer.indifference_curve(ax,u_max_new,par,label='final')

    # C1
    if steps > 3:
        ax.plot(x1_fixI,x2_fixI,'ro',color='black')
        ax.text(x1_fixI*1.03,x2_fixI*1.03,'$C_1$')
        consumer.indifference_curve(ax,u_max_fixI,par,ls='-',label='constant income',color='firebrick')

    
    if steps > 3:
        line = f'subtitution: $B-A$   = ({h1-x1:5.2f},{h2-x2:5.2f})\n'
        line += f'     income: $C_1-B$  = ({x1_fixI-h1:5.2f},{x2_fixI-h2:5.2f})\n'
        line += f'     wealth: $C_2-C_1$ = ({x1_new-x1_fixI:5.2f},{x2_new-x2_fixI:5.2f})'        
        ax.text(0.45*par.x1_max,0.85*par.x2_max,line)
    
    # d. basic layout
    legend = ax.legend(loc='right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    

    ax.grid(ls='--',lw=1)
    ax.set_xlim([0,par.x1_max])
    ax.set_ylim([0,par.x2_max])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.show()

def figure(par):

    widgets.interact(_figure,
        par=widgets.fixed(par), 
        steps=widgets.IntSlider(description='steps',min=1, max=4, step=1, value=1),
        p1_old=widgets.FloatSlider(description='$p_1$',min=par.p1_min, max=par.p1_max, step=par.p1_step, value=par.p1),
        p1_new=widgets.FloatSlider(description='$p_1^{\\prime}$',min=par.p1_min, max=par.p1_max, step=par.p1_step, value=par.p1_new),
        e1=widgets.FloatSlider(description='$e_1$',min=par.e1_min, max=par.e1_max, step=par.e1_step, value=par.e1),
        e2=widgets.FloatSlider(description='$e_2$',min=par.e2_min, max=par.e2_max, step=par.e2_step, value=par.e2),
        alpha=widgets.FloatSlider(description='$\\alpha$',min=par.alpha_min, max=par.alpha_max, step=par.alpha_step, value=par.alpha),
        beta=widgets.FloatSlider(description='$\\beta$',min=par.beta_min, max=par.beta_max, step=par.beta_step, value=par.beta))

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

    # d. utility
    par.u = None
    par.g = None
    par.g_inv = None
    par.monotone = True
    par.alpha = 1.00
    par.beta = 1.00

    # e. budgetset
    par.p1 = 1
    par.p1_new = 3
    par.p2 = 1
    par.e1 = 4
    par.e2 = 2

    # f. slider
    par.alpha_min = 0.05
    par.alpha_max = 4.00
    par.alpha_step = 0.05

    par.beta_min = 0.05
    par.beta_max = 4.00
    par.beta_step = 0.05

    par.p1_min = 0.05   
    par.p1_max = 4.00    
    par.p1_step = 0.05

    par.I_min = 0.5
    par.I_max = 20
    par.I_step = 0.05

    par.e1_min = 0.0
    par.e1_max = 5
    par.e1_step = 0.05

    par.e2_min = 0.0
    par.e2_max = 5
    par.e2_step = 0.05
        
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

    par.alpha = 0.50
    par.beta = 0.85

    par.alpha_min = 0.05
    par.alpha_max = 0.99

    par.beta_min = -0.95
    par.beta_max = 10.01

    figure(par)

def perfect_complements():

    par = settings()
    consumer.utility_functions(par,'leontief')
    figure(par)

def perfect_substitutes():

    par = settings()
    par.p1 = 0.75
    par.e1 = 2
    par.e2 = 2
    consumer.utility_functions(par,'perfect_substitutes')
    figure(par)
