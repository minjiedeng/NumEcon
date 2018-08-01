from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import ipywidgets as widgets

import utility
import budgetset

####################
## 1. maximization #
####################

def maximization(par):

    if par.monotone:

        # a. target
        u,_eps = utility.func(par)
        def x2_func(x1):
            return (par.I - par.p1*x1)/par.p2
        def target(x1):
            x2 = x2_func(x1)
            return -u(x1,x2)
            
        # b. solve
        xopt = optimize.fminbound(target, 0, par.I/par.p1)

        # c. save
        x1 = xopt
        par.x1 = [0.8*x1,x1,1.2*x1]
        x2 = x2_func(x1)
        par.x2 = [0.8*x2,x2,1.2*x2]

        return par

    else:

        # a. target
        u,_eps = utility.func(par)
        def target_2d(x): 
            return -u(x[0],x[1]) + 1000*(np.fmax(par.p1*x[0]+par.p2*x[1]-par.I,0))
            
        # b. solve
        x0 = [par.I/2/par.p1,par.I/2/par.p2]
        res = optimize.minimize(target_2d,x0,method='Nelder-Mead')

        # c. save
        x1 = res.x[0]
        par.x1 = [0.8*x1,x1,1.2*x1]
        x2 = res.x[1]
        par.x2 = [0.8*x2,x2,1.2*x2]

        return par

###########
# 2. draw #
###########

def draw_figure(par):      
    
    # a. calculations
    par = maximization(par)
    indiff_sets, u0s = utility.indifference_set(par)
    x1s,x2s, _slope_xy,_slope = budgetset.calc_exogenous(par.p1,par.p2,par.I)

    # b. figure
    fig = plt.figure(frameon=False, figsize=(6,6), dpi=100)
    ax = fig.add_subplot(1,1,1)

    # c. basic layout
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    ax.grid(True)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    # d. draw axes
    budgetset.draw(ax,x1s,x2s)
    utility.draw(par,ax,indiff_sets,u0s)

    plt.show()

############
# 3. basic #
############

def settings():
    
    # a. setup
    par = dict()
    par = SimpleNamespace(**par)

    # b. curves
    par.upper = 10
    par.N = 50

    # d. figure
    par.plot_type = 'line'

    # e. update
    par.interact_basic = True
    par.p1_step = 0.05
    par.p2_step = 0.05
    par.I_step = 0.05
    par.alpha_step = 0.05
    par.beta_step = 0.05

    # f. budgetset
    par.p1 = 1
    par.p2 = 1
    par.I = 8
    par.p1_min = 0.05   
    par.p1_max = 4.00    
    par.p2_min = 0.05
    par.p2_max = 4.00    
    par.I_min = 0.5
    par.I_max = 20

    # g. preferences
    par.monotone = True
    
    return par

def update(par,p1,p2,I,alpha,beta):

    par.p1 = p1
    par.p2 = p2
    par.I = I
    par.alpha = alpha
    par.beta = beta

    draw_figure(par)

def interact(par):

    widgets.interact(update,
                     par=widgets.fixed(par), 
                     p1=widgets.FloatSlider(min=par.p1_min, max=par.p1_max, step=par.p1_step, value=par.p1),
                     p2=widgets.FloatSlider(min=par.p2_min, max=par.p2_max, step=par.p2_step, value=par.p2),
                     I=widgets.FloatSlider(min=par.I_min, max=par.I_max, step=par.I_step, value=par.I),
                     alpha=widgets.FloatSlider(min=par.alpha_min, max=par.alpha_max, step=par.alpha_step, value=par.alpha),
                     beta=widgets.FloatSlider(min=par.beta_min, max=par.beta_max, step=par.beta_step, value=par.beta))

def cobb_douglas():

    par = settings()

    par.uname = 'cobb_douglas'

    par.alpha = 0.50
    par.beta = 0.50

    par.alpha_min = 0.05
    par.alpha_max = 0.95

    par.beta_min = 0.05
    par.beta_max = 0.95

    interact(par)

def ces():

    par = settings()

    par.uname = 'ces'

    par.alpha = 0.50
    par.beta = 0.85

    par.alpha_min = 0.05
    par.alpha_max = 0.95

    par.beta_min = -0.95
    par.beta_max = 10

    interact(par)

def perfect_substitutes():

    par = settings()

    par.uname = 'perfect_substitutes'

    par.p1 = 1.5
    par.I = 5

    par.alpha = 1.00
    par.beta = 1.00

    par.alpha_min = 0.05
    par.alpha_max = 3.00

    par.beta_min = 0.05
    par.beta_max = 3.00

    interact(par)

def perfect_complements():

    par = settings()

    par.uname = 'perfect_complements'

    par.alpha = 1.00
    par.beta = 1.00

    par.alpha_min = 0.05
    par.alpha_max = 3.00

    par.beta_min = 0.05
    par.beta_max = 3.00

    interact(par)

def quasi_linear_case_1():

    par = settings()

    par.uname = 'quasi_linear_case_1'

    par.alpha = 1.00
    par.beta = 1.00

    par.alpha_min = 0.05
    par.alpha_max = 3.00

    par.beta_min = 0.05
    par.beta_max = 3.00

    interact(par)

def quasi_linear_case_2():

    par = settings()

    par.uname = 'quasi_linear_case_2'

    par.alpha = 1.00
    par.beta = 1.00

    par.alpha_min = 0.05
    par.alpha_max = 3.00

    par.beta_min = 0.05
    par.beta_max = 3.00

    interact(par)

def arbitrary(uname,alpha,beta,alpha_bounds,beta_bounds,p1,p2,I,p1_bounds,p2_bounds,I_bounds,plot_type='scatter',monotone=True):

    par = settings()
    par.plot_type = plot_type

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
    par.uname = uname
    par.monotone = monotone

    par.alpha = alpha
    par.beta = beta

    par.alpha_min = alpha_bounds[0]
    par.alpha_max = alpha_bounds[1]

    par.beta_min = beta_bounds[0]
    par.beta_max = beta_bounds[1]

    interact(par)
