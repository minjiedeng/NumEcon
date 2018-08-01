from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import ipywidgets as widgets

 ########################
 ## 1. indifference set #
 ########################

def func(par):

    eps = 0 # hack to help numerical optimizer
    if callable(par.uname):
        u = lambda x1,x2: par.uname(x1,x2,par.alpha,par.beta)
    elif par.uname == 'cobb_douglas':
        u = lambda x1,x2: x1**par.alpha*x2**par.beta
    elif par.uname == 'ces':
        if par.beta == 0:
            u = lambda x1,x2: x1**par.alpha*x2**(1-par.alpha)
        else:
            u = lambda x1,x2: (par.alpha*x1**(-par.beta)+(1-par.alpha)*x2**(-par.beta))**(-1.0/par.beta)
            eps = 1e-8
    elif par.uname == 'perfect_substitutes':
        u = lambda x1,x2: par.alpha*x1+par.beta*x2
    elif par.uname == 'perfect_complements':
        u = lambda x1,x2: np.fmin(par.alpha*x1,par.beta*x2) + 1e-8*(par.alpha*x1-par.beta*x2)**2 # hack to help numerical optimizer
    elif par.uname == 'quasi_linear_case_1':
        u = lambda x1,x2: par.alpha*np.log(x1)+par.beta*x2
        eps = 1e-8
    elif par.uname == 'quasi_linear_case_2':
        u = lambda x1,x2: par.alpha*np.sqrt(x1)+par.beta*x2
        eps = 1e-8
    
    return u,eps

def indifference_set(par):

    indiff_sets = []
    u0s = []
    for x1,x2 in zip(par.x1,par.x2):
        
        # a. utility function
        u,eps = func(par)

        # b. baseline
        u0 = u(x1,x2)
        u0s.append(u0)

        x1s = []    
        x2s = []

        # c. special points
        if par.uname == 'perfect_complements':
            if par.alpha < par.beta:
                x1s.append(x1)
                x2s.append(par.alpha*x1/par.beta)
                x1s.append(x1+0.01)
                x2s.append(par.alpha*x1/par.beta)
            else:
                x1s.append(par.beta*x2/par.alpha)
                x2s.append(x2)
                x1s.append(par.beta*x2/par.alpha+0.01)
                x2s.append(x2)

        # d. fixed x1
        x_vec = np.linspace(eps,par.upper,par.N)
        for x1_now in x_vec:

            def target_for_x2(x2):
                return u(x1_now,np.fmax(x2,eps))-u0

            x_A,_infodict_A,ier_A,_mesg_A = optimize.fsolve(target_for_x2, 0, full_output=True)
            x_B,_infodict_B,ier_B,_mesg_B = optimize.fsolve(target_for_x2, par.upper, full_output=True)        

            if ier_A == 1:
                x1s.append(x1_now)
                x2s.append(x_A[0])
            else:
                x1s.append(np.nan)
                x2s.append(np.nan)

            if ier_B == 1 and np.abs(x_A[0]-x_B[0]) > 0.01:
                x1s.append(x1_now)
                x2s.append(x_B[0])
            else:
                x1s.append(np.nan)
                x2s.append(np.nan)

        # e. fixed x2
        x_vec = np.linspace(eps,par.upper,par.N)
        for x2_now in x_vec:

            def target_for_x1(x1):
                return u(np.fmax(x1,eps),x2_now)-u0

            x_A,_infodict_A,ier_A,_mesg_A = optimize.fsolve(target_for_x1, 0, full_output=True)
            x_B,_infodict_B,ier_B,_mesg_B = optimize.fsolve(target_for_x1, par.upper, full_output=True)
            
            if ier_A == 1:
                x1s.append(x_A[0])
                x2s.append(x2_now)
            else:
                x1s.append(np.nan)
                x2s.append(np.nan)

            if ier_B == 1 and np.abs(x_A[0]-x_B[0]) > 0.01:
                x1s.append(x_B[0])
                x2s.append(x2_now)
            else:
                x1s.append(np.nan)
                x2s.append(np.nan)

        # f. sort
        x1s = np.array(x1s)
        x2s = np.array(x2s)
        I = np.argsort(x1s)
        x1s = x1s[I]
        x2s = x2s[I]

        # g. append
        indiff_sets.append([x1s,x2s])

    return indiff_sets, u0s

###########
# 2. draw #
###########

def draw_figure(par):      
    
    # a. calculations
    indiff_sets, u0s = indifference_set(par)

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
    draw(par,ax,indiff_sets,u0s)
    draw_45(ax)

    plt.show()

def draw(par,ax,indiff_sets,u0s):

    for indiff_set in indiff_sets:
        if par.plot_type == 'line':
            ax.plot(indiff_set[0],indiff_set[1],linewidth=2,color="navy",zorder=2)
        elif par.plot_type == 'scatter':
            ax.scatter(indiff_set[0],indiff_set[1],color="navy",zorder=2)

    ax.scatter(par.x1,par.x2,color='black',zorder=3)

    for i,u0 in enumerate(u0s):
        ax.text(par.x1[i]*1.03,par.x2[i]*1.03,'u = {:3.2f}'.format(u0))

def draw_45(ax):

    ax.plot([0,10],[0,10],'--',color="black",zorder=1,alpha=0.1)

def update(par,alpha,beta):

    par.alpha = alpha
    par.beta = beta

    draw_figure(par)

def interact(par):

    if par.interact:
        
        widgets.interact(update,
                        par=widgets.fixed(par), 
                        alpha=widgets.FloatSlider(min=par.alpha_min, max=par.alpha_max, step=par.alpha_step, value=par.alpha),
                        beta=widgets.FloatSlider(min=par.beta_min, max=par.beta_max, step=par.beta_step, value=par.beta))


############
# 3. basic #
############

def settings():
    
    # a. setup
    par = dict()
    par = SimpleNamespace(**par)

    #. b. points
    par.x1 = [2,3,4]
    par.x2 = [2,3,4]

    # c. curves
    par.upper = 10
    par.N = 50

    # d. figure
    par.plot_type = 'line'

    # e. update
    par.interact = True
    par.alpha_step = 0.05
    par.beta_step = 0.05

    return par

def cobb_douglas():

    par = settings()

    par.uname = 'cobb_douglas'

    par.alpha = 0.50
    par.beta = 0.50

    par.alpha_min = 0.05
    par.alpha_max = 0.99

    par.beta_min = 0.05
    par.beta_max = 0.99

    interact(par)

def ces():

    par = settings()

    par.uname = 'ces'

    par.alpha = 0.50
    par.beta = 0.85

    par.alpha_min = 0.05
    par.alpha_max = 0.99

    par.beta_min = -0.95
    par.beta_max = 10.01

    interact(par)

def perfect_substitutes():

    par = settings()

    par.uname = 'perfect_substitutes'

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

def arbitrary(uname,alpha,beta,alpha_bounds,beta_bounds,plot_type='scatter'):

    par = settings()
    par.plot_type = plot_type

    par.uname = uname

    par.alpha = alpha
    par.beta = beta

    par.alpha_min = alpha_bounds[0]
    par.alpha_max = alpha_bounds[1]

    par.beta_min = beta_bounds[0]
    par.beta_max = beta_bounds[1]

    interact(par)