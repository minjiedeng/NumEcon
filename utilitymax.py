from types import SimpleNamespace
import numpy as np
from scipy import optimize

import ipywidgets as widgets
from bokeh.io import push_notebook, show
from bokeh.plotting import figure
from bokeh.models import Range1d, Label

import utility
import budgetset

####################
## 1. maximization #
####################

def maximization(par):

    u,_eps = utility.func(par)
    def x2_func(x1):
        return (par.I - par.p1*x1)/par.p2
    def target(x1):
        x2 = x2_func(x1)
        return -u(x1,x2)
        
    xopt = optimize.fminbound(target, 0, par.I/par.p1)

    par.x1 = [0.95*xopt,xopt,1.05*xopt]
    x2 = x2_func(xopt)
    par.x2 = [0.8*x2,x2,1.2*x2]

    return par

###########
# 2. draw #
###########

def update(par):      
    
    # a. calculate
    par = maximization(par)
    indiff_sets, u0s = utility.indifference_set(par)
    budgetset_x1s,budgetset_x2s, _slope_xy,_slope = budgetset.exogenous(par.p1,par.p2,par.I)
    
    # b. indifference curves
    for i in range(len(indiff_sets)):
        par.ax[i].data_source.data['x'] = indiff_sets[i][0]
        par.ax[i].data_source.data['y'] = indiff_sets[i][1]
        par.labels[i].x = par.x1[i]
        par.labels[i].y = par.x2[i]
        par.labels[i].text = 'u = {:3.2f}'.format(u0s[i])
    
    # c. points
    par.ax_circle.data_source.data['x'] = par.x1 
    par.ax_circle.data_source.data['y'] = par.x2

    # d. budgetset
    par.ax_patch.data_source.data['x'] = budgetset_x1s
    par.ax_patch.data_source.data['y'] = budgetset_x2s

    push_notebook()

def draw(par):

    # a. calculate
    par = maximization(par)
    indiff_sets, u0s = utility.indifference_set(par)
    budgetset_x1s,budgetset_x2s, _slope_xy,_slope = budgetset.exogenous(par.p1,par.p2,par.I)

    # b. basic figure
    fig =  figure(plot_height=400, plot_width=400, x_range=(0,10), y_range=(0,10))
    fig.xaxis.axis_label = 'x1'
    fig.yaxis.axis_label = 'x2'

    # c. indifference curves
    par.ax = []
    for i in range(len(indiff_sets)):
        if par.plot_type == 'line':
            par.ax.append(fig.line(indiff_sets[i][0],indiff_sets[i][1], color="#2222aa", line_width=3, alpha=0.5))
        elif par.plot_type == 'scatter':
            par.ax.append(fig.circle(indiff_sets[i][0],indiff_sets[i][1], color="#2222aa", size=5, alpha=0.5))

    # d. points
    par.ax_circle = fig.circle(par.x1,par.x2,color='black')

    par.labels = []
    for i in range(len(indiff_sets)):
        par.labels.append(Label(x=par.x1[i],y=par.x2[i],render_mode='css',text_font_size='10pt',text='u = {:3.2f}'.format(u0s[i])))
        fig.add_layout(par.labels[i])

    # e. budgetset
    par.ax_patch = fig.patch(budgetset_x1s,budgetset_x2s, color="#2222aa", line_width=3, alpha=0.5)

    show(fig,notebook_handle=True)

    # f. interact
    if par.interact_basic:
        interact_basic(par)

############
# 3. basic #
############

def basic_settings():
    
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
    par.I = 10
    par.p1_min = 0.05   
    par.p1_max = 4.00    
    par.p2_min = 0.05
    par.p2_max = 4.00    
    par.I_min = 0.5
    par.I_max = 20
    
    return par

def update_basic(par,p1,p2,I,alpha,beta):

    par.p1 = p1
    par.p2 = p2
    par.I = I
    par.alpha = alpha
    par.beta = beta

    update(par)

def interact_basic(par):

    widgets.interact(update_basic,
                    par=widgets.fixed(par), 
                    p1=widgets.FloatSlider(min=par.p1_min, max=par.p1_max, step=par.p1_step, value=par.p1),
                    p2=widgets.FloatSlider(min=par.p2_min, max=par.p2_max, step=par.p2_step, value=par.p2),
                    I=widgets.FloatSlider(min=par.I_min, max=par.I_max, step=par.I_step, value=par.I),
                    alpha=widgets.FloatSlider(min=par.alpha_min, max=par.alpha_max, step=par.alpha_step, value=par.alpha),
                    beta=widgets.FloatSlider(min=par.beta_min, max=par.beta_max, step=par.beta_step, value=par.beta))

def arbitrary(uname,alpha,beta,alpha_bounds,beta_bounds,p1,p2,I,p1_bounds,p2_bounds,I_bounds,plot_type='scatter'):

    par = basic_settings()
    par.plot_type = plot_type

    # a. budget
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

    par.alpha = alpha
    par.beta = beta

    par.alpha_min = alpha_bounds[0]
    par.alpha_max = alpha_bounds[1]

    par.beta_min = beta_bounds[0]
    par.beta_max = beta_bounds[1]

    draw(par)

def cobb_douglas():

    par = basic_settings()

    par.uname = 'cobb_douglas'

    par.alpha = 0.50
    par.beta = 0.50

    par.alpha_min = 0.05
    par.alpha_max = 0.95

    par.beta_min = 0.05
    par.beta_max = 0.95

    draw(par)

def ces():

    par = basic_settings()

    par.uname = 'ces'

    par.alpha = 0.50
    par.beta = 0.85

    par.alpha_min = 0.05
    par.alpha_max = 0.95

    par.beta_min = -0.95
    par.beta_max = 10

    draw(par)

def perfect_substitutes():

    par = basic_settings()

    par.uname = 'perfect_substitutes'

    par.alpha = 1.00
    par.beta = 1.00

    par.alpha_min = 0.05
    par.alpha_max = 3.00

    par.beta_min = 0.05
    par.beta_max = 3.00

    draw(par)

def perfect_complements():

    par = basic_settings()

    par.uname = 'perfect_complements'

    par.alpha = 1.00
    par.beta = 1.00

    par.alpha_min = 0.05
    par.alpha_max = 3.00

    par.beta_min = 0.05
    par.beta_max = 3.00

    draw(par)

def quasi_linear_case_1():

    par = basic_settings()

    par.uname = 'quasi_linear_case_1'

    par.alpha = 1.00
    par.beta = 1.00

    par.alpha_min = 0.05
    par.alpha_max = 3.00

    par.beta_min = 0.05
    par.beta_max = 3.00

    draw(par)

def quasi_linear_case_2():

    par = basic_settings()

    par.uname = 'quasi_linear_case_2'

    par.alpha = 1.00
    par.beta = 1.00

    par.alpha_min = 0.05
    par.alpha_max = 3.00

    par.beta_min = 0.05
    par.beta_max = 3.00

    draw(par)
