import numpy as np

import ipywidgets as widgets
from bokeh.io import push_notebook, show
from bokeh.plotting import figure
from bokeh.models import Range1d, Label

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

################
# 1. exogenous #
################

def exogenous(p1=2,p2=1,I=10):
    
    # a. (0,0)
    x1 = [0]
    x2 = [0]
    
    # b. (0,max)
    x1.append(0)
    x2.append(I/p2)
    
    # c. (max,0)
    x1.append(I/p1)
    x2.append(0)
    
    # d. slope
    slope_xy = [(I/p1)/1.95,(I/p2)/1.95] # placement
    slope = p1/p2 # value
    
    return x1,x2,slope_xy,slope

def update_exogenous(ax,slope_label,p1=2,p2=1,I=10):  
    
    # a. calculate
    x1, x2,slope_xy,slope = exogenous(p1,p2,I)
    
    # b. data
    ax.data_source.data['x'] = x1
    ax.data_source.data['y'] = x2
    
    # c. slope_label
    slope_label.x = slope_xy[0]
    slope_label.y = slope_xy[1]    
    slope_label.text = 'slope = -{:3.2f}'.format(slope)

    push_notebook()

def interact_exogenous(ax,slope_label):

    widgets.interact(update_exogenous,
                    ax=widgets.fixed(ax),
                    slope_label=widgets.fixed(slope_label),                    
                    p1=(0.1,5,0.01), 
                    p2=(0.1,5,0.1),
                    I=(0.1,20,0.1))

def draw_exogenous():

    # a. calculate
    x1,x2,slope_xy,slope = exogenous()

    # b. basic figure
    fig =  figure(plot_height=400, plot_width=400, x_range=(0,10), y_range=(0,10))
    ax = fig.patch(x1,x2, color="firebrick", line_width=3, alpha=0.5)
    fig.xaxis.axis_label = 'x1'
    fig.yaxis.axis_label = 'x2'

    # c. slope
    slope_label = Label(x=slope_xy[0],y=slope_xy[1], text_font_size='10pt', text='slope = -{}'.format(slope))                    
    fig.add_layout(slope_label)

    # d. interatc
    show(fig,notebook_handle=True)
    interact_exogenous(ax,slope_label)

################
# 2. endowment #
################

def endowment(p1=2,p2=3,e1=2.5,e2=4): 
    
    I = p1*e1 + p2*e2
    
    # a. (0,0)
    x1 = [0]
    x2 = [0]
    
    # b. (0,max)
    x1.append(0)
    x2.append(I/p2)
    
    # c. (max,0)
    x1.append(I/p1)
    x2.append(0)
    
    # d. slope
    slope_xy = [(I/p1)/1.95,(I/p2)/1.95] # placement
    slope = p1/p2 # value

    # e. endowment
    e = (e1,e2) 
    
    return x1,x2,slope_xy,slope,e

def update_endowment(ax,slope_label,endowment_point,endowment_label,p1=2,p2=3,e1=2.5,e2=4): 
    
    # a. calculate
    x1,x2,slope_xy,slope,e = endowment(p1,p2,e1,e2)
    
    # b. data
    ax.data_source.data['x'] = x1
    ax.data_source.data['y'] = x2
    
    # c. slope_label
    slope_label.x = slope_xy[0]
    slope_label.y = slope_xy[1]    
    slope_label.text = 'slope = -{:3.2f}'.format(slope)

    # d. endowment
    endowment_point.data_source.data['x'] = [e[0],e[0]]
    endowment_point.data_source.data['y'] = [e[1],e[1]]
    endowment_label.x = e[0]
    endowment_label.y = e[1]

    push_notebook()

def interact_endowment(ax,slope_label,endowment_point,endowment_label):

    widgets.interact(update_endowment,
                    ax=widgets.fixed(ax),
                    slope_label=widgets.fixed(slope_label),   
                    endowment_point=widgets.fixed(endowment_point),    
                    endowment_label=widgets.fixed(endowment_label),                                                            
                    p1=(0.1,5,0.01), 
                    p2=(0.1,5,0.1),
                    e1=(0.1,5,0.01), 
                    e2=(0.1,5,0.1))

def draw_endowment():

    # a. calculate
    x1,x2,slope_xy,slope,e = endowment()

    # b. basic figure
    fig =  figure(plot_height=400, plot_width=400, x_range=(0,10), y_range=(0,10))
    ax = fig.patch(x1,x2, color="firebrick", line_width=3, alpha=0.5)
    fig.xaxis.axis_label = 'x1'
    fig.yaxis.axis_label = 'x2'

    # c. slope
    slope_label = Label(x=slope_xy[0],y=slope_xy[1], text_font_size='10pt', text='slope = -{}'.format(slope))                
    fig.add_layout(slope_label)
           
    # d. endowment
    endowment_point = fig.circle([e[0],e[0]],[e[1],e[1]], color="black")
    endowment_label = Label(x=e[0],y=e[1], text_font_size='10pt', text='endowment')         
    fig.add_layout(endowment_label)

    # e. interatc
    show(fig,notebook_handle=True)
    interact_endowment(ax,slope_label,endowment_point,endowment_label)

#############
# 3. kinked #
#############

def kink(p1_A=1,p1_B=4,p2=1,x1_kink=5,I=10): 
    
    x1 = [0]
    x2 = [0]
    
    # a. x1 = 0
    x1.append(0)
    x2.append(I/p2)
    
    # b. kink
    I_kink = I - x1_kink*p1_A
    if I_kink <= 0:
        x1.append(I/p1_A)
        x2.append(0)
        x1.append(I/p1_A) # hack -> ensures number of points are constant
        x2.append(0) # hack -> ensures number of points are constant
    else:
        x1.append(x1_kink)
        x2.append(I_kink/p2)
        x1.append(x1_kink+I_kink/p1_B)
        x2.append(0)        
    
    return x1,x2

def update_kink(ax,p1_A=1,p1_B=2,p2=1,x1_kink=5,I=10):      
    
    # a. calculate
    x1, x2 = kink(p1_A,p1_B,p2,x1_kink,I)
    
    # b. redo basic
    ax.data_source.data['x'] = x1
    ax.data_source.data['y'] = x2
    
    push_notebook()

def interact_kink(ax):

    widgets.interact(update_kink,
                    ax=widgets.fixed(ax),                
                    p1_A=(0.1,5,0.01), 
                    p1_B=(0.1,5,0.01),                     
                    p2=(0.1,5,0.1),
                    I=(0.1,20,0.1))

def draw_kink():

    # a. calculate
    x1,x2 = kink()

    # b. basic figure
    fig =  figure(plot_height=400, plot_width=400, x_range=(0,10), y_range=(0,10))
    ax = fig.patch(x1,x2, color="firebrick", line_width=3, alpha=0.5)
    fig.xaxis.axis_label = 'x1'
    fig.yaxis.axis_label = 'x2'

    # c. interatc
    show(fig,notebook_handle=True)
    interact_kink(ax)

#########
# 3. 3D #
#########

def draw_3D(price_vectors,I):

    import math
    nrows = math.ceil(len(price_vectors)/3)

    fig = plt.figure(figsize=(5*3,5*nrows))
    for i,price_vector in enumerate(price_vectors):

        ax = fig.add_subplot(3,3,i+1,projection='3d')
        # a. find edges
        xmax = I/price_vector[0]
        ymax = I/price_vector[1]
        zmax = I/price_vector[2]

        edges = [[[0,0,0],[0,0,zmax],[0,ymax,0]]]
        edges.append([[0,0,0],[0,0,zmax],[xmax,0,0]])
        edges.append([[0,0,0],[0,ymax,0],[xmax,0,0]])
        edges.append([[0,0,zmax],[0,ymax,0],[xmax,0,0]])
        
        # b. figure
        faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
        faces.set_facecolor((0,0,1,0.1))
        ax.add_collection3d(faces)

        # c. title
        ax.set_title('$p = ({},{},{})$'.format(price_vector[0],price_vector[1],price_vector[2]))

        # d. details
        ax.set_aspect('equal')
        ax.set_xlim(0,10)
        ax.set_ylim(0,10)
        ax.set_zlim(0,10)        
        ax.invert_yaxis()    
        ax.set_xlabel('$x_1$')        
        ax.set_ylabel('$x_2$')        
        ax.set_zlabel('$x_3$')                        