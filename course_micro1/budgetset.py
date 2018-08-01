from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import ipywidgets as widgets

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

################
# 1. exogenous #
################

def calc_exogenous(p1,p2,I):

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

def draw_figure_exogenous(p1=2,p2=1,I=10):
        
    # a. calculations
    x1,x2,slope_xy,slope = calc_exogenous(p1,p2,I)

    # b. figure
    fig = plt.figure(frameon=False, figsize=(6,6), dpi=100)
    ax = fig.add_subplot(1,1,1)

    # c. basic layout
    ax.grid(True)
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    # d. draw axes
    draw(ax,x1,x2)
    draw_slope(ax,slope_xy,slope)
    
    plt.show()

def draw(ax,x1,x2):
    ax.fill(x1,x2, color="firebrick", linewidth=2, alpha=0.5, zorder=1)

def draw_slope(ax,slope_xy,slope):
    ax.text(slope_xy[0],slope_xy[1],'slope = -{:3.2f}'.format(slope))

def exogenous():

    widgets.interact(draw_figure_exogenous,        
                     p1=(0.1,5,0.01), 
                     p2=(0.1,5,0.1),
                     I=(0.1,20,0.1))

################
# 2. endowment #
################

def calc_endowment(p1,p2,e1,e2):
    
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

def draw_figure_endowment(p1=2,p2=3,e1=2.5,e2=4): 
        
    # a. calculations
    x1,x2,slope_xy,slope,e = calc_endowment(p1,p2,e1,e2)

    # b. figure
    fig = plt.figure(frameon=False, figsize=(6,6), dpi=100)
    ax = fig.add_subplot(1,1,1)

    # c. basic layout
    ax.grid(True)
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    # d. draw axes
    draw(ax,x1,x2)
    draw_slope(ax,slope_xy,slope)
    draw_endowment(ax,e)

    plt.show()
    
def draw_endowment(ax,e):
    ax.scatter(e[0],e[1],color='black',zorder=2)
    ax.text(e[0]*1.025,e[1]*1.025,'endowment')

def endowment():

    widgets.interact(draw_figure_endowment,
                    calc=widgets.fixed(calc_endowment),               
                    p1=(0.1,5,0.01), 
                    p2=(0.1,5,0.1),
                    e1=(0.1,5,0.01), 
                    e2=(0.1,5,0.1))

###########
# 3. kink #
###########

def calc_kink(p1_A,p1_B,p2,x1_kink,I): 
    
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

def draw_figure_kink(p1_A=1,p1_B=2,p2=1,x1_kink=5,I=10):      

    # a. calculations
    x1,x2= calc_kink(p1_A,p1_B,p2,x1_kink,I)

    # b. figure
    fig = plt.figure(frameon=False, figsize=(6,6), dpi=100)
    ax = fig.add_subplot(1,1,1)

    # c. basic layout
    ax.grid(True)
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    # d. draw axes
    draw(ax,x1,x2)
    
    plt.show()
    
def kink():

    widgets.interact(draw_figure_kink,
                    p1_A=(0.1,5,0.01), 
                    p1_B=(0.1,5,0.01),                     
                    p2=(0.1,5,0.1),
                    I=(0.1,20,0.1))

#########
# 4. 3D #
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