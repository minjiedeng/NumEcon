from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import ipywidgets as widgets

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import numecon.course_micro1.consumer as consumer

################
# 1. exogenous #
################

def _exogenous(p1,p2,I):

    # a. figure
    fig = plt.figure(frameon=False, figsize=(6,6),dpi=100)
    ax = fig.add_subplot(1,1,1)

    # b. line
    consumer.budgetline(ax,p1,p2,I)
    consumer.budgetset(ax,p1,p2,I)
    consumer.budgetline_slope(ax,p1,p2,I)

    # c. basic layout
    ax.grid(ls='--',lw=1)
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

def exogenous():

    widgets.interact(_exogenous,        
        p1=widgets.FloatSlider(description='$p_1$',min=0.1, max=5, step=0.05, value=2), 
        p2=widgets.FloatSlider(description='$p_2$',min=0.1, max=5, step=0.05, value=1), 
        I=widgets.FloatSlider(description='$I$',min=0.1, max=20, step=0.10, value=5))

################
# 2. endowment #
################

def _endowment(p1,p2,e1,e2):

    I = p1*e1+p2*e2

    # a. figure
    fig = plt.figure(frameon=False, figsize=(6,6),dpi=100)
    ax = fig.add_subplot(1,1,1)

    # b. line
    consumer.budgetline(ax,p1,p2,I)
    consumer.budgetset(ax,p1,p2,I)
    consumer.budgetline_slope(ax,p1,p2,I)
    consumer.endowment(ax,e1,e2)

    # c. basic layout
    ax.grid(ls='--',lw=1)
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

def endowment():

    widgets.interact(_endowment,             
        p1=widgets.FloatSlider(description='$p_1$',min=0.1, max=5, step=0.05, value=2), 
        p2=widgets.FloatSlider(description='$p_2$',min=0.1, max=5, step=0.05, value=1),
        e1=widgets.FloatSlider(description='$e_1$',min=0.0, max=5, step=0.05, value=3), 
        e2=widgets.FloatSlider(description='$e_2$',min=0.0, max=5, step=0.05, value=4))

###########
# 3. kink #
###########

def _kink(p1_A,p1_B,p2,I,xbar):

    # a. figure
    fig = plt.figure(figsize=(6,6),dpi=100)
    ax = fig.add_subplot(1,1,1)

    # b. line
    consumer.budgetline_with_kink(ax,p1_A,p1_B,p2,I,xbar)
    consumer.budgetset_with_kink(ax,p1_A,p1_B,p2,I,xbar)

    # c. basic layout
    ax.grid(ls='--',lw=1)
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    
def kink():

    widgets.interact(_kink,
                    p1_A=widgets.FloatSlider(description='$p_1^A$',min=0.1, max=5, step=0.05, value=1), 
                    p1_B=widgets.FloatSlider(description='$p_1^B$',min=0.1, max=5, step=0.05, value=2),                   
                    p2=widgets.FloatSlider(description='$p_2$',min=0.1, max=5, step=0.05, value=1), 
                    I=widgets.FloatSlider(description='$I$',min=0.10, max=20, step=0.10, value=10),
                    xbar=widgets.FloatSlider(description='$\\overline{x}_1$',min=0.1, max=10, step=0.10, value=5))                     

#########
# 4. 3D #
#########

def _D3(p1,p2,p3,I,elev,angle):

    # a. figure
    fig = plt.figure(figsize=(6,6),dpi=100)
    ax = fig.add_subplot(1,1,1,projection='3d')
        
    # b. edges
    xmax = I/p1
    ymax = I/p2
    zmax = I/p3

    edges = [[[0,0,0],[0,0,zmax],[0,ymax,0]]]
    edges.append([[0,0,0],[0,0,zmax],[xmax,0,0]])
    edges.append([[0,0,0],[0,ymax,0],[xmax,0,0]])
    edges.append([[0,0,zmax],[0,ymax,0],[xmax,0,0]])
        
    # c. faces
    faces = Poly3DCollection(edges, linewidths=1, edgecolors='black')
    faces.set_alpha(0.35)
    faces.set_facecolor('firebrick')
    ax.add_collection3d(faces)

    # d. details
    ax.set_aspect('equal')
    ax.set_xlim(0,15)
    ax.set_ylim(0,15)
    ax.set_zlim(0,15)        
    ax.invert_yaxis()    
    ax.set_xlabel('$x_1$')        
    ax.set_ylabel('$x_2$')        
    ax.set_zlabel('$x_3$')

    # e. rotation
    ax.view_init(elev,angle)

def D3():

    widgets.interact(_D3,
                    p1=widgets.FloatSlider(description='$p_1$',min=0.1, max=5, step=0.05, value=1), 
                    p2=widgets.FloatSlider(description='$p_2$',min=0.1, max=5, step=0.05, value=1), 
                    p3=widgets.FloatSlider(description='$p_3$',min=0.1, max=5, step=0.05, value=1), 
                    I=widgets.FloatSlider(description='$I$',min=0.10, max=20, step=0.10, value=10),
                    elev=widgets.FloatSlider(description='elevation',min=0, max=180, step=5, value=30), 
                    angle=widgets.FloatSlider(description='angle',min=0, max=360, step=5, value=300))
