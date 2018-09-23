from math import pi
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

#############
# 1. budget #
#############

def budgetset(ax,p1,p2,I,**kwargs):

    x = [0,0,I/p1]
    y = [0,I/p2,0]

    ax.fill(x,y,color="firebrick",lw=2,alpha=0.5,**kwargs)

def budgetline(ax,p1,p2,I,**kwargs):
    
    if 'color' not in kwargs:
        kwargs['color'] = 'black'
    if 'lw' not in kwargs:
        kwargs['lw'] = 2        

    x = [0,I/p1]
    y = [I/p2,0]

    ax.plot(x,y,**kwargs)

def budgetline_slope(ax,p1,p2,I,scalex=1.03,scaley=1.03,**kwargs):

    x = (I/p1)/2*scalex
    y = (I/p2)/2*scaley

    ax.text(x,y,f'slope = -{p1/p2:3.2f}',**kwargs)

def endowment(ax,e1,e2,scalex=1.03,scaley=1.03,**kwargs):

    ax.scatter(e1,e2,color='black',**kwargs)
    ax.text(e1*scalex,e2*scaley,'endowment')

def budgetline_with_kink(ax,p1_A,p1_B,p2,I,xbar,**kwargs):
    
    x = [0,xbar,xbar+(I-p1_A*xbar)/p1_B]
    y = [I/p2,(I-p1_A*xbar)/p2,0]

    ax.plot(x,y,color='black',lw=2,**kwargs)    

def budgetset_with_kink(ax,p1_A,p1_B,p2,I,xbar,**kwargs):

    x = [0,0,xbar,xbar+(I-p1_A*xbar)/p1_B]
    y = [0,I/p2,(I-p1_A*xbar)/p2,0]

    ax.fill(x,y,color="firebrick",lw=2,alpha=0.5,**kwargs)

##############
# 2. utility #
##############

def utility_functions(par,uname,v=None):

    par.uname = uname
    par.g_inv = None
    if par.uname == 'cobb_douglas':
        par.u = lambda x1,x2,alpha,beta: x1**alpha*x2**beta 
        par.g = lambda u,x1,alpha,beta: (u/x1**alpha)**(1/beta)
    elif par.uname == 'ces':
        def u(x1,x2,alpha,beta):
            if beta == 0:
                return  x1**alpha*x2**(1-alpha)
            else:
                return (alpha*x1**(-beta)+(1-alpha)*x2**(-beta))**(-1.0/beta)
        par.u = u
        def g(u,x1,alpha,beta):
            if beta == 0:
                return (u/x1**alpha)**(1/(1-alpha))
            else:
                return ((u**(-beta)-alpha*x1**(-beta))/(1-alpha))**(-1.0/beta)      
        par.g = g 
        def g_inv(u,x2,alpha,beta):
            if beta == 0:
                return (u/x2**alpha)**(1/(1-alpha))
            else:
                return ((u**(-beta)-(1-alpha)*x2**(-beta))/alpha)**(-1.0/beta)
        par.g_inv = g_inv       
    elif par.uname == 'perfect_substitutes':
        par.u = lambda x1,x2,alpha,beta: par.alpha*x1+par.beta*x2
        par.g = lambda u,x1,alpha,beta: (u-par.alpha*x1)/par.beta    
    elif par.uname == 'leontief':
        par.u = lambda x1,x2,alpha,beta: np.fmin(par.alpha*x1,par.beta*x2)
        par.g = ''
    elif par.uname ==  'quasi_linear':
        assert callable(v), 'v is not callable'
        par.u = lambda x1,x2,alpha,beta: alpha*v(x1)+beta*x2
        par.g = lambda u,x1,alpha,beta: (u-alpha*v(x1))/beta
    elif par.uname == 'concave':
        par.u = lambda x1,x2,alpha,beta: alpha*x1**2+beta*x2**2
        par.g = lambda u,x1,alpha,beta: np.sqrt((u-alpha*x1**2)/beta)
        par.g_inv = lambda u,x2,alpha,beta: np.sqrt((u-beta*x2**2)/alpha)
    elif par.uname == 'quasi_quasi_linear':
        par.u = lambda x1,x2,alpha,beta: x1**alpha*(x2+beta)
        par.g = lambda u,x1,alpha,beta: (u/x1**alpha-beta)
    elif par.uname == 'saturated':
        par.u = lambda x1,x2,alpha,beta: -(x1-alpha)**2-(x2-beta)**2
        par.g = 'saturated'
    else:
        raise ValueError('unknown utility function')

def find_indifference_curve(u,par):

    if callable(par.g_inv): 
    
        # given x1
        x1_x1 = np.linspace(0,par.x1_max,par.N)
        with np.errstate(divide='ignore',invalid='ignore'):
            x2_x1 = par.g(u,x1_x1,par.alpha,par.beta)

        # given x2
        x2_x2 = np.linspace(0,par.x2_max,par.N)
        with np.errstate(divide='ignore',invalid='ignore'):
            x1_x2 = par.g_inv(u,x2_x2,par.alpha,par.beta)

        x1 = np.hstack([x1_x1,x1_x2])
        x2 = np.hstack([x2_x1,x2_x2])
        I = np.argsort(x1)
        x1 = x1[I]
        x2 = x2[I]

    elif callable(par.g):

        x1 = np.linspace(0,par.x1_max,par.N)
        with np.errstate(divide='ignore',invalid='ignore'):
            x2 = par.g(u,x1,par.alpha,par.beta)

    else: # arbitrary utility function

        x1 = []
        x2 = []

        x1_vec = np.linspace(par.eps,par.x1_max,par.N)
        for x1_now in x1_vec:

            def target_for_x2(x2):
                return par.u(x1_now,np.fmax(x2,par.eps),par.alpha,par.beta)-u

            x_A,_infodict_A,ier_A,_mesg_A = optimize.fsolve(target_for_x2, 0, full_output=True)
            x_B,_infodict_B,ier_B,_mesg_B = optimize.fsolve(target_for_x2, par.x2_max, full_output=True)        

            if ier_A == 1:
                x1.append(x1_now)
                x2.append(x_A[0])
            else:
                x1.append(np.nan)
                x2.append(np.nan)

            if ier_B == 1 and np.abs(x_A[0]-x_B[0]) > 0.01:
                x1.append(x1_now)
                x2.append(x_B[0])
            else:
                x1.append(np.nan)
                x2.append(np.nan)

        # e. fixed x2
        x2_vec = np.linspace(par.eps,par.x2_max,par.N)
        for x2_now in x2_vec:

            def target_for_x1(x1):
                return par.u(np.fmax(x1,par.eps),x2_now,par.alpha,par.beta)-u

            x_A,_infodict_A,ier_A,_mesg_A = optimize.fsolve(target_for_x1, 0, full_output=True)
            x_B,_infodict_B,ier_B,_mesg_B = optimize.fsolve(target_for_x1, par.x1_max, full_output=True)
            
            if ier_A == 1:
                x1.append(x_A[0])
                x2.append(x2_now)
            else:
                x1.append(np.nan)
                x2.append(np.nan)

            if ier_B == 1 and np.abs(x_A[0]-x_B[0]) > 0.01:
                x1.append(x_B[0])
                x2.append(x2_now)
            else:
                x1.append(np.nan)
                x2.append(np.nan)

        # f. sort
        x1 = np.array(x1)
        x2 = np.array(x2)
        I = np.argsort(x1)
        x1 = x1[I]
        x2 = x2[I]

    # clean
    with np.errstate(divide='ignore',invalid='ignore'):
        u_curve = par.u(x1,x2,par.alpha,par.beta)
        I = (u-u_curve)**2 > 1e-4 
    x2[I] = np.nan
        
    return x1,x2

def indifference_curve(ax,u,par,inv=False,**kwargs):

    if 'color' not in kwargs:
        kwargs['color'] = 'navy'
    if 'lw' not in kwargs:
        kwargs['lw'] = 2
        
    if par.uname == 'leontief':

        corner_x1 = u/par.alpha
        x1 = [corner_x1,corner_x1,par.x1_max]

        corner_x2 = u/par.beta
        x2 = [par.x2_max,corner_x2,corner_x2]

        if inv: 
            ax.plot(1-np.array(x1),1-np.array(x2),**kwargs) 
        else:
            ax.plot(x1,x2,**kwargs) 
                
    elif par.uname == 'saturated':

        radius = np.sqrt(-u)
        if inv:
            circle = plt.Circle((1-par.alpha,1-par.beta),radius,fill=False,**kwargs)
        else:
            circle = plt.Circle((par.alpha,par.beta),radius,fill=False,**kwargs)
        ax.add_artist(circle)

    else:
        
        x1,x2 = find_indifference_curve(u,par)

        if inv:
            ax.plot(1-x1,1-x2,**kwargs)                
        else:
            ax.plot(x1,x2,**kwargs)                
    
def monotonicity(ax,par,x1,x2,text='north-east'):

    polygon = plt.Polygon([[x1, x2],[par.x1_max, x2],[par.x1_max, par.x2_max],[x1, par.x2_max],[x1, x2]],
        color='black',alpha=0.10)
    ax.add_patch(polygon)
    ax.text(x1*1.05,0.95*par.x2_max,text)
        
def convex_combination(ax,par,x1,x2,u,text='mix'):

    diff = 5

    if par.uname == 'leontief':

        x1 = u/par.alpha 
        x2 = u/par.beta

        x1_low = x1
        x2_low = x2 + diff
    
        x1_high = x1 + diff
        x2_high = x2
        
    elif par.uname == 'saturated':
        
        diff_circle = 35

        radius = np.sqrt(-u)
        
        low = 45 + diff_circle
        x1_low = par.alpha - radius*np.cos(low/180*pi)
        x2_low = par.beta - radius*np.sin(low/180*pi)

        high = 45 - diff_circle
        x1_high = par.alpha - radius*np.cos(high/180*pi)
        x2_high = par.beta - radius*np.sin(high/180*pi)
                    
    else:
        
        x1, x2 = find_indifference_curve(u,par)

        I = (np.isnan(x1) == False) & (np.isnan(x2) == False)
        J = (x1[I] > 0) & (x2[I] > 0) & (x1[I] < par.x1_max) & (x2[I] < par.x2_max) 

        x1 = x1[I][J]
        x2 = x2[I][J]
        N = x1.size        
        i_low = np.int(N/3)
        i_high = np.int(N*2/3)

        x1_low = x1[i_low]
        x2_low = x2[i_low]

        x1_high = x1[i_high]
        x2_high = x2[i_high]

    ax.plot([x1_low,x1_high],[x2_low,x2_high],ls='--',marker='o',markerfacecolor='none',color='black')

    x1 = 0.05*x1_low + 0.95*x1_high
    x2 = 0.05*x2_low + 0.95*x2_high
    ax.text(x1,x2*1.05,text)


###################
# 3. optimization #
###################

def maximization(par):

    if par.monotone:

        # a. target
        def x2_func(x1):
            return (par.I - par.p1*x1)/par.p2
        def target(x1):
            x2 = x2_func(x1)
            return -par.u(x1,x2,par.alpha,par.beta)
            
        # b. solve
        xopt = optimize.fminbound(target, 0, par.I/par.p1)

        # c. save
        x1 = xopt
        x2 = x2_func(x1)

    else:

        # a. target
        def target_2d(x): 
            excess_spending = par.p1*x[0]+par.p2*x[1]-par.I
            return -par.u(x[0],x[1],par.alpha,par.beta) + 1000*np.max([excess_spending,-x[0],-x[1],0])
            
        # b. solve
        x0 = np.array([par.I/par.p1,par.I/par.p2])/2
        res = optimize.minimize(target_2d,x0,method='Nelder-Mead')

        # c. save
        x1 = res.x[0]
        x2 = res.x[1]

    u = par.u(x1,x2,par.alpha,par.beta)
    return x1,x2,u

def costminimization(par,u):

    # a. target
    def target_2d(x):
        x1 = x[0]
        x2 = x[1]
        udiff = (par.u(x1,x2,par.alpha,par.beta)-u)**2
        
        return par.p1*x1+par.p2*x2 + 1000*udiff + 1000*np.max([-x[0],-x[1],0])
        
    # b. solve
    x0 = np.array([par.I/par.p1,par.I/par.p2])/2
    res = optimize.minimize(target_2d,x0,method='Nelder-Mead')

    # c. save
    x1 = res.x[0]
    x2 = res.x[1]
    
    return x1,x2
