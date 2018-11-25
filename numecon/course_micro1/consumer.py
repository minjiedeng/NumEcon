import numpy as np
import matplotlib.pyplot as plt
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

class ConsumerClass():

    def __init__(self,**kwargs):

        # a. baseline setup
        
        # utility
        self.utility = 'cobb_douglas'
        self.alpha = 1
        self.beta = 1

        # prices and income
        self.p1 = 1
        self.p2 = 1
        self.I = 5

        # figures
        self.x1label = '$x_1$'
        self.x2label = '$x_2$'
        self.N = 100

        # b. update 
        for key,val in kwargs.items():
            setattr(self,key,val)

        # c. calculations
        self.calculations()

    def calculations(self):

        # a. utility function      
        if self.utility == 'cobb_douglas':
            self.u = lambda x1,x2,alpha,beta: x1**alpha*x2**beta
            self.indiff_func_x1 = lambda u0,x1,alpha,beta: (u0/x1**alpha)**(1/beta)
            self.indiff_func_x2 = lambda u0,x2,alpha,beta: (u0/x2**beta)**(1/alpha)
        elif self.utility == 'quasi_log':
            self.u = lambda x1,x2,alpha,beta: alpha*np.log(x1)+x2*beta            
            self.indiff_func_x1 = lambda u0,x1,alpha,beta: (u0-alpha*np.log(x1))/beta
            self.indiff_func_x2 = lambda u0,x2,alpha,beta: np.exp((u0-x2*beta)/alpha)
        elif self.utility == 'concave':
            self.u = lambda x1,x2,alpha,beta: alpha*x1**2+beta*x2**2            
            self.indiff_func_x1 = lambda u0,x1,alpha,beta: np.sqrt((u0-alpha*x1**2)/beta)
            self.indiff_func_x2 = lambda u0,x2,alpha,beta: np.sqrt((u0-beta*x2**2)/alpha)      
        else:
            raise ValueError('unknown utility function')

        # b. figures
        self.x1max = 1.2*self.I/self.p1
        self.x2max = 1.2*self.I/self.p2

    def figure(self):

        fig = plt.figure(frameon=False,figsize=(6,6),dpi=100)
        ax = fig.add_subplot(1,1,1)
        
        ax.set_xlim([0,self.x1max])
        ax.set_ylim([0,self.x2max])
        ax.set_xlabel(self.x1label)
        ax.set_ylabel(self.x2label)

        fig.tight_layout()

        return fig,ax

    def legend(self,ax,**kwargs):

        kwargs.setdefault('loc','upper right')
        kwargs.setdefault('frameon',True)

        legend = ax.legend(**kwargs)
        frame = legend.get_frame()
        frame.set_facecolor('white')        

    ##########
    # choice #
    ##########

    def maximize_utility(self,**kwargs):

        # a. update
        for key,val in kwargs.items():
            setattr(self,key,val)

        # b. solve
        if self.utility == 'cobb_douglas':
            self.x1_ast = self.alpha/(self.alpha+self.beta)*self.I/self.p1
            self.x2_ast = self.alpha/(self.alpha+self.beta)*self.I/self.p2
        elif self.utility == 'quasi_log':
            self.x1_ast = (self.alpha*self.p2)/(self.beta*self.p1)
            self.x2_ast = self.I-self.p1*self.x1_ast
            if self.x2_ast < 0:
                self.x1_ast = self.I/self.p1
                self.x2_ast = 0
        elif self.utility == 'concave':
            self.x1_ast = self.I/self.p1
            self.x2_ast = self.I/self.p2
            if self.x1_ast**self.alpha < self.x2_ast**self.beta:
                self.x1_ast = 0
            else:
                self.x2_ast = 0                                
        else:
            raise ValueError('unknown utility function')

        # c. utility
        self.u_ast = self.u(self.x1_ast,self.x2_ast,self.alpha,self.beta)

        # d. return
        return np.array([self.x1_ast,self.x2_ast,self.u_ast])

    def plot_max(self,ax,**kwargs):

        kwargs.setdefault('ls','')
        kwargs.setdefault('marker','*')
        kwargs.setdefault('markersize',7)
        kwargs.setdefault('color','black')
        kwargs.setdefault('label',f'$u({self.x1_ast:.2f},{self.x2_ast:.2f}) = {self.u_ast:.2f}$')

        ax.plot(self.x1_ast,self.x2_ast,**kwargs)

    def minimize_cost(self,u,**kwargs):

        # a. update
        for key,val in kwargs.items():
            setattr(self,key,val)

        # b. solve
        if self.utility == 'cobb_douglas':
            self.h1_ast = (u*(self.p1/self.p2)**(-self.beta))**(1/(self.alpha+self.beta))
            self.h2_ast = (u*(self.p2/self.p1)**(-self.alpha))**(1/(self.alpha+self.beta))
        elif self.utility == 'quasi_log':
            self.h1_ast = (self.alpha*self.p2)/(self.beta*self.p1)
            self.h2_ast = (u-np.log(self.h1_ast))/self.beta
        else:
            raise ValueError('unknown utility function')

        # c. cost
        self.h_cost = self.p1*self.h1_ast+self.p2*self.h2_ast

        # d. return
        return np.array([self.h1_ast,self.h2_ast,self.h_cost])

    #############
    # budgetset #
    #############

    def plot_budgetline(self,ax,**kwargs):
        
        kwargs.setdefault('lw',2)

        x = [0,self.I/self.p1]
        y = [self.I/self.p2,0]

        ax.plot(x,y,**kwargs)
 
    #######################
    # indifference curves #
    #######################

    def find_indifference_curve(self,u0=None):

        if u0 == None:
            u0 = self.u_ast

        # a. fix x1
        x1_x1 = np.linspace(0,self.x1max,self.N)
        with np.errstate(divide='ignore',invalid='ignore'):
            x2_x1 = self.indiff_func_x1(u0,x1_x1,self.alpha,self.beta)

        # b. fix x2
        x2_x2 = np.linspace(0,self.x2max,self.N)
        with np.errstate(divide='ignore',invalid='ignore'):
            x1_x2 = self.indiff_func_x2(u0,x2_x2,self.alpha,self.beta)

        # c. combine
        x1 = np.hstack([x1_x1,x1_x2])
        x2 = np.hstack([x2_x1,x2_x2])
        
        # d. sort
        I = np.argsort(x1)
        x1 = x1[I]
        x2 = x2[I]

        return x1,x2

    def plot_indifference_curves(self,ax,u0s=[],do_label=True,**kwargs):

        # a. find utility levels
        if len(u0s ) == 0:
            self.maximize_utility()
            u0s = [self.u_ast,
                   self.u(0.8*self.x1_ast,0.8*self.x2_ast,self.alpha,self.beta),
                   self.u(1.2*self.x1_ast,1.2*self.x2_ast,self.alpha,self.beta)]

        # b. construct and plot indifference curves
        for u0 in u0s:
            
            x1,x2 = self.find_indifference_curve(u0)

            if do_label:
                ax.plot(x1,x2,label=f'$u = {u0:.2f}$',**kwargs)
            else:
                ax.plot(x1,x2,**kwargs)

    def plot_monotonicity_check(self,ax,x1=None,x2=None,text='north-east'):

        # a. value
        if x1 == None:
            x1 = self.x1_ast
        if x2 == None:
            x2 = self.x2_ast

        # b. plot
        polygon = plt.Polygon([[x1, x2],[self.x1max, x2],[self.x1max, self.x2max],[x1, self.x2max],[x1, x2]],
            color='black',alpha=0.10)
        ax.add_patch(polygon)
        ax.text(x1*1.05,0.95*self.x2max,text)

    def plot_convexity_check(self,ax,u=None,text='mix'):

        # a. value
        if u == None:
            u = self.u_ast

        # b. indifference curve
        x1, x2 = self.find_indifference_curve(u)

        # c. select
        I = (np.isnan(x1) == False) & (np.isnan(x2) == False)
        J = (x1[I] > 0) & (x2[I] > 0) & (x1[I] < self.x1max) & (x2[I] < self.x2max) 

        x1 = x1[I][J]
        x2 = x2[I][J]
        N = x1.size        
        i_low = np.int(N/3)
        i_high = np.int(N*2/3)

        x1_low = x1[i_low]
        x2_low = x2[i_low]

        x1_high = x1[i_high]
        x2_high = x2[i_high]

        # d. plot
        ax.plot([x1_low,x1_high],[x2_low,x2_high],ls='--',marker='o',
            markerfacecolor='none',color='black')

        x1 = 0.05*x1_low + 0.95*x1_high
        x2 = 0.05*x2_low + 0.95*x2_high
        ax.text(x1,x2*1.05,text)

    ###########
    # slutsky #
    ###########

    def plot_slutsky_endogenous(self,ax,p1_old,p1_new,p2,e1,e2,steps=4):
        
        I_old = p1_old*e1 + p2*e2
        I_new = p1_new*e1 + p2*e2
   
        # a. calculations
        A = self.maximize_utility(p1=p1_old,p2=p2,I=I_old)
        x1,x2,u_max = A

        C1 = self.maximize_utility(p1=p1_new,p2=p2,I=I_old)
        x1_fixI,x2_fixI,u_max_fixI = C1

        B = self.minimize_cost(u_max,p1=p1_new,p2=p2)
        h1, h2, h_cost = B

        C2 = self.maximize_utility(p1=p1_new,p2=p2,I=I_new)
        x1_new,x2_new,u_max_new = C2

        # b. plots
        self.p1,self.p2,self.I = p1_old,p2,I_old
        self.plot_budgetline(ax,ls='-',label='original',color=colors[0])
        if steps > 1:
            self.p1,self.p2,self.I = p1_new,p2,I_new
            self.plot_budgetline(ax,ls='-',label='final',color=colors[1])
        if steps > 2:
            self.p1,self.p2,self.I = p1_new,p2,h_cost
            self.plot_budgetline(ax,ls='-',alpha=0.50,label='compensated',color=colors[2])
        if steps > 3:
            self.p1,self.p2,self.I = p1_new,p2,I_old
            self.plot_budgetline(ax,ls='-',label='constant income',color=colors[3])
        ax.plot(e1,e2,'ro',color='black')
        ax.text(e1*1.03,e2*1.03,'$E$')

        # A
        ax.plot(x1,x2,'ro',color='black')
        ax.text(x1*1.03,x2*1.03,'$A$')
        self.plot_indifference_curves(ax,[u_max],do_label=False,ls='--',color=colors[0])

        # B
        if steps > 2:
            ax.plot(h1,h2,'ro',color='black')
            ax.text(h1*1.03,h2*1.03,'$B$')
        
        # C2
        if steps > 1:
            ax.plot(x1_new,x2_new,'ro',color='black')
            ax.text(x1_new*1.03,x2_new*1.03,'$C_2$')
            self.plot_indifference_curves(ax,[u_max_new],do_label=False,ls='--',color=colors[1])

        # C1
        if steps > 3:
            ax.plot(x1_fixI,x2_fixI,'ro',color='black')
            ax.text(x1_fixI*1.03,x2_fixI*1.03,f'$C_1$')
            self.plot_indifference_curves(ax,[u_max_fixI],do_label=False,ls='--',color=colors[3])
        
        if steps > 3:
            line = f'subtitution: $B-A$ = ({h1-x1:5.2f},{h2-x2:5.2f})\n'
            line += f'income: $C_1-B$ = ({x1_fixI-h1:5.2f},{x2_fixI-h2:5.2f})\n'
            line += f'wealth: $C_2-C_1$ = ({x1_new-x1_fixI:5.2f},{x2_new-x2_fixI:5.2f})'        
            ax.text(0.55*self.x1max,0.87*self.x2max,line)    