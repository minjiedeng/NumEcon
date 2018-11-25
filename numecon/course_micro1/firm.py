import numpy as np
import matplotlib.pyplot as plt

class FirmClass1D():

    def __init__(self,**kwargs):

        # a. baseline setup

        # technology
        self.technology = 'baseline'
        self.A = 1
        self.gamma = 0.5
        
        # prices
        self.w = 1
        self.p = 1

        # figures
        self.x1label = '$\\ell$'
        self.x2label = '$x$'
        self.N = 100

        # b. update 
        for key,val in kwargs.items():
            setattr(self,key,val) # like par.key = val

        # c. calculations
        self.calculations()

    def calculations(self):
        
        if self.technology == 'baseline':
            self.g = lambda l,A,gamma: A*l**gamma    
        else:
            raise ValueError('unknown technology')

    ##########
    # choice #
    ##########

    def maximize_profit(self,**kwargs):

        # a. update
        for key,val in kwargs.items():
            setattr(self,key,val)

        # b. solve
        if self.technology == 'baseline':
            if self.gamma < 1:
                nom = self.w
                denom = self.p*self.A*self.gamma
                exp = 1/(self.gamma-1)
                self.l_ast = (nom/denom)**exp
                self.y_ast = self.g(self.l_ast,self.A,self.gamma)
                self.pi_ast = self.p*self.y_ast - self.w*self.l_ast
            else:
                self.l_ast = np.nan 
                self.y_ast = np.nan 
                self.pi_ast = np.nan
        else:
            raise ValueError('unknown production function')

        return np.array([self.l_ast,self.y_ast])

    def plot_max(self,ax,**kwargs):

        kwargs.setdefault('ls','')
        kwargs.setdefault('marker','*')
        kwargs.setdefault('markersize',7)
        kwargs.setdefault('color','black')
        kwargs.setdefault('label',f'$g({self.l_ast:.2f}) = {self.y_ast:.2f}$')

        ax.plot(self.l_ast,self.y_ast,**kwargs)

    ##########
    # figure #
    ##########

    def plot_pmo_line(self,ax,lmax=10,**kwargs):
        
        kwargs.setdefault('lw',2)
        l = np.linspace(1e-8,lmax,self.N)
        ax.plot(l,self.g(l,self.A,self.gamma),**kwargs)

    def plot_profit_line(self,ax,lmax=10,**kwargs):
        
        kwargs.setdefault('lw',2)
        l = np.linspace(1e-8,lmax,self.N)
        ax.plot(l,(self.pi_ast+self.w*l)/self.p,**kwargs)