import numpy as np
import matplotlib.pyplot as plt
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

from .consumer import ConsumerClass
from .firm import FirmClass1D

class KoopmanModel():

    def __init__(self,consumer,firm,L=24,e=0,w=1,**kwargs):

        # a. baseline setup
        self.consumer = ConsumerClass(**consumer)
        self.firm = FirmClass1D(**firm)

        # endowment        
        self.L = 24
        self.e = 0
        
        # prices
        self.w = w

        # figure
        self.lmax = L
        self.xmax = self.firm.g(L,self.firm.A,self.firm.gamma)*1.5

        self.x1label = '$\\ell,L-f$'
        self.x2label = '$y,x$'

        self.N = 100

        # b. update 
        for key,val in kwargs.items():
            setattr(self,key,val)

    ##########
    # figure #
    ##########

    def walras_figure(self):

        fig = plt.figure(frameon=False, figsize=(6,6),dpi=100)
        ax = fig.add_subplot(1,1,1)
        
        # a. set prices
        self.firm.w = self.w
        self.firm.p = 1
        self.consumer.p1 = self.w
        self.consumer.p2 = 1

        # b. firm
        self.firm.maximize_profit()
        self.firm.plot_max(ax,color=colors[0],label=f'firm choice')
        self.firm.plot_pmo_line(ax,color=colors[0],lmax=self.lmax,label='$g(\\ell)$')
        self.firm.plot_profit_line(ax,lmax=self.lmax,color='black',label='budgetline: $(\\pi-w\\ell)/p$')

        # c. consumer
        self.consumer.I = self.w*self.L + 1*self.e + self.firm.pi_ast
        self.consumer.maximize_utility()
        indiff_f,indiff_x = self.consumer.find_indifference_curve(u0=self.consumer.u_ast)

        # convert
        indiff_l = self.L-indiff_f
        f_ast = self.L-self.consumer.x1_ast
        x_ast = self.consumer.x2_ast-self.e

        # plot
        ax.plot(f_ast,x_ast,ls='',marker='*',markersize=7,color=colors[1],
            label=f'consumer choice')
        ax.plot(indiff_l,indiff_x,color=colors[1],label=f'$u(L-\\ell,x) = {self.consumer.u_ast:.2f}$')

        # d. layout
        ax.set_xlim([0,self.lmax])
        ax.set_ylim([0,self.xmax])
        ax.set_xlabel(self.x1label)
        ax.set_ylabel(self.x2label)

        legend = ax.legend(loc='lower right')
        frame = legend.get_frame()
        frame.set_facecolor('white')        

        fig.tight_layout()