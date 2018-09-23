import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import ipywidgets as widgets

class RamseyModel():

    ############
    # 1. setup #
    ############

    def __init__(self,name='',**kwargs):
        
        self.name = name
        self.baseline_parameters()
        self.update_parameters(kwargs)
        self.primitive_functions()
    
    def baseline_parameters(self):

        # a. model parameters  
        self.beta = 0.97 # discount factor
        self.utility_function = 'crra'
        self.sigma = 2 # crra coefficient
        self.production_function = 'cobb_douglas'
        self.alpha = 1/3 # cobb-dogulas coefficient
        self.delta = 0.10 # depreciation rate

        # b. solution
        self.tau = 1e-8 # tolerance
        
        # c. saddle-path
        self.k0 = 1.00 # initial level of capital
        self.saddlepath_max_iter = 1000 # maximum number of iterations
        self.saddlepath_T = 1000 # maximum number of time periods to reach steady state

        # d. loci in phase diagram
        self.Nk = 100 # num of points
        self.k_max = 8 # maximum capital
        self.c_max = 2 # maximum consumption 
     
        # e. simulation        
        self.T = 500 # number of periods
        self.c_upper_fac = 5 # exit if consumption > c_upper_fac*consumption in steady state
        self.c_lower_fac = 0.05 # exit if consumption < c_lower_fac*consumption in steady state
        self.force_steady_state = 1e-4 # assume in steady state if close enough  

        # f. misc
        self.do_print = True

    def update_parameters(self,kwargs):
        for key, value in kwargs.items():
            setattr(self,key,value)

    def primitive_functions(self):
        
        eps = 1e-12 # minimum evaluation value to avoid errors

        # a. utility function
        if self.utility_function == 'crra':
            if self.sigma == 1:
                self.u = lambda c: np.log(np.fmax(c,eps))
            else:
                self.u = lambda c: np.fmax(c,eps)**(1-self.sigma)/(1-self.sigma)
            self.u_prime = lambda c: np.fmax(c,eps)**(-self.sigma)
        else:
            raise ValueError('unknown utility function')


        # b. production function
        if self.production_function == 'cobb_douglas':
            self.f = lambda k: np.fmax(k,eps)**self.alpha
            self.f_prime = lambda k: self.alpha*np.fmax(k,eps)**(self.alpha-1)
            self.f_prime_inv = lambda x: (x/self.alpha)**(1/(self.alpha-1))
        else:
            raise ValueError('unknown production function')


    ######################
    # 2. model functions #
    ######################

    def k_plus_func(self,k,c): # next period capital
        return k*(1-self.delta) + self.f(k) - c

    def c_func(self,k,k_plus): # backwards calculation of consumption
        return k*(1-self.delta) + self.f(k) - k_plus

    def euler_LHS_func(self,c,c_plus): # left-hand-side of euler equation
        return self.u_prime(c)/self.u_prime(c_plus)
    
    def euler_RHS_func(self,k_plus): # right-hand-side of euler equation
        return self.beta*(1+self.f_prime(k_plus)-self.delta)

    def r_func(self,k): # interest rate
        return self.f_prime(k) - self.delta

    def w_func(self,k): # wage
        return self.f(k)-self.f_prime(k)*k

    def Gamma(self,k,c): # law-of-motion

        # i. capital
        k_plus = self.k_plus_func(k,c) 
                
        # ii. consumption
        if self.production_function == 'cobb_douglas':
            c_plus = self.euler_RHS_func(k_plus)**(1/self.sigma)*c
        else: # numerical solution
            obj = lambda c_plus: self.euler_LHS_func(c,c_plus)-self.euler_RHS_func(k_plus)
            c_plus = optimize.fsolve(obj, c)[0]

        return k_plus,c_plus

    def dist_to_ss(self,k,c): # distance to steady state
        
        return np.sqrt((k-self.k_ss)**2 + (c-self.c_ss)**2)

    ############
    # 3. solve #
    ############

    def find_steady_state(self):

        # a. objective euler_RHS_func == 1          
        obj = lambda k: self.euler_RHS_func(k)-1

        # b. find capital in steady state 
        if self.production_function == 'cobb_douglas':
            self.k_ss = self.f_prime_inv(1/self.beta-1+self.delta)
        else: # numerical solution
            self.k_ss = optimize.fsolve(obj,1)[0]
        
        # c. find implied steady consumption, interest rate and wage
        self.c_ss = self.c_func(self.k_ss,self.k_ss)
        self.r_ss = self.r_func(self.k_ss)
        self.w_ss = self.w_func(self.k_ss)

        # d. print
        if self.do_print:
            print(f'k_ss = {self.k_ss:.4f}')
            print(f'c_ss = {self.c_ss:.4f}')
            print(f'r_ss = {self.r_ss:.4f}')
            print(f'w_ss = {self.w_ss:.4f}')        
    
    def find_c0_on_saddlepath(self): # initial consumption

        # a. initial capital
        k0 = self.k0
        
        # b. initial interval for consumption
        if k0 < self.k_ss:
            c_low = 0
            c_high = self.c_func(k0,k0)
        else:
            c_low = self.c_func(k0,k0)
            c_high = k0**self.alpha
        
        # for plotting
        self.c0_path = np.nan*np.ones(self.saddlepath_max_iter)
        self.c_high_path = np.nan*np.ones(self.saddlepath_max_iter)
        self.c_low_path = np.nan*np.ones(self.saddlepath_max_iter)

        # c. algorithm
        t = 0 # time
        it = 0 # iteration counter
        while it < self.saddlepath_max_iter:

            # i. initilize
            if t == 0: 
                c0 = (c_low + c_high)/2                
                k = k0
                c = c0

                 # for plotting
                self.c0_path[it] = c0
                self.c_low_path[it] = c_low
                self.c_high_path[it] = c_high

                it += 1

            t += 1 # increment time

            # ii. update and distance
            k,c = self.Gamma(k,c)
            dist = self.dist_to_ss(k,c)
            
            # iii. finish, forward or restart?
            if dist < self.tau or t > self.saddlepath_T: # finish
                break
            elif k0 <= self.k_ss: 
                if c <= self.c_ss and k <= self.k_ss: 
                    continue # simulate forward
                else: # restart
                    t = 0                                       
                    if k > self.k_ss:
                        c_low = c0 # c in [c0,c_high]
                    elif c > self.c_ss:
                        c_high = c0 # c in [c_low,c0]
            else:
                if c >= self.c_ss and k >= self.k_ss: 
                    continue # simulate forward
                else: # restart
                    t = 0                                       
                    if k < self.k_ss:
                        c_high = c0 # c in [c_low,c0]
                    elif c < self.c_ss:
                        c_low = c0 # c in [c0,c_high]
                                
        self.k0,self.c0 = k0,c0

        # d. print result
        if self.do_print:
            print(f'c0 = {self.c0:.4f}')
        
    def calculate_loci(self):

        # a. setup
        self.c_loci = dict()
        self.k_loci = dict()

        # b. c loci
        self.c_loci['x'] = [self.k_ss,self.k_ss]
        self.c_loci['y'] = [0,self.c_max]

        # c. k loci
        self.k_loci['x'] = np.linspace(0,self.k_max,self.Nk)
        self.k_loci['y'] = self.c_func(self.k_loci['x'],self.k_loci['x']) 

    ###############
    # 4. simulate #
    ###############

    def simulate(self,c0=None):

        # a. allocate
        self.sim = dict()
        self.sim['k'] = np.empty(self.T)
        self.sim['c'] = np.empty(self.T)
        self.sim['r'] = np.empty(self.T)
        self.sim['w'] = np.empty(self.T)

        # b. time loop
        for t in range(self.T):
            
            # i. initial values 
            if t == 0:
            
                self.sim['k'][t] = self.k0
                if c0 == None:
                    self.sim['c'][t] = self.c0 # model state
                else:
                    self.sim['c'][t] = c0 # input
            
            # ii. forward
            else:

                if self.dist_to_ss(self.sim['k'][t-1],self.sim['c'][t-1]) < self.force_steady_state:
                    self.sim['k'][t] = self.k_ss
                    self.sim['c'][t] = self.c_ss
                else:
                    self.sim['k'][t],self.sim['c'][t] = self.Gamma(self.sim['k'][t-1],self.sim['c'][t-1])

            # iii. prices
            self.sim['r'][t] = self.r_func(self.sim['k'][t])
            self.sim['w'][t] = self.w_func(self.sim['k'][t])

            # iv. disconvergence
            if self.sim['c'][t] > self.c_upper_fac*self.c_ss or self.sim['c'][t] < self.c_lower_fac*self.c_ss:
                self.sim['k'][t:] = np.nan
                self.sim['c'][t:] = np.nan
                break

    def optimal_initial_consumption_level(self):

        if not self.utility_function == 'crra':
                raise ValueError('consumption level only implemented for crra utility') 
        
        r_path = self.sim['r']
        w_path = self.sim['w']

        # a. initialize sums
        c_sum_rel = 1 # discounted sum of consumption relative to c0
        wealth_sum = (1+r_path[0])*self.k0 + w_path[0] # discounted wealth

        # b. initialize factors
        c_fac = np.ones(self.T) # growth factor of consumption
        R_fac = 1 # total discount factor

        # c. calculate sums
        for t in range(1,self.T): # t = 1,2,3,...

            # i. factors
            c_fac[t] = c_fac[t-1]*((1+r_path[t])*self.beta)**(1/self.sigma)/(1+r_path[t])
            R_fac /= (1+r_path[t])

            # ii. sums
            c_sum_rel += c_fac[t]
            wealth_sum += w_path[t]*R_fac

        # d. return optimal consumption choice
        c0 = wealth_sum/c_sum_rel
        return c0
    
    ###########
    # 5. plot #
    ###########

    def plot_loci(self,ax,**kwargs):

        self.calculate_loci()
        self.plot_c_loci(ax,**kwargs)
        self.plot_k_loci(ax,**kwargs)

        ax.grid(ls='--',lw=1)
        ax.set_xlim([0,self.k_max])
        ax.set_ylim([0,self.c_max])
        ax.set_xlabel('$k_t$')
        ax.set_ylabel('$c_t$')

    def plot_c_loci(self,ax,**kwargs):
        ax.plot(self.c_loci['x'],self.c_loci['y'],color='black',**kwargs)

    def plot_k_loci(self,ax,**kwargs):
        ax.plot(self.k_loci['x'],self.k_loci['y'],color='black',**kwargs)

    def plot_steady_state(self,ax,**kwargs):
        ax.plot(self.k_ss,self.c_ss,**kwargs)

    def plot_kc_path(self,ax,**kwargs):
        ax.plot(self.sim['k'],self.sim['c'],'-o',lw=0.5,MarkerSize=2,**kwargs)

    def plot_sim_time(self,ax,varname,**kwargs):
        ax.plot(self.sim[varname],'o',MarkerSize=2,**kwargs)

def interactive_phase_diagram():

   widgets.interact(phase_diagram,        
                    beta=widgets.FloatSlider(description='$\\beta$',min=0.90, max=0.99, step=0.01, value=0.97,continuous_update=False),
                    sigma=widgets.FloatSlider(description='$\\sigma$',min=0.5, max=4, step=0.01, value=2,continuous_update=False),
                    delta=widgets.FloatSlider(description='$\\delta$',min=0.05, max=0.20, step=0.01, value=0.10,continuous_update=False),
                    alpha=widgets.FloatSlider(description='$\\alpha$',min=0.10, max=0.50, step=0.01, value=1/3,continuous_update=False),
                    k0=widgets.FloatSlider(description='$k_0$',min=0, max=8, step=0.05, value=1,continuous_update=False)
                    ) 

def phase_diagram(beta,sigma,delta,alpha,k0):
    
    model = RamseyModel(beta=beta,sigma=sigma,delta=delta,alpha=alpha,k0=k0,do_print=False)
    model.find_steady_state()
    model.find_c0_on_saddlepath()

    # a. setup figure
    fig = plt.figure(figsize=(12,4),dpi=100)
    
    # b. phase diagram
    ax = fig.add_subplot(1,2,1) 
    model.plot_loci(ax)
    model.simulate()
    model.plot_kc_path(ax)

    # c. time profile
    ax = fig.add_subplot(1,2,2) 
    model.plot_sim_time(ax,'k')
    ax.plot([0,100],[model.k_ss,model.k_ss],'--',color='black')
    ax.grid(ls='--',lw=1)
    ax.set_xlim([0,100])
    ax.set_ylim([0,model.k_max])
    ax.set_xlabel('time')
    ax.set_ylabel('$k_t$')    

