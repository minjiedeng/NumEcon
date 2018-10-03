import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import ipywidgets as widgets

class OLGModel():

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
        self.A = 7 # technology level
        self.n = 0.2 # population growth rate
        self.beta = 1/(1+0.4) # discount factor

        self.utility_function = 'crra'
        self.sigma = 8 # crra coefficient
        
        self.production_function = 'ces'
        self.alpha = 1/3 # distirubtion parameter
        self.rho = -4 # substitution parameter
        self.delta = 0.6 # depreciation rate
        
        # b. transition curve
        self.Nk = 1000 # num of points
        self.k_min = 1e-6 # minimum capital
        self.k_max = 6 # maximum capital

        # c. simulation
        self.T = 50 # num of periods
        self.seed = 1999 # seed for choosing between equilibria

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
        cobb_douglas_from_ces = self.production_function == 'ces' and self.rho == 0
        if self.production_function == 'cobb_douglas' or cobb_douglas_from_ces:
            self.f = lambda k: self.A*np.fmax(k,eps)**self.alpha
            self.f_prime = lambda k: self.A*self.alpha*np.fmax(k,eps)**(self.alpha-1)
        elif self.production_function == 'ces':
            self.f_raw = lambda k: (self.alpha*k**(self.rho) + (1-self.alpha))
            self.f = lambda k: self.A*self.f_raw(k)**(1/self.rho)
            self.f_prime = lambda k: self.A*self.alpha*k**(self.rho-1)*self.f_raw(k)**(1/self.rho-1)
        else:
            raise ValueError('unknown production function')


    ############
    # 2. solve #
    ############

    def solve_firm_problem(self,k):
        
        # a. interest rate factor 
        R = 1 + self.f_prime(k) - self.delta

        # b. wage rate
        w = self.f(k)-self.f_prime(k)*k

        return R,w

    def life_time_utility(self,c,w,R_plus):
        
        # a. consumption as old
        c_plus = R_plus*(w-c)

        # b. life-time utility
        U = self.u(c) + self.beta*self.u(c_plus)

        return U
        
    def solve_household_problem(self,w,R_plus):
        
        if self.utility_function == 'crra':
            
            s = (1-1/(1+R_plus**(1.0/self.sigma-1)*self.beta**(1.0/self.sigma)))*w

        else:

            # a. objective funciton to minimize
            obj = lambda c: -self.life_time_utility(c,w,R_plus)
            
            # b. optimal consumption
            c = optimize.fminbound(obj,0,w)

            # c. implied savings
            s = w - c

        return s 

    def find_equilibrium(self,k_plus,disp=0):

        # a. find future factor prices
        R_plus, _w_plus = self.solve_firm_problem(k_plus)

        # b. objective function to minimize
        def obj(k):
            
            # i. current factor prices
            _R, w = self.solve_firm_problem(k)

            # ii. saving
            s = self.solve_household_problem(w,R_plus)
            
            # iii. deviation
            return (k_plus-s/(1+self.n))**2

        # c. find k
        k_min = 0
        k_max = self.k_max+1
        k = optimize.fminbound(obj,k_min,k_max,disp=disp)

        return k

    def find_transition_curve(self):
        
        # a. future capital
        self.k_plus_grid = np.linspace(self.k_min,self.k_max,self.Nk)
        
        # b. implied current capital 
        self.k_grid = np.empty(self.Nk)
        for i,k_plus in enumerate(self.k_plus_grid):
            k = self.find_equilibrium(k_plus)
            self.k_grid[i] = k

    def find_interest_rate_function(self):

        # a. allocate
        self.R_plus_grid = np.empty(self.Nk)
        
        # b. interest rate factor
        for i,k in enumerate(self.k_plus_grid):
            self.R_plus_grid[i], _w = self.solve_firm_problem(k)

    def find_saving_function(self):

        # a. allocate
        self.s_grid = np.empty(self.Nk)

        # b. implied s
        for i,R_plus in enumerate(self.R_plus_grid):
            self.s_grid[i] = self.solve_household_problem(1,R_plus)

    def simulate(self,reset_seed=True):

        if reset_seed:
            np.random.seed(self.seed)

        # a. initialize
        self.sim_k = np.empty(self.T)
        self.sim_k[0] = 1

        # b. time loop
        for t in range(self.T-1):
            
            # i. current
            k = self.sim_k[t]

            # ii. list of potential future values 
            k_plus_list = []
       
            for i in range(1,self.Nk):

                if k >= self.k_grid[i-1] and k < self.k_grid[i]: # between grid points
                    
                    # o. linear interpolation
                    dy = (self.k_plus_grid[i]-self.k_plus_grid[i-1])
                    dx = (self.k_grid[i]-self.k_grid[i-1])
                    k_plus_interp = self.k_plus_grid[i-1] + dy/dx*(k-self.k_grid[i-1])
                    
                    # oo. append
                    k_plus_list.append(k_plus_interp)

            # iii. random draw of future value
            if len(k_plus_list) > 0:
                self.sim_k[t+1] = np.random.choice(k_plus_list,size=1)[0]
            else:
                self.sim_k[t+1] = 0
            
    ###########
    # 3. plot #
    ###########

    def plot_45(self,ax,**kwargs):

        if not 'color' in kwargs:
            kwargs['color'] = 'black'
        if not 'ls' in kwargs:
            kwargs['ls'] = '--'

        ax.plot([self.k_min,self.k_max],[self.k_min,self.k_max],**kwargs)

    def plot_transition_curve(self,ax,**kwargs):

        ax.plot(self.k_grid,self.k_plus_grid,**kwargs)

        ax.grid(ls='--',lw=1)
        ax.set_xlim([0,self.k_max])
        ax.set_ylim([0,self.k_max])
        ax.set_xlabel('$k_t$')
        ax.set_ylabel('$k_{t+1}$')

    def plot_interest_rate_function(self,ax,R_min=0,R_max=4,**kwargs):

        ax.plot(self.k_plus_grid,self.R_plus_grid,**kwargs)

        ax.grid(ls='--',lw=1)
        ax.set_xlim([0,self.k_max])
        ax.set_ylim([R_min,R_max])
        ax.set_xlabel('$k_t$')
        ax.set_ylabel('$R_t$')

    def plot_saving_function(self,ax,R_min=0,R_max=4,**kwargs):

        ax.plot(self.R_plus_grid,self.s_grid,**kwargs)

        ax.grid(ls='--',lw=1)
        ax.set_xlim([R_min,R_max])
        ax.set_ylim([0,1])
        ax.set_xlabel('$R_{t+1}$')
        ax.set_ylabel('$s_t$')
    
    def plot_sim_k(self,ax,**kwargs):

        if not 'ls' in kwargs:
            kwargs['ls'] = '-'
        if not 'marker' in kwargs:
            kwargs['marker'] = 'o'            
        if not 'MarkerSize' in kwargs:
            kwargs['MarkerSize'] = 2

        ax.plot(self.sim_k,**kwargs)

        ax.grid(ls='--',lw=1)
        ax.set_xlim([0,self.T])
        ax.set_xlabel('time')
        ax.set_ylabel('$k_t$')

#####################
# interactive plots #
#####################

def interactive_interest_rate_function():

   widgets.interact(interest_rate_function,        
                    alpha=widgets.FloatSlider(description='$\\alpha$',min=0.01, max=0.99, step=0.01, value=0.33,continuous_update=False),
                    rho=widgets.FloatSlider(description='$\\rho$',min=-16, max=0.99, step=0.01, value=-4,continuous_update=False),
                    delta=widgets.FloatSlider(description='$\\delta$',min=0.0, max=1, step=0.01, value=0.6,continuous_update=False),
                    ) 

def interest_rate_function(alpha,rho,delta):
    
    # a. solve model
    model = OLGModel(alpha=alpha,rho=rho,delta=delta)
    model.find_transition_curve()
    model.find_interest_rate_function()

    # b. setup figure
    fig = plt.figure(figsize=(6,4),dpi=100)
    ax = fig.add_subplot(1,1,1)

    # c. plot
    model.plot_interest_rate_function(ax)

def interactive_saving_function():

   widgets.interact(saving_function,        
                    beta=widgets.FloatSlider(description='$\\beta$',min=0.00, max=2.00, step=0.01, value=1/(1+0.4),continuous_update=False),
                    sigma=widgets.FloatSlider(description='$\\sigma$',min=0.5, max=16, step=0.01, value=8,continuous_update=False),
                    ) 

def saving_function(beta,sigma):
    
    # a. solve model
    model = OLGModel(beta=beta,sigma=sigma)
    model.find_transition_curve()
    model.find_interest_rate_function()
    model.find_saving_function()

    # b. setup figure
    fig = plt.figure(figsize=(6,4),dpi=100)
    ax = fig.add_subplot(1,1,1)

    # c. model
    model.plot_saving_function(ax)

def interactive_transition_curve():

   widgets.interact(transition_curve,        
                    sigma=widgets.FloatSlider(description='$\\sigma$',min=0.5, max=16, step=0.01, value=2,continuous_update=False),
                    rho=widgets.FloatSlider(description='$\\rho$',min=-16, max=0.99, step=0.01, value=0,continuous_update=False),
                    ) 

def transition_curve(sigma,rho):
    
    # a. solve model
    model = OLGModel(sigma=sigma,rho=rho)
    model.find_transition_curve()

    # b. setup figure
    fig = plt.figure(figsize=(6,4),dpi=100)
    ax = fig.add_subplot(1,1,1)

    # c. plot
    model.plot_45(ax)
    model.plot_transition_curve(ax)
