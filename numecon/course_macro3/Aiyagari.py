import time
import numba
import numpy as np
from scipy import optimize
from scipy import interpolate
from types import SimpleNamespace
import matplotlib.pyplot as plt
 
class AiyagariModel():

    ############
    # 1. setup #
    ############

    def __init__(self,name='',**kwargs):
        
        self.name = name
        self.setup_parameters()
        self.update_parameters(kwargs)
        self.setup_primitive_functions()
        self.setup_misc()
    
    def setup_parameters(self):

        # a. model parameters  
        self.beta = 0.96 # discount factor
        self.delta = 0.08 # depreciation rate
        self.sigma = 4 # crra coefficient
        self.alpha = 1/3 # cobb-douglas coeffient

        # b. solution
        self.tol_cfunc_inf = 1e-6 # tolerance for consumption function
        self.cfunc_maxiter = 2000 # maximum number of iterations when finding consumption function

        # income
        self.unemp_p = 0.05 # unemployment probability
        self.unemp_b = 0.15 # unemployment benefits
        self.Nz = 2 # number of productivity states
        self.grid_z = np.array([0.90,1.10]) # productivity values
        self.trans_p_z = np.array([[0.95,0.05],[0.05,0.95]]) # transition probabilities

        # end-of-period assets grid
        self.Na = 200
        self.a_min = 0
        self.a_max = 20
        self.a_phi = 1.1

        # cash-on-hand grid
        self.Nm = 500
        self.m_max = 20
        self.m_phi = 1.1

        # c. simulation
        self.seed = 2018        

        # d. steady state
        self.ss_R_tol = 1e-7 # tolerance for finding interest rate
        self.ss_a0 = 4.0 # initial cash-on-hand (homogenous)
        self.ss_simN = 50000 # number of households
        self.ss_simT = 2000 # number of time-periods
        self.ss_sim_burnin = 1000 # burn-in periods before calculating average savings

        # e. transition path
        self.transN = 50000 # number of households
        self.transT = 200 # number of periods
        self.trans_maxiter = 200
        self.trans_tol = 1e-4 # tolerance for convergence

    def update_parameters(self,kwargs):
        for key, value in kwargs.items():
            setattr(self,key,value)

    def setup_primitive_functions(self):
        
        eps = 1e-8

        # a. utility function
        if self.sigma == 1:
            self.u = lambda c: np.log(np.fmax(c,eps))
        else:
            self.u = lambda c: np.fmax(c,eps)**(1-self.sigma)/(1-self.sigma)        
        self.u_prime = lambda c: np.fmax(c,eps)**(-self.sigma)
        self.u_prime_inv = lambda x: x**(-1/self.sigma)

        # b. production function
        self.f = lambda k: np.fmax(k,eps)**self.alpha
        self.f_prime = lambda k: self.alpha*np.fmax(k,eps)**(self.alpha-1)
        self.f_prime_inv = lambda x: (np.fmax(x,eps)/self.alpha)**(1/(self.alpha-1))

    def setup_misc(self):

        def nonlinspace(min_val,max_val,num,phi): # phi up, more points close to min_val
            x = np.zeros(num)
            x[0] = min_val
            for i in range(1,num):
                x[i] = x[i-1] + (max_val-x[i-1]) / (num-i)**phi
            return x
            
        # a. grids
        self.grid_a = nonlinspace(self.a_min,self.a_max,self.Na,self.a_phi)
        self.grid_m = nonlinspace(0,self.m_max,self.Nm,self.m_phi)

        # b. initial distribution of z
        z_diag = np.diag(self.trans_p_z**1000)
        self.ini_p_z = z_diag/np.sum(z_diag)

        avg_z = np.sum(self.grid_z*self.ini_p_z)
        self.grid_z = self.grid_z/avg_z # force mean one

        # c. bounds on interst factor
        self.R_high = 1/self.beta + 0.005
        self.R_low = 1/self.beta - 0.005 

        # d. misc
        self.c_transition_path = np.empty((1,1,1)) # raw allocate

    ######################
    # 2. model functions #
    ######################

    def R_func(self,k):
        return 1 + self.f_prime(k) - self.delta

    def w_func(self,k):
        return self.f(k)-self.f_prime(k)*k
    
    def w_from_R_func(self,R):
        k = self.f_prime_inv(R-1+self.delta)
        return self.w_func(k)

    ############
    # 3. solve #
    ############

    def solve_step(self,c_plus_interp,R,w):
        
        c_func = [] 
        for i_z in range(self.Nz):
            
            # a. find next-period average marginal utility
            avg_marg_u_plus = np.zeros(self.Na)            
            for i_zplus in range(self.Nz):
                for u in [0,1]:

                    # i. future cash-on-hand
                    if u == 0:
                        m_plus = R*self.grid_a + w*(self.grid_z[i_zplus]-self.unemp_p*self.unemp_b)/(1-self.unemp_p)
                    else:
                        m_plus = R*self.grid_a + w*self.unemp_b

                    # ii. future consumption
                    c_plus = c_plus_interp[i_zplus](m_plus)

                    # iii. future marginal utility
                    marg_u_plus = self.u_prime(c_plus)

                    # iv. accumulate average marginal utility
                    weight = self.trans_p_z[i_z,i_zplus]
                    if u == 0:
                        weight *= 1-self.unemp_p
                    else:
                        weight *= self.unemp_p

                    avg_marg_u_plus += weight*marg_u_plus 
                
            # b. find current consumption and cash-on-hand
            c = self.u_prime_inv(R*self.beta*avg_marg_u_plus)
            m = self.grid_a + c
            
            m = np.insert(m,0,0) # add 0 in beginning
            c = np.insert(c,0,0) # add 0 in beginning

            # c. interpolate to common grid
            c_raw_func = interpolate.RegularGridInterpolator([m],c,method='linear', bounds_error=False, fill_value=None)

            # d. construct interpolator at common grid
            c_func_now = interpolate.RegularGridInterpolator([self.grid_m],c_raw_func(self.grid_m),method='linear', bounds_error=False, fill_value=None)
            c_func.append(c_func_now)

        return c_func

    def solve_inf_horizon(self):

        # a. initial guess (consume everything)
        c_func_inf = []
        for i_z in range(self.Nz):
            
            # i. consume everything
            m = self.grid_m
            c = m

            # ii. create linear interpolator
            interp = interpolate.RegularGridInterpolator([m],c,method='linear', bounds_error=False, fill_value=None)

            # iii. append
            c_func_inf.append(interp)

        # b. solve household problem
        diff_cfunc = np.inf
        it = 0    
        while diff_cfunc > self.tol_cfunc_inf:

            it += 1

            # i. remember previous
            c_func_inf_old = c_func_inf

            # ii. solve one step further 
            c_func_inf = self.solve_step(c_func_inf_old,self.R_ss,self.w_ss)
       
            # iii. maximum absolute difference
            diff_cfunc = []
            for i_z in range(self.Nz):
                diff_cfunc.append(np.amax(np.abs(c_func_inf_old[i_z].values-c_func_inf[i_z].values)))
            diff_cfunc = max(diff_cfunc)

            # iv. do not reach 2000 iterations
            if it > self.cfunc_maxiter:
                break

        # c. save interpolators
        self.c_func_inf = c_func_inf

        # d. save values
        self.c_inf = np.empty((self.Nz,self.Nm))
        for z in range(self.Nz):
            self.c_inf[z,:] = c_func_inf[z].values
            
    def solve_transition_path(self):

        # a. allocate memory
        self.c_func_transition_path = [None]*self.transT
        self.c_transition_path = np.empty((self.transT,self.Nz,self.Nm))

        # b. solve backwards along transition path
        for t in reversed(range(self.transT)):
            
            # i. solve
            if t == self.transT-1:
                c_plus_func = self.c_func_inf
                self.c_func_transition_path[t] = self.solve_step(c_plus_func,self.R_ss,self.w_ss)
            else:
                c_plus_func = self.c_func_transition_path[t+1]
                self.c_func_transition_path[t] = self.solve_step(c_plus_func,self.sim_R[t+1],self.sim_w[t+1])
            
            # ii. save values
            for z in range(self.Nz):
                self.c_transition_path[t,z,:] = self.c_func_transition_path[t][z].values 
    
    #############################
    # 4. stationary equilibrium #
    #############################

    def check_supply_and_demand(self,R_ss_guess,a0,z0,print_results=False):
    
        # a. prices
        self.R_ss = R_ss_guess
        self.w_ss = self.w_from_R_func(self.R_ss)    
                
        # b. solve infinite horizon problem
        t0 = time.time()
        self.solve_inf_horizon()
        time_sol = time.time() - t0

        # c. simulate
        t0 = time.time()

        # prices
        self.ss_sim_R = self.R_ss*np.ones(self.ss_simT)
        self.ss_sim_w = self.w_ss*np.ones(self.ss_simT)

        # simulate
        
        self.ss_sim_k,self.ss_sim_a,self.ss_sim_z = simulate(
            a0,z0,self.ss_sim_R,self.ss_sim_w,self.ss_simN,self.ss_simT,
            self.grid_z,self.grid_m,self.c_inf,
            self.trans_p_z,self.unemp_p,self.unemp_b,
            self.c_transition_path,0,self.seed)

        time_sim = time.time() - t0

        # d. calculate difference
        self.k_ss = np.mean(self.ss_sim_k[self.ss_sim_burnin:])
        R_ss_implied = self.R_func(self.k_ss)
        diff = R_ss_implied-R_ss_guess
    
        # e. print results              
        if print_results:
            print(f'  guess on R = {R_ss_guess:.5f} -> implied R = {R_ss_implied:.5f} (diff = {diff:8.5f})')
            #print(f'  time to solve = {time_sol:.1f}, time to simulate = {time_sim:.1f}')
        return diff

    def find_stationary_equilibrium(self,print_results=True):
        
        print(f'find stationary equilibrium (R in [{self.R_low:.5f};{self.R_high:.5f}]')

        # a. initial values
        a0 = self.ss_a0*np.ones(self.ss_simN)
        z0 = np.zeros(self.ss_simN,dtype=np.int32)
        z0[np.linspace(0,1,self.ss_simN) > self.ini_p_z[0]] = 1

        # b. find R_ss (first go)
        self.R_ss = optimize.bisect(self.check_supply_and_demand,self.R_low,self.R_high,args=(a0,z0,print_results),xtol=self.ss_R_tol*100)
        self.check_supply_and_demand(self.R_ss,a0,z0)

        print(f' update initial distribution')

        # b. find R_ss (second go)
        a0 = np.copy(self.ss_sim_a)
        z0 = np.copy(self.ss_sim_z)
        
        self.R_ss = optimize.bisect(self.check_supply_and_demand,self.R_low,self.R_high,args=(a0,z0,print_results),xtol=self.ss_R_tol)
        self.check_supply_and_demand(self.R_ss,a0,z0)

        print(f'steady state R = {self.R_ss:.5f} with k = {self.k_ss:.2f}')

    ######################
    # 5. transition path #
    ######################

    def find_transition_path(self,mu,**kwargs):
        
        print('finding transition path')

        # a. guess on interest rate
        self.sim_R = np.zeros(self.transT)
        for t in range(self.transT):
            if t == 0:
                self.sim_R[0] = self.R_func(np.mean(self.trans_sim_a0))
            elif t < self.transT/2:
                self.sim_R[t] = self.sim_R[t-1] + mu*(self.R_ss-self.sim_R[t-1]) 
            else:
                self.sim_R[t] = self.R_ss

        # b. update guess
        count = 0
        while True:

            # i. implied wage path
            self.sim_w = self.w_from_R_func(self.sim_R)

            # ii. solve
            self.solve_transition_path()

            # iii. simulate
            self.sim_k,self.sim_a,self.sim_z = simulate(
                self.trans_sim_a0,self.trans_sim_z0,self.sim_R,self.sim_w,self.transN,self.transT,
                self.grid_z,self.grid_m,self.c_inf,
                self.trans_p_z,self.unemp_p,self.unemp_b,
                self.c_transition_path,self.transT,self.seed)

            # iv. new R path
            R_old = self.sim_R
            R_new = self.R_func(self.sim_k)

            # v. done or update
            max_diff = np.amax(np.abs(R_new-R_old))
            if max_diff < self.trans_tol or count >= self.trans_maxiter: # done                
                #raise Exception('transition path has not converged')
                break
            else: # update
                self.sim_R = 0.9*R_old + 0.1*R_new
            
            if count == 1 or count%5 == 0:
                print(f'{count:3d}: {max_diff:.8f}')

            count += 1
    
        # c. save
        self.sim_R = R_new
        self.sim_w = self.w_from_R_func(self.sim_R)


@numba.njit(nogil)
def simulate(a0,z0,sim_R,sim_w,simN,simT,grid_z,grid_m,c_inf,trans_p_z,unemp_p,unemp_b,c_transition_path,transT,seed):

    np.random.seed(seed)

    # 1. allocate
    sim_a = np.zeros(simN)
    sim_z = np.zeros(simN,np.int32)
    sim_k = np.zeros(simT)
    
    # 2. simulate
    for t in range(simT):

        draw = np.linspace(0,1,simN)
        np.random.shuffle(draw)

        draw_uemp = np.linspace(0,1,simN)
        np.random.shuffle(draw_uemp)

        if t <= 0:
            sim_k[t] = np.mean(a0)     
        else:
            sim_k[t] = np.mean(sim_a)

        for i in numba.prange(simN):

            # a. states
            if t == 0:
                z_lag = np.int32(z0[i])
                a_lag = a0[i]
            else:
                z_lag = sim_z[i]
                a_lag = sim_a[i]

            # b. producitivty
            if draw[i] <= trans_p_z[z_lag,0]:
                sim_z[i] = 0
            else:
                sim_z[i] = 1

            # c. income
            if draw_uemp[i] <= unemp_p:
                y = sim_w[t]*unemp_b
            else:
                y = sim_w[t]*(grid_z[sim_z[i]]-unemp_p*unemp_b)/(1-unemp_p)

            # d. cash-on-hand
            m = sim_R[t]*a_lag + y

            # e. consumption
            if m <= grid_m[-1]:
                j = np.searchsorted(grid_m,m,side='right')
            else: # extrapolation
                j = grid_m.size-1 

            if t >= transT:
                c_left = c_inf[sim_z[i],j-1]
                c_right = c_inf[sim_z[i],j]         
            else:
                c_left = c_transition_path[t,sim_z[i],j-1]
                c_right = c_transition_path[t,sim_z[i],j]         

            c_diff = c_right - c_left
            
            m_left = grid_m[j-1]
            m_right = grid_m[j]
            m_diff = m_right - m_left
        
            c = c_left + c_diff * (m-m_left)/m_diff

            # f. savings
            sim_a[i] = m - c
            
    return sim_k,sim_a,sim_z
