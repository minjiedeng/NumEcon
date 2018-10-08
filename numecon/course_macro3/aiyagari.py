import time
import numba
import numpy as np
from scipy import optimize
from scipy import interpolate
from types import SimpleNamespace
import matplotlib.pyplot as plt
import linear_interp

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
        self.beta = 0.97 # discount factor
        self.delta = 0.10 # depreciation
        self.sigma = 2 # crra coefficient
        self.alpha = 1/3 # cobb-douglas coeffient

        # b. solution
        self.tol_cfunc = 1e-6 # tolerance for consumption function
        self.R_tol = 1e-6 # tolerance for finding interest rate

        # income
        self.unemp_p = 0.05
        self.unemp_b = 0.30
        self.Nz = 2
        self.grid_z = np.array([0.8,1.20])
        self.trans_p_z = np.array([[0.90,0.10],[0.10,0.90]])

        # end-of-period assets
        self.Na = 200
        self.a_min = 0
        self.a_max = 20
        self.a_phi = 1.1

        # cash-on-hand
        self.Nm = 500
        self.m_max = 20
        self.m_phi = 1.1

        # c. simulation
        self.seed = 2018        
        self.simN = 100000
        self.simT = 1000
        self.sim_z0 = np.zeros(self.simN,dtype=np.int)
        self.sim_m0 = 4.5*np.ones(self.simN)

        # d. transition path
        self.transT = 200
        self.trans_tol = 1e-4
        self.c_func_transition_path = [None]

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
        self.grid_z = self.grid_z/avg_z

        # c. bounds on interst factor
        self.R_high = 1/self.beta - 1e-8
        self.R_low = self.R_high - 0.01 # a looser bound is 1-self.delta + 1e-8


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

                # i. future cash-on-hand
                m_plus = R*self.grid_a + w*self.grid_z[i_z]
                
                # ii. future consumption
                #c_plus = c_plus_interp[i_zplus](m_plus)
                c_plus = c_plus_interp[i_zplus].evaluate(m_plus)
                c_plus.reshape(m_plus.shape)

                # iii. future marginal utility
                marg_u_plus = self.u_prime(c_plus)

                # iv. accumulate average marginal utility
                avg_marg_u_plus += self.trans_p_z[i_z,i_zplus]*marg_u_plus 
                
            # b. find current consumption and cash-on-hand
            c = self.u_prime_inv(R*self.beta*avg_marg_u_plus)
            m = self.grid_a + c
            
            m = np.insert(m,0,0) # add 0 in beginning
            c = np.insert(c,0,0) # add 0 in beginning

            # c. interpolate to common grid
            c_raw_func = linear_interp.create_interpolator(grids=[m],values=[c],Nxi=self.grid_m.size)
            #c_raw_func = interpolate.RegularGridInterpolator([m],c,method='linear', bounds_error=False, fill_value=None)

            # d. construct interpolator at common grid
            c_func_now = linear_interp.create_interpolator(grids=[self.grid_m],values=[c_raw_func.evaluate(self.grid_m)],Nxi=self.grid_a.size)
            #c_func_now = interpolate.RegularGridInterpolator([self.grid_m],c_raw_func(self.grid_m),method='linear', bounds_error=False, fill_value=None)
            c_func.append(c_func_now)

        return c_func

    def solve_inf_horizon(self):

        # 1. initial guess (consume everything)
        c_func_inf = []
        for i_z in range(self.Nz):
            
            # a. consume everything
            m = self.grid_m
            c = m

            # b. create linear interpolator
            interp = linear_interp.create_interpolator(grids=[m],values=[c],Nxi=self.grid_a.size)
            #interp = interpolate.RegularGridInterpolator([m],c,method='linear', bounds_error=False, fill_value=None)

            # c. append
            c_func_inf.append(interp)

        # 2. solve household problem
        diff_cfunc = np.inf
        it = 0    
        while diff_cfunc > self.tol_cfunc:

            it += 1

            # a. remember previous
            c_func_inf_old = c_func_inf

            # b. solve one step further 
            c_func_inf = self.solve_step(c_func_inf_old,self.R_ss,self.w_ss)
       
            # c. maximum absolute difference
            diff_cfunc = []
            for i_z in range(self.Nz):
                #diff_cfunc.append(np.amax(np.abs(c_func_inf_old[i_z].values-c_func_inf[i_z].values)))
                diff_cfunc.append(np.amax(np.abs(c_func_inf_old[i_z].y-c_func_inf[i_z].y)))
            diff_cfunc = max(diff_cfunc)

            #print(f'{it:4d}: {diff_cfunc:.6f}')
            
            # d. do not reach 2000 iterations
            if it > 2000:
                break

        self.c_func_inf = c_func_inf
        self.c_func_inf_sim = []
        for i_z in range(self.Nz):
            self.c_func_inf_sim.append(linear_interp.create_interpolator(grids=[c_func_inf[i_z].x],values=[c_func_inf[i_z].y],Nxi=self.simN))

    def solve_transition_path(self):

        self.c_func_transition_path = [None]*self.transT
        for t in reversed(range(self.transT)):

            if t == self.transT-1:
                c_plus_func = self.c_func_inf
            else:
                c_plus_func = self.c_func_transition_path[t+1]
                
            self.c_func_transition_path[t] = self.solve_step(c_plus_func,self.sim_R[t],self.sim_w[t])


    ###############
    # 4. simulate #
    ###############

    def simulate_old(self,transT=0):

        #print('simulating')

        # 1. set seed for random number generator
        np.random.seed(self.seed)

        # 2. allocate
        self.sim_m = np.zeros(self.simN)
        self.sim_z = np.zeros(self.simN,dtype=np.int)
        self.sim_a = np.zeros(self.simN)

        self.sim_k = np.zeros(self.simT)
        self.sim_y = np.zeros(self.simT)

        # 3. simulate
        for t in range(self.simT):

            # a. states
            if t == 0:
                z = self.sim_z0
                m = self.sim_m0
            else:
                z = np.copy(self.sim_z)
                m = np.copy(self.sim_m)
            
            # b. household decistions
            for i_z in range(self.Nz):
                
                # i. selection
                I = z == i_z 
                I_sum = np.sum(I)
                if I_sum == 0:
                    continue

                # ii. consumption choice
                if t < transT:
                    c = self.c_func_transition_path[t][i_z](m[I]) 
                else:
                    c = self.c_func_inf_sim[i_z].evaluate() 

                # iii. savings
                self.sim_a[I] = m[I] - c

                # iv. next-period
                if t < self.simT-1:
                    
                    #z_plus = np.random.choice(self.Nz, I_sum, p=self.trans_p_z[i_z,:])
                    
                    z_plus = np.empty(I_sum) 
                    draw = np.linspace(0,1,I_sum)
                    np.random.shuffle(draw)
                    J = draw < self.trans_p_z[i_z,0]
                    z_plus[J] = 0
                    z_plus[~J] = 1
                    self.sim_z[I] = z_plus
                    
                    # cash-on-hand
                    y = np.empty(I_sum)
                    y_unemp = self.unemp_p*self.unemp_b
                    y_work = (self.sim_w[t]*self.grid_z[z[I]]-y_unemp)/(1-self.unemp_p)
                    draw = np.linspace(0,1,I_sum)
                    J = draw < self.unemp_p
                    y[~J] = y_work[~J]
                    y[J] = y_unemp
                    self.sim_m[I] = self.sim_R[t]*self.sim_a[I] + y

            # c. aggregate capital
            self.sim_k[t] = np.mean(self.sim_a)            
            self.sim_y[t] = np.mean(self.sim_w[t]*self.grid_z[z])    

    def simulate(self,transT=0):
        self.sim_k,self.sim_m,self.sim_z,self.sim_a = simulate(self.sim_m0,self.sim_z0,self.sim_R,self.sim_w,self.seed,self.simN,self.simT,self.c_func_inf_sim,
            self.trans_p_z,self.unemp_p,self.unemp_b,self.grid_z)
        #print(self.sim_k)
        #print(np.amax(self.sim_k))

    #############################
    # 5. stationary equilibrium #
    #############################

    def check_supply_and_demand(self,R_ss_guess,print_results=False):
    
        # a. prices
        self.R_ss = R_ss_guess
        self.w_ss = self.w_from_R_func(self.R_ss)    
        
        if print_results:
            print(f'guessed R: {self.R_ss:.4f}')
            print(f'implied w: {self.w_ss:.4f}')        

        # b. solve infinite horizon problem
        t0 = time.time()

        self.solve_inf_horizon()

        if print_results:
            print(f' solved in {time.time()-t0:.1f} secs')

        # c. simulate
        t0 = time.time()

        self.sim_R = self.R_ss*np.ones(self.simT)
        self.sim_w = self.w_ss*np.ones(self.simT)
        self.simulate()

        if print_results:
            print(f' simulated in {time.time()-t0:.1f} secs')

        # d. calculate difference
        k_ss = self.sim_k[-1]
        R_ss_implied = self.R_func(k_ss)
        diff = R_ss_implied-R_ss_guess
    
        # e. print results              
        if print_results:
            print(f' implied k = {k_ss:.4f}')
            if diff > 0:
                print(f' implied R = {R_ss_implied:.5f} > {R_ss_guess:.5f} (diff = {diff:.5})\n')
            else:
                print(f' implied R = {R_ss_implied:.5f} < {R_ss_guess:.5f} (diff = {diff:.5})\n')   
        
        return diff

    def find_stationary_equilibrium(self,print_results):
                
        # a. find startionary equilibrium
        self.R_ss = optimize.bisect(self.check_supply_and_demand,self.R_low,self.R_high,args=(print_results),xtol=self.R_tol)
        
        
        #self.R_ss = 1.02950
        self.check_supply_and_demand(self.R_ss)
        self.sim_z0 = self.sim_z
        self.sim_m0 = self.sim_m
        #self.check_supply_and_demand(self.R_ss)

        #t0 = time.time()
        #self.simulate()

        #if print_results:
        #    print(f'R_ss = {self.R_ss:.4f} (found in {time.time()-t0:.1f} secs)')
    
    #############################
    # 6. stationary equilibrium #
    #############################

    def find_transition_path(self,mu,R0,**kwargs):
        
        print('finding transition path')

        # a. guess on interest rate
        self.sim_R = np.zeros(self.simT)
        for t in range(self.simT):
            if t == 0:
                self.sim_R[0] = R0
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
            self.simulate(transT=self.transT)
            
            # iv. new R path
            R_old = self.sim_R[:self.transT]
            R_new = self.R_func(self.sim_k[:self.transT])

            # v. done or update
            max_diff = np.amax(np.abs(R_new-R_old))
            if max_diff < self.trans_tol or count > 200: # done
                break
            else: # update
                self.sim_R[:self.transT] = 0.9*R_old + 0.1*R_new
            
            if count == 1 or count%5 == 0:
                print(f'{count}: {max_diff:.8f}')

            count += 1

@numba.njit(nogil=True)
def simulate(m0,z0,sim_R,sim_w,seed,simN,simT,c_func_inf_sim,trans_p_z,unemp_p,unemp_b,grid_z):

    #print('simulating')

    # 1. set seed for random number generator
    np.random.seed(seed)

    # 2. allocate
    sim_m = np.zeros(simN)
    sim_z = np.zeros(simN,dtype=np.int32)
    sim_a = np.zeros(simN)

    sim_k = np.zeros(simT)
    
    z = np.zeros(simN,dtype=np.int32)
    m = np.zeros(simN)
    c = np.zeros(simN)

    # 3. simulate
    for t in range(simT):

        draw = np.linspace(0,1,simN)
        np.random.shuffle(draw)

        draw_uemp = np.linspace(0,1,simN)
        np.random.shuffle(draw_uemp)

        for i in range(simN):

            # a. states
            if t == 0:
                z[i] = z0[i]
                m[i] = m0[i]
            else:
                z[i] = sim_z[i]
                m[i] = sim_m[i]

        # b. consumption choice
        c0 = c_func_inf_sim[0].evaluate_par(m)
        for i in range(simN):
            if z[i] == 0:
                c[i] = c0[i]

        c1 = c_func_inf_sim[1].evaluate_par(m)
        for i in range(simN):
            if z[i] == 1:
                c[i] = c1[i]

        for i in range(simN):

            # c. savings
            sim_a[i] = m[i] - c[i]

            # iv. next-period
            if t < simT-1:
                
                #z_plus = np.random.choice(Nz, I_sum, p=trans_p_z[i_z,:])
                if draw[i] <= trans_p_z[z[i],0]:
                    sim_z[i] = 0
                else:
                    sim_z[i] = 1
                
                # cash-on-hand
                if draw_uemp[i] <= unemp_p:
                    y = unemp_p*unemp_b
                else:
                    y = (sim_w[t]*grid_z[z[i]]-unemp_p*unemp_b)/(1-unemp_p)
                sim_m[i] = sim_R[t]*sim_a[i] + y

        # c. aggregate capital
        sim_k[t] = np.mean(sim_a)            

    return sim_k,sim_m,sim_z,sim_a


if __name__ == '__main__':

    # a. model
    model = AiyagariModel()
    
    # b. find stationary equilibrium
    model.find_stationary_equilibrium(print_results=True)

    # c. plot convergence
    fig = plt.figure(figsize=(6,6),dpi=100)
    ax = fig.add_subplot(1,1,1)

    ts = np.arange(model.simT)
    ax.plot(ts,model.R_ss*np.ones(model.simT),'-',color='black')
    ax.plot(ts,model.R_func(model.sim_k),'--',color='firebrick')
    #ax.plot(ts,model.sim_k,'--',color='firebrick')

    ax.grid(ls='--',lw=1)

    plt.show()

    # d. transition path
    model.sim_m0 *= 0.95
    model.find_transition_path(0.05,model.R_ss+0.01)

    fig = plt.figure(figsize=(6,6),dpi=100)
    ax = fig.add_subplot(1,1,1)

    ts = np.arange(model.simT)
    ax.plot(ts,model.sim_R,'-',color='black')
    ax.plot(ts,model.R_func(model.sim_k),'--',color='firebrick')

    ax.grid(ls='--',lw=1)

    fig = plt.figure(figsize=(6,6),dpi=100)
    ax = fig.add_subplot(1,1,1)

    ts = np.arange(model.simT)
    ax.plot(ts[:400],model.sim_R[:400],'-',color='black')
    ax.plot(ts[:400],model.R_func(model.sim_k[:400]),'--',color='firebrick')

    ax.grid(ls='--',lw=1)


    plt.show()
