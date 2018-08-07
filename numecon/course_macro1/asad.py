import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

def simulate(a=0.4,gamma=0.1,phi=0.9,delta=0.8,omega=0.15,sigma_x=1,sigma_c=0.2,T=100):

  widgets.interact(simulate_,    
        a=widgets.fixed(a),
        gamma=widgets.fixed(gamma),
        phi=widgets.fixed(phi),
        delta=widgets.fixed(delta),
        omega=widgets.fixed(omega),
        sigma_x=widgets.FloatSlider(description='$\\sigma_{x}$',min=0.00, max=2.0, step=0.01, value=sigma_x),
        sigma_c=widgets.FloatSlider(description='$\\sigma_{c}$',min=0.00, max=2.0, step=0.01, value=sigma_c),
        T=widgets.fixed(T),
  )

def simulate_(a,phi,gamma,delta,omega,sigma_x,sigma_c,T):

    np.random.seed(2015)    
    
    # a. parameters
    b = (1+a*phi*gamma)/(1+a*gamma)
    beta = 1/(1+a*gamma)

    # b. function
    y_hat_func = lambda y_hat_lag,z,z_lag,s,s_lag: b*y_hat_lag + beta*(z-z_lag) - a*beta*s + a*beta*phi*s_lag
    pi_hat_func = lambda pi_lag,z,z_lag,s,s_lag: b*pi_lag + beta*gamma*z - beta*phi*gamma*z_lag + beta*s - beta*phi*s_lag
    z_func = lambda z_lag,x: delta*z_lag + x
    s_func = lambda s_lag,c: omega*s_lag + c
    
    # c. simulation
    x = np.random.normal(loc=0,scale=sigma_x,size=T)
    c = np.random.normal(loc=0,scale=sigma_c,size=T)
    z = np.zeros(T)
    s = np.zeros(T)
    y_hat = np.zeros(T)
    pi_hat = np.zeros(T)

    for t in range(1,T):

        # i. update z and s
        z[t] = z_func(z[t-1],x[t])
        s[t] = s_func(s[t-1],c[t])

        # ii. compute y og pi 
        y_hat[t] = y_hat_func(y_hat[t-1],z[t],z[t-1],s[t],s[t-1])
        pi_hat[t] = pi_hat_func(pi_hat[t-1],z[t],z[t-1],s[t],s[t-1])
    
    # d. figure
    fig = plt.figure(figsize=(8,6),dpi=100)
    ax = fig.add_subplot(1,1,1)

    ax.plot(y_hat,label='$\\hat{y}$')
    ax.plot(pi_hat,label='$\\hat{pi}$')

    ax.set_xlabel('time')
    
    ax.set_ylabel('percent')
    ax.set_ylim([-8,8])

    ax.grid(ls='--', lw=1)

    legend = ax.legend(loc='upper left', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
