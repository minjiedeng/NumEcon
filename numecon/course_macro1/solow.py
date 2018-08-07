import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

def solow_equation(k,alpha,delta,s):

    saving = s*k**alpha
    depreciation = delta*k
    k_plus = k + saving - depreciation    
    return k_plus, saving, depreciation  
              
def simulate_solow_model(k0,alpha,delta,s,T):

    k_path = [k0]    
    saving_path = []
    depreciation_path = []
    
    for t in range(1,T):

        k_plus,saving,depreciation = solow_equation(k_path[t-1],alpha,delta,s)    

        k_path.append(k_plus)
        saving_path.append(saving)
        depreciation_path.append(depreciation)
        
    return k_path,saving_path,depreciation_path


def simulate(k_max=10,T=150):

   widgets.interact(simulate_,        
                    k0=widgets.FloatSlider(description='$k_0$',min=0, max=k_max, step=0.05, value=2), 
                    alpha=widgets.FloatSlider(description='$\\alpha$',min=0.01, max=0.99, step=0.01, value=0.3), 
                    delta=widgets.FloatSlider(description='$\\delta$',min=0.01, max=0.50, step=0.01, value=0.1), 
                    s=widgets.FloatSlider(description='$s$',min=0.01, max=0.99, step=0.01, value=0.3), 
                    T=widgets.fixed(T), 
                    k_max=widgets.fixed(k_max)) 

def simulate_(k0,alpha,delta,s,T,k_max):

    k_path,saving_path,depreciation_path = simulate_solow_model(k0,alpha,delta,s,T*5)

    fig = plt.figure(figsize=(16,6),dpi=100)
    ax1 = fig.add_subplot(1,2,1)

    ax1.plot(k_path[:T],lw=2)
    ax1.grid(ls='--',lw=1)
    ax1.set_title('Capital ($k_t$)')
    ax1.set_xlim([0,T])
    ax1.set_ylim([0,k_max])

    k_path = np.array(k_path)
    I = np.abs(np.log(k_path) - np.log(k_path[-1])) < 0.01
    t_converge = T*5 - k_path[I].size
    ax1.plot([t_converge,t_converge],[0,k_max],ls='--',color='black',label='$|\\log(k_t)-\\log(k^\\ast)| < 0.01$')

    legend = ax1.legend(loc='lower right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(saving_path[:T],label='saving')
    ax2.plot(depreciation_path[:T],label='depreciation')
    ax2.grid(ls='--',lw=1)
    ax2.set_title('Saving ($sk^\\alpha$) and depreciation ($\\delta k_t$)')
    ax2.set_xlim([0,T])
    ax2.set_ylim([0,0.5*k_max])

    legend = ax2.legend(loc='upper left', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
