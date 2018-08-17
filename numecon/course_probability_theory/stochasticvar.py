from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

def dice(Nmax=30):

    widgets.interact_manual.opts['manual_name'] = 'Roll!'
    widgets.interact_manual(dice_,
        N=widgets.IntSlider(description='rolls',min=1, max=Nmax*5, step=1, value=1),
        Nmax=widgets.fixed(Nmax),
        density=widgets.fixed(0)
        )

def dice_density(continuous_update=True,Nmax=50000):


    widgets.interact(dice_,
        N=widgets.IntSlider(description='rolls',min=50, max=Nmax, step=50, value=50, continuous_update=continuous_update),
        Nmax=widgets.fixed(Nmax),
        density=widgets.fixed(1)
    )

def dice_(N,Nmax,density):

    # a. roll dice
    die_1 = np.random.random_integers(1,high=6,size=N)
    die_2 = np.random.random_integers(1,high=6,size=N)
    dice_sum = die_1 + die_2

    # b. figure
    fig = plt.figure(figsize=(8,6))
    ax_hist = fig.add_subplot(1,1,1)

    # c. histogram
    width = 0.1
    bins = []
    for i in range(1,14):
        bins.append(i-0.5*width)
    ax_hist.hist(dice_sum,bins=bins,width=width,density=density)
    ax_hist.set_xlabel('sum of the two dice')
    ax_hist.set_xticks(np.arange(2,12+1,1))
    if not density:
        ax_hist.set_ylabel('number of rolls')
        ax_hist.set_ylim([0,Nmax])
        ax_hist.set_yticks(np.arange(0,Nmax+1,1))
    else:
        ax_hist.set_ylabel('share of rolls')
        ax_hist.set_ylim([0,0.25])

    plt.show()