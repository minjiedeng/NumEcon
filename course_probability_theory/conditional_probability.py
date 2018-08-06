from types import SimpleNamespace
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import ipywidgets as widgets

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def bivariate_normal(continuous_update=True):

    widgets.interact(bivariate_normal_,
                    mu1=widgets.FloatSlider(description='$\\mu_1$',min=-1, max=1, step=0.05, value=0, continuous_update=continuous_update),
                    mu2=widgets.FloatSlider(description='$\\mu_2$',min=-1, max=1, step=0.05, value=0, continuous_update=continuous_update),
                    sigma1=widgets.FloatSlider(description='$\\sigma_1$',min=0.05, max=2, step=0.05, value=1, continuous_update=continuous_update),
                    sigma2=widgets.FloatSlider(description='$\\sigma_2$',min=0.05, max=2, step=0.05, value=1.5, continuous_update=continuous_update),
                    rho=widgets.FloatSlider(description='$\\rho$',min=-0.99, max=0.99, step=0.05, value=0.75, continuous_update=continuous_update),
                    y=widgets.FloatSlider(description='$X_2$',min=-3, max=3, step=0.05, value=-1.7, continuous_update=continuous_update))

def bivariate_normal_(mu1,mu2,sigma1,sigma2,rho,y):

    # a. grids
    N_X = 50
    N_Y = 50

    X_vec = np.linspace(-5, 5, N_X)
    Y_vec = np.linspace(-5, 5, N_Y)
    X, Y = np.meshgrid(X_vec, Y_vec)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    pos_cond = np.copy(pos)
    pos_cond[:,:,1] = y

    # b. covariance
    mu = [mu1,mu2]
    sigma = [sigma1,sigma2]
    corr_mat = np.array([[1,rho],[rho,1]])
    diag_sigma = np.diag(sigma)
    cov = diag_sigma@corr_mat@diag_sigma

    # c. pdf
    F = stats.multivariate_normal(mu, cov)
    Z = F.pdf(pos)       
    Z_cond = F.pdf(pos_cond)     

    # d. join distribution
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1,2,1,projection='3d')

    # surface
    ax.plot_surface(X, Y, Z, linewidth=1, cmap=cm.viridis)
    
    # lines
    zmax = 0.20
    xmin = np.min(X_vec)
    xmax = np.max(X_vec)   
    ax.plot([xmin,xmin,xmax,xmax],[y,y,y,y],[0,zmax,zmax,00],color='black',lw=2)
    ax.plot(X_vec,np.repeat(y,N_X),np.mean(Z_cond,axis=0),color='black',lw=2)

    # details
    ax.set_title('joint density of $X_1$ and $X_2$',pad=20)    

    ax.set_xlabel('$X_1$')
    ax.invert_xaxis()
    ax.set_ylabel('$X_2$')

    ax.set_zlabel('density')
    ax.set_zlim(0,0.2)
    ax.set_zticks(np.linspace(0,0.2,5))
    ax.view_init(50, -21)

    # e. marginal distribution
    ax = fig.add_subplot(1,2,2)
    ax.plot(X_vec,np.mean(Z_cond,axis=0),color='black',lw=2)

    ax.set_title('marginal density of $X_1$ given $X_2$')        
    ax.set_ylim(0,0.20)

    plt.show()

