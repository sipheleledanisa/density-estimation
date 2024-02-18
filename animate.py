import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter as writer
from svgd import Kernel, SVGD_model
#If you want to save
#plt.rcParams['animation.ffmpeg_path'] = "C:\\{PATH TO}\\ffmpeg.exe"

# Define prob dist model/form
class MVN():
    """Multivariate normal with derivative on log density function."""
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
    def dlnprob(self, x):
        return -1 * np.matmul((x - self.mean), np.linalg.inv(self.cov))
    def dist(self):
        return multivariate_normal(self.mean,self.cov)
    
# Distributional Details
    
# Bounds for search of random parameters
bd=4
# Initialize distribution components
cov = np.array([[0.2260, 0.1652], [0.1652, 0.6779]])
mean = np.zeros_like(np.random.uniform(-bd,bd,size=(2)))
mvn_model = MVN(mean, cov)
dlnprob = mvn_model.dlnprob

# SVGD initialize and initialization of particles
svgd_model = SVGD_model(Kernel())
x0 = np.random.uniform(-bd,bd,size=(15,2))

# Run & provide summary
n_iter = 2000
x, history,full_history = svgd_model.update(x0, dlnprob, n_iter=n_iter, stepsize=1e-1)
print ("Mean ground truth: ", mean)
print ("Mean obtained by svgd: ", np.mean(x, axis=0))
print("Cov ground truth: ", cov)
print("Cov obtianed by svgd: ", np.cov(x.T))

# Prepare for plots by setting grid and initializing distribution
x = np.linspace(-bd,bd,n_iter)
X, Y = np.meshgrid(x,x)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv = mvn_model.dist()

# Animating function.
def animate(i):
    """
    This will animate the results using 'i' as the frame index.
    We access needed variables through their global scope here, because it is not clear how to pass these through matplotlib's FuncAnimation.
    """
    ax.clear()
    #Make a 3D plot
    ax.set_title('A Simple Particle Estimation Process')
    # Plot distribution for context.
    ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0,alpha=0.5)
    # Plot iterations with points using the full history
    ax.scatter(full_history[i][:,0], full_history[i][:,1],rv.pdf(full_history[i])+1/1000,marker='x', color='k')
    # Set parameters for the plot (optional details)
    ax.view_init(elev=85, azim=0, roll=15)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlim(-bd-1/9,bd+1/9)
    ax.set_ylim(-bd-1/9,bd+1/9)
    ax.grid(False)

# Initialize figure for 3D plotting   
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'},figsize=(10,10))
# Do animation computations
ani = FuncAnimation(fig, animate, interval=10,frames=20)
plt.show()

# If one wants to save, uncomment.
#FFwriter = writer(fps=10)
#ani.save('C:\\{PATH TO}\\animation.mp4', writer = FFwriter)