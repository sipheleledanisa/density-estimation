import numpy as np
from scipy.spatial.distance import pdist, squareform

# Kernel Class
class Kernel():
    """Kernel specifications:
    K(x,y)  = exp((-1/h)*||x-y||**2)
    dK(x,y) = (2/h)*K(x,y)*||x-y||
    """
    def __init__(self):
        pass
    def Kernel(self, x):
        # Take all the points, compute pairwise dists
        init_dist = pdist(x)
        pairwise_dists = squareform(init_dist)
        # Definite the h factor using information that is relevant to the data
        # Helps fix the scale of this constant so that the kernel is meaningful
        h = np.median(pairwise_dists)
        # Define the kernel
        kernel_xj_xi = np.exp(- pairwise_dists ** 2 / h)
        # Get the derivatives
        d_kernal_xi = np.zeros(x.shape)
        for i_index in range(x.shape[0]):
            d_kernal_xi[i_index] = np.matmul(kernel_xj_xi[i_index], (x[i_index] - x)) * 2 / h
        return kernel_xj_xi, d_kernal_xi

# SVGD Class 
class SVGD_model():
    """
    Uses kernel to do gradient updates. This is a vanilla implementation (Stein Variational Gradient Descent: 
    A General Purpose Bayesian Inference Algorithm by Liu et al. (2016))
    """
    def __init__(self,kernel):
        self.kernel= kernel.Kernel
    def update(self, x0, lnprob, n_iter=1000, stepsize=1e-3):
        # Gradient updates for SVGD
        x = np.copy(x0)
        mean_history = [np.mean(x,0)] 
        full_history = [x]
        for iter in range(n_iter):
            # Get kernel information given the data
            kernal_xj_xi, d_kernal_xi = self.kernel(x)
            # Gradient update
            current_grad = (np.matmul(kernal_xj_xi, lnprob(x)) + d_kernal_xi) / x.shape[0]
            # Step in parameter space
            x = x + stepsize * current_grad
            mean_history.append((np.mean(x,0)))
            full_history.append(x)
        return x, mean_history,full_history