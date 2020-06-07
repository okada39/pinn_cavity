import lib.tf_silent
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from lib.pinn import PINN
from lib.network import Network
from lib.optimizer import L_BFGS_B

def uv(network, xy):
    """
    Compute flow velocities (u, v) for the network with output (psi, p).

    Args:
        xy: network input variables as ndarray.

    Returns:
        (u, v) as ndarray.
    """

    xy = tf.constant(xy)
    with tf.GradientTape() as g:
        g.watch(xy)
        psi_p = network(xy)
    psi_p_j = g.batch_jacobian(psi_p, xy)
    u =  psi_p_j[..., 0, 1]
    v = -psi_p_j[..., 0, 0]
    return u.numpy(), v.numpy()

def contour(grid, x, y, z, title, levels=50):
    """
    Contour plot.

    Args:
        grid: plot position.
        x: x-array.
        y: y-array.
        z: z-array.
        title: title string.
        levels: number of contour lines.
    """

    # get the value range
    vmin = np.min(z)
    vmax = np.max(z)
    # plot a contour
    plt.subplot(grid)
    plt.contour(x, y, z, colors='k', linewidths=0.2, levels=levels)
    plt.contourf(x, y, z, cmap='rainbow', levels=levels, norm=Normalize(vmin=vmin, vmax=vmax))
    plt.title(title)
    cbar = plt.colorbar(pad=0.03, aspect=25, format='%.0e')
    cbar.mappable.set_clim(vmin, vmax)

if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model
    for the cavity flow governed by the steady Navier-Stokes equation.
    """

    # number of training samples
    num_train_samples = 10000
    # number of test samples
    num_test_samples = 100

    # inlet flow velocity
    u0 = 1
    # density
    rho = 1
    # viscosity
    nu = 0.01

    # build a core network model
    network = Network().build()
    network.summary()
    # build a PINN model
    pinn = PINN(network, rho, nu).build()

    # create training input
    xy_eqn = np.random.rand(num_train_samples, 2)
    xy_ub = np.random.rand(num_train_samples//2, 2)
    xy_ub[..., 1] = np.round(xy_ub[..., 1])
    xy_lr = np.random.rand(num_train_samples//2, 2)
    xy_lr[..., 0] = np.round(xy_lr[..., 0])
    xy_bnd = np.random.permutation(np.concatenate([xy_ub, xy_lr]))

    # create training output
    zeros = np.zeros((num_train_samples, 2))
    uv_bnd = np.zeros((num_train_samples, 2))
    uv_bnd[..., 0] = u0 * np.floor(xy_bnd[..., 1])

    # train the model using L-BFGS-B algorithm
    x_train = [xy_eqn, xy_bnd]
    y_train = [zeros, zeros, uv_bnd]
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
    lbfgs.fit()

    # create meshgrid coordinates (x, y) for test plots
    x = np.linspace(0, 1, num_test_samples)
    y = np.linspace(0, 1, num_test_samples)
    x, y = np.meshgrid(x, y)
    xy = np.stack([x.flatten(), y.flatten()], axis=-1)
    # predict (psi, p)
    psi_p = network.predict(xy, batch_size=len(xy))
    psi, p = [ psi_p[..., i].reshape(x.shape) for i in range(psi_p.shape[-1]) ]
    # compute (u, v)
    u, v = uv(network, xy)
    u = u.reshape(x.shape)
    v = v.reshape(x.shape)
    # plot test results
    fig = plt.figure(figsize=(6, 5))
    gs = GridSpec(2, 2)
    contour(gs[0, 0], x, y, psi, 'psi')
    contour(gs[0, 1], x, y, p, 'p')
    contour(gs[1, 0], x, y, u, 'u')
    contour(gs[1, 1], x, y, v, 'v')
    plt.tight_layout()
    plt.show()
