# Data-driven energy function learning algorithm
# see our TNNLS paper "Learning an autonomous dynamic
# system to encode periodic human motion skills"
import numpy as np
from algorithms.GPR import sgpr
import matplotlib.pyplot as plt
np.random.seed(5)


def set_normalization(X, alpha):
    '''
    Used to construct the data-set for GP, see Eq. (2) of the paper
    :param X: original inputs, (N, 2)
    :param alpha: positive scalar, see definitions in the paper
    :return: X_ and y, inputs and outputs of the GP model, (N, 2) and (N,)
    '''
    norm = np.sqrt(np.sum(X * X, 1))
    X_ = (1 / norm * X.T).T * alpha
    y = np.log(norm / alpha)
    return X_, y


class LearnEnergyFunction:
    def __init__(self, X, alpha, c, likelihood_noise=0.01):
        '''
        :param X: original inputs, (N, 2)
        :param alpha: positive scalar, see paper
        :param b: positive scalar, see paper
        :param c: positive scalar, see paper
        '''
        self.X = X
        self.X_, self.y = set_normalization(X=X, alpha=alpha)
        self.alpha = alpha
        self.c = c
        self.gp = sgpr(X=self.X_, y=self.y, likelihood_noise=likelihood_noise)

    def train(self, path=None):
        '''
        Training the energy function
        :param path: path to save the parameter
        :return:
        '''
        self.gp.train()
        self.gp.save_param(direction=path)

    def load_para(self, path):
        param = np.loadtxt(path)
        self.gp.set_param(param=param)

    def V(self, x):
        '''
        Compute the energy function value, see Eq.(6)
        :param x: robot position
        :return: V(x)
        '''
        if x is np.zeros(2):
            return 0
        else:
            x_norm = np.sqrt(x.dot(x))
            x_ = x / x_norm * self.alpha
            den, _ = self.gp.predict_determined_input(x_.reshape(1, -1))
            V = x_norm / (np.exp(den[0, 0]) * self.alpha) * self.c
            return V

    def plot_V(self, area, handle=None, gap=1, scatter_flag=True):
        '''
        plot the energy function, only used for the 2-D case
        '''
        plot_flag = False
        if handle is None:
            handle = plt
            plot_flag = True
        step = area['step']
        x = np.arange(area['x_min'], area['x_max'], step)
        y = np.arange(area['y_min'], area['y_max'], step)
        length_x = np.shape(x)[0]
        length_y = np.shape(y)[0]
        X, Y = np.meshgrid(x, y)
        V = np.zeros((length_y, length_x))
        for i in range(length_y):
            for j in range(length_x):
                pose = np.array([x[j], y[i]])
                V[i, j] = self.V(pose)
        levels = np.linspace(np.min(V), np.max(V), 10)
        contour = handle.contour(X, Y, V, levels=levels, alpha=1.0, linewidths=1.0)
        handle.clabel(contour, fontsize=8)

        mark_size = np.ones(np.shape(self.X[0::gap, :])[0]) * 10
        demonstration_points = None
        if scatter_flag is True:
            demonstration_points = handle.scatter(self.X[0::gap, 0], self.X[0::gap, 1], c='blue', alpha=1.0, s=mark_size, marker='x')
        if plot_flag is True:
            handle.show()
        return demonstration_points

    def dVdx(self, x):
        '''
        Compute the gradient of the energy function V
        with respect to x
        '''
        x_norm = np.sqrt(x.dot(x))
        x_ = x / x_norm * self.alpha
        h, _ = self.gp.predict_determined_input(x_.reshape(1, -1))
        g = np.exp(h[0, 0])
        dgdh = g
        dhdx_ = self.gp.gradient2input(x_)
        dgdx_ = dgdh * dhdx_
        num = self.alpha * x * x_norm * g - self.alpha**2 * np.dot((x_norm**2 * np.eye(2) - np.dot(x.reshape(-1, 1), x.reshape(1, -1))), dgdx_)
        den = g**2 * self.alpha**2 * x_norm**2
        return num * self.c / den

    def abs_diff(self, x):
        x_norm = np.sqrt(x.dot(x))
        x_ = x / x_norm * self.alpha
        h, _ = self.gp.predict_determined_input(x_.reshape(1, -1))
        g = np.exp(h[0, 0])
        return 1 / self.c * np.abs(self.alpha * g - x_norm)
