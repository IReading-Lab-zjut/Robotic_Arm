import numpy as np
from algorithms.GPR import mgpr
from cvxopt import solvers, matrix, spdiag, sqrt, div, exp, spmatrix, log
import matplotlib.pyplot as plt
np.random.seed(5)


class LearnAds:
    def __init__(self, X, Y, b, max_v, likelihood_noise=1e-2):
        '''
        :param X: Inputs, (N, 2)
        :param Y: Outputs, (N, 2)
        :param b: positive scalar, see paper
        :param max_v: positive scalar, see paper
        '''
        self.X = X
        self.Y = Y
        self.b = b
        self.max_v = max_v
        self.original_ads = mgpr(X, Y, likelihood_noise=likelihood_noise)

    def train_original_ads(self):
        self.original_ads.train()

    def ads_evolution(self, x, lf, training_options):
        o_x_dot, _ = self.original_ads.predict_determined_input(x.reshape(1, -1))
        o_x_dot = matrix(o_x_dot)
        max_v = np.max(np.array(self.max_v, self.b * lf.abs_diff(x)))

        def obj(u=None, z=None):
            if u is None:
                return 1, matrix(0.0, (2, 1))
            f_value = matrix(0.0, (2, 1))
            f_gradient = matrix(0.0, (2, 2))
            f_value[0, 0] = sum(u.T * u)
            f_value[1, 0] = sum((u + o_x_dot).T * (u + o_x_dot) - max_v ** 2)
            f_gradient[0, :] = 2 * u.T
            f_gradient[1, :] = 2 * (u + o_x_dot).T
            if z is None:
                return f_value, f_gradient
            I = spmatrix(1.0, range(2), range(2))
            return f_value, f_gradient, 2 * (z[0] + z[1]) * I

        dv_dx = lf.dVdx(x)
        A = matrix(dv_dx, (1, 2))
        b = -self.b / lf.c * lf.V(x) + self.b - dv_dx.dot(np.array(o_x_dot).reshape(-1))
        b = matrix(b, (1, 1))

        solvers.options['feastol'] = training_options['feastol']
        solvers.options['abstol'] = training_options['abstol']
        solvers.options['reltol'] = training_options['reltol']
        solvers.options['maxiters'] = training_options['maxiters']
        solvers.options['show_progress'] = training_options['show_progress']
        u = solvers.cp(obj, A=A, b=b)['x']
        return np.array(o_x_dot).reshape(-1), np.array(u).reshape(-1)

    def plot_vector_field(self, area, lf, training_options, handle=None, gap=1):
        plot_flag = False
        if handle is None:
            handle = plt
            plot_flag = True
        step = area['step']
        x = np.arange(area['x_min'], area['x_max'], step)
        y = np.arange(area['y_min'], area['y_max'], step)
        X, Y = np.meshgrid(x, y)
        length_x = np.shape(x)[0]
        length_y = np.shape(y)[0]
        Dot_x = np.zeros((length_y, length_x))
        Dot_y = np.zeros((length_y, length_x))
        for i in range(length_y):
            for j in range(length_x):
                pose = np.array([x[j], y[i]])
                o_x_dot, u = self.ads_evolution(pose, lf, training_options)
                Dot_x[i, j], Dot_y[i, j] = o_x_dot + u
                # print(t2 - t1)
        # fig, ax = handle.subplots()
        vector_fields = handle.streamplot(X, Y, Dot_x, Dot_y, density=1.0, color='red', linewidth=0.3, maxlength=0.2, minlength=0.1, arrowstyle='simple', arrowsize=0.5)
        demonstration_points = handle.scatter(self.X[0::gap, 0], self.X[0::gap, 1], c='blue', alpha=1.0, s=10, marker='x')
        if plot_flag is True:
            handle.show()
        return vector_fields, demonstration_points
